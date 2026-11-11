#include "kvcache.h"
#include <algorithm>
#include <memory>
#include <string>
#include <cassert>
#include <functional>
#include <stdexcept>

// ====== 调度策略 ======
Scheduler::Scheduler(BlockManager& manager, int num_workers)
    : manager_(manager), num_workers_(num_workers > 0 ? num_workers : 1) {}

// 采样器在运行前设置好，取消加锁，防止死锁
int Scheduler::run_sample(const Request& req)
{
    std::function<int(const Request&)> fn;
    {
        fn = sample_func_;
    }
    if(!fn)
    {
        throw std::runtime_error(
            "Scheduler::run_sample: sample_func not set, call set_sampler() first");
    }
    return fn(req);
}

void Scheduler::set_sampler(std::function<int(const Request&)> sampler)
{
    // 采样器只允许在启动前设置：Worker 会并发只读 sample_func_，运行中替换是数据竞争
    assert(!started_.load() && "set_sampler() must be called before start()");
    std::lock_guard<std::mutex> lock(sched_mutex_);
    sample_func_ = std::move(sampler);
}

void Scheduler::set_step_hook(std::function<void(const Scheduler&)> hook)
{
    // 回调在调度线程里执行，运行中被替换同样是数据竞争
    assert(!started_.load() && "set_step_hook() must be called before start()");
    std::lock_guard<std::mutex> lock(sched_mutex_);
    step_hook_ = std::move(hook);
}

std::future<void> Scheduler::add_request(std::shared_ptr<Request> req)
{
    // 已请求停机：不再接收，直接返回一个已完成、状态 ABORTED 的 future
    if(stop_flag_.load() || req->all_tokens.empty() || req->max_new_tokens <= 0)
    {
        req->s = ReqState::ABORTED;
        std::promise<void> p;
        auto f = p.get_future();
        p.set_value();
        return f;
    }

    // 对inbox_和pending_操作，加锁
    std::unique_lock<std::mutex> lock1(sched_mutex_);
    std::unique_lock<std::mutex> lock2(pending_mutex_);
    if(stop_flag_.load()) // 拿锁期间可能刚好被 stop
    {
        lock2.unlock();
        lock1.unlock();
        req->s = ReqState::ABORTED;
        std::promise<void> p;
        auto f = p.get_future();
        p.set_value();
        return f;
    }
    int id = next_arrival_id_++;
    req->arrival_id = id;
    auto [it, ok] = pending_.try_emplace(id);
    assert(ok && "add_request: duplicated arrival_id");
    inbox_.push_back(req);
    unfinished++;
    sched_cv_.notify_all();
    return it->second.done.get_future();
}

// 判断请求是否能继续decode
bool Scheduler::can_append(std::shared_ptr<Request> req)
{
    int offset = req->sequence_length % BLOCK_SIZE;

    // 是否需要分配新物理块（新增 / 写时复制）
    bool if_new;
    if(offset == 0 || req->block_table.empty())
    {
        // 新块边界：必然要新块
        // block_table 为空（防御 seq_len == 0 或外部误用）：没有尾块可写，同样要新块
        if_new = true;
    }
    else
    {
        int tail_block_physical_id = req->block_table.back();
        auto& tail_block = manager_.get_allocator()->get_block(tail_block_physical_id);
        if_new = (tail_block.ref_count > 1);
    }

    bool if_add = (!if_new || manager_.get_allocator()->num_free_blocks() >= 1);
    return if_add;
}

int Scheduler::blocks_needed_for_prefill(std::shared_ptr<Request> req)
{
    int total = (req->all_tokens.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    auto match = manager_.match_prefix(req->all_tokens);
    auto allocator_ = manager_.get_allocator();
    int match_block_num{0};
    for(auto it : match.cached_blocks) // 只有命中且ref_count > 0 的块才不占空间
    {
        const auto& block = allocator_->get_block(it);
        if(block.ref_count > 0) match_block_num++;
    }

    int block_needed = total - match_block_num;
    return block_needed;
}

void Scheduler::preempt()
{
    // 前提条件：能重算，不满足就是逻辑bug，直接断言
    auto victim = running_queue_.back();
    bool re_com_able = (static_cast<int>(victim->all_tokens.size()) == victim->sequence_length);
    assert(re_com_able && "preempt: victim cannot be recomputed, invalid precondition");

    // 释放victim的资源
    manager_.release(*victim);
    victim->s = ReqState::WAITING;
    auto temp = running_queue_.back();
    running_queue_.pop_back();
    waiting_queue_.push_front(temp); // 放入队首，以便下次调度时优先执行
    ++num_preemptions_;
}

void Scheduler::worker_loop(int id)
{
    uint64_t seen = 0; // 该 Worker 已完成到第几轮；局部变量，每个线程各一份

    while(true)
    {
        const std::vector<std::shared_ptr<Request>>* batch = nullptr;
        std::vector<int>* out = nullptr;
        size_t begin = 0, end = 0;

        {
            std::unique_lock<std::mutex> lock(pool_mutex_);
            task_cv_.wait(lock, [&]
                          { return workers_stop_.load() || round_ != seen; });
            if(round_ == seen) return; // 只可能是停机
            seen = round_;
            batch = cur_batch_;
            out = cur_out_;
            begin = cur_begin_[id];
            end = cur_end_[id];
        }

        // 计算时不持锁，否则并行退化成串行
        for(size_t i = begin; batch && out && i < end; ++i)
        {
            try
            {
                (*out)[i] = run_sample(*(*batch)[i]);
            }
            catch(...)
            {
                std::lock_guard<std::mutex> lk(pool_mutex_);
                if(!worker_error_) worker_error_ = std::current_exception();
            }
        }

        {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            if(--remaining_ == 0) done_cv_.notify_one();
        }
    }
}

void Scheduler::sample_batch_parallel(const std::vector<std::shared_ptr<Request>>& batch, std::vector<int>& out)
{
    out.assign(batch.size(), 0);
    if(batch.empty()) return; // 空批次不翻牌，否则 remaining_ 与唤醒次数对不上

    const int n = static_cast<int>(batch.size());

    // 单 Worker 或池未启动：串行，行为与旧版完全一致
    if(num_workers_ <= 1 || !pool_running_)
    {
        for(int i = 0; i < n; ++i)
            out[i] = run_sample(*batch[i]);
        return;
    }

    std::unique_lock<std::mutex> lock(pool_mutex_);
    cur_batch_ = &batch;
    cur_out_ = &out;
    cur_begin_.resize(num_workers_);
    cur_end_.resize(num_workers_);

    const int chunk = n / num_workers_;
    const int extra = n % num_workers_;
    size_t pos = 0;
    for(int w = 0; w < num_workers_; ++w)
    {
        size_t len = static_cast<size_t>(chunk) + (w < extra ? 1 : 0);
        cur_begin_[w] = pos;
        pos += len;
        cur_end_[w] = pos;
    }

    worker_error_ = nullptr;
    remaining_ = num_workers_; // 必须在 notify 之前设好；空区间的 Worker 也会减
    ++round_;                  // 翻牌
    task_cv_.notify_all();
    done_cv_.wait(lock, [&]
                  { return remaining_ == 0; });

    cur_batch_ = nullptr;
    cur_out_ = nullptr;

    if(worker_error_)
    {
        auto e = worker_error_;
        worker_error_ = nullptr;
        std::rethrow_exception(e);
    }
}
void Scheduler::step()
{
    std::lock_guard<std::mutex> step_lock(step_mutex_); // 同一时刻只允许一个线程推进调度

    BlockAllocator* allocator = manager_.get_allocator();
    int num_blocks = allocator->num_blocks();

    // 阶段一：先清空收件箱进入waiting_queue
    std::vector<PendingCompletion> finished; // 收集需在锁外唤醒的 promise
    {
        std::unique_lock<std::mutex> lock(sched_mutex_);     // 对inbox_操作（与add_request竞争）
        std::unique_lock<std::mutex> lock_p(pending_mutex_); // 对pending_操作
        std::unique_lock<std::mutex> lock_w(waiting_mutex_); // 对waiting_queue_操作（与waiting()竞争）
        while(!inbox_.empty())
        {
            auto temp = inbox_.begin();
            int N = (*temp)->all_tokens.size();
            int max_ = (*temp)->max_new_tokens;
            if((N + max_ + BLOCK_SIZE - 1) / BLOCK_SIZE > num_blocks - WATER_MARK)
            {
                int id = (*temp)->arrival_id;
                (*temp)->s = ReqState::ABORTED;
                auto node = pending_.extract(id);
                if(!node.empty()) finished.push_back(std::move(node.mapped()));
                inbox_.erase(temp);
                unfinished--;
                continue;
            }
            waiting_queue_.push_back(*temp);
            inbox_.erase(temp);
        }
    }
    // 锁外 set_value：唤醒的调用方不会卡在 add_request 的锁上
    for(auto& pc : finished)
        pc.done.set_value();
    if(!finished.empty() && unfinished.load() == 0) sched_cv_.notify_all();

    // 阶段二：挨个看看是否该抢占
    bool preempted = false;
    std::vector<std::shared_ptr<Request>> batch; // 本step能执行的请求

    for(size_t i = 0; i < running_queue_.size(); ++i)
    {
        while(!can_append(running_queue_[i]))
        {
            preempt();
            preempted = true;
            if(i >= running_queue_.size()) break; // 抢占自己不行
        }
        if(i < running_queue_.size())
        {
            append_slot(manager_, *running_queue_[i]);
            batch.push_back(running_queue_[i]);
        }
    }

    // 阶段三：判断新请求是否该加入进来
    if(!preempted)
    {
        // 没发生抢占，说明空间还很多，可以加入新的请求
        // 首先进行挨个判空
        std::lock_guard<std::mutex> lock_w(waiting_mutex_);
        while(!waiting_queue_.empty())
        {
            auto temp = waiting_queue_.front();
            int block_needed = blocks_needed_for_prefill(temp);
            if(block_needed + WATER_MARK > allocator->num_free_blocks()) break; // 判断进去后是否要抢占，要抢的话还不如不进
            // 为什么不是continue呢，如果放过大的去塞小的，那么大的就一直轮不到了
            // prefill新请求，加入running_queue_和batch等待运行吧
            bool ok = prefill(temp->all_tokens, *temp, manager_);
            assert(ok);
            temp->s = ReqState::RUNNING;
            waiting_queue_.pop_front();
            running_queue_.push_back(temp);
            // 此时先不放入batch，因为没有进行新slot的计算，留着下一轮再decode
        }
    }

    // 阶段四：先跑采样（这一步不需要任何锁，Worker 只读 batch、各写各的下标）
    std::vector<int> new_tokens;
    sample_batch_parallel(batch, new_tokens);

    // 收集已结束请求的 promise，锁外统一 fire
    finished.clear();
    {
        std::unique_lock<std::mutex> lock_p(pending_mutex_); // 保护 pending_
        std::unique_lock<std::mutex> lock_r(running_mutex_); // 保护 running_queue_

        for(int i = 0; i < (int)batch.size(); i++)
        {
            int new_token = new_tokens[i];
            append_token(manager_, *batch[i], new_token);
            batch[i]->num_generated_tokens++;

            if(batch[i]->num_generated_tokens >= batch[i]->max_new_tokens)
            {
                batch[i]->s = ReqState::FINISHED;
                manager_.release(*batch[i]); // 内部只用 alloc_mutex_，不冲突

                auto it = std::find(running_queue_.begin(), running_queue_.end(), batch[i]);
                if(it != running_queue_.end()) running_queue_.erase(it);
                unfinished--; // 结束就一定减，与队列无关

                auto node = pending_.extract(batch[i]->arrival_id);
                if(!node.empty()) finished.push_back(std::move(node.mapped()));
            }
        }
    }
    // 两把锁在这里析构释放

    // 锁外 set_value：唤醒不再堵 add_request
    for(auto& pc : finished)
    {
        pc.done.set_value();
    }
    if(!finished.empty() && unfinished.load() == 0) sched_cv_.notify_all();

    // 本步到此结束：Worker 已全部收工、锁已释放，可以安全地做检查/统计
    ++num_steps_;
    if(step_hook_) step_hook_(*this);
}

void Scheduler::run()
{
    while(true)
    {
        {
            std::unique_lock<std::mutex> lk(sched_mutex_);
            sched_cv_.wait(lk, [&]
                           { return stop_flag_.load() || unfinished.load() > 0; });
            if(stop_flag_.load()) break;
        } // 释放 sched_mutex_，因为 step 的阶段一要用
        step();
    }
    // 停机后不再有线程修改队列/pending_，统一清理
    abort_all_pending();
}

void Scheduler::start_pool()
{
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if(num_workers_ <= 1 || pool_running_) return; // 单 Worker 走串行，无需池

    pool_running_ = true;
    workers_stop_ = false;
    round_ = 0;
    remaining_ = 0;
    cur_begin_.assign(num_workers_, 0);
    cur_end_.assign(num_workers_, 0);
    workers_.reserve(num_workers_);
    for(int i = 0; i < num_workers_; ++i)
        workers_.emplace_back(&Scheduler::worker_loop, this, i);
}

void Scheduler::stop_pool()
{
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        if(!pool_running_) return;
        workers_stop_ = true;
    }
    task_cv_.notify_all();
    for(auto& t : workers_)
        if(t.joinable()) t.join();
    workers_.clear();
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        pool_running_ = false;
        remaining_ = 0;
    }
}

void Scheduler::start()
{
    if(sched_thread_.joinable()) return; // 已经启动过
    assert(sample_func_ && "start(): sampler not set, call set_sampler() first");
    started_ = true;

    // 先启动 Worker 池：调度线程一启动就可能调用 step() 派活
    start_pool();
    sched_thread_ = std::thread(&Scheduler::run, this);
}

void Scheduler::stop(bool drain)
{
    {
        std::unique_lock<std::mutex> lk(sched_mutex_);
        if(!stop_flag_.load())
        {
            if(drain && sched_thread_.joinable())
            {
                // 等待调度线程把已提交的请求跑完（run() 会在 unfinished>0 时继续 step）
                sched_cv_.wait(lk, [&]
                               { return unfinished.load() == 0; });
            }
            stop_flag_ = true;
        }
    }
    sched_cv_.notify_all();

    // 顺序很重要：先停调度线程（它可能正卡在 done_cv_ 上等 Worker 收工）
    if(sched_thread_.joinable() && sched_thread_.get_id() != std::this_thread::get_id())
        sched_thread_.join();

    // 再停 Worker 池
    stop_pool();

    // 未启动线程时（纯 step() 驱动），也要保证 pending_ 被清空
    abort_all_pending();
}

Scheduler::~Scheduler()
{
    stop(true);
}

void Scheduler::abort_all_pending()
{
    std::vector<PendingCompletion> to_fire;
    {
        std::unique_lock<std::mutex> lock1(sched_mutex_);
        std::unique_lock<std::mutex> lock2(pending_mutex_);
        std::unique_lock<std::mutex> lock_r(running_mutex_);
        std::unique_lock<std::mutex> lock_w(waiting_mutex_);

        for(auto& req : running_queue_)
        {
            if(req->s != ReqState::FINISHED)
            {
                manager_.release(*req);
                req->s = ReqState::ABORTED;
            }
        }
        for(auto& req : waiting_queue_)
        {
            if(req->s != ReqState::FINISHED) req->s = ReqState::ABORTED;
        }
        for(auto& req : inbox_) // 还没进入等待队列的也算未完成
        {
            if(req->s != ReqState::FINISHED) req->s = ReqState::ABORTED;
        }
        running_queue_.clear();
        waiting_queue_.clear();
        inbox_.clear();

        for(auto& kv : pending_)
        {
            to_fire.push_back(std::move(kv.second));
        }
        pending_.clear();
        unfinished.store(0);
    }
    for(auto& pc : to_fire)
        pc.done.set_value();
    sched_cv_.notify_all();
}
