#include "kvcache.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <memory>
#include <string>
#include <cassert>
#include <functional>
#include <stdexcept>

// ====== 调度策略 ======
Scheduler::Scheduler(BlockManager& manager, int num_workers)
    : Scheduler(manager, SchedulerConfig{num_workers, DEFAULT_WATER_MARK, -1.0, 0, 0, true, 0}) {}

Scheduler::Scheduler(BlockManager& manager, SchedulerConfig config)
    : manager_(manager), config_(config), num_workers_(config.num_workers > 0 ? config.num_workers : 1)
{
    config_.num_workers = num_workers_;
    if(config_.watermark_blocks < 0) config_.watermark_blocks = 0;
    if(config_.max_num_batched_tokens < 0) config_.max_num_batched_tokens = 0;
    if(config_.max_num_seqs < 0) config_.max_num_seqs = 0;
    if(config_.waiting_token_reserve < 0) config_.waiting_token_reserve = 0;
}

int Scheduler::watermark_blocks() const
{
    const int total = manager_.get_allocator()->num_blocks();
    if(config_.watermark_ratio >= 0.0)
    {
        int wm = static_cast<int>(std::ceil(static_cast<double>(total) * config_.watermark_ratio));
        return std::max(0, std::min(wm, std::max(0, total - 1)));
    }
    return config_.watermark_blocks;
}

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
    const int blk_size = manager_.block_size();
    int offset = req->sequence_length % blk_size;

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
    const int blk_size = manager_.block_size();
    int total = (static_cast<int>(req->all_tokens.size()) + blk_size - 1) / blk_size;
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
    // 分块预填充下，running_queue_ 中的请求可能处于“prefill 到一半”状态
    // （num_computed_tokens < all_tokens.size()），因此前置条件为 num_computed_tokens <= all_tokens.size()。
    // 抢占只释放物理块并清零 num_computed_tokens；已登记进前缀缓存的满块仍留在 CACHED 空闲链表中，
    // 下次恢复时 match_prefix() 会自动复用存活块并从断点续算剩余 chunk。
    auto victim = running_queue_.back();
    assert(victim->num_computed_tokens <= static_cast<int>(victim->all_tokens.size()) &&
           "preempt: victim cannot be recomputed, invalid precondition");

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
    (void)id;
    uint64_t seen = 0; // 该 Worker 已完成到第几轮；局部变量，每个线程各一份

    while(true)
    {
        const std::vector<std::shared_ptr<Request>>* batch = nullptr;
        std::vector<int>* out = nullptr;
        size_t batch_size = 0;

        {
            std::unique_lock<std::mutex> lock(pool_mutex_);
            task_cv_.wait(lock, [&]
                          { return workers_stop_.load() || round_ != seen; });
            if(round_ == seen) return; // 只可能是停机
            seen = round_;
            batch = cur_batch_;
            out = cur_out_;
            batch_size = cur_batch_size_;
        }

        // 动态分发：共享原子游标按需领取任务，消除静态分片的长尾空等
        while(batch && out)
        {
            size_t i = next_task_idx_.fetch_add(1, std::memory_order_relaxed);
            if(i >= batch_size) break;
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

    const size_t n = batch.size();

    // 单 Worker、池未启动或单请求批次：直接串行，避免无意义的线程唤醒开销
    if(num_workers_ <= 1 || !pool_running_ || n == 1)
    {
        for(size_t i = 0; i < n; ++i)
            out[i] = run_sample(*batch[i]);
        return;
    }

    std::unique_lock<std::mutex> lock(pool_mutex_);
    cur_batch_ = &batch;
    cur_out_ = &out;
    cur_batch_size_ = n;
    next_task_idx_.store(0, std::memory_order_relaxed);

    worker_error_ = nullptr;
    remaining_ = num_workers_; // 必须在 notify 之前设好；没领到任务的 Worker 也会减
    ++round_;                  // 翻牌
    task_cv_.notify_all();
    done_cv_.wait(lock, [&]
                  { return remaining_ == 0; });

    cur_batch_ = nullptr;
    cur_out_ = nullptr;
    cur_batch_size_ = 0;

    if(worker_error_)
    {
        auto e = worker_error_;
        worker_error_ = nullptr;
        std::rethrow_exception(e);
    }
}
SchedulerOutput Scheduler::schedule()
{
    std::lock_guard<std::mutex> step_lock(step_mutex_);
    return schedule_locked();
}

void Scheduler::execute(SchedulerOutput& output)
{
    std::lock_guard<std::mutex> step_lock(step_mutex_);
    execute_locked(output);
}

void Scheduler::step()
{
    std::lock_guard<std::mutex> step_lock(step_mutex_); // 同一时刻只允许一个线程推进调度
    SchedulerOutput output = schedule_locked();
    execute_locked(output);
}

SchedulerOutput Scheduler::schedule_locked()
{
    SchedulerOutput output;
    BlockAllocator* allocator = manager_.get_allocator();
    const int num_blocks = allocator->num_blocks();
    const int blk_size = manager_.block_size();
    const int watermark = watermark_blocks();
    const int cur_step = static_cast<int>(num_steps_.load());

    const bool has_token_budget = (config_.max_num_batched_tokens > 0);
    const bool chunked = (has_token_budget && config_.enable_chunked_prefill);

    // 阶段一：先清空收件箱进入 waiting_queue_
    // 本步内准入拒绝的 promise 放入 output.aborted_promises，统一由 execute_locked 在 step_hook_ 之后锁外唤醒
    {
        std::unique_lock<std::mutex> lock(sched_mutex_);     // 对inbox_操作（与add_request竞争）
        std::unique_lock<std::mutex> lock_p(pending_mutex_); // 对pending_操作
        std::unique_lock<std::mutex> lock_w(waiting_mutex_); // 对waiting_queue_操作（与waiting()竞争）
        while(!inbox_.empty())
        {
            auto temp = inbox_.begin();
            int N = static_cast<int>((*temp)->all_tokens.size());
            int max_ = (*temp)->max_new_tokens;
            if((N + max_ + blk_size - 1) / blk_size > num_blocks - watermark)
            {
                int id = (*temp)->arrival_id;
                (*temp)->s = ReqState::ABORTED;
                auto node = pending_.extract(id);
                if(!node.empty()) output.aborted_promises.push_back(std::move(node.mapped().done));
                inbox_.erase(temp);
                unfinished--;
                continue;
            }
            waiting_queue_.push_back(*temp);
            inbox_.erase(temp);
        }
    }

    int remaining_budget = chunked ? config_.max_num_batched_tokens : INT_MAX;

    // 阶段二：running 优先（处理正在 decode 或续算 chunked prefill 的请求，必要时抢占队尾）
    // 严格遵守全局锁顺序：running_mutex_ -> waiting_mutex_
    {
        std::unique_lock<std::mutex> lock_r(running_mutex_);
        std::unique_lock<std::mutex> lock_w(waiting_mutex_);

        // 若配置了 waiting_token_reserve 且 waiting 队列有资格准入，为新请求保留一小份预算防饥饿
        int reserve = 0;
        if(chunked && config_.waiting_token_reserve > 0 && !waiting_queue_.empty() &&
           (config_.max_num_seqs <= 0 || static_cast<int>(running_queue_.size()) < config_.max_num_seqs))
        {
            reserve = std::min(config_.waiting_token_reserve, std::max(0, config_.max_num_batched_tokens - 1));
        }
        int running_budget = chunked ? (remaining_budget - reserve) : INT_MAX;

        for(size_t i = 0; i < running_queue_.size(); ++i)
        {
            if(chunked && running_budget <= 0) break;

            auto req = running_queue_[i];
            if(!req->is_prefill_done())
            {
                // 分块预填充续算分支
                int rem_prefill = static_cast<int>(req->all_tokens.size()) - req->num_computed_tokens;
                int chunk_new = chunked ? std::min(rem_prefill, running_budget) : rem_prefill;
                if(chunk_new <= 0) continue;

                int old_len = req->num_computed_tokens;
                int target_blocks = (old_len + chunk_new + blk_size - 1) / blk_size;
                int extra_blocks = std::max(0, target_blocks - static_cast<int>(req->block_table.size()));
                if(old_len % blk_size != 0 && !req->block_table.empty() &&
                   allocator->get_block(req->block_table.back()).ref_count > 1)
                {
                    ++extra_blocks; // 防御性 CoW 所需块
                }

                while(allocator->num_free_blocks() < extra_blocks)
                {
                    preempt();
                    output.preempted = true;
                    ++output.num_preemptions;
                    if(i >= running_queue_.size()) break;
                }
                if(i < running_queue_.size())
                {
                    int computed = 0;
                    bool ok = prefill_chunk(*req, chunk_new, manager_, nullptr, nullptr, &computed);
                    assert(ok);
                    output.prefill_reqs.push_back(SchedulerOutput::ScheduledPrefill{req, 0, computed, false});
                    output.num_prefill_tokens += computed;
                    if(chunked)
                    {
                        running_budget -= computed;
                        remaining_budget -= computed;
                    }
                }
            }
            else
            {
                // Decode 分支：每请求消耗 1 token 预算
                while(!can_append(req))
                {
                    preempt();
                    output.preempted = true;
                    ++output.num_preemptions;
                    if(i >= running_queue_.size()) break; // 抢占自己不行
                }
                if(i < running_queue_.size())
                {
                    append_slot(manager_, *req);
                    output.decode_reqs.push_back(req);
                    output.num_decode_tokens += 1;
                    if(chunked)
                    {
                        running_budget -= 1;
                        remaining_budget -= 1;
                    }
                }
            }
        }
    }

    // 阶段三：准入新请求（仅在阶段二未发生抢占时执行，同时受 max_num_seqs 与 token budget 约束）
    if(!output.preempted)
    {
        std::unique_lock<std::mutex> lock_r(running_mutex_);
        std::unique_lock<std::mutex> lock_w(waiting_mutex_);
        while(!waiting_queue_.empty())
        {
            if(config_.max_num_seqs > 0 && static_cast<int>(running_queue_.size()) >= config_.max_num_seqs)
                break;

            auto temp = waiting_queue_.front();
            // 只算一次 match_prefix，复用到 prefill_chunk() 避免重复哈希
            auto match = manager_.match_prefix(temp->all_tokens);
            const int total_prompt = static_cast<int>(temp->all_tokens.size());
            const int cached_tokens = match.num_matched_tokens;
            const int uncached_tokens = std::max(0, total_prompt - cached_tokens);

            int chunk_new = uncached_tokens;
            if(chunked)
            {
                if(uncached_tokens > 0 && remaining_budget <= 0) break;
                chunk_new = std::min(uncached_tokens, remaining_budget);
            }

            const int new_len = cached_tokens + chunk_new;
            const int target_blocks = (new_len + blk_size - 1) / blk_size;
            int active_hit = 0;
            for(int id : match.cached_blocks)
            {
                if(allocator->get_block(id).ref_count > 0) ++active_hit;
            }
            const int block_needed = target_blocks - active_hit;
            if(block_needed + watermark > allocator->num_free_blocks()) break; // 空间不足则停止准入，避免大请求饥饿

            int out_cached = 0;
            int out_computed = 0;
            bool ok = prefill_chunk(*temp, chunk_new, manager_, &match, &out_cached, &out_computed);
            assert(ok);
            temp->s = ReqState::RUNNING;
            if(temp->admit_step < 0) temp->admit_step = cur_step;
            waiting_queue_.pop_front();
            running_queue_.push_back(temp);

            output.prefill_reqs.push_back(SchedulerOutput::ScheduledPrefill{temp, out_cached, out_computed, true});
            output.num_prefill_tokens += out_computed;
            if(chunked) remaining_budget -= out_computed;
        }
    }

    return output;
}

void Scheduler::execute_locked(SchedulerOutput& output)
{
    const int cur_step = static_cast<int>(num_steps_.load());

    // 阶段四：先跑并行采样（不持任何锁，Worker 只读 decode_reqs、各写各的下标）
    std::vector<int> new_tokens;
    sample_batch_parallel(output.decode_reqs, new_tokens);

    std::vector<std::promise<void>> finished;
    for(auto& p : output.aborted_promises)
        finished.push_back(std::move(p));
    output.aborted_promises.clear();

    {
        std::unique_lock<std::mutex> lock_p(pending_mutex_); // 保护 pending_
        std::unique_lock<std::mutex> lock_r(running_mutex_); // 保护 running_queue_

        for(int i = 0; i < static_cast<int>(output.decode_reqs.size()); i++)
        {
            auto& req = output.decode_reqs[i];
            int new_token = new_tokens[i];
            if(req->first_token_step < 0) req->first_token_step = cur_step;
            append_token(manager_, *req, new_token);
            req->num_generated_tokens++;

            if(req->num_generated_tokens >= req->max_new_tokens)
            {
                req->finish_step = cur_step;
                req->s = ReqState::FINISHED;
                manager_.release(*req); // 内部只用 alloc_mutex_，不冲突

                auto it = std::find(running_queue_.begin(), running_queue_.end(), req);
                if(it != running_queue_.end()) running_queue_.erase(it);
                unfinished--; // 结束就一定减，与队列无关

                auto node = pending_.extract(req->arrival_id);
                if(!node.empty()) finished.push_back(std::move(node.mapped().done));
            }
        }
    }
    // 两把锁在这里析构释放

    last_step_prefill_tokens_.store(output.num_prefill_tokens, std::memory_order_relaxed);
    last_step_decode_tokens_.store(output.num_decode_tokens, std::memory_order_relaxed);

    // 本步到此结束：Worker 已全部收工、锁已释放，可以安全地做检查/统计
    ++num_steps_;
    if(step_hook_) step_hook_(*this);

    // 锁外 set_value：唤醒不再堵 add_request。
    // 必须放在 step_hook_ 之后，保证本步的全部写入都 happens-before 调用方 wait() 返回
    for(auto& p : finished)
    {
        p.set_value();
    }
    if(!finished.empty() && unfinished.load() == 0) sched_cv_.notify_all();
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
    cur_batch_size_ = 0;
    next_task_idx_.store(0, std::memory_order_relaxed);
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
