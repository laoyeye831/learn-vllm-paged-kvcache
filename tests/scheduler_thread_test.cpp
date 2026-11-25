#include "test_util.h"

#include <chrono>
#include <memory>
#include <thread>

using namespace std::chrono;

// =========================================================
// 阶段四（M2）：run() 生命周期与完成通知
// 所有等待都带超时，出错时测试失败而不是挂死
// =========================================================

static std::shared_ptr<Request> make_req(int num_tokens, int base, int max_new)
{
    auto req = std::make_shared<Request>();
    req->all_tokens.resize(num_tokens);
    for(int i = 0; i < num_tokens; ++i)
        req->all_tokens[i] = base + i;
    req->max_new_tokens = max_new;
    return req;
}

// 等待 future 就绪，超时返回 false
static bool wait_ready(std::future<void>& f, milliseconds timeout = milliseconds(5000))
{
    return f.wait_for(timeout) == std::future_status::ready;
}

void test_thread_basic_flow()
{
    BlockAllocator alloc(64);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr);
    sched.set_sampler(fake_sample);
    sched.start();

    std::vector<std::shared_ptr<Request>> reqs;
    std::vector<std::future<void>> futures;
    for(int i = 0; i < 3; ++i)
    {
        auto r = make_req(16, 1000 * (i + 1), 20);
        reqs.push_back(r);
        futures.push_back(sched.add_request(r));
    }

    bool all_ready = true;
    for(auto& f : futures)
        all_ready = wait_ready(f) && all_ready;
    EXPECT_TRUE(all_ready);

    for(auto& r : reqs)
    {
        EXPECT_TRUE(r->s == ReqState::FINISHED);
        EXPECT_EQ(r->num_generated_tokens, 20);
        EXPECT_EQ((int)r->all_tokens.size(), 36);
    }

    sched.stop(true); // 已跑完，立即返回
    EXPECT_FALSE(sched.has_work());
    EXPECT_EQ(alloc.num_free_blocks(), 64);
}

// future 只能在请求真正结束时就绪，不能在生成第一个 token 时就绪
void test_thread_future_ready_only_at_end()
{
    BlockAllocator alloc(64);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr);
    // 采样加 1ms 延迟，放大"提前就绪"的窗口
    sched.set_sampler([](const Request& r)
                      {
        std::this_thread::sleep_for(milliseconds(1));
        return fake_sample(r); });
    sched.start();

    auto r = make_req(16, 5000, 30);
    auto f = sched.add_request(r);
    EXPECT_TRUE(wait_ready(f));

    // 就绪时一定已经全部生成完
    EXPECT_EQ(r->num_generated_tokens, 30);
    EXPECT_EQ((int)r->all_tokens.size(), 46);
    EXPECT_TRUE(r->s == ReqState::FINISHED);

    sched.stop(true);
    EXPECT_EQ(alloc.num_free_blocks(), 64);
}

void test_thread_concurrent_producers()
{
    BlockAllocator alloc(128);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr);
    sched.set_sampler(fake_sample);
    sched.start();

    constexpr int kProducers = 4;
    constexpr int kPerProducer = 10;
    std::vector<std::vector<std::shared_ptr<Request>>> reqs(kProducers);
    std::vector<std::vector<std::future<void>>> futures(kProducers);

    std::vector<std::thread> producers;
    for(int p = 0; p < kProducers; ++p)
    {
        producers.emplace_back([&, p]
                               {
            for (int i = 0; i < kPerProducer; ++i) {
                auto r = make_req(8 + i, 100 * (p + 1) + i * 7, 5 + i);
                reqs[p].push_back(r);
                futures[p].push_back(sched.add_request(r));
            } });
    }
    for(auto& t : producers)
        t.join();

    bool all_ready = true;
    for(int p = 0; p < kProducers; ++p)
        for(auto& f : futures[p])
            all_ready = wait_ready(f) && all_ready;
    EXPECT_TRUE(all_ready);

    int finished = 0;
    for(int p = 0; p < kProducers; ++p)
        for(auto& r : reqs[p])
        {
            if(r->s == ReqState::FINISHED) ++finished;
            EXPECT_EQ(r->num_generated_tokens, r->max_new_tokens);
        }
    EXPECT_EQ(finished, kProducers * kPerProducer);

    sched.stop(true);
    EXPECT_FALSE(sched.has_work());
    EXPECT_EQ(alloc.num_free_blocks(), 128);
}

// 空闲时不应忙等：没有请求时 stop(true) 应该很快返回
void test_thread_idle_stop_is_fast()
{
    BlockAllocator alloc(16);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr);
    sched.set_sampler(fake_sample);
    sched.start();

    std::this_thread::sleep_for(milliseconds(200)); // 期间调度器应处于等待状态

    auto t0 = steady_clock::now();
    sched.stop(true);
    auto elapsed = duration_cast<milliseconds>(steady_clock::now() - t0);
    EXPECT_TRUE(elapsed < milliseconds(1000));
    EXPECT_FALSE(sched.has_work());
}

// 立即停机：所有已提交请求都必须被通知，不能有 future 永远不就绪
void test_thread_stop_without_drain_notifies_all()
{
    BlockAllocator alloc(16);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr);
    sched.set_sampler([](const Request& r)
                      {
        std::this_thread::sleep_for(milliseconds(1));
        return fake_sample(r); });
    sched.start();

    std::vector<std::shared_ptr<Request>> reqs;
    std::vector<std::future<void>> futures;
    for(int i = 0; i < 50; ++i)
    {
        auto r = make_req(16, 300 * (i + 1), 40);
        reqs.push_back(r);
        futures.push_back(sched.add_request(r));
    }

    sched.stop(false); // 不排空，立刻退出

    bool all_ready = true;
    for(auto& f : futures)
        all_ready = wait_ready(f, milliseconds(5000)) && all_ready;
    EXPECT_TRUE(all_ready);

    for(auto& r : reqs)
        EXPECT_TRUE(r->s == ReqState::FINISHED || r->s == ReqState::ABORTED);

    EXPECT_FALSE(sched.has_work());
    EXPECT_EQ(alloc.num_free_blocks(), 16); // 被中止的请求也必须归还块
}

// 停机后 add_request 立即得到已完成的 future，且状态为 ABORTED
void test_thread_add_request_after_stop()
{
    BlockAllocator alloc(16);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr);
    sched.set_sampler(fake_sample);
    sched.start();
    sched.stop(true);

    auto r = make_req(16, 1234, 5);
    auto f = sched.add_request(r);
    EXPECT_TRUE(wait_ready(f, milliseconds(1000)));
    EXPECT_TRUE(r->s == ReqState::ABORTED);
    EXPECT_EQ(alloc.num_free_blocks(), 16);
}

// =========================================================
// M3：Worker 池
// =========================================================

// 跑一批请求，返回每个请求最终的 all_tokens，用于比较不同 worker 数的输出
static std::vector<std::vector<int>> run_with_workers(int num_workers, int pool_blocks,
                                                      std::chrono::microseconds sample_delay)
{
    BlockAllocator alloc(pool_blocks);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr, num_workers);
    if(sample_delay.count() > 0)
    {
        sched.set_sampler([sample_delay](const Request& r)
                          {
            std::this_thread::sleep_for(sample_delay);
            return fake_sample(r); });
    }
    else
    {
        sched.set_sampler(fake_sample);
    }
    sched.start();

    std::vector<std::shared_ptr<Request>> reqs;
    std::vector<std::future<void>> futures;
    for(int i = 0; i < 8; ++i)
    {
        auto r = make_req(16, 700 * (i + 1), 25);
        reqs.push_back(r);
        futures.push_back(sched.add_request(r));
    }
    for(auto& f : futures)
        wait_ready(f);

    std::vector<std::vector<int>> out;
    for(auto& r : reqs)
        out.push_back(r->all_tokens);

    sched.stop(true);
    EXPECT_EQ(alloc.num_free_blocks(), pool_blocks);
    return out;
}

void test_pool_output_identical()
{
    auto serial = run_with_workers(1, 64, microseconds(0));
    auto parallel2 = run_with_workers(2, 64, microseconds(0));
    auto parallel4 = run_with_workers(4, 64, microseconds(0));

    EXPECT_EQ((int)serial.size(), 8);
    for(size_t i = 0; i < serial.size(); ++i)
    {
        EXPECT_EQ(parallel2[i], serial[i]);
        EXPECT_EQ(parallel4[i], serial[i]);
    }
}

void test_pool_speeds_up_with_delay()
{
    const auto delay = microseconds(3000); // 每个请求 3ms 的"计算"

    auto timeit = [&](int workers)
    {
        auto t0 = steady_clock::now();
        run_with_workers(workers, 64, delay);
        return duration_cast<microseconds>(steady_clock::now() - t0).count();
    };

    long long serial = timeit(1);
    long long parallel = timeit(4);

    std::cout << "    serial=" << serial / 1000 << "ms, 4 workers=" << parallel / 1000 << "ms\n";
    // 8 个请求 * 25 token = 200 次采样，4 线程理想加速 4 倍，这里只要求明显更快
    EXPECT_TRUE(parallel * 2 < serial);
}

// 分摊不均：请求数少于 worker 数（有的 Worker 分到空区间），仍必须能收工
void test_pool_more_workers_than_requests()
{
    BlockAllocator alloc(64);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr, 8); // 8 个 Worker，只提交 2 个请求
    sched.set_sampler(fake_sample);
    sched.start();

    std::vector<std::future<void>> futures;
    std::vector<std::shared_ptr<Request>> reqs;
    for(int i = 0; i < 2; ++i)
    {
        auto r = make_req(16, 2500 + i * 100, 10);
        reqs.push_back(r);
        futures.push_back(sched.add_request(r));
    }
    bool ready = true;
    for(auto& f : futures)
        ready = wait_ready(f) && ready;
    EXPECT_TRUE(ready);
    for(auto& r : reqs)
        EXPECT_EQ(r->num_generated_tokens, 10);

    sched.stop(true);
    EXPECT_EQ(alloc.num_free_blocks(), 64);
}

// 多 Worker 下立即停机不能挂住
void test_pool_stop_without_drain()
{
    BlockAllocator alloc(32);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr, 4);
    sched.set_sampler([](const Request& r)
                      {
        std::this_thread::sleep_for(microseconds(500));
        return fake_sample(r); });
    sched.start();

    std::vector<std::future<void>> futures;
    std::vector<std::shared_ptr<Request>> reqs;
    for(int i = 0; i < 40; ++i)
    {
        auto r = make_req(16, 400 * (i + 1), 40);
        reqs.push_back(r);
        futures.push_back(sched.add_request(r));
    }
    sched.stop(false);

    bool ready = true;
    for(auto& f : futures)
        ready = wait_ready(f, milliseconds(10000)) && ready;
    EXPECT_TRUE(ready);
    for(auto& r : reqs)
        EXPECT_TRUE(r->s == ReqState::FINISHED || r->s == ReqState::ABORTED);
    EXPECT_EQ(alloc.num_free_blocks(), 32);
}

// 动态分发验证：批次内存在长尾慢请求时，其他 Worker 应继续领取剩余快任务而非空等
void test_pool_dynamic_dispatch_handles_straggler()
{
    BlockAllocator alloc(64);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr, 4);

    // arrival_id == 0 为慢请求（12ms），其余 7 个为快请求（1ms）
    // 静态分片下，与慢请求同组的快请求必须等慢请求结束；动态分发下由其他 3 个 Worker 立即消化
    sched.set_sampler([](const Request& r)
                      {
        if(r.arrival_id == 0)
            std::this_thread::sleep_for(milliseconds(12));
        else
            std::this_thread::sleep_for(milliseconds(1));
        return fake_sample(r); });
    sched.start();

    std::vector<std::shared_ptr<Request>> reqs;
    std::vector<std::future<void>> futures;
    for(int i = 0; i < 8; ++i)
    {
        auto r = make_req(16, 6000 + i * 100, 5);
        reqs.push_back(r);
        futures.push_back(sched.add_request(r));
    }

    bool ready = true;
    for(auto& f : futures)
        ready = wait_ready(f, milliseconds(5000)) && ready;
    EXPECT_TRUE(ready);

    for(auto& r : reqs)
    {
        EXPECT_TRUE(r->s == ReqState::FINISHED);
        EXPECT_EQ(r->num_generated_tokens, 5);
    }

    sched.stop(true);
    EXPECT_EQ(alloc.num_free_blocks(), 64);
}

// 并发观察者验证：调度线程在频繁抢占/准入期间，外部线程并发调用 running()/waiting() 不得出现数据竞争
void test_concurrent_queue_observers_safe()
{
    BlockAllocator alloc(12);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr, 2);
    sched.set_sampler(fake_sample);
    sched.start();

    std::atomic<bool> done{false};
    std::thread observer([&]
                         {
        while(!done.load())
        {
            auto r = sched.running();
            auto w = sched.waiting();
            (void)r;
            (void)w;
        } });

    std::vector<std::future<void>> futures;
    std::vector<std::shared_ptr<Request>> reqs;
    for(int i = 0; i < 8; ++i)
    {
        auto r = make_req(16, 8000 + i * 100, 32);
        reqs.push_back(r);
        futures.push_back(sched.add_request(r));
    }

    bool ready = true;
    for(auto& f : futures)
        ready = wait_ready(f, milliseconds(10000)) && ready;
    done.store(true);
    observer.join();

    EXPECT_TRUE(ready);
    EXPECT_TRUE(sched.num_preemptions() > 0);
    sched.stop(true);
    EXPECT_EQ(alloc.num_free_blocks(), 12);
}

int main()
{
    std::cout << "========== 阶段四：Scheduler 线程生命周期（M2） ==========\n";
    RUN_TEST(test_thread_basic_flow);
    RUN_TEST(test_thread_future_ready_only_at_end);
    RUN_TEST(test_thread_concurrent_producers);
    RUN_TEST(test_thread_idle_stop_is_fast);
    RUN_TEST(test_thread_stop_without_drain_notifies_all);
    RUN_TEST(test_thread_add_request_after_stop);

    std::cout << "\n========== 阶段四：Worker 池（M3） ==========\n";
    RUN_TEST(test_pool_output_identical);
    RUN_TEST(test_pool_speeds_up_with_delay);
    RUN_TEST(test_pool_more_workers_than_requests);
    RUN_TEST(test_pool_stop_without_drain);
    RUN_TEST(test_pool_dynamic_dispatch_handles_straggler);
    RUN_TEST(test_concurrent_queue_observers_safe);

    if(g_fail_count == 0)
    {
        std::cout << "\n所有测试通过 ✅\n";
        return 0;
    }
    std::cout << "\n失败 " << g_fail_count << " 项 ❌\n";
    return 1;
}