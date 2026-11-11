#include "test_util.h"
#include <memory>

// =========================================================
// 阶段四（Recompute）：Scheduler::step() 的抢占与接纳
// 注意：阶段四（执行/后处理）尚未实现，这里只断言队列、块表与空闲块数
// =========================================================

// 每个请求使用互不相同的 token，避免触发前缀缓存共享而干扰空闲块计数
static std::shared_ptr<Request> make_request(int num_tokens, int base, int max_new = 16)
{
    auto req = std::make_shared<Request>();
    req->all_tokens.resize(num_tokens);
    for(int i = 0; i < num_tokens; ++i)
        req->all_tokens[i] = base + i;
    req->max_new_tokens = max_new;
    return req;
}

// 空闲块刚好差 1 块：A、B、C 各占 1 个满块，下一步每个都要新块，只剩 1 个空闲块
void test_sched_preempt_last_arrival()
{
    BlockAllocator alloc(8);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr);
    sched.set_sampler(fake_sample);

    auto a = make_request(16, 1000);
    auto b = make_request(16, 2000);
    auto c = make_request(16, 3000);
    sched.add_request(a);
    sched.add_request(b);
    sched.add_request(c);

    // 第 1 步：三个请求都被接纳并 prefill
    sched.step();
    EXPECT_EQ((int)sched.running().size(), 3);
    EXPECT_EQ((int)sched.waiting().size(), 0);
    EXPECT_EQ(alloc.num_free_blocks(), 5);

    // 模拟其他占用者，把空闲块压到 1
    for(int i = 0; i < 4; ++i)
        EXPECT_TRUE(alloc.allocate() >= 0);
    EXPECT_EQ(alloc.num_free_blocks(), 1);

    // 第 2 步：A 拿走最后一块；B 不够 → 抢占队尾的 C；B 随后拿到 C 释放的块
    sched.step();

    EXPECT_EQ((int)sched.running().size(), 2);
    EXPECT_TRUE(sched.running()[0] == a);
    EXPECT_TRUE(sched.running()[1] == b);
    EXPECT_EQ((int)sched.waiting().size(), 1);
    EXPECT_TRUE(sched.waiting().front() == c);

    // C 被抢占：资源释放，但 token 保留，可以重算
    EXPECT_EQ((int)c->block_table.size(), 0);
    EXPECT_EQ(c->sequence_length, 0);
    EXPECT_EQ((int)c->all_tokens.size(), 16);
    EXPECT_TRUE(c->s == ReqState::WAITING);

    // A、B 各自拿到了新的尾块
    EXPECT_EQ((int)a->block_table.size(), 2);
    EXPECT_EQ((int)b->block_table.size(), 2);
    EXPECT_TRUE(a->s == ReqState::RUNNING);
    EXPECT_TRUE(b->s == ReqState::RUNNING);

    EXPECT_EQ(alloc.num_free_blocks(), 0);
}

// 空间充足、没有抢占时，等待队列里的请求会在一步内被全部接纳
void test_sched_admit_when_space_enough()
{
    BlockAllocator alloc(16);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr);
    sched.set_sampler(fake_sample);

    auto a = make_request(16, 1000);
    auto b = make_request(16, 2000);
    auto c = make_request(16, 3000);
    sched.add_request(a);
    sched.add_request(b);
    sched.add_request(c);

    EXPECT_EQ((int)sched.running().size(), 0);
    sched.step();

    EXPECT_EQ((int)sched.running().size(), 3);
    EXPECT_EQ((int)sched.waiting().size(), 0);
    EXPECT_TRUE(sched.running()[0] == a); // 按到达顺序
    EXPECT_TRUE(sched.running()[1] == b);
    EXPECT_TRUE(sched.running()[2] == c);

    for(auto& r : sched.running())
    {
        EXPECT_TRUE(r->s == ReqState::RUNNING);
        EXPECT_EQ((int)r->block_table.size(), 1);
        EXPECT_EQ(r->sequence_length, 16);
    }
    EXPECT_EQ(alloc.num_free_blocks(), 16 - 3);
}

// ---------------- 端到端：驱动到全部完成 ----------------

struct RunResult
{
    int steps{0};
    int preemptions{0};
    int free_blocks{0};
    std::vector<std::shared_ptr<Request>> reqs;
};

// 在 pool_blocks 大小的池子里跑完一批请求
static RunResult run_batch(int pool_blocks, const std::vector<std::vector<int>>& prompts, int max_new)
{
    BlockAllocator alloc(pool_blocks);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr);
    sched.set_sampler(fake_sample);

    RunResult r;
    for(const auto& p : prompts)
    {
        auto req = std::make_shared<Request>();
        req->all_tokens = p;
        req->max_new_tokens = max_new;
        r.reqs.push_back(req);
        sched.add_request(req);
    }
    r.steps = run_until_done(sched);
    r.preemptions = sched.num_preemptions();
    r.free_blocks = alloc.num_free_blocks();
    return r;
}

static std::vector<int> tokens_range(int n, int base)
{
    std::vector<int> v(n);
    for(int i = 0; i < n; ++i)
        v[i] = base + i;
    return v;
}

void test_sched_has_work_lifecycle()
{
    BlockAllocator alloc(16);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr);
    sched.set_sampler(fake_sample);
    EXPECT_FALSE(sched.has_work());

    sched.add_request(make_request(16, 1000, 5));
    EXPECT_TRUE(sched.has_work()); // 仍在收件箱

    int steps = run_until_done(sched);
    EXPECT_TRUE(steps > 0); // -1 表示卡死
    EXPECT_FALSE(sched.has_work());
}

void test_sched_run_to_completion()
{
    std::vector<std::vector<int>> prompts = {
        tokens_range(16, 1000), tokens_range(16, 2000), tokens_range(16, 3000)};
    auto r = run_batch(32, prompts, 20);

    EXPECT_TRUE(r.steps > 0);
    EXPECT_EQ(r.preemptions, 0);
    for(auto& req : r.reqs)
    {
        EXPECT_TRUE(req->s == ReqState::FINISHED);
        EXPECT_EQ(req->num_generated_tokens, 20);
        EXPECT_EQ((int)req->all_tokens.size(), 16 + 20); // prompt + 生成
        EXPECT_EQ((int)req->block_table.size(), 0);      // 块已释放
    }
    EXPECT_EQ(r.free_blocks, 32);
}

void test_sched_prompt_longer_than_max_new()
{
    // prompt（40）比 max_new_tokens（5）长，也必须正常结束
    auto r = run_batch(32, {tokens_range(40, 100)}, 5);
    EXPECT_TRUE(r.steps > 0);
    EXPECT_TRUE(r.reqs[0]->s == ReqState::FINISHED);
    EXPECT_EQ(r.reqs[0]->num_generated_tokens, 5);
    EXPECT_EQ((int)r.reqs[0]->all_tokens.size(), 45);
    EXPECT_EQ(r.free_blocks, 32);
}

void test_sched_abort_oversized_request()
{
    BlockAllocator alloc(16);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr);
    sched.set_sampler(fake_sample);

    auto big = make_request(16, 1000, 1000); // 需要远超池子的块数
    sched.add_request(big);
    sched.step();

    EXPECT_TRUE(big->s == ReqState::ABORTED);
    EXPECT_EQ((int)sched.waiting().size(), 0);
    EXPECT_EQ((int)sched.running().size(), 0);
    EXPECT_FALSE(sched.has_work());
    EXPECT_EQ(alloc.num_free_blocks(), 16);
}

// 核心验证：Recompute 抢占不应改变任何请求的最终输出
static void check_preemption_preserves_output(const std::vector<std::vector<int>>& prompts, int max_new,
                                              int small_pool, int big_pool)
{
    auto big = run_batch(big_pool, prompts, max_new);
    auto small = run_batch(small_pool, prompts, max_new);

    EXPECT_TRUE(big.steps > 0);
    EXPECT_TRUE(small.steps > 0);
    EXPECT_EQ(big.preemptions, 0);      // 大池子不应抢占
    EXPECT_TRUE(small.preemptions > 0); // 小池子必须真的抢占过，否则测试无意义

    for(size_t i = 0; i < prompts.size(); ++i)
    {
        EXPECT_TRUE(small.reqs[i]->s == ReqState::FINISHED);
        EXPECT_EQ(small.reqs[i]->all_tokens, big.reqs[i]->all_tokens);
        EXPECT_EQ(small.reqs[i]->num_generated_tokens, max_new); // 抢占不重复计数
    }
    EXPECT_EQ(small.free_blocks, small_pool); // 无泄漏
    EXPECT_EQ(big.free_blocks, big_pool);
}

void test_sched_preemption_preserves_output()
{
    // 4 个请求各需 ceil((16+40)/16)=4 块，共 16 块，小池子只有 10 块
    std::vector<std::vector<int>> prompts;
    for(int i = 0; i < 4; ++i)
        prompts.push_back(tokens_range(16, 1000 * (i + 1)));
    check_preemption_preserves_output(prompts, 40, 10, 64);
}

void test_sched_preemption_with_shared_prefix()
{
    // 4 个请求共享首块（16 个相同 token），尾部各不相同，叠加前缀缓存
    std::vector<std::vector<int>> prompts;
    for(int i = 0; i < 4; ++i)
    {
        auto p = tokens_range(16, 5000);
        auto tail = tokens_range(4, 9000 + 100 * i);
        p.insert(p.end(), tail.begin(), tail.end());
        prompts.push_back(p);
    }
    check_preemption_preserves_output(prompts, 40, 10, 64);
}

void test_sched_preempted_request_resumes_first()
{
    // 被抢占的请求回到等待队首，不会被后到的新请求插队
    std::vector<std::vector<int>> prompts;
    for(int i = 0; i < 4; ++i)
        prompts.push_back(tokens_range(16, 1000 * (i + 1)));

    BlockAllocator alloc(10);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr);
    sched.set_sampler(fake_sample);
    std::vector<std::shared_ptr<Request>> reqs;
    for(auto& p : prompts)
    {
        auto r = std::make_shared<Request>();
        r->all_tokens = p;
        r->max_new_tokens = 40;
        reqs.push_back(r);
        sched.add_request(r);
    }

    int guard = 0;
    while(sched.has_work() && sched.num_preemptions() == 0 && guard++ < 1000)
        sched.step();
    EXPECT_TRUE(sched.num_preemptions() > 0);

    // 等待队列应按到达顺序排列（较早到达的在前）
    const auto& w = sched.waiting();
    for(size_t i = 1; i < w.size(); ++i)
        EXPECT_TRUE(w[i - 1]->arrival_id < w[i]->arrival_id);

    EXPECT_TRUE(run_until_done(sched) > 0);
    EXPECT_EQ(alloc.num_free_blocks(), 10);
}

int main()
{
    std::cout << "========== 阶段四：Scheduler（Recompute 抢占与接纳） ==========\n";
    RUN_TEST(test_sched_preempt_last_arrival);
    RUN_TEST(test_sched_admit_when_space_enough);

    std::cout << "\n========== 阶段四：端到端（驱动到全部完成） ==========\n";
    RUN_TEST(test_sched_has_work_lifecycle);
    RUN_TEST(test_sched_run_to_completion);
    RUN_TEST(test_sched_prompt_longer_than_max_new);
    RUN_TEST(test_sched_abort_oversized_request);
    RUN_TEST(test_sched_preemption_preserves_output);
    RUN_TEST(test_sched_preemption_with_shared_prefix);
    RUN_TEST(test_sched_preempted_request_resumes_first);

    if(g_fail_count == 0)
    {
        std::cout << "\n所有测试通过 ✅\n";
        return 0;
    }
    std::cout << "\n失败 " << g_fail_count << " 项 ❌\n";
    return 1;
}