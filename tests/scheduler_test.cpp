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
        EXPECT_TRUE(mgr.allocate_block() >= 0);
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

void test_sched_custom_block_size_and_watermark()
{
    // block_size=8，16 个块，watermark_blocks=2 → 可用 14 块
    BlockAllocator alloc(16, 8);
    BlockManager mgr(&alloc);
    SchedulerConfig cfg;
    cfg.watermark_blocks = 2;
    Scheduler sched(mgr, cfg);
    sched.set_sampler(fake_sample);
    sched.set_step_hook([&](const Scheduler& s)
                        { check_invariants(s, alloc); });
    EXPECT_EQ(sched.watermark_blocks(), 2);

    auto r1 = make_request(12, 1000, 20); // 32 token -> 4 块
    auto r2 = make_request(12, 2000, 20); // 32 token -> 4 块
    sched.add_request(r1);
    sched.add_request(r2);

    EXPECT_TRUE(run_until_done(sched) > 0);
    EXPECT_TRUE(r1->s == ReqState::FINISHED);
    EXPECT_TRUE(r2->s == ReqState::FINISHED);
    EXPECT_EQ(alloc.num_free_blocks(), 16);
}

void test_sched_watermark_ratio_aborts_oversized()
{
    // 16 块，watermark_ratio=0.25 → watermark=4，最大单请求 12 块（192 token）
    BlockAllocator alloc(16, 16);
    BlockManager mgr(&alloc);
    SchedulerConfig cfg;
    cfg.watermark_ratio = 0.25;
    Scheduler sched(mgr, cfg);
    sched.set_sampler(fake_sample);
    EXPECT_EQ(sched.watermark_blocks(), 4);

    auto too_big = make_request(16, 1000, 180); // 196 token -> 13 块 > 12 → ABORTED
    auto ok_req = make_request(16, 2000, 16);   // 32 token -> 2 块 <= 12 → FINISHED
    sched.add_request(too_big);
    sched.add_request(ok_req);

    EXPECT_TRUE(run_until_done(sched) > 0);
    EXPECT_TRUE(too_big->s == ReqState::ABORTED);
    EXPECT_TRUE(ok_req->s == ReqState::FINISHED);
    EXPECT_EQ(alloc.num_free_blocks(), 16);
}

// =========================================================
// v0.4.0：Schedule/Execute 解耦与 Chunked Prefill + Token Budget
// =========================================================

void test_sched_decoupled_schedule_execute()
{
    BlockAllocator alloc(32, 16);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr);
    sched.set_sampler(fake_sample);

    auto r = make_request(32, 1000, 4);
    sched.add_request(r);

    // 第 1 步：schedule() 准入并 prefill 32 token，无 decode
    SchedulerOutput out1 = sched.schedule();
    EXPECT_EQ((int)out1.prefill_reqs.size(), 1);
    EXPECT_EQ(out1.num_prefill_tokens, 32);
    EXPECT_EQ(out1.num_decode_tokens, 0);
    sched.execute(out1);
    EXPECT_EQ(sched.last_step_prefill_tokens(), 32);
    EXPECT_EQ(sched.last_step_decode_tokens(), 0);
    EXPECT_EQ(r->num_generated_tokens, 0);

    // 第 2 步：schedule() 调度 1 个 decode token
    SchedulerOutput out2 = sched.schedule();
    EXPECT_EQ(out2.num_prefill_tokens, 0);
    EXPECT_EQ(out2.num_decode_tokens, 1);
    sched.execute(out2);
    EXPECT_EQ(r->num_generated_tokens, 1);

    // 跑完剩余步骤
    while(sched.has_work())
    {
        auto out = sched.schedule();
        sched.execute(out);
    }
    EXPECT_TRUE(r->s == ReqState::FINISHED);
    EXPECT_EQ(r->num_generated_tokens, 4);
    EXPECT_EQ(alloc.num_free_blocks(), 32);
}

void test_sched_chunked_prefill_basic()
{
    // 100 token prompt，单步预算 32 → 应拆成 32 + 32 + 32 + 4 四步 prefill，输出与非分块完全一致
    auto prompt = tokens_range(100, 500);
    auto baseline = run_batch(64, {prompt}, 10);

    BlockAllocator alloc(64, 16);
    BlockManager mgr(&alloc);
    SchedulerConfig cfg;
    cfg.max_num_batched_tokens = 32;
    cfg.enable_chunked_prefill = true;
    Scheduler sched(mgr, cfg);
    sched.set_sampler(fake_sample);
    sched.set_step_hook([&](const Scheduler& s)
                        { check_invariants(s, alloc, &mgr); });

    auto r = std::make_shared<Request>();
    r->all_tokens = prompt;
    r->max_new_tokens = 10;
    sched.add_request(r);

    std::vector<int> prefill_chunks;
    while(sched.has_work())
    {
        auto out = sched.schedule();
        EXPECT_TRUE(out.total_scheduled_tokens() <= 32);
        if(out.num_prefill_tokens > 0) prefill_chunks.push_back(out.num_prefill_tokens);
        sched.execute(out);
    }

    EXPECT_EQ(prefill_chunks, (std::vector<int>{32, 32, 32, 4}));
    EXPECT_TRUE(r->s == ReqState::FINISHED);
    EXPECT_EQ(r->all_tokens, baseline.reqs[0]->all_tokens);
    EXPECT_EQ(alloc.num_free_blocks(), 64);
}

void test_sched_chunked_prefill_mixed_with_decode()
{
    // 2 个短请求正在 decode 时，1 个 96-token 长请求到达；budget=20 应实现 decode + prefill chunk 同批混排
    BlockAllocator alloc(64, 16);
    BlockManager mgr(&alloc);
    SchedulerConfig cfg;
    cfg.max_num_batched_tokens = 20;
    cfg.enable_chunked_prefill = true;
    Scheduler sched(mgr, cfg);
    sched.set_sampler(fake_sample);
    sched.set_step_hook([&](const Scheduler& s)
                        { check_invariants(s, alloc, &mgr); });

    auto d1 = make_request(8, 1000, 10);
    auto d2 = make_request(8, 2000, 10);
    sched.add_request(d1);
    sched.add_request(d2);
    sched.step(); // 首步完成 d1/d2 的 prefill（共 16 token <= 20）

    // 注入长 prompt 请求
    auto long_req = make_request(96, 5000, 5);
    sched.add_request(long_req);

    bool saw_mixed_batch = false;
    while(sched.has_work())
    {
        auto out = sched.schedule();
        EXPECT_TRUE(out.total_scheduled_tokens() <= 20);
        if(out.num_decode_tokens == 2 && out.num_prefill_tokens == 18) saw_mixed_batch = true;
        sched.execute(out);
    }
    EXPECT_TRUE(saw_mixed_batch);
    EXPECT_TRUE(d1->s == ReqState::FINISHED);
    EXPECT_TRUE(d2->s == ReqState::FINISHED);
    EXPECT_TRUE(long_req->s == ReqState::FINISHED);
    EXPECT_EQ(alloc.num_free_blocks(), 64);
}

void test_sched_preempt_mid_chunked_prefill()
{
    // 紧池（12 块，watermark=2）+ 分块预填充（budget=16）：
    // 多个长 prompt 同时进入分块预填充，必然在“prefill 到一半”时触发抢占，
    // 验证不触发断言崩溃、不变量始终成立，且最终输出与大池基线逐 token 一致。
    std::vector<std::vector<int>> prompts = {
        tokens_range(64, 1000),
        tokens_range(64, 2000),
        tokens_range(64, 3000),
        tokens_range(64, 4000)};
    auto big = run_batch(128, prompts, 16);

    BlockAllocator alloc(12, 16);
    BlockManager mgr(&alloc);
    SchedulerConfig cfg;
    cfg.watermark_blocks = 2;
    cfg.max_num_batched_tokens = 16;
    cfg.enable_chunked_prefill = true;
    Scheduler sched(mgr, cfg);
    sched.set_sampler(fake_sample);
    sched.set_step_hook([&](const Scheduler& s)
                        { check_invariants(s, alloc, &mgr); });

    std::vector<std::shared_ptr<Request>> reqs;
    for(const auto& p : prompts)
    {
        auto r = std::make_shared<Request>();
        r->all_tokens = p;
        r->max_new_tokens = 16;
        reqs.push_back(r);
        sched.add_request(r);
    }

    int steps = run_until_done(sched, 20000);
    EXPECT_TRUE(steps > 0);
    EXPECT_TRUE(sched.num_preemptions() > 0);
    for(size_t i = 0; i < reqs.size(); ++i)
    {
        EXPECT_TRUE(reqs[i]->s == ReqState::FINISHED);
        EXPECT_EQ(reqs[i]->all_tokens, big.reqs[i]->all_tokens);
    }
    EXPECT_EQ(alloc.num_free_blocks(), 12);
}

void test_sched_max_num_seqs_and_waiting_reserve()
{
    // max_num_seqs=2 限制并发 running 上限；waiting_token_reserve=8 防止长 chunked prefill 饿死等待队列
    BlockAllocator alloc(64, 16);
    BlockManager mgr(&alloc);
    SchedulerConfig cfg;
    cfg.max_num_batched_tokens = 32;
    cfg.max_num_seqs = 2;
    cfg.enable_chunked_prefill = true;
    cfg.waiting_token_reserve = 8;
    Scheduler sched(mgr, cfg);
    sched.set_sampler(fake_sample);

    auto r1 = make_request(80, 1000, 4); // 长 prompt
    auto r2 = make_request(8, 2000, 4);  // 短 prompt
    auto r3 = make_request(8, 3000, 4);  // 第 3 个请求，应被 max_num_seqs=2 挡在 waiting
    sched.add_request(r1);
    sched.add_request(r2);
    sched.add_request(r3);

    // 第 1 步：r1 先拿 32 token（剩余 48）；r2 还没进
    sched.step();
    EXPECT_TRUE((int)sched.running().size() <= 2);

    // 第 2 步：因为 waiting_token_reserve=8，r1 续算最多拿 32-8=24 token，留出 8 token 让 r2 立刻准入！
    auto out2 = sched.schedule();
    EXPECT_TRUE(out2.total_scheduled_tokens() <= 32);
    EXPECT_EQ((int)sched.running().size(), 2); // r1 + r2 已同时在 running，r3 被 max_num_seqs=2 挡住
    EXPECT_EQ((int)sched.waiting().size(), 1);
    sched.execute(out2);

    int steps = run_until_done(sched);
    EXPECT_TRUE(steps > 0);
    EXPECT_TRUE(r1->s == ReqState::FINISHED);
    EXPECT_TRUE(r2->s == ReqState::FINISHED);
    EXPECT_TRUE(r3->s == ReqState::FINISHED);
    EXPECT_EQ(alloc.num_free_blocks(), 64);
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
    RUN_TEST(test_sched_custom_block_size_and_watermark);
    RUN_TEST(test_sched_watermark_ratio_aborts_oversized);

    std::cout << "\n========== v0.4.0：Schedule/Execute 与 Chunked Prefill ==========\n";
    RUN_TEST(test_sched_decoupled_schedule_execute);
    RUN_TEST(test_sched_chunked_prefill_basic);
    RUN_TEST(test_sched_chunked_prefill_mixed_with_decode);
    RUN_TEST(test_sched_preempt_mid_chunked_prefill);
    RUN_TEST(test_sched_max_num_seqs_and_waiting_reserve);

    if(g_fail_count == 0)
    {
        std::cout << "\n所有测试通过 ✅\n";
        return 0;
    }
    std::cout << "\n失败 " << g_fail_count << " 项 ❌\n";
    return 1;
}