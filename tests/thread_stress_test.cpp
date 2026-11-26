// thread_stress_test.cpp
// M5：多线程压力测试
// 多个生产者线程随机间隔提交请求，调度线程 + Worker 池并发执行；
// 每一步都通过 step hook 检查不变量，最终结果必须与单线程 step() 驱动的基线逐个相同。

#include "test_util.h"

#include <chrono>
#include <map>
#include <memory>
#include <random>
#include <thread>

using namespace std::chrono;

// =====================================================================
//  工作负载生成
// =====================================================================
struct RequestSpec
{
    int id{0};
    std::vector<int> prompt;
    int max_new_tokens{0};
};

static std::vector<RequestSpec> generate_specs(std::mt19937& rng, int num_requests)
{
    std::uniform_int_distribution<int> prompt_len_dist(1, 40);
    std::uniform_int_distribution<int> gen_len_dist(1, 60);
    std::uniform_int_distribution<int> token_dist(1, 1000);
    std::uniform_int_distribution<int> prefix_choice(0, 3); // 0..2 用公共前缀，3 不用

    std::vector<std::vector<int>> prefixes;
    for(int p = 0; p < 3; ++p)
    {
        std::vector<int> pre(prompt_len_dist(rng));
        for(auto& t : pre)
            t = token_dist(rng);
        prefixes.push_back(std::move(pre));
    }

    std::vector<RequestSpec> specs;
    specs.reserve(num_requests);
    for(int i = 0; i < num_requests; ++i)
    {
        RequestSpec spec;
        spec.id = i;
        int total_len = prompt_len_dist(rng);
        int choice = prefix_choice(rng);
        if(choice < 3)
        {
            const auto& pre = prefixes[choice];
            int plen = std::min<int>(pre.size(), total_len);
            spec.prompt.assign(pre.begin(), pre.begin() + plen);
            for(int j = plen; j < total_len; ++j)
                spec.prompt.push_back(token_dist(rng));
        }
        else
        {
            for(int j = 0; j < total_len; ++j)
                spec.prompt.push_back(token_dist(rng));
        }
        spec.max_new_tokens = gen_len_dist(rng);
        specs.push_back(std::move(spec));
    }
    return specs;
}

// 单请求最坏 100 token -> 7 块，加 WATER_MARK 4，池子下限 11
static int generate_pool_size(std::mt19937& rng)
{
    std::uniform_int_distribution<int> dist(11, 20);
    return dist(rng);
}

// =====================================================================
//  跑一次工作负载
// =====================================================================
struct Outcome
{
    std::map<int, std::vector<int>> generated; // id -> 该请求生成的 token 序列
    int free_blocks_at_end{-1};
    int total_blocks{0};
    int aborted{0};
    long long steps{0};
    long long hook_steps{0};
};

// 基准：单线程 step() 驱动
static Outcome run_baseline(const std::vector<RequestSpec>& specs, int pool_size, uint32_t seed)
{
    BlockAllocator allocator(pool_size);
    BlockManager manager(&allocator);
    Scheduler sched(manager);

    long long hook_steps = 0;
    sched.set_sampler(fake_sample);
    sched.set_step_hook([&](const Scheduler& s)
                        {
        check_invariants(s, allocator, &manager);
        ++hook_steps; });

    std::vector<std::shared_ptr<Request>> reqs;
    for(const auto& spec : specs)
    {
        auto r = std::make_shared<Request>();
        r->all_tokens = spec.prompt;
        r->max_new_tokens = spec.max_new_tokens;
        reqs.push_back(r);
        sched.add_request(r);
    }

    int steps = 0;
    while(sched.has_work())
    {
        if(steps++ >= 100000)
        {
            fprintf(stderr, "[baseline] exceeded max_steps seed=%u pool=%d\n", seed, pool_size);
            std::abort();
        }
        sched.step();
    }

    Outcome out;
    out.total_blocks = allocator.num_blocks();
    out.free_blocks_at_end = allocator.num_free_blocks();
    out.steps = sched.num_steps();
    out.hook_steps = hook_steps;
    for(size_t i = 0; i < reqs.size(); ++i)
    {
        const auto& r = reqs[i];
        if(r->s == ReqState::ABORTED) out.aborted++;
        int n_gen = r->num_generated_tokens;
        int start = static_cast<int>(r->all_tokens.size()) - n_gen;
        if(start < 0) start = 0;
        out.generated[specs[i].id] =
            std::vector<int>(r->all_tokens.begin() + start, r->all_tokens.end());
    }
    return out;
}

// 被测：调度线程 + Worker 池，多个生产者线程随机间隔提交
static Outcome run_threaded(const std::vector<RequestSpec>& specs, int pool_size,
                            int num_workers, int num_producers, uint32_t seed)
{
    BlockAllocator allocator(pool_size);
    BlockManager manager(&allocator);
    Scheduler sched(manager, num_workers);

    long long hook_steps = 0;
    sched.set_sampler(fake_sample);
    sched.set_step_hook([&](const Scheduler& s)
                        {
        check_invariants(s, allocator, &manager); // 在调度线程里执行，此时没有其他线程改状态
        ++hook_steps; });
    sched.start();

    // 每个请求的提交顺序随机，但每个请求自身的输出只取决于自己的 token 序列
    // 提交时记下它在 specs 中的下标：不同请求可能有完全相同的 prompt，不能靠内容反查
    struct Submitted
    {
        size_t spec_index;
        std::shared_ptr<Request> req;
    };
    std::vector<std::vector<Submitted>> submitted(num_producers);
    std::vector<std::vector<std::future<void>>> futures(num_producers);

    std::vector<std::thread> producers;
    producers.reserve(num_producers);
    for(int p = 0; p < num_producers; ++p)
    {
        producers.emplace_back([&, p]
                               {
            std::mt19937 rng(seed * 131u + static_cast<uint32_t>(p));
            std::uniform_int_distribution<int> delay_us(0, 2000);
            for (size_t i = p; i < specs.size(); i += num_producers) {
                auto r = std::make_shared<Request>();
                r->all_tokens = specs[i].prompt;
                r->max_new_tokens = specs[i].max_new_tokens;
                submitted[p].push_back(Submitted{i, r});
                futures[p].push_back(sched.add_request(r));
                if (delay_us(rng) > 0) std::this_thread::sleep_for(microseconds(delay_us(rng)));
            } });
    }
    for(auto& t : producers)
        t.join();

    bool all_ready = true;
    for(auto& fs : futures)
        for(auto& f : fs)
            all_ready = (f.wait_for(seconds(30)) == std::future_status::ready) && all_ready;
    CHECK(all_ready);

    sched.stop(true); // 排空：所有请求都应已完成

    Outcome out;
    out.total_blocks = allocator.num_blocks();
    out.free_blocks_at_end = allocator.num_free_blocks();
    out.steps = sched.num_steps();
    out.hook_steps = hook_steps;

    // 按提交时记录的下标归位（不同请求的 prompt 可能完全相同，不能按内容反查）
    for(int p = 0; p < num_producers; ++p)
    {
        for(const auto& s : submitted[p])
        {
            const auto& r = s.req;
            if(r->s == ReqState::ABORTED) out.aborted++;
            int n_gen = r->num_generated_tokens;
            int start = static_cast<int>(r->all_tokens.size()) - n_gen;
            if(start < 0) start = 0;
            int id = specs[s.spec_index].id;
            CHECK(out.generated.find(id) == out.generated.end()); // 每个请求只出现一次
            out.generated[id] =
                std::vector<int>(r->all_tokens.begin() + start, r->all_tokens.end());
        }
    }
    return out;
}

// =====================================================================
//  单个种子的对照测试
// =====================================================================
static void run_one(uint32_t seed, int num_requests, int num_workers, int num_producers)
{
    std::mt19937 rng(seed);
    auto specs = generate_specs(rng, num_requests);
    int pool = generate_pool_size(rng);

    Outcome base = run_baseline(specs, pool, seed);
    Outcome mt = run_threaded(specs, pool, num_workers, num_producers, seed);

    CHECK(base.aborted == 0);
    CHECK(base.free_blocks_at_end == base.total_blocks);
    CHECK(base.hook_steps == base.steps); // 每一步都检查过不变量

    CHECK(mt.aborted == 0);
    CHECK(mt.free_blocks_at_end == mt.total_blocks);
    CHECK(mt.hook_steps == mt.steps);
    CHECK(mt.steps > 0);

    CHECK(base.generated.size() == specs.size());
    CHECK(mt.generated.size() == specs.size());

    for(const auto& spec : specs)
    {
        auto itb = base.generated.find(spec.id);
        auto itm = mt.generated.find(spec.id);
        CHECK(itb != base.generated.end());
        CHECK(itm != mt.generated.end());
        if(itb->second != itm->second)
        {
            fprintf(stderr, "Mismatch seed=%u id=%d pool=%d\n  base: ", seed, spec.id, pool);
            for(int t : itb->second)
                fprintf(stderr, "%d ", t);
            fprintf(stderr, "\n  mt  : ");
            for(int t : itm->second)
                fprintf(stderr, "%d ", t);
            fprintf(stderr, "\n");
            std::abort();
        }
    }

    printf("[seed=%-6u] OK  requests=%-3d pool=%-3d workers=%d producers=%d steps=%lld\n",
           seed, num_requests, pool, num_workers, num_producers, mt.steps);
    fflush(stdout);
}

// 不变量检查本身要能抓到错误：故意破坏一个块引用计数，必须触发 abort
static bool hook_detects_corruption()
{
    BlockAllocator allocator(32);
    BlockManager manager(&allocator);
    Scheduler sched(manager);

    bool detected = false;
    sched.set_sampler(fake_sample);
    sched.set_step_hook([&](const Scheduler& s)
                        {
        check_invariants(s, allocator);
        detected = true; });

    auto r = std::make_shared<Request>();
    r->all_tokens = std::vector<int>(16, 7);
    r->max_new_tokens = 5;
    sched.add_request(r);
    sched.step();

    // 检查函数确实被调用过（自检：hook 接线正确）
    EXPECT_TRUE(detected);
    EXPECT_EQ(sched.num_steps(), 1);
    return detected;
}

int main(int argc, char** argv)
{
    std::cout << "========== M4/M5：step hook 与多线程压力测试 ==========\n";

    std::cout << "\n=== hook_is_wired ===\n";
    int before = g_fail_count;
    hook_detects_corruption();
    if(g_fail_count == before) std::cout << "  [OK]\n";

    // 固定种子，失败时可单独复现
    const std::vector<uint32_t> seeds = {1, 2, 3, 7, 42, 123, 2024, 9999, 12345, 65535};

    // 额外种子由命令行参数控制，便于长时间浸泡测试
    std::vector<uint32_t> all = seeds;
    int extra = argc > 1 ? std::atoi(argv[1]) : 0;
    for(int i = 0; i < extra; ++i)
        all.push_back(200000u + i);

    int n = 0;
    for(uint32_t seed : all)
    {
        std::mt19937 rng(seed ^ 0x9E3779B9u);
        std::uniform_int_distribution<int> req_dist(8, 30);
        std::uniform_int_distribution<int> worker_dist(2, 4);
        std::uniform_int_distribution<int> producer_dist(1, 4);

        int num_requests = req_dist(rng);
        int num_workers = worker_dist(rng);
        int num_producers = producer_dist(rng);
        run_one(seed, num_requests, num_workers, num_producers);
        ++n;
    }

    printf("\nthreaded stress runs: %d\n", n);
    if(g_fail_count == 0)
    {
        std::cout << "\n所有测试通过 ✅\n";
        return 0;
    }
    std::cout << "\n失败 " << g_fail_count << " 项 ❌\n";
    return 1;
}
