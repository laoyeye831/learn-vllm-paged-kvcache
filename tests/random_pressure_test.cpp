// random_pressure_test.cpp
// 随机压力测试：同一批随机请求在小池（频繁抢占）与大池（无抢占）下运行，
// 每步检查块引用计数等不变量，并要求两边的生成结果逐个相同。
// 不变量检查函数与 CHECK 宏见 test_util.h。

#include "test_util.h"

#include <map>
#include <random>

// =====================================================================
//  二、随机工作负载
// =====================================================================
struct RequestSpec {
    int id{0};
    std::vector<int> prompt;
    int max_new_tokens{0};
};

// 随机生成一批请求规格：随机 prompt 长度、随机 max_new_tokens、随机公共前缀
std::vector<RequestSpec> generate_specs(std::mt19937& rng, int num_requests)
{
    std::uniform_int_distribution<int> prompt_len_dist(1, 40);
    std::uniform_int_distribution<int> gen_len_dist(1, 60);
    std::uniform_int_distribution<int> token_dist(1, 1000);
    std::uniform_int_distribution<int> prefix_choice(0, 3); // 0/1/2 用前缀，3 不用

    // 3 个公共前缀
    std::vector<std::vector<int>> prefixes;
    for (int p = 0; p < 3; p++) {
        int len = prompt_len_dist(rng);
        std::vector<int> pre(len);
        for (auto& t : pre) t = token_dist(rng);
        prefixes.push_back(std::move(pre));
    }

    std::vector<RequestSpec> specs;
    specs.reserve(num_requests);
    for (int i = 0; i < num_requests; i++) {
        RequestSpec spec;
        spec.id = i;
        int total_len = prompt_len_dist(rng);
        int choice = prefix_choice(rng);
        if (choice < 3) {
            const auto& pre = prefixes[choice];
            int plen = std::min<int>(pre.size(), total_len);
            spec.prompt.assign(pre.begin(), pre.begin() + plen);
            for (int j = plen; j < total_len; j++)
                spec.prompt.push_back(token_dist(rng));
        } else {
            for (int j = 0; j < total_len; j++)
                spec.prompt.push_back(token_dist(rng));
        }
        spec.max_new_tokens = gen_len_dist(rng);
        specs.push_back(std::move(spec));
    }
    return specs;
}

// 池大小：需 > WATER_MARK + 单请求最坏块数
// 单请求最坏: prompt 40 + gen 60 = 100 tokens -> 7 blocks; WATER_MARK = 4
// 因此最小安全值 = 11，这里取 [11, 20]，让运行中的请求增长时频繁耗尽余量
int generate_pool_size(std::mt19937& rng)
{
    std::uniform_int_distribution<int> dist(11, 20);
    return dist(rng);
}


// =====================================================================
//  三、跑一次工作负载
// =====================================================================
struct RunResult {
    int num_finished{0};
    int num_aborted{0};
    int num_preemptions{0};
    int total_blocks{0};
    int free_blocks_at_end{-1};
    // key = RequestSpec::id，value = 该请求生成的 token 序列
    std::map<int, std::vector<int>> generated_tokens;
};

RunResult run_workload(int pool_size,
                       const std::vector<RequestSpec>& specs,
                       bool check_inv,
                       uint32_t seed,
                       int num_workers = 1,
                       int max_steps = 100000)
{
    BlockAllocator allocator(pool_size);
    BlockManager   manager(&allocator);
    Scheduler      sched(manager, num_workers);
    sched.set_sampler(fake_sample); // 使用内置模拟生成器
    sched.start_pool();             // 仍由本线程驱动 step()，只启用 Worker 池做采样

    std::vector<std::shared_ptr<Request>> requests;
    requests.reserve(specs.size());
    for (const auto& spec : specs) {
        auto r = std::make_shared<Request>();
        r->all_tokens      = spec.prompt;
        r->max_new_tokens  = spec.max_new_tokens;
        requests.push_back(r);
        sched.add_request(r);
    }

    int steps = 0;
    while (sched.has_work()) {
        if (steps++ >= max_steps) {
            fprintf(stderr,
                "[run] exceeded max_steps=%d seed=%u pool=%d\n",
                max_steps, seed, pool_size);
            std::abort();
        }
        sched.step();
        if (check_inv) check_invariants(sched, allocator);
    }
    sched.stop_pool(); // 主线程已收工，回收 Worker

    RunResult res;
    res.num_preemptions   = sched.num_preemptions();
    res.total_blocks      = allocator.num_blocks();
    res.free_blocks_at_end = allocator.num_free_blocks();

    for (size_t i = 0; i < requests.size(); i++) {
        const auto& r = requests[i];
        if (r->s == ReqState::FINISHED) res.num_finished++;
        else if (r->s == ReqState::ABORTED) res.num_aborted++;

        int n_gen = r->num_generated_tokens;
        std::vector<int> gen;
        if (n_gen > 0) {
            int start = static_cast<int>(r->all_tokens.size()) - n_gen;
            if (start < 0) start = 0;
            gen.assign(r->all_tokens.begin() + start, r->all_tokens.end());
        }
        res.generated_tokens[specs[i].id] = std::move(gen);
    }
    return res;
}


// =====================================================================
//  四、单次压力测试：小池子 vs 大池子
// =====================================================================
int run_stress_test(uint32_t seed, int num_requests)
{
    std::mt19937 rng(seed);
    auto specs        = generate_specs(rng, num_requests);
    int pool_small    = generate_pool_size(rng);
    const int pool_large = 4096; // 足够大，不会触发抢占也不会 abort
    const int workers = 4;       // 并行采样路径

    // 小池：会触发抢占（串行采样 / 并行采样各跑一次）
    RunResult small = run_workload(pool_small, specs, /*check_inv=*/true, seed, 1);
    RunResult small_mt = run_workload(pool_small, specs, /*check_inv=*/true, seed, workers);
    // 大池：对照组，作为"无抢占"基准
    RunResult large = run_workload(pool_large, specs, /*check_inv=*/true, seed, 1);
    RunResult large_mt = run_workload(pool_large, specs, /*check_inv=*/true, seed, workers);

    // --- 最终状态检查 ---
    CHECK(small.num_finished == static_cast<int>(specs.size()));
    CHECK(small.num_aborted  == 0);
    CHECK(small.free_blocks_at_end == small.total_blocks);

    CHECK(large.num_finished == static_cast<int>(specs.size()));
    CHECK(large.num_aborted  == 0);
    CHECK(large.free_blocks_at_end == large.total_blocks);
    CHECK(large.num_preemptions == 0); // 大池不应该抢占

    // --- 并行采样不得改变结果 ---
    CHECK(small_mt.num_finished == small.num_finished);
    CHECK(small_mt.num_aborted  == small.num_aborted);
    CHECK(small_mt.num_preemptions == small.num_preemptions);
    CHECK(small_mt.free_blocks_at_end == small.total_blocks);
    CHECK(small_mt.generated_tokens == small.generated_tokens);

    CHECK(large_mt.num_finished == large.num_finished);
    CHECK(large_mt.free_blocks_at_end == large.total_blocks);
    CHECK(large_mt.generated_tokens == large.generated_tokens);

    // --- 输出逐个对比 ---
    CHECK(small.generated_tokens.size() == large.generated_tokens.size());
    for (const auto& kv : small.generated_tokens) {
        int id = kv.first;
        const auto& gen_small = kv.second;
        auto it = large.generated_tokens.find(id);
        CHECK(it != large.generated_tokens.end());
        const auto& gen_large = it->second;

        if (gen_small != gen_large) {
            fprintf(stderr,
                "Mismatch seed=%u request_id=%d (pool_small=%d)\n",
                seed, id, pool_small);
            fprintf(stderr, "  small: ");
            for (int t : gen_small) fprintf(stderr, "%d ", t);
            fprintf(stderr, "\n  large: ");
            for (int t : gen_large) fprintf(stderr, "%d ", t);
            fprintf(stderr, "\n");
            std::abort();
        }
    }

    printf("[seed=%-6u] OK  requests=%-3d pool_small=%-3d preemptions=%d\n",
           seed, num_requests, pool_small, small.num_preemptions);
    fflush(stdout);
    return small.num_preemptions;
}


// =====================================================================
//  五、入口
// =====================================================================
// 用法: random_pressure_test [额外随机种子数]，额外种子从 100000 起顺序生成
int main(int argc, char** argv)
{
    // 固定种子集合：失败时可用对应 seed 单跑复现
    const std::vector<uint32_t> seeds = {
        1, 2, 3, 7, 42, 123, 2024, 9999, 12345, 65535
    };

    std::vector<uint32_t> all_seeds = seeds;
    int extra = argc > 1 ? std::atoi(argv[1]) : 0;
    for (int i = 0; i < extra; i++) all_seeds.push_back(100000u + i);

    int total_preemptions = 0;
    for (uint32_t seed : all_seeds) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> req_dist(10, 40); // 随机请求数
        int num_requests = req_dist(rng);
        total_preemptions += run_stress_test(seed, num_requests);
    }

    // 压力测试必须真的触发过抢占，否则等于没测
    CHECK(total_preemptions > 0);
    printf("All stress tests passed.\n");
    return 0;
}