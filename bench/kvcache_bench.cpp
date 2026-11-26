// bench/kvcache_bench.cpp
// 可复现的前缀缓存与淘汰策略对照基准：
//   支持 sysprompt / chat / zipf 三种工作负载，对比 FIFO / LRU / prefix-off。
//
// 用法示例：
//   ./build/kvcache_bench --sweep
//   ./build/kvcache_bench --workload zipf --blocks 32 --policy lru --prefix on --seed 42
//   ./build/kvcache_bench --workload chat --sessions 8 --turns 6 --blocks 32 --csv

#include "kvcache.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

struct BenchOptions
{
    int blocks{32};
    int block_size{16};
    int watermark{4};
    FreePolicy policy{FreePolicy::LRU};
    bool prefix_on{true};
    std::string workload{"zipf"};
    int num_requests{80};
    int sessions{8};
    int turns{6};
    uint32_t seed{42};
    int workers{1};
    bool csv{false};
    bool sweep{false};
    bool timed{false};
};

struct ReqSpec
{
    std::vector<int> prompt;
    int max_new_tokens{0};
};

struct BenchResult
{
    std::string workload;
    std::string policy_name;
    bool prefix_on{true};
    int blocks{0};
    uint64_t query_blocks{0};
    uint64_t hit_blocks{0};
    double hit_pct{0.0};
    double hit_tok_pct{0.0};
    uint64_t evictions{0};
    uint64_t revivals{0};
    int preemptions{0};
    uint64_t cow_copies{0};
    double peak_util_pct{0.0};
    double avg_util_pct{0.0};
    double avg_cached_pct{0.0};
    long long steps{0};
    long long gen_tokens{0};
    double tok_per_step{0.0};
    double avg_ttft{0.0};
    double p99_ttft{0.0};
    double avg_tpot{0.0};
    double wall_ms{0.0};
};

static std::vector<int> random_tokens(std::mt19937& rng, int len, int lo = 1, int hi = 40000)
{
    std::uniform_int_distribution<int> dist(lo, hi);
    std::vector<int> out(static_cast<size_t>(len));
    for(int i = 0; i < len; ++i)
        out[static_cast<size_t>(i)] = dist(rng);
    return out;
}

// 1) sysprompt：共享长 system prompt + 独立短问题（中间穿插无前缀冷请求）
static std::vector<ReqSpec> gen_sysprompt(const BenchOptions& opt)
{
    std::mt19937 rng(opt.seed);
    // 2 条不同 system prompt（各 6 块 = 96 token）+ 偶尔无前缀冷请求制造 CLEAN 块
    std::vector<std::vector<int>> sys_prompts = {
        random_tokens(rng, opt.block_size * 6, 1, 10000),
        random_tokens(rng, opt.block_size * 6, 10001, 20000)};

    std::uniform_int_distribution<int> q_len(12, 40);
    std::uniform_int_distribution<int> gen_len(8, 20);
    std::uniform_int_distribution<int> kind(0, 9);

    std::vector<ReqSpec> specs;
    specs.reserve(static_cast<size_t>(opt.num_requests));
    for(int i = 0; i < opt.num_requests; ++i)
    {
        ReqSpec s;
        int k = kind(rng);
        if(k < 6)
        {
            // 60% 热点 system prompt #0
            s.prompt = sys_prompts[0];
        }
        else if(k < 8)
        {
            // 20% 次热 system prompt #1
            s.prompt = sys_prompts[1];
        }
        // 20% 无共享前缀（冷请求，释放后产生 CLEAN 块，检验 LRU 保温能力）
        auto q = random_tokens(rng, q_len(rng), 20001, 49000);
        s.prompt.insert(s.prompt.end(), q.begin(), q.end());
        s.max_new_tokens = gen_len(rng);
        specs.push_back(std::move(s));
    }
    return specs;
}

// 2) chat：多会话多轮对话，离线推演每轮完整历史 prompt（完全确定性）
static std::vector<ReqSpec> gen_chat(const BenchOptions& opt)
{
    std::mt19937 rng(opt.seed);
    std::uniform_int_distribution<int> user_len(16, 32);
    std::uniform_int_distribution<int> reply_len(10, 18);

    std::vector<std::vector<int>> history(static_cast<size_t>(opt.sessions));
    for(int s = 0; s < opt.sessions; ++s)
    {
        // 每个会话初始一段 2 块（32 token）的上下文
        history[static_cast<size_t>(s)] = random_tokens(rng, opt.block_size * 2, 1, 30000);
    }

    std::vector<ReqSpec> specs;
    specs.reserve(static_cast<size_t>(opt.sessions * opt.turns));
    for(int t = 0; t < opt.turns; ++t)
    {
        for(int s = 0; s < opt.sessions; ++s)
        {
            auto& h = history[static_cast<size_t>(s)];
            auto u = random_tokens(rng, user_len(rng), 1, 45000);
            h.insert(h.end(), u.begin(), u.end());

            int g = reply_len(rng);
            ReqSpec spec;
            spec.prompt = h;
            spec.max_new_tokens = g;
            specs.push_back(std::move(spec));

            // 用同一确定性 fake_sample 离线推演下一轮历史
            Request sim;
            sim.all_tokens = h;
            sim.sequence_length = static_cast<int>(h.size());
            for(int k = 0; k < g; ++k)
            {
                int tok = fake_sample(sim);
                sim.all_tokens.push_back(tok);
                sim.sequence_length += 1;
            }
            h = std::move(sim.all_tokens);
        }
    }
    return specs;
}

// 3) zipf：K 个文档前缀按 Zipf(s=1.2) 分布抽取 + 独立后缀（LRU 经典热点场景）
static std::vector<ReqSpec> gen_zipf(const BenchOptions& opt)
{
    std::mt19937 rng(opt.seed);
    constexpr int kDocs = 16;
    std::vector<std::vector<int>> docs;
    docs.reserve(kDocs);
    for(int d = 0; d < kDocs; ++d)
    {
        // 每个文档前缀 4 块（64 token）
        docs.push_back(random_tokens(rng, opt.block_size * 4, 1 + d * 2000, 1999 + d * 2000));
    }

    // 构造 Zipf 离散分布
    std::vector<double> weights(kDocs);
    for(int d = 0; d < kDocs; ++d)
        weights[static_cast<size_t>(d)] = 1.0 / std::pow(static_cast<double>(d + 1), 1.2);
    std::discrete_distribution<int> doc_dist(weights.begin(), weights.end());
    std::uniform_int_distribution<int> q_len(8, 24);
    std::uniform_int_distribution<int> gen_len(8, 16);

    std::vector<ReqSpec> specs;
    specs.reserve(static_cast<size_t>(opt.num_requests));
    for(int i = 0; i < opt.num_requests; ++i)
    {
        int doc_id = doc_dist(rng);
        ReqSpec s;
        s.prompt = docs[static_cast<size_t>(doc_id)];
        auto q = random_tokens(rng, q_len(rng), 35000, 49000);
        s.prompt.insert(s.prompt.end(), q.begin(), q.end());
        s.max_new_tokens = gen_len(rng);
        specs.push_back(std::move(s));
    }
    return specs;
}

static std::vector<ReqSpec> build_workload(const BenchOptions& opt)
{
    if(opt.workload == "sysprompt") return gen_sysprompt(opt);
    if(opt.workload == "chat") return gen_chat(opt);
    return gen_zipf(opt);
}

static BenchResult run_once(const BenchOptions& opt, const std::vector<ReqSpec>& specs)
{
    BlockAllocator alloc(opt.blocks, opt.block_size, opt.policy);
    BlockManager mgr(&alloc, BlockManagerConfig{opt.prefix_on});
    SchedulerConfig scfg;
    scfg.num_workers = opt.workers;
    scfg.watermark_blocks = std::min(opt.watermark, std::max(0, opt.blocks / 4));
    Scheduler sched(mgr, scfg);
    sched.set_sampler(fake_sample);

    int peak_used = 0;
    long long sum_used = 0;
    long long sum_cached_free = 0;
    long long hook_samples = 0;
    sched.set_step_hook([&](const Scheduler&)
                        {
        int used = alloc.num_blocks() - alloc.num_free_blocks();
        if(used > peak_used) peak_used = used;
        sum_used += used;
        sum_cached_free += alloc.num_free_cached();
        ++hook_samples; });

    if(opt.workers > 1) sched.start_pool();

    std::vector<std::shared_ptr<Request>> reqs;
    reqs.reserve(specs.size());

    auto t0 = std::chrono::steady_clock::now();
    // 分批提交（每 2 个请求推进到有进展，模拟持续到达流并制造空闲块淘汰窗口）
    const size_t stride = 2;
    size_t idx = 0;
    while(idx < specs.size() || sched.has_work())
    {
        for(size_t k = 0; k < stride && idx < specs.size(); ++k, ++idx)
        {
            auto r = std::make_shared<Request>();
            r->all_tokens = specs[idx].prompt;
            r->max_new_tokens = specs[idx].max_new_tokens;
            reqs.push_back(r);
            sched.add_request(r);
        }
        // 推进若干步，让前一批有机会完成释放块，同时后一批不断到达
        int burst = (idx < specs.size()) ? 6 : 100000;
        for(int s = 0; s < burst && sched.has_work(); ++s)
            sched.step();
    }
    auto t1 = std::chrono::steady_clock::now();

    if(opt.workers > 1) sched.stop_pool();

    BenchResult res;
    res.workload = opt.workload;
    res.policy_name = (opt.policy == FreePolicy::LRU) ? "lru" : "fifo";
    res.prefix_on = opt.prefix_on;
    res.blocks = opt.blocks;
    res.query_blocks = mgr.prefix_cache_queries();
    res.hit_blocks = mgr.prefix_cache_hits();
    res.hit_pct = res.query_blocks > 0
                      ? 100.0 * static_cast<double>(res.hit_blocks) / static_cast<double>(res.query_blocks)
                      : 0.0;
    res.hit_tok_pct = mgr.prompt_tokens() > 0
                          ? 100.0 * static_cast<double>(res.hit_blocks * static_cast<uint64_t>(opt.block_size)) /
                                static_cast<double>(mgr.prompt_tokens())
                          : 0.0;
    res.evictions = alloc.num_evictions();
    res.revivals = alloc.num_revivals();
    res.preemptions = sched.num_preemptions();
    res.cow_copies = mgr.num_cow_copies();
    res.peak_util_pct = 100.0 * static_cast<double>(peak_used) / static_cast<double>(opt.blocks);
    res.avg_util_pct = hook_samples > 0
                           ? 100.0 * static_cast<double>(sum_used) /
                                 (static_cast<double>(hook_samples) * static_cast<double>(opt.blocks))
                           : 0.0;
    res.avg_cached_pct = hook_samples > 0
                             ? 100.0 * static_cast<double>(sum_cached_free) /
                                   (static_cast<double>(hook_samples) * static_cast<double>(opt.blocks))
                             : 0.0;
    res.steps = sched.num_steps();

    std::vector<double> ttfts;
    std::vector<double> tpots;
    for(const auto& r : reqs)
    {
        if(r->s != ReqState::FINISHED) continue;
        res.gen_tokens += r->num_generated_tokens;
        if(r->admit_step >= 0 && r->first_token_step >= r->admit_step)
        {
            ttfts.push_back(static_cast<double>(r->first_token_step - r->admit_step));
            double denom = static_cast<double>(std::max(1, r->num_generated_tokens - 1));
            tpots.push_back(static_cast<double>(r->finish_step - r->first_token_step) / denom);
        }
    }
    res.tok_per_step = res.steps > 0 ? static_cast<double>(res.gen_tokens) / static_cast<double>(res.steps) : 0.0;
    if(!ttfts.empty())
    {
        std::sort(ttfts.begin(), ttfts.end());
        res.avg_ttft = std::accumulate(ttfts.begin(), ttfts.end(), 0.0) / static_cast<double>(ttfts.size());
        size_t p99_idx = std::min(ttfts.size() - 1, static_cast<size_t>(std::ceil(0.99 * ttfts.size())) - 1);
        res.p99_ttft = ttfts[p99_idx];
    }
    if(!tpots.empty())
    {
        res.avg_tpot = std::accumulate(tpots.begin(), tpots.end(), 0.0) / static_cast<double>(tpots.size());
    }
    res.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

static void print_header(bool csv, bool timed)
{
    if(csv)
    {
        printf("workload,policy,prefix,blocks,q_blk,hit_blk,hit_pct,hit_tok_pct,evict,revive,preempt,cow,peak_util,avg_util,cached_free,steps,gen_tok,tok_per_step,avg_ttft,p99_ttft,avg_tpot%s\n",
               timed ? ",wall_ms" : "");
    }
    else
    {
        printf("%-10s %-5s %-6s %6s %6s %7s %7s %8s %6s %6s %7s %4s %6s %6s %6s %6s %7s %8s %7s %7s %7s%s\n",
               "workload", "pol", "prefix", "blocks", "q_blk", "hit_blk", "hit%", "hit_tok%",
               "evict", "revive", "preempt", "cow", "peak%", "avg%", "cach%", "steps",
               "gen_tok", "tok/step", "ttft", "p99", "tpot", timed ? "   ms" : "");
    }
}

static void print_row(const BenchResult& r, bool csv, bool timed)
{
    if(csv)
    {
        printf("%s,%s,%s,%d,%llu,%llu,%.2f,%.2f,%llu,%llu,%d,%llu,%.2f,%.2f,%.2f,%lld,%lld,%.2f,%.2f,%.2f,%.2f",
               r.workload.c_str(), r.policy_name.c_str(), r.prefix_on ? "on" : "off",
               r.blocks, static_cast<unsigned long long>(r.query_blocks),
               static_cast<unsigned long long>(r.hit_blocks), r.hit_pct, r.hit_tok_pct,
               static_cast<unsigned long long>(r.evictions), static_cast<unsigned long long>(r.revivals),
               r.preemptions, static_cast<unsigned long long>(r.cow_copies),
               r.peak_util_pct, r.avg_util_pct, r.avg_cached_pct,
               r.steps, r.gen_tokens, r.tok_per_step, r.avg_ttft, r.p99_ttft, r.avg_tpot);
        if(timed) printf(",%.2f", r.wall_ms);
        printf("\n");
    }
    else
    {
        printf("%-10s %-5s %-6s %6d %6llu %7llu %6.1f%% %7.1f%% %6llu %6llu %7d %4llu %5.1f%% %5.1f%% %5.1f%% %6lld %7lld %8.2f %7.2f %7.1f %7.2f",
               r.workload.c_str(), r.policy_name.c_str(), r.prefix_on ? "on" : "off",
               r.blocks, static_cast<unsigned long long>(r.query_blocks),
               static_cast<unsigned long long>(r.hit_blocks), r.hit_pct, r.hit_tok_pct,
               static_cast<unsigned long long>(r.evictions), static_cast<unsigned long long>(r.revivals),
               r.preemptions, static_cast<unsigned long long>(r.cow_copies),
               r.peak_util_pct, r.avg_util_pct, r.avg_cached_pct,
               r.steps, r.gen_tokens, r.tok_per_step, r.avg_ttft, r.p99_ttft, r.avg_tpot);
        if(timed) printf(" %6.1f", r.wall_ms);
        printf("\n");
    }
}

static void usage(const char* argv0)
{
    printf("usage: %s [options]\n", argv0);
    printf("  --workload <sysprompt|chat|zipf>  workload type (default: zipf)\n");
    printf("  --blocks N                        physical block pool size (default: 32)\n");
    printf("  --block-size N                    tokens per block (default: 16)\n");
    printf("  --policy <fifo|lru>               free block eviction policy (default: lru)\n");
    printf("  --prefix <on|off>                 enable prefix caching (default: on)\n");
    printf("  --requests N                      requests for sysprompt/zipf (default: 80)\n");
    printf("  --sessions N                      sessions for chat (default: 8)\n");
    printf("  --turns N                         turns per session for chat (default: 6)\n");
    printf("  --seed N                          RNG seed (default: 42)\n");
    printf("  --workers N                       worker threads (default: 1)\n");
    printf("  --sweep                           run built-in FIFO vs LRU vs prefix-off sweep\n");
    printf("  --csv                             CSV output\n");
    printf("  --timed                           include wall-clock milliseconds\n");
}

int main(int argc, char** argv)
{
    BenchOptions opt;
    for(int i = 1; i < argc; ++i)
    {
        auto next_arg = [&]() -> const char*
        {
            if(i + 1 >= argc)
            {
                fprintf(stderr, "missing value for %s\n", argv[i]);
                std::exit(2);
            }
            return argv[++i];
        };

        if(std::strcmp(argv[i], "--blocks") == 0)
            opt.blocks = std::atoi(next_arg());
        else if(std::strcmp(argv[i], "--block-size") == 0)
            opt.block_size = std::atoi(next_arg());
        else if(std::strcmp(argv[i], "--policy") == 0)
        {
            const char* p = next_arg();
            opt.policy = (std::strcmp(p, "fifo") == 0) ? FreePolicy::FIFO : FreePolicy::LRU;
        }
        else if(std::strcmp(argv[i], "--prefix") == 0)
        {
            const char* p = next_arg();
            opt.prefix_on = (std::strcmp(p, "off") != 0);
        }
        else if(std::strcmp(argv[i], "--workload") == 0)
            opt.workload = next_arg();
        else if(std::strcmp(argv[i], "--requests") == 0)
            opt.num_requests = std::atoi(next_arg());
        else if(std::strcmp(argv[i], "--sessions") == 0)
            opt.sessions = std::atoi(next_arg());
        else if(std::strcmp(argv[i], "--turns") == 0)
            opt.turns = std::atoi(next_arg());
        else if(std::strcmp(argv[i], "--seed") == 0)
            opt.seed = static_cast<uint32_t>(std::strtoul(next_arg(), nullptr, 10));
        else if(std::strcmp(argv[i], "--workers") == 0)
            opt.workers = std::atoi(next_arg());
        else if(std::strcmp(argv[i], "--csv") == 0)
            opt.csv = true;
        else if(std::strcmp(argv[i], "--sweep") == 0)
            opt.sweep = true;
        else if(std::strcmp(argv[i], "--timed") == 0)
            opt.timed = true;
        else if(std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0)
        {
            usage(argv[0]);
            return 0;
        }
        else
        {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    print_header(opt.csv, opt.timed);

    if(opt.sweep)
    {
        const std::vector<std::string> workloads = {"sysprompt", "chat", "zipf"};
        const std::vector<int> pool_sizes = {24, 32, 48, 64};
        for(const auto& wl : workloads)
        {
            BenchOptions base = opt;
            base.workload = wl;
            auto specs = build_workload(base);

            for(int blocks : pool_sizes)
            {
                // 对照下界：prefix off
                BenchOptions o_off = base;
                o_off.blocks = blocks;
                o_off.policy = FreePolicy::LRU;
                o_off.prefix_on = false;
                print_row(run_once(o_off, specs), opt.csv, opt.timed);

                // FIFO + prefix on
                BenchOptions o_fifo = base;
                o_fifo.blocks = blocks;
                o_fifo.policy = FreePolicy::FIFO;
                o_fifo.prefix_on = true;
                print_row(run_once(o_fifo, specs), opt.csv, opt.timed);

                // LRU + prefix on
                BenchOptions o_lru = base;
                o_lru.blocks = blocks;
                o_lru.policy = FreePolicy::LRU;
                o_lru.prefix_on = true;
                print_row(run_once(o_lru, specs), opt.csv, opt.timed);
            }
        }
        return 0;
    }

    auto specs = build_workload(opt);
    print_row(run_once(opt, specs), opt.csv, opt.timed);
    return 0;
}
