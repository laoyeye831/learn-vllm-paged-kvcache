// bench/kvcache_bench.cpp
// 可复现的前缀缓存与调度策略对照基准：
//   1) --sweep         : 对比 FIFO / LRU / prefix-off（sysprompt / chat / zipf）
//   2) --chunked-sweep : 对比 Chunked Prefill 改造前后的 TTFT、ITL p50、ITL p99 与吞吐量（mixed 混合长短请求负载）
//
// 用法示例：
//   ./build/kvcache_bench --sweep
//   ./build/kvcache_bench --chunked-sweep
//   ./build/kvcache_bench --workload mixed --blocks 256 --chunked on --budget 64

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
    bool chunked_prefill{false};
    int token_budget{0};
    int max_num_seqs{0};
    int waiting_reserve{0};
    std::string workload{"zipf"};
    int num_requests{80};
    int sessions{8};
    int turns{6};
    uint32_t seed{42};
    int workers{1};
    bool csv{false};
    bool sweep{false};
    bool chunked_sweep{false};
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
    bool chunked_on{false};
    int budget{0};
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
    // 基于确定性 step-cost 模型（base + per_tok * scheduled_tokens）的模拟延迟指标（单位：ms）
    double avg_ttft_ms{0.0};
    double p99_ttft_ms{0.0};
    double itl_p50_ms{0.0};
    double itl_p99_ms{0.0};
    double throughput_tok_s{0.0};
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

static double percentile(std::vector<double>& vals, double p)
{
    if(vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    size_t idx = std::min(vals.size() - 1, static_cast<size_t>(std::ceil(p * static_cast<double>(vals.size()))) - 1);
    return vals[idx];
}

// 1) sysprompt：共享长 system prompt + 独立短问题（中间穿插无前缀冷请求）
static std::vector<ReqSpec> gen_sysprompt(const BenchOptions& opt)
{
    std::mt19937 rng(opt.seed);
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
            s.prompt = sys_prompts[0];
        else if(k < 8)
            s.prompt = sys_prompts[1];
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
        history[static_cast<size_t>(s)] = random_tokens(rng, opt.block_size * 2, 1, 30000);

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
        docs.push_back(random_tokens(rng, opt.block_size * 4, 1 + d * 2000, 1999 + d * 2000));

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

// 4) mixed：Sarathi-Serve / vLLM v1 经典长短混合负载
//    约 80% 短 prompt（24~48 token，长 decode 32~48 token）+ 20% 超长 prompt（384~512 token）
//    用于量化 Chunked Prefill 对 decode ITL p50/p99 与 TTFT 的影响
static std::vector<ReqSpec> gen_mixed(const BenchOptions& opt)
{
    std::mt19937 rng(opt.seed);
    std::uniform_int_distribution<int> short_p(24, 48);
    std::uniform_int_distribution<int> short_g(32, 48);
    std::uniform_int_distribution<int> long_p(384, 512);
    std::uniform_int_distribution<int> long_g(16, 24);

    std::vector<ReqSpec> specs;
    specs.reserve(static_cast<size_t>(opt.num_requests));
    for(int i = 0; i < opt.num_requests; ++i)
    {
        ReqSpec s;
        if(i % 5 == 3)
        {
            s.prompt = random_tokens(rng, long_p(rng), 1, 49000);
            s.max_new_tokens = long_g(rng);
        }
        else
        {
            s.prompt = random_tokens(rng, short_p(rng), 1, 49000);
            s.max_new_tokens = short_g(rng);
        }
        specs.push_back(std::move(s));
    }
    return specs;
}

static std::vector<ReqSpec> build_workload(const BenchOptions& opt)
{
    if(opt.workload == "sysprompt") return gen_sysprompt(opt);
    if(opt.workload == "chat") return gen_chat(opt);
    if(opt.workload == "mixed") return gen_mixed(opt);
    return gen_zipf(opt);
}

static BenchResult run_once(const BenchOptions& opt, const std::vector<ReqSpec>& specs)
{
    BlockAllocator alloc(opt.blocks, opt.block_size, opt.policy);
    BlockManager mgr(&alloc, BlockManagerConfig{opt.prefix_on});
    SchedulerConfig scfg;
    scfg.num_workers = opt.workers;
    scfg.watermark_blocks = std::min(opt.watermark, std::max(0, opt.blocks / 4));
    scfg.max_num_batched_tokens = opt.chunked_prefill ? opt.token_budget : 0;
    scfg.enable_chunked_prefill = opt.chunked_prefill;
    scfg.max_num_seqs = opt.max_num_seqs;
    scfg.waiting_token_reserve = opt.waiting_reserve;
    Scheduler sched(mgr, scfg);
    sched.set_sampler(fake_sample);

    struct ReqTrace
    {
        std::shared_ptr<Request> req;
        double arrival_ms{0.0};
        double last_tok_ms{-1.0};
        int prev_gen{0};
    };
    std::vector<ReqTrace> traces;
    traces.reserve(specs.size());

    // 确定性 step-cost 模拟时钟：每步耗时 = 基础开销 5ms + 0.25ms * 本步计算总 token 数
    // 该模型反映真实推理中「大 prefill 步拉长全批迭代时间」的物理特性
    constexpr double kBaseStepMs = 5.0;
    constexpr double kPerTokenMs = 0.25;
    double sim_time_ms = 0.0;

    std::vector<double> ttfts_ms;
    std::vector<double> itls_ms;

    int peak_used = 0;
    long long sum_used = 0;
    long long sum_cached_free = 0;
    long long hook_samples = 0;
    sched.set_step_hook([&](const Scheduler& s)
                        {
        int used = alloc.num_blocks() - alloc.num_free_blocks();
        if(used > peak_used) peak_used = used;
        sum_used += used;
        sum_cached_free += alloc.num_free_cached();
        ++hook_samples;

        const int step_tokens = s.last_step_scheduled_tokens();
        sim_time_ms += kBaseStepMs + kPerTokenMs * static_cast<double>(step_tokens);

        for(auto& tr : traces)
        {
            int cur_gen = tr.req->num_generated_tokens;
            if(cur_gen > tr.prev_gen)
            {
                if(tr.prev_gen == 0)
                    ttfts_ms.push_back(sim_time_ms - tr.arrival_ms);
                else
                    itls_ms.push_back(sim_time_ms - tr.last_tok_ms);
                tr.last_tok_ms = sim_time_ms;
                tr.prev_gen = cur_gen;
            }
        } });

    if(opt.workers > 1) sched.start_pool();

    auto t0 = std::chrono::steady_clock::now();
    const size_t stride = 2;
    size_t idx = 0;
    while(idx < specs.size() || sched.has_work())
    {
        for(size_t k = 0; k < stride && idx < specs.size(); ++k, ++idx)
        {
            auto r = std::make_shared<Request>();
            r->all_tokens = specs[idx].prompt;
            r->max_new_tokens = specs[idx].max_new_tokens;
            traces.push_back(ReqTrace{r, sim_time_ms, -1.0, 0});
            sched.add_request(r);
        }
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
    res.chunked_on = opt.chunked_prefill;
    res.budget = opt.chunked_prefill ? opt.token_budget : 0;
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

    for(const auto& tr : traces)
    {
        if(tr.req->s == ReqState::FINISHED) res.gen_tokens += tr.req->num_generated_tokens;
    }
    res.tok_per_step = res.steps > 0 ? static_cast<double>(res.gen_tokens) / static_cast<double>(res.steps) : 0.0;
    if(!ttfts_ms.empty())
    {
        res.avg_ttft_ms = std::accumulate(ttfts_ms.begin(), ttfts_ms.end(), 0.0) / static_cast<double>(ttfts_ms.size());
        res.p99_ttft_ms = percentile(ttfts_ms, 0.99);
    }
    res.itl_p50_ms = percentile(itls_ms, 0.50);
    res.itl_p99_ms = percentile(itls_ms, 0.99);
    res.throughput_tok_s = sim_time_ms > 0.0 ? static_cast<double>(res.gen_tokens) / (sim_time_ms / 1000.0) : 0.0;
    res.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

static void print_header(bool csv, bool timed)
{
    if(csv)
    {
        printf("workload,policy,prefix,chunked,budget,blocks,q_blk,hit_blk,hit_pct,evict,revive,preempt,peak_util,steps,gen_tok,tok_per_step,avg_ttft_ms,p99_ttft_ms,itl_p50_ms,itl_p99_ms,throughput_tok_s%s\n",
               timed ? ",wall_ms" : "");
    }
    else
    {
        printf("%-9s %-4s %-3s %-3s %4s %4s %6s %6s %5s %5s %4s %6s %6s %7s %7s %7s %7s %8s%s\n",
               "workload", "pol", "pre", "chk", "bgt", "blks", "hit%", "evict", "rev", "preem",
               "stps", "tok/st", "ttft", "ttft99", "itl50", "itl99", "tok/s", "peak%", timed ? "   ms" : "");
    }
}

static void print_row(const BenchResult& r, bool csv, bool timed)
{
    if(csv)
    {
        printf("%s,%s,%s,%s,%d,%d,%llu,%llu,%.2f,%llu,%llu,%d,%.2f,%lld,%lld,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",
               r.workload.c_str(), r.policy_name.c_str(), r.prefix_on ? "on" : "off",
               r.chunked_on ? "on" : "off", r.budget, r.blocks,
               static_cast<unsigned long long>(r.query_blocks), static_cast<unsigned long long>(r.hit_blocks),
               r.hit_pct, static_cast<unsigned long long>(r.evictions),
               static_cast<unsigned long long>(r.revivals), r.preemptions, r.peak_util_pct,
               r.steps, r.gen_tokens, r.tok_per_step,
               r.avg_ttft_ms, r.p99_ttft_ms, r.itl_p50_ms, r.itl_p99_ms, r.throughput_tok_s);
        if(timed) printf(",%.2f", r.wall_ms);
        printf("\n");
    }
    else
    {
        printf("%-9s %-4s %-3s %-3s %4d %4d %5.1f%% %6llu %5llu %5d %4lld %6.2f %6.1f %7.1f %7.1f %7.1f %7.1f %7.1f%%",
               r.workload.c_str(), r.policy_name.c_str(), r.prefix_on ? "on" : "off",
               r.chunked_on ? "on" : "off", r.budget, r.blocks,
               r.hit_pct, static_cast<unsigned long long>(r.evictions),
               static_cast<unsigned long long>(r.revivals), r.preemptions,
               r.steps, r.tok_per_step,
               r.avg_ttft_ms, r.p99_ttft_ms, r.itl_p50_ms, r.itl_p99_ms, r.throughput_tok_s,
               r.peak_util_pct);
        if(timed) printf(" %6.1f", r.wall_ms);
        printf("\n");
    }
}

static void usage(const char* argv0)
{
    printf("usage: %s [options]\n", argv0);
    printf("  --workload <sysprompt|chat|zipf|mixed>  workload type (default: zipf)\n");
    printf("  --blocks N                              physical block pool size (default: 32)\n");
    printf("  --block-size N                          tokens per block (default: 16)\n");
    printf("  --policy <fifo|lru>                     free block eviction policy (default: lru)\n");
    printf("  --prefix <on|off>                       enable prefix caching (default: on)\n");
    printf("  --chunked <on|off>                      enable chunked prefill (default: off)\n");
    printf("  --budget N                              max_num_batched_tokens when chunked=on\n");
    printf("  --max-seqs N                            max_num_seqs limit (0=unlimited)\n");
    printf("  --reserve N                             waiting_token_reserve (default: 0)\n");
    printf("  --requests N                            requests for sysprompt/zipf/mixed (default: 80)\n");
    printf("  --sessions N                            sessions for chat (default: 8)\n");
    printf("  --turns N                               turns per session for chat (default: 6)\n");
    printf("  --seed N                                RNG seed (default: 42)\n");
    printf("  --workers N                             worker threads (default: 1)\n");
    printf("  --sweep                                 run FIFO vs LRU vs prefix-off sweep\n");
    printf("  --chunked-sweep                         run chunked prefill off vs on (TTFT/ITL/throughput)\n");
    printf("  --csv                                   CSV output\n");
    printf("  --timed                                 include wall-clock milliseconds\n");
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
        else if(std::strcmp(argv[i], "--chunked") == 0)
        {
            const char* p = next_arg();
            opt.chunked_prefill = (std::strcmp(p, "on") == 0);
        }
        else if(std::strcmp(argv[i], "--budget") == 0)
            opt.token_budget = std::atoi(next_arg());
        else if(std::strcmp(argv[i], "--max-seqs") == 0)
            opt.max_num_seqs = std::atoi(next_arg());
        else if(std::strcmp(argv[i], "--reserve") == 0)
            opt.waiting_reserve = std::atoi(next_arg());
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
        else if(std::strcmp(argv[i], "--chunked-sweep") == 0)
            opt.chunked_sweep = true;
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

    if(opt.chunked_sweep)
    {
        BenchOptions base = opt;
        base.workload = "mixed";
        base.blocks = 256;
        base.num_requests = 50;
        base.policy = FreePolicy::LRU;
        base.prefix_on = true;
        base.max_num_seqs = 16;
        auto specs = build_workload(base);

        // 改造前：关闭 chunked prefill（整条长 prompt 一步算完，阻塞同批 decode）
        BenchOptions o_off = base;
        o_off.chunked_prefill = false;
        o_off.token_budget = 0;
        print_row(run_once(o_off, specs), opt.csv, opt.timed);

        // 改造后：开启 chunked prefill，对比不同 token budget（32 / 64 / 128）
        for(int bgt : {32, 64, 128})
        {
            BenchOptions o_on = base;
            o_on.chunked_prefill = true;
            o_on.token_budget = bgt;
            o_on.waiting_reserve = std::min(16, bgt / 4);
            print_row(run_once(o_on, specs), opt.csv, opt.timed);
        }
        return 0;
    }

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
                BenchOptions o_off = base;
                o_off.blocks = blocks;
                o_off.policy = FreePolicy::LRU;
                o_off.prefix_on = false;
                print_row(run_once(o_off, specs), opt.csv, opt.timed);

                BenchOptions o_fifo = base;
                o_fifo.blocks = blocks;
                o_fifo.policy = FreePolicy::FIFO;
                o_fifo.prefix_on = true;
                print_row(run_once(o_fifo, specs), opt.csv, opt.timed);

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
