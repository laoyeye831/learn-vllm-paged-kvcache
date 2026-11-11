// examples/demo.cpp
// 三个可观测的演示：
//   A. 前缀命中 + 写时复制：单线程直接操作 BlockManager，看 ref_count 变化
//   B. 前缀缓存跨批次复用：第一波请求把满块登记进哈希表，第二波请求直接命中
//   C. 空间不足时的抢占与重算：小池 + 长请求，观察 preemptions 与最终输出
//
// 构建：cmake --build build --target kvcache_demo
// 运行：./build/kvcache_demo           （ctest 用的就是这条：无停顿、无转义序列）
//       ./build/kvcache_demo --paced   （录制 GIF 用：每步停顿 + 配色）

#include "kvcache.h"

#include <algorithm>
#include <climits>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <map>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX // windows.h 的 max() 宏会破坏 std::max
#include <windows.h>
#endif

using namespace std::chrono;

// ---------- 讲解节奏（只在 --paced 时生效） ----------
// ctest 里跑的 kvcache_demo 不带参数：零停顿、无转义序列，输出与默认行为一致
namespace
{
constexpr const char* C_RESET = "\033[0m";
constexpr const char* C_TITLE = "\033[1;36m"; // 粗体青色：章节标题
constexpr const char* C_OK = "\033[32m";      // 绿色：命中 / 复用
constexpr const char* C_REF = "\033[33m";     // 黄色：引用计数变化
constexpr const char* C_HOT = "\033[1;31m";   // 红色：CoW 分裂

bool g_paced = false;
int g_beat_ms = 900; // 每个节拍的默认停顿，可用 --beat-ms=N 调整

const char* tint(const char* code)
{
    return g_paced ? code : "";
}

// Windows 10+ 的控制台默认不解析 ANSI，需要显式打开 VT 处理
void enable_ansi()
{
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if(h != INVALID_HANDLE_VALUE && GetConsoleMode(h, &mode))
        SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
}

// 停顿一拍，让录屏（VHS）有机会截到这一屏
void beat(int ms = 0)
{
    if(!g_paced) return;
    fflush(stdout);
    std::this_thread::sleep_for(milliseconds(ms > 0 ? ms : g_beat_ms));
}

void section(const char* title)
{
    printf("\n%s=== %s ===%s\n", tint(C_TITLE), title, tint(C_RESET));
    beat();
}
} // namespace

// ---------- 小工具 ----------

static std::vector<int> make_tokens(int base, int len)
{
    std::vector<int> v(static_cast<size_t>(len));
    for(int i = 0; i < len; ++i)
        v[static_cast<size_t>(i)] = base + i;
    return v;
}

static const char* state_name(ReqState s)
{
    switch(s)
    {
    case ReqState::WAITING:
        return "WAITING";
    case ReqState::RUNNING:
        return "RUNNING";
    case ReqState::SWAPPED:
        return "SWAPPED";
    case ReqState::FINISHED:
        return "FINISHED";
    case ReqState::ABORTED:
        return "ABORTED";
    }
    return "?";
}

static void print_blocks(const char* tag, const Request& r, BlockAllocator& alloc)
{
    printf("  %-18s block_table=[", tag);
    for(size_t i = 0; i < r.block_table.size(); ++i)
    {
        const int ref = alloc.ref_count_of(r.block_table[i]);
        // 共享块（ref>1）用黄色标出，方便一眼看到复用发生在哪儿
        printf("%s%s%d(ref=%d)%s", i ? " " : "", tint(ref > 1 ? C_REF : ""),
               r.block_table[i], ref, tint(ref > 1 ? C_RESET : ""));
    }
    printf("]\n");
}

struct Spec
{
    int prompt_len;
    int max_new;
};

// 每步只在状态变化时打印一行，避免刷屏；同时快照两个指标：
//   min_free —— 本轮最小空闲块数，用来说明「缓存省了多少块」
//   hits     —— 每个请求被观察到的最多已缓存块数（release() 会把它清零，所以要在运行中取）
struct Trace
{
    size_t prev_run = static_cast<size_t>(-1);
    size_t prev_wait = 0;
    int prev_pre = -1;
    int min_free = INT_MAX;
    std::map<const Request*, int> hits;
};

static std::shared_ptr<Trace> attach_trace(Scheduler& sched, BlockAllocator& alloc)
{
    auto t = std::make_shared<Trace>();
    sched.set_step_hook([&alloc, t](const Scheduler& s)
                        {
        const int free_blocks = alloc.num_free_blocks();
        if (free_blocks < t->min_free) t->min_free = free_blocks;

        const auto running = s.running();
        const auto waiting = s.waiting();
        for (const auto& r : running)
            t->hits[r.get()] = std::max(t->hits[r.get()], r->num_cached_blocks);
        for (const auto& r : waiting)
            t->hits[r.get()] = std::max(t->hits[r.get()], r->num_cached_blocks);

        const int pre = s.num_preemptions();
        if (running.size() != t->prev_run || waiting.size() != t->prev_wait || pre != t->prev_pre)
        {
            printf("    step=%3lld  running=%zu  waiting=%zu  free=%2d  preemptions=%d\n",
                   s.num_steps(), running.size(), waiting.size(), free_blocks, pre);
            t->prev_run = running.size();
            t->prev_wait = waiting.size();
            t->prev_pre = pre;
        } });
    return t;
}

// 采样器：模拟一点计算耗时，让 Worker 池的并行有意义
static int slow_sample(const Request& r)
{
    std::this_thread::sleep_for(microseconds(1000));
    return fake_sample(r);
}

static void print_results(const char* tag,
                          const std::vector<Spec>& specs,
                          const std::vector<std::shared_ptr<Request>>& reqs,
                          const std::shared_ptr<Trace>& trace)
{
    printf("  %s:\n", tag);
    printf("    %-5s %-8s %-8s %-10s %-11s %s\n",
           "req", "prompt", "max_new", "generated", "cached_blk", "state");
    for(size_t i = 0; i < reqs.size(); ++i)
    {
        auto it = trace->hits.find(reqs[i].get());
        int cached = (it == trace->hits.end()) ? 0 : it->second;
        const char* ok = tint(cached > 0 ? C_OK : "");
        printf("    #%-4zu %-8d %-8d %-10d %s%-11d%s %s\n",
               i, specs[i].prompt_len, specs[i].max_new,
               reqs[i]->num_generated_tokens, ok, cached, tint(cached > 0 ? C_RESET : ""),
               state_name(reqs[i]->s));
    }
}

// ============================================================
//  A. 前缀缓存命中 + 尾块写时复制
// ============================================================
static void demo_prefix_and_cow()
{
    section("A. Prefix reuse and copy-on-write");
    printf("  BLOCK_SIZE=%d, 16 physical blocks\n", BLOCK_SIZE);

    BlockAllocator alloc(16);
    BlockManager mgr(&alloc);

    // 40 个 token = 2 个满块(32) + 1 个半块(8)
    // 前缀缓存只登记满块，所以第二个请求命中 2 块，第 3 块必须新分配
    auto prompt = make_tokens(1000, 40);

    Request r1;
    bool ok = prefill(prompt, r1, mgr);
    printf("\n  [1] r1 prefill(%d tokens): ok=%d, registered full blocks=%d, free=%d\n",
           static_cast<int>(prompt.size()), ok, r1.num_cached_blocks, alloc.num_free_blocks());
    print_blocks("r1", r1, alloc);
    beat();

    Request r2;
    ok = prefill(prompt, r2, mgr);
    printf("\n  [2] r2 prefill same prompt: ok=%d, hit blocks=%d, free=%d\n",
           ok, r2.num_cached_blocks, alloc.num_free_blocks());
    printf("      %s-> first %d blocks reuse r1's blocks (ref 1 -> 2), only 1 block is new%s\n",
           tint(C_OK), r2.num_cached_blocks, tint(C_RESET));
    print_blocks("r2", r2, alloc);
    beat();

    // fork：共享父请求的全部块（对应 beam search / 并行采样）
    Request son = fork_sequence(r2, alloc);
    printf("\n  [3] fork_sequence(r2) -> ref_count +1 for every block, free still %d\n",
           alloc.num_free_blocks());
    print_blocks("son", son, alloc);
    beat();

    // 尾块被两条序列共享，此时写入必须触发 CoW
    int tail_before = son.block_table.back();
    int tail_ref_before = alloc.ref_count_of(tail_before);
    int slot = append_slot(mgr, son);
    append_token(mgr, son, 7);
    printf("\n  [4] son decodes one token (tail block ref_count>1 -> CoW)\n");
    printf("      %sBLK #%d (ref=%d) splits -> BLK #%d (ref=%d); #%d stays shared by r2 (ref=%d)%s\n",
           tint(C_HOT), tail_before, tail_ref_before, son.block_table.back(),
           alloc.ref_count_of(son.block_table.back()), tail_before,
           alloc.ref_count_of(tail_before), tint(C_RESET));
    printf("      tail %d -> %d, slot=%d, free=%d\n",
           tail_before, son.block_table.back(), slot, alloc.num_free_blocks());
    print_blocks("son", son, alloc);
    print_blocks("r2 (untouched)", r2, alloc);
    beat(1500);

    // 引用计数归零才真正回收到空闲链表
    mgr.release(son);
    mgr.release(r2);
    mgr.release(r1);
    printf("\n  [5] all requests released: free = %d/%d\n",
           alloc.num_free_blocks(), alloc.num_blocks());
    beat();
}

// ============================================================
//  B. 前缀缓存跨批次复用
// ============================================================
static void demo_prefix_cache_waves()
{
    section("B. Cross-batch prefix cache reuse");

    BlockAllocator alloc(32);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr, 2);
    sched.set_sampler(slow_sample);

    // 48 token = 3 个满块，正好全部可被前缀缓存登记
    auto common = make_tokens(2000, 48);

    printf("  common prefix: %d tokens (= %d full blocks)\n",
           static_cast<int>(common.size()), static_cast<int>(common.size()) / BLOCK_SIZE);

    // ---- 第一波：3 个完全相同 prompt 的请求同一步准入 ----
    // 第一个请求登记 3 个满块，同一步内后两个请求立刻命中
    printf("\n  [wave 1] 3 requests, all with the identical 48-token prompt\n");
    // 回调必须在 start() 之前设置（运行中替换 std::function 会与 Worker 竞争）
    auto trace = attach_trace(sched, alloc);

    std::vector<Spec> specs1;
    std::vector<std::shared_ptr<Request>> reqs1;
    std::vector<std::future<void>> futures1;
    for(int i = 0; i < 3; ++i)
    {
        specs1.push_back(Spec{static_cast<int>(common.size()), 2});
        auto r = std::make_shared<Request>();
        r->all_tokens = common;
        r->max_new_tokens = 2;
        reqs1.push_back(r);
        futures1.push_back(sched.add_request(r));
    }
    sched.start();
    for(auto& f : futures1)
        f.wait();
    print_results("wave 1 (cached_blk=3 means 3 full blocks; #1/#2 hit blocks #0 just registered)",
                  specs1, reqs1, trace);
    printf("    peak usage %d blocks = 3 shared prefix + 3 private decode (12 without sharing)\n",
           alloc.num_blocks() - trace->min_free);
    printf("    free=%d/%d (all finished: refs hit zero, blocks are back on the free list)\n",
           alloc.num_free_blocks(), alloc.num_blocks());
    beat();

    // ---- 第二波：共享同一前缀，尾部各自不同 ----
    // 第一波已结束，但满块的哈希仍登记在 hash_to_block_ 中 → 仍可命中
    printf("\n  [wave 2] 3 requests: 48-token common prefix + 16 private tokens (4 blocks each)\n");
    beat(600);

    // 跨波的 happens-before 由 future 保证：scheduler 线程不会再动这些字段
    trace->min_free = INT_MAX;
    trace->hits.clear();
    trace->prev_run = static_cast<size_t>(-1);

    std::vector<Spec> specs2;
    std::vector<std::shared_ptr<Request>> reqs2;
    std::vector<std::future<void>> futures2;
    for(int i = 0; i < 3; ++i)
    {
        std::vector<int> p = common;
        for(int t = 0; t < 16; ++t)
            p.push_back(9000 + i * 100 + t);

        // 提交前查一次，能看到「已被第一波登记」的 3 个块
        // 预查必须放在所有 add_request() 之前：BlockManager 只由调度线程改，
        // 一旦有请求入队，调度线程就可能 prefill 并写哈希表，主线程再查就是数据竞争
        auto preview = mgr.match_prefix(p);
        printf("    req#%d pre-check before submit: %d of %d blocks already cached\n", i,
               static_cast<int>(preview.cached_blocks.size()),
               static_cast<int>((p.size() + BLOCK_SIZE - 1) / BLOCK_SIZE));

        specs2.push_back(Spec{static_cast<int>(p.size()), 2});
        auto r = std::make_shared<Request>();
        r->all_tokens = p;
        r->max_new_tokens = 2;
        reqs2.push_back(r);
    }
    for(auto& r : reqs2)
        futures2.push_back(sched.add_request(r));

    for(auto& f : futures2)
        f.wait();
    sched.stop(true);
    print_results("wave 2 (pre-check already hit 3 blocks; cached_blk also counts the 4th)",
                  specs2, reqs2, trace);
    printf("    peak usage %d blocks (without sharing, 3 concurrent requests need 4 each = 12)\n",
           alloc.num_blocks() - trace->min_free);
    printf("    steps=%lld, preemptions=%d, free=%d/%d\n",
           sched.num_steps(), sched.num_preemptions(),
           alloc.num_free_blocks(), alloc.num_blocks());
    beat();
}

// ============================================================
//  C. 抢占与重算
// ============================================================
static void demo_preemption()
{
    constexpr int kBlocks = 16;

    section("C. Preemption and recompute");
    printf("  %d physical blocks (%d usable)\n", kBlocks, kBlocks - WATER_MARK);

    BlockAllocator alloc(kBlocks);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr, 2);
    sched.set_sampler(slow_sample);
    auto trace = attach_trace(sched, alloc);

    // 每个请求最终需要 ceil((32+56)/16) = 6 块，4 个共 24 块 >> 16 → 多次抢占
    constexpr int kNumReq = 4;
    std::vector<Spec> specs;
    std::vector<std::shared_ptr<Request>> reqs;
    std::vector<std::future<void>> futures;

    printf("  submit %d requests: prompt=32 tokens, max_new=56 (6 blocks each at the end)\n", kNumReq);
    for(int i = 0; i < kNumReq; ++i)
    {
        auto p = make_tokens(4000 + i * 700, 32);
        specs.push_back(Spec{32, 56});
        auto r = std::make_shared<Request>();
        r->all_tokens = p;
        r->max_new_tokens = 56;
        reqs.push_back(r);
        futures.push_back(sched.add_request(r));
    }

    sched.start();
    for(auto& f : futures)
        f.wait();
    sched.stop(true);

    print_results("results (no output lost across preemption; recovered by recompute)", specs, reqs, trace);
    printf("    steps=%lld, preemptions=%d, free=%d/%d\n",
           sched.num_steps(), sched.num_preemptions(),
           alloc.num_free_blocks(), alloc.num_blocks());
    printf("    preemption: free the victim's blocks -> mark it WAITING -> push to the queue front;\n");
    printf("                on resume, prefill all_tokens (prompt + generated) again, i.e. recompute\n");
    beat();
}

static void usage(const char* argv0)
{
    printf("usage: %s [options]\n", argv0);
    printf("  --paced        pause after each step and colorize the output (for GIF recording; off by default)\n");
    printf("  --beat-ms=N    milliseconds per beat (default 900, minimum 100)\n");
    printf("  -h, --help     show this help\n");
}

int main(int argc, char** argv)
{
#ifdef _WIN32
    // Windows 控制台默认按本地代码页（如 GBK）解码，统一切到 UTF-8 避免非 ASCII 字符乱码
    SetConsoleOutputCP(CP_UTF8);
#endif

    for(int i = 1; i < argc; ++i)
    {
        if(std::strcmp(argv[i], "--paced") == 0)
        {
            g_paced = true;
            enable_ansi();
        }
        else if(std::strncmp(argv[i], "--beat-ms=", 10) == 0)
        {
            g_beat_ms = std::max(100, std::atoi(argv[i] + 10));
        }
        else if(std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0)
        {
            usage(argv[0]);
            return 0;
        }
        else
        {
            printf("unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    printf("%smini KV Cache demo%s - reproducing vLLM's paged KV cache and scheduler\n",
           tint(C_TITLE), tint(C_RESET));
    beat();
    demo_prefix_and_cow();
    demo_prefix_cache_waves();
    demo_preemption();
    printf("\n%sdone%s. Mechanics: docs/ARCHITECTURE.md | vLLM comparison: docs/VLLM_MAPPING.md\n",
           tint(C_OK), tint(C_RESET));
    beat(1200);
    return 0;
}
