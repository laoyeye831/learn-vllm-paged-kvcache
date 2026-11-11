// examples/demo.cpp
// 三个可观测的演示：
//   A. 前缀命中 + 写时复制：单线程直接操作 BlockManager，看 ref_count 变化
//   B. 前缀缓存跨批次复用：第一波请求把满块登记进哈希表，第二波请求直接命中
//   C. 空间不足时的抢占与重算：小池 + 长请求，观察 preemptions 与最终输出
//
// 构建：cmake --build build --target kvcache_demo
// 运行：./build/kvcache_demo

#include "kvcache.h"

#include <algorithm>
#include <climits>
#include <chrono>
#include <cstdio>
#include <future>
#include <map>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

using namespace std::chrono;

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
        printf("%s%d(ref=%d)", i ? " " : "", r.block_table[i], alloc.ref_count_of(r.block_table[i]));
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
    printf("  %s：\n", tag);
    printf("    %-5s %-8s %-8s %-10s %-11s %s\n",
           "req", "prompt", "max_new", "generated", "cached_blk", "state");
    for(size_t i = 0; i < reqs.size(); ++i)
    {
        auto it = trace->hits.find(reqs[i].get());
        int cached = (it == trace->hits.end()) ? 0 : it->second;
        printf("    #%-4zu %-8d %-8d %-10d %-11d %s\n",
               i, specs[i].prompt_len, specs[i].max_new,
               reqs[i]->num_generated_tokens, cached,
               state_name(reqs[i]->s));
    }
}

// ============================================================
//  A. 前缀缓存命中 + 尾块写时复制
// ============================================================
static void demo_prefix_and_cow()
{
    printf("\n=== A. 前缀命中与写时复制 ===\n");
    printf("  BLOCK_SIZE=%d，16 个物理块\n", BLOCK_SIZE);

    BlockAllocator alloc(16);
    BlockManager mgr(&alloc);

    // 40 个 token = 2 个满块(32) + 1 个半块(8)
    // 前缀缓存只登记满块，所以第二个请求命中 2 块，第 3 块必须新分配
    auto prompt = make_tokens(1000, 40);

    Request r1;
    bool ok = prefill(prompt, r1, mgr);
    printf("\n  [1] r1 prefill(%d tokens): ok=%d, 登记满块=%d, 空闲块=%d\n",
           static_cast<int>(prompt.size()), ok, r1.num_cached_blocks, alloc.num_free_blocks());
    print_blocks("r1", r1, alloc);

    Request r2;
    ok = prefill(prompt, r2, mgr);
    printf("\n  [2] r2 prefill 同一 prompt: ok=%d, 命中块=%d, 空闲块=%d\n",
           ok, r2.num_cached_blocks, alloc.num_free_blocks());
    printf("      → 前 %d 块直接复用 r1 的物理块（ref_count 变 2），只新分配 1 块\n",
           r2.num_cached_blocks);
    print_blocks("r2", r2, alloc);

    // fork：共享父请求的全部块（对应 beam search / 并行采样）
    Request son = fork_sequence(r2, alloc);
    printf("\n  [3] fork_sequence(r2) → ref_count 全部 +1，空闲块仍为 %d\n",
           alloc.num_free_blocks());
    print_blocks("son", son, alloc);

    // 尾块被两条序列共享，此时写入必须触发 CoW
    int tail_before = son.block_table.back();
    int slot = append_slot(mgr, son);
    append_token(mgr, son, 7);
    printf("\n  [4] son decode 一个 token（尾块 ref_count>1 → CoW）\n");
    printf("      尾块 %d → %d，slot=%d，空闲块=%d\n",
           tail_before, son.block_table.back(), slot, alloc.num_free_blocks());
    print_blocks("son", son, alloc);
    print_blocks("r2（未受影响）", r2, alloc);

    // 引用计数归零才真正回收到空闲链表
    mgr.release(son);
    mgr.release(r2);
    mgr.release(r1);
    printf("\n  [5] 全部释放后：空闲块 = %d/%d\n",
           alloc.num_free_blocks(), alloc.num_blocks());
}

// ============================================================
//  B. 前缀缓存跨批次复用
// ============================================================
static void demo_prefix_cache_waves()
{
    printf("\n=== B. 前缀缓存跨批次复用 ===\n");

    BlockAllocator alloc(32);
    BlockManager mgr(&alloc);
    Scheduler sched(mgr, 2);
    sched.set_sampler(slow_sample);

    // 48 token = 3 个满块，正好全部可被前缀缓存登记
    auto common = make_tokens(2000, 48);

    printf("  公共前缀：%d tokens（= %d 个满块）\n",
           static_cast<int>(common.size()), static_cast<int>(common.size()) / BLOCK_SIZE);

    // ---- 第一波：3 个完全相同 prompt 的请求同一步准入 ----
    // 第一个请求登记 3 个满块，同一步内后两个请求立刻命中
    printf("\n  [第一波] 3 个请求 prompt 均为 48 tokens（完全相同）\n");
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
    print_results("第一波结果（cached_blk=3 是 3 个满块；#1/#2 命中的是 #0 刚登记的块）",
                  specs1, reqs1, trace);
    printf("    峰值占用 %d 块 = 3 个共享前缀块 + 3 个各自 decode 块（不共享则需 12 块）\n",
           alloc.num_blocks() - trace->min_free);
    printf("    空闲块=%d/%d（请求已结束：引用归零，块回到空闲链表）\n",
           alloc.num_free_blocks(), alloc.num_blocks());

    // ---- 第二波：共享同一前缀，尾部各自不同 ----
    // 第一波已结束，但满块的哈希仍登记在 hash_to_block_ 中 → 仍可命中
    printf("\n  [第二波] 3 个请求 prompt = 48 公共前缀 + 16 各自私有（共 4 块）\n");

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
        auto preview = mgr.match_prefix(p);
        printf("    req#%d 提交前预查：命中 %d 块 / 共需 %d 块\n", i,
               static_cast<int>(preview.cached_blocks.size()),
               static_cast<int>((p.size() + BLOCK_SIZE - 1) / BLOCK_SIZE));

        specs2.push_back(Spec{static_cast<int>(p.size()), 2});
        auto r = std::make_shared<Request>();
        r->all_tokens = p;
        r->max_new_tokens = 2;
        reqs2.push_back(r);
        futures2.push_back(sched.add_request(r));
    }

    for(auto& f : futures2)
        f.wait();
    sched.stop(true);
    print_results("第二波结果（提交前预查已命中 3 块；cached_blk 还含自己登记的第 4 块）",
                  specs2, reqs2, trace);
    printf("    峰值占用 %d 块（若不共享前缀，同一时刻 3 个请求各需 4 块 = 12 块）\n",
           alloc.num_blocks() - trace->min_free);
    printf("    总步数=%lld，抢占次数=%d，剩余空闲块=%d/%d\n",
           sched.num_steps(), sched.num_preemptions(),
           alloc.num_free_blocks(), alloc.num_blocks());
}

// ============================================================
//  C. 抢占与重算
// ============================================================
static void demo_preemption()
{
    constexpr int kBlocks = 16;

    printf("\n=== C. 空间不足时的抢占与重算 ===\n");
    printf("  %d 个物理块（可用 %d）\n", kBlocks, kBlocks - WATER_MARK);

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

    printf("  提交 %d 个请求：prompt=32 tokens, max_new=56（每个最终需 6 块）\n", kNumReq);
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

    print_results("结果（抢占不丢输出，靠重算恢复）", specs, reqs, trace);
    printf("    总步数=%lld，抢占次数=%d，剩余空闲块=%d/%d\n",
           sched.num_steps(), sched.num_preemptions(),
           alloc.num_free_blocks(), alloc.num_blocks());
    printf("    抢占语义：释放 victim 的块 → 标记 WAITING → 放回等待队列队首；\n");
    printf("              恢复时对 all_tokens（prompt + 已生成 token）重新 prefill，即「重算」\n");
}

int main()
{
    printf("mini KV Cache demo — 复现 vLLM 的分页 KV Cache 与调度\n");
    demo_prefix_and_cow();
    demo_prefix_cache_waves();
    demo_preemption();
    printf("\n完成。机制细节见 docs/ARCHITECTURE.md，与 vLLM 的对照见 docs/VLLM_MAPPING.md\n");
    return 0;
}
