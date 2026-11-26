#pragma once
#include "kvcache.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>

inline int g_fail_count = 0;

// 防御性断言：失败即 abort，用于不变量检查（不受 NDEBUG 影响）
#define CHECK(cond)                                                                    \
    do                                                                                 \
    {                                                                                  \
        if(!(cond))                                                                    \
        {                                                                              \
            fprintf(stderr, "CHECK FAILED: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
            std::abort();                                                              \
        }                                                                              \
    } while(0)

// 让 EXPECT_EQ 能打印任意可流式输出的类型
template <typename T>
static std::string test_repr(const T& v)
{
    std::ostringstream os;
    os << v;
    return os.str();
}

// 容器没有 operator<<，单独提供
inline std::string test_repr(const std::vector<int>& v)
{
    std::ostringstream os;
    os << "[";
    for(size_t i = 0; i < v.size(); ++i)
    {
        if(i) os << ", ";
        os << v[i];
    }
    os << "]";
    return os.str();
}

#define EXPECT_EQ(a, b)                                                                       \
    do                                                                                        \
    {                                                                                         \
        auto _va = (a);                                                                       \
        auto _vb = (b);                                                                       \
        if(!(_va == _vb))                                                                     \
        {                                                                                     \
            std::cerr << "    [FAIL] " << #a << " == " << #b                                  \
                      << "  (got " << test_repr(_va) << ", want " << test_repr(_vb) << ")\n"; \
            ++g_fail_count;                                                                   \
        }                                                                                     \
    } while(0)

#define EXPECT_TRUE(x)                                \
    do                                                \
    {                                                 \
        if(!(x))                                      \
        {                                             \
            std::cerr << "    [FAIL] " << #x << "\n"; \
            ++g_fail_count;                           \
        }                                             \
    } while(0)

#define EXPECT_FALSE(x)                                     \
    do                                                      \
    {                                                       \
        if((x))                                             \
        {                                                   \
            std::cerr << "    [FAIL] NOT (" << #x << ")\n"; \
            ++g_fail_count;                                 \
        }                                                   \
    } while(0)

#define RUN_TEST(fn)                                        \
    do                                                      \
    {                                                       \
        std::cout << "\n=== " << #fn << " ===\n";           \
        int before = g_fail_count;                          \
        fn();                                               \
        if(g_fail_count == before) std::cout << "  [OK]\n"; \
    } while(0)


// 阶段一/二测试沿用的 decode 语义：先预留槽位，再追加 token
// token < 0 表示不关心内容（块写满后不会注册前缀缓存）
inline int decode(BlockManager& manager, Request& request, int token = -1)
{
    int slot = append_slot(manager, request);
    if(slot < 0) return -1;
    append_token(manager, request, token);
    return slot;
}

// 驱动调度器直到所有请求完成；超过 max_steps 视为卡死，返回 -1
inline int run_until_done(Scheduler& sched, int max_steps = 10000)
{
    int steps = 0;
    while(sched.has_work())
    {
        if(steps >= max_steps) return -1;
        sched.step();
        ++steps;
    }
    return steps;
}

inline void check_invariants(const Scheduler& sched, const BlockAllocator& allocator,
                             const BlockManager* manager = nullptr)
{
    const int n = allocator.num_blocks();

    // 1) 收集所有"活跃请求" = RUNNING ∪ WAITING
    std::vector<std::shared_ptr<Request>> active;
    for(const auto& r : sched.running())
        active.push_back(r);
    for(const auto& r : sched.waiting())
        active.push_back(r);

    // 2) 统计每个物理块的期望引用计数
    std::vector<int> expected_ref(n, 0);
    for(const auto& r : active)
    {
        for(int id : r->block_table)
        {
            CHECK(id >= 0 && id < n);
            expected_ref[id]++;
        }
    }

    // 3) ref_count、tier、clean/cached 计数与 has_hash 一致性（I1, I2, I3, I6）
    int zero_count = 0;
    int clean_count = 0;
    int cached_count = 0;
    int has_hash_count = 0;
    for(int i = 0; i < n; i++)
    {
        const PhysicalBlock& b = allocator.get_block(i);
        int actual = allocator.ref_count_of(i);
        if(actual != expected_ref[i])
        {
            fprintf(stderr,
                    "[invariant] block %d ref_count=%d expected=%d\n",
                    i, actual, expected_ref[i]);
            std::abort();
        }
        if(b.has_hash) ++has_hash_count;
        if(actual == 0)
        {
            ++zero_count;
            CHECK(b.tier != FreeTier::NONE); // I1
            if(b.tier == FreeTier::CACHED)
            {
                CHECK(b.has_hash); // I3
                ++cached_count;
            }
            else
            {
                CHECK(!b.has_hash); // I3
                ++clean_count;
            }
        }
        else
        {
            CHECK(b.tier == FreeTier::NONE); // I1
            CHECK(b.free_prev == -1 && b.free_next == -1);
        }
    }
    CHECK(zero_count == allocator.num_free_blocks());
    CHECK(clean_count == allocator.num_free_clean());
    CHECK(cached_count == allocator.num_free_cached());
    CHECK(allocator.num_free_blocks() == allocator.num_free_clean() + allocator.num_free_cached()); // I2

    // 3.5) 侵入式空闲链表完整性 + LRU 布局不变量（I4）
    int visited = 0;
    bool seen_cached = false;
    uint64_t prev_cached_ts = 0;
    int prev_id = -1;
    for(int cur = allocator.free_head(); cur >= 0;)
    {
        CHECK(cur >= 0 && cur < n);
        CHECK(visited < n); // 防止成环
        const PhysicalBlock& b = allocator.get_block(cur);
        CHECK(b.free_prev == prev_id);
        CHECK(b.ref_count == 0 && b.tier != FreeTier::NONE);
        if(allocator.policy() == FreePolicy::LRU)
        {
            if(b.tier == FreeTier::CACHED)
            {
                seen_cached = true;
                CHECK(b.last_accessed >= prev_cached_ts); // I4：cached 后缀时间戳单调不减
                prev_cached_ts = b.last_accessed;
            }
            else
            {
                CHECK(!seen_cached); // I4：clean 前缀不可出现在 cached 后缀之后
            }
        }
        prev_id = cur;
        cur = b.free_next;
        ++visited;
    }
    CHECK(visited == zero_count);
    CHECK(prev_id == allocator.free_tail());

    // 3.6) 哈希表有界不变量（I5）
    if(manager != nullptr)
    {
        CHECK(static_cast<int>(manager->num_hash_entries()) <= has_hash_count);
    }

    // 4) RUNNING 请求的 block_table 长度
    const int blk_size = allocator.block_size();
    for(const auto& r : sched.running())
    {
        int expected_blocks = (r->sequence_length + blk_size - 1) / blk_size;
        if(static_cast<int>(r->block_table.size()) != expected_blocks)
        {
            fprintf(stderr,
                    "[invariant] RUNNING req seq_len=%d block_table=%zu expected=%d\n",
                    r->sequence_length, r->block_table.size(), expected_blocks);
            std::abort();
        }
    }

    // 5) WAITING 请求的 block_table 必须为空（被抢占时已 release）
    for(const auto& r : sched.waiting())
    {
        if(!r->block_table.empty())
        {
            fprintf(stderr,
                    "[invariant] WAITING req has non-empty block_table (size=%zu)\n",
                    r->block_table.size());
            std::abort();
        }
    }
}