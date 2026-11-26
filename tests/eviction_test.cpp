// tests/eviction_test.cpp
// v0.3.0：O(1) 侵入式空闲队列、LRU/FIFO 淘汰策略、哈希表主动清理与指标自洽测试

#include "test_util.h"
#include <numeric>
#include <random>
#include <set>

static std::vector<int> make_tokens(int n, int base = 0)
{
    std::vector<int> v(static_cast<size_t>(n));
    for(int i = 0; i < n; ++i)
        v[static_cast<size_t>(i)] = base + i;
    return v;
}

// 1) O(1) 摘链与归还：随机乱序释放/复活，链表无环无漏无重
void test_intrusive_list_unlink_and_revive()
{
    constexpr int N = 32;
    BlockAllocator alloc(N, 4, FreePolicy::LRU);
    BlockManager mgr(&alloc);

    std::vector<Request> reqs(N);
    for(int i = 0; i < N; ++i)
    {
        EXPECT_TRUE(prefill(make_tokens(4, 1000 + i * 10), reqs[i], mgr));
    }
    EXPECT_EQ(alloc.num_free_blocks(), 0);

    // 乱序释放全部块（均带 hash → 进入 CACHED 后缀）
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(2026);
    std::shuffle(order.begin(), order.end(), rng);

    std::vector<int> blk_ids(N);
    for(int i = 0; i < N; ++i)
        blk_ids[i] = reqs[i].block_table[0];

    for(int idx : order)
        mgr.release(reqs[idx]);

    EXPECT_EQ(alloc.num_free_blocks(), N);
    EXPECT_EQ(alloc.num_free_clean(), 0);
    EXPECT_EQ(alloc.num_free_cached(), N);

    // 随机挑一半块复活（O(1) unlink）
    std::shuffle(order.begin(), order.end(), rng);
    for(int k = 0; k < N / 2; ++k)
        alloc.inc_ref(blk_ids[order[k]]);

    EXPECT_EQ(alloc.num_free_blocks(), N - N / 2);
    EXPECT_EQ(alloc.num_free_cached(), N - N / 2);
    EXPECT_EQ(alloc.num_revivals(), static_cast<uint64_t>(N / 2));

    // 遍历剩余空闲链表，验证前后指针完全一致且无重复
    std::set<int> seen;
    int prev = -1;
    for(int cur = alloc.free_head(); cur >= 0;)
    {
        EXPECT_TRUE(seen.insert(cur).second);
        const PhysicalBlock& b = alloc.get_block(cur);
        EXPECT_EQ(b.free_prev, prev);
        EXPECT_EQ(b.ref_count, 0);
        prev = cur;
        cur = b.free_next;
    }
    EXPECT_EQ(static_cast<int>(seen.size()), N - N / 2);
    EXPECT_EQ(prev, alloc.free_tail());

    // 归还复活的块
    for(int k = 0; k < N / 2; ++k)
        alloc.dec_ref(blk_ids[order[k]]);
    EXPECT_EQ(alloc.num_free_blocks(), N);
}

// 2) LRU 顺序：优先吃 CLEAN 块，再淘汰最久未用的 CACHED 块
void test_lru_prefers_clean_then_oldest_cached()
{
    BlockAllocator alloc(4, 4, FreePolicy::LRU);
    BlockManager mgr(&alloc);

    // A、B、C 各占 1 个满块并登记哈希；D 为半块（不登记哈希 → 释放后为 CLEAN）
    Request ra, rb, rc, rd;
    EXPECT_TRUE(prefill(make_tokens(4, 100), ra, mgr));
    EXPECT_TRUE(prefill(make_tokens(4, 200), rb, mgr));
    EXPECT_TRUE(prefill(make_tokens(4, 300), rc, mgr));
    EXPECT_TRUE(prefill(make_tokens(2, 400), rd, mgr));

    const int id_a = ra.block_table[0];
    const int id_b = rb.block_table[0];
    const int id_c = rc.block_table[0];
    const int id_d = rd.block_table[0];

    // 依次释放 A → B → C → D：D 进入 CLEAN 队首，CACHED 队列顺序为 [A, B, C]
    mgr.release(ra);
    mgr.release(rb);
    mgr.release(rc);
    mgr.release(rd);
    EXPECT_EQ(alloc.num_free_clean(), 1);
    EXPECT_EQ(alloc.num_free_cached(), 3);

    // 第 1 次分配：必须先拿到 CLEAN 的 D（不触发淘汰）
    int p1 = mgr.allocate_block();
    EXPECT_EQ(p1, id_d);
    EXPECT_EQ(alloc.num_evictions(), 0u);
    EXPECT_EQ(alloc.num_free_clean(), 0);

    // 复活再释放 A，使 A 变为最新，此时 CACHED 顺序应变为 [B, C, A]
    alloc.inc_ref(id_a);
    alloc.dec_ref(id_a);

    // 第 2 次分配：必须淘汰最久未用的 B
    int p2 = mgr.allocate_block();
    EXPECT_EQ(p2, id_b);
    EXPECT_EQ(alloc.num_evictions(), 1u);
    EXPECT_FALSE(alloc.get_block(id_b).has_hash);

    // 第 3 次分配：淘汰 C，A 仍幸存
    int p3 = mgr.allocate_block();
    EXPECT_EQ(p3, id_c);
    EXPECT_EQ(alloc.num_evictions(), 2u);
    EXPECT_EQ(mgr.match_prefix(make_tokens(4, 100)).num_matched_tokens, 4);
    EXPECT_EQ(mgr.match_prefix(make_tokens(4, 200)).num_matched_tokens, 0);
    EXPECT_EQ(mgr.match_prefix(make_tokens(4, 300)).num_matched_tokens, 0);

    alloc.dec_ref(p1);
    alloc.dec_ref(p2);
    alloc.dec_ref(p3);
    EXPECT_EQ(alloc.num_free_blocks(), 4);
}

// 3) 淘汰主动清理哈希表：hash_to_block_.size() 随淘汰立即下降
void test_eviction_cleans_hash_table_eagerly()
{
    BlockAllocator alloc(2, 4, FreePolicy::LRU);
    BlockManager mgr(&alloc);

    auto pa = make_tokens(4, 100);
    auto pb = make_tokens(4, 200);
    Request ra, rb;
    EXPECT_TRUE(prefill(pa, ra, mgr));
    EXPECT_TRUE(prefill(pb, rb, mgr));
    EXPECT_EQ(mgr.num_hash_entries(), 2u);

    mgr.release(ra);
    mgr.release(rb);
    EXPECT_EQ(mgr.num_hash_entries(), 2u);

    // 分配 1 个新块触发淘汰 A，哈希表条目应立刻从 2 降到 1（不依赖 match_prefix 路过清理）
    int id = mgr.allocate_block();
    EXPECT_TRUE(id >= 0);
    EXPECT_EQ(alloc.num_evictions(), 1u);
    EXPECT_EQ(mgr.num_hash_entries(), 1u);
    EXPECT_EQ(mgr.match_prefix(pa).num_matched_tokens, 0);
    EXPECT_EQ(mgr.match_prefix(pb).num_matched_tokens, 4);

    alloc.dec_ref(id);
}

// 4) 同内容双块保护：淘汰旧块时不得误删新块的哈希映射（验证 it->second == block_id 守卫）
void test_eviction_guard_duplicate_hash_blocks()
{
    BlockAllocator alloc(3, 4, FreePolicy::LRU);
    BlockManager mgr(&alloc);

    auto tokens = make_tokens(4, 500);
    Request r1, r2;
    EXPECT_TRUE(prefill(tokens, r1, mgr));
    const int b1 = r1.block_table[0];

    // 手工构造一个持有相同内容但不同物理块的 r2 并登记哈希（模拟并发未命中后双写覆盖）
    int b2 = mgr.allocate_block();
    EXPECT_TRUE(b2 >= 0 && b2 != b1);
    r2.block_table.push_back(b2);
    r2.all_tokens = tokens;
    r2.sequence_length = 4;
    mgr.register_block(r2, 0); // hash_to_block_[h] 现在指向 b2

    // 先释放 b1（使其成为最久未用的 CACHED 块），保持 b2 在用
    mgr.release(r1);
    // 再消耗掉第 3 个 CLEAN 块，然后触发淘汰 b1
    int b3 = mgr.allocate_block();
    EXPECT_TRUE(b3 >= 0);
    int evicted_id = mgr.allocate_block();
    EXPECT_EQ(evicted_id, b1);

    // b1 被淘汰后，b2 的缓存映射必须完好无损
    auto m = mgr.match_prefix(tokens);
    EXPECT_EQ(m.num_matched_tokens, 4);
    EXPECT_EQ(m.cached_blocks[0], b2);

    alloc.dec_ref(b3);
    alloc.dec_ref(evicted_id);
    mgr.release(r2);
    EXPECT_EQ(alloc.num_free_blocks(), 3);
}

// 5) 关闭前缀缓存：命中 0、淘汰 0、无 CACHED 空闲块
void test_prefix_off_zero_hits_and_evictions()
{
    BlockAllocator alloc(4, 4, FreePolicy::LRU);
    BlockManager mgr(&alloc, BlockManagerConfig{/*enable_prefix_caching=*/false});

    auto prompt = make_tokens(8, 100);
    for(int i = 0; i < 6; ++i)
    {
        Request r;
        EXPECT_TRUE(prefill(prompt, r, mgr));
        mgr.release(r);
    }
    EXPECT_EQ(mgr.prefix_cache_hits(), 0u);
    EXPECT_EQ(mgr.num_hash_entries(), 0u);
    EXPECT_EQ(alloc.num_evictions(), 0u);
    EXPECT_EQ(alloc.num_free_cached(), 0);
    EXPECT_EQ(alloc.num_free_clean(), 4);
}

// 6) FIFO vs LRU 对照：存在 CLEAN 块与热缓存块时，LRU 能保住热缓存而 FIFO 会先误杀旧缓存
void test_lru_beats_fifo_with_clean_and_hot_blocks()
{
    auto run_scenario = [](FreePolicy policy) -> uint64_t
    {
        // 3 个物理块，block_size=4
        BlockAllocator alloc(3, 4, policy);
        BlockManager mgr(&alloc);

        auto hot = make_tokens(4, 1000);
        auto cold = make_tokens(2, 2000); // 半块 → CLEAN
        auto noise1 = make_tokens(2, 3000);
        auto noise2 = make_tokens(2, 4000);

        // 先缓存 hot 并释放，再放两个 CLEAN 块并释放
        // FIFO 空闲队：[hot, cold, noise1] → 下一请求分配 noise2 会直接把队首 hot 淘汰！
        // LRU 空闲队：[noise1, cold, hot] → 分配 noise2 吃掉 CLEAN 块，hot 存活！
        Request r_hot, r_cold, r_n1, r_n2, r_hot2;
        EXPECT_TRUE(prefill(hot, r_hot, mgr));
        mgr.release(r_hot);
        EXPECT_TRUE(prefill(cold, r_cold, mgr));
        EXPECT_TRUE(prefill(noise1, r_n1, mgr));
        mgr.release(r_cold);
        mgr.release(r_n1);

        EXPECT_TRUE(prefill(noise2, r_n2, mgr));
        EXPECT_TRUE(prefill(hot, r_hot2, mgr));

        uint64_t hits = mgr.prefix_cache_hits();
        mgr.release(r_n2);
        mgr.release(r_hot2);
        return hits;
    };

    uint64_t fifo_hits = run_scenario(FreePolicy::FIFO);
    uint64_t lru_hits = run_scenario(FreePolicy::LRU);
    EXPECT_EQ(fifo_hits, 0u);
    EXPECT_EQ(lru_hits, 1u);
}

// 7) 指标自洽性：queries / hits / allocations / evictions / revivals / cow_tokens / TTFT 步数戳
void test_metrics_self_consistency()
{
    BlockAllocator alloc(16, 4, FreePolicy::LRU);
    BlockManager mgr(&alloc);
    SchedulerConfig cfg;
    cfg.watermark_blocks = 2;
    Scheduler sched(mgr, cfg);
    sched.set_sampler(fake_sample);
    sched.set_step_hook([&](const Scheduler& s)
                        { check_invariants(s, alloc, &mgr); });

    auto shared_prefix = make_tokens(8, 700); // 2 个满块
    std::vector<std::shared_ptr<Request>> reqs;
    for(int i = 0; i < 4; ++i)
    {
        auto r = std::make_shared<Request>();
        r->all_tokens = shared_prefix;
        r->all_tokens.push_back(9000 + i);
        r->max_new_tokens = 5;
        reqs.push_back(r);
        sched.add_request(r);
    }

    int steps = run_until_done(sched);
    EXPECT_TRUE(steps > 0);

    // 4 个请求每个 9 token（2 个满块）：第 1 个未命中，后 3 个各命中 2 块 → queries=8, hits=6
    EXPECT_EQ(mgr.prefix_cache_queries(), 8u);
    EXPECT_EQ(mgr.prefix_cache_hits(), 6u);
    EXPECT_EQ(mgr.prompt_tokens(), 36u);
    EXPECT_TRUE(alloc.num_evictions() <= alloc.num_allocations());
    EXPECT_TRUE(alloc.num_revivals() <= mgr.prefix_cache_hits());

    // 步数戳合法性：admit_step >= 0，first_token_step > admit_step，finish_step >= first_token_step
    for(const auto& r : reqs)
    {
        EXPECT_TRUE(r->admit_step >= 0);
        EXPECT_TRUE(r->first_token_step > r->admit_step);
        EXPECT_TRUE(r->finish_step >= r->first_token_step);
    }
    EXPECT_EQ(alloc.num_free_blocks(), 16);
}

int main()
{
    std::cout << "========== v0.3.0：O(1) 空闲队列、LRU 淘汰与指标测试 ==========\n";
    RUN_TEST(test_intrusive_list_unlink_and_revive);
    RUN_TEST(test_lru_prefers_clean_then_oldest_cached);
    RUN_TEST(test_eviction_cleans_hash_table_eagerly);
    RUN_TEST(test_eviction_guard_duplicate_hash_blocks);
    RUN_TEST(test_prefix_off_zero_hits_and_evictions);
    RUN_TEST(test_lru_beats_fifo_with_clean_and_hot_blocks);
    RUN_TEST(test_metrics_self_consistency);

    if(g_fail_count == 0)
    {
        std::cout << "\n所有测试通过 ✅\n";
        return 0;
    }
    std::cout << "\n失败 " << g_fail_count << " 项 ❌\n";
    return 1;
}
