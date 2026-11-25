#include "test_util.h"
#include <set>
#include <algorithm>

// ---------------- 工具：请求释放 ----------------
static void release_request(Request& req, BlockAllocator& alloc)
{
    for(int id : req.block_table)
        alloc.dec_ref(id);
    req.block_table.clear();
    req.sequence_length = 0;
    req.slot_mapping.clear();
}


// =========================================================
// 阶段一：分页与 Slot Mapping
// =========================================================

void test_allocator_allocate_unique()
{
    BlockAllocator alloc(4);
    BlockManager mgr(&alloc);
    int b0 = alloc.allocate();
    int b1 = alloc.allocate();
    int b2 = alloc.allocate();
    int b3 = alloc.allocate();
    EXPECT_TRUE(b0 >= 0 && b1 >= 0 && b2 >= 0 && b3 >= 0);

    std::set<int> ids{b0, b1, b2, b3};
    EXPECT_EQ((int)ids.size(), 4); // 互不相同

    EXPECT_EQ(alloc.get_block(b0).ref_count, 1);
    EXPECT_EQ(alloc.get_block(b1).ref_count, 1);
    EXPECT_EQ(alloc.get_block(b2).ref_count, 1);
    EXPECT_EQ(alloc.get_block(b3).ref_count, 1);
}

void test_allocator_exhaust()
{
    BlockAllocator alloc(2);
    BlockManager mgr(&alloc);
    EXPECT_TRUE(alloc.allocate() >= 0);
    EXPECT_TRUE(alloc.allocate() >= 0);
    EXPECT_EQ(alloc.allocate(), -1); // 用尽
    EXPECT_EQ(alloc.num_free_blocks(), 0);
}

void test_allocator_reclaim()
{
    BlockAllocator alloc(2);
    BlockManager mgr(&alloc);
    int b0 = alloc.allocate();
    int b1 = alloc.allocate();
    EXPECT_EQ(alloc.num_free_blocks(), 0);

    alloc.dec_ref(b0);
    EXPECT_EQ(alloc.get_block(b0).ref_count, 0);
    EXPECT_EQ(alloc.num_free_blocks(), 1);

    alloc.dec_ref(b1);
    EXPECT_EQ(alloc.num_free_blocks(), 2);

    // 回收后可再次分配
    EXPECT_TRUE(alloc.allocate() >= 0);
    EXPECT_EQ(alloc.num_free_blocks(), 1);
}

void test_allocator_inc_dec_balance()
{
    BlockAllocator alloc(2);
    BlockManager mgr(&alloc);
    int b0 = alloc.allocate();
    EXPECT_EQ(alloc.get_block(b0).ref_count, 1);

    alloc.inc_ref(b0);
    EXPECT_EQ(alloc.get_block(b0).ref_count, 2);

    alloc.dec_ref(b0);
    EXPECT_EQ(alloc.get_block(b0).ref_count, 1);
    EXPECT_EQ(alloc.num_free_blocks(), 1); // 还没归零，不能回收

    alloc.dec_ref(b0);
    EXPECT_EQ(alloc.get_block(b0).ref_count, 0);
    EXPECT_EQ(alloc.num_free_blocks(), 2);
}

void test_prefill_empty()
{
    BlockAllocator alloc(4);
    BlockManager mgr(&alloc);
    Request req;
    EXPECT_TRUE(prefill({}, req, mgr));
    EXPECT_EQ(req.sequence_length, 0);
    EXPECT_EQ((int)req.block_table.size(), 0);
    EXPECT_EQ((int)req.slot_mapping.size(), 0);
    EXPECT_EQ(alloc.num_free_blocks(), 4);
}

void test_prefill_one_token()
{
    BlockAllocator alloc(4);
    BlockManager mgr(&alloc);
    Request req;
    EXPECT_TRUE(prefill({42}, req, mgr));
    EXPECT_EQ(req.sequence_length, 1);
    EXPECT_EQ((int)req.block_table.size(), 1);
    EXPECT_EQ((int)req.slot_mapping.size(), 1);

    int pb = req.block_table[0];
    EXPECT_EQ(req.slot_mapping[0], pb * BLOCK_SIZE + 0);
    EXPECT_EQ(alloc.num_free_blocks(), 3);
    EXPECT_EQ(alloc.get_block(pb).ref_count, 1);
}

void test_prefill_exactly_one_block()
{
    BlockAllocator alloc(4);
    BlockManager mgr(&alloc);
    Request req;
    EXPECT_TRUE(prefill(std::vector<int>(16, 1), req, mgr));
    EXPECT_EQ((int)req.block_table.size(), 1);
    EXPECT_EQ((int)req.slot_mapping.size(), 16);

    int pb = req.block_table[0];
    for(int t = 0; t < 16; ++t)
        EXPECT_EQ(req.slot_mapping[t], pb * BLOCK_SIZE + t);
}

void test_prefill_spans_two_blocks()
{
    BlockAllocator alloc(4);
    BlockManager mgr(&alloc);
    Request req;
    EXPECT_TRUE(prefill(std::vector<int>(17, 1), req, mgr));
    EXPECT_EQ((int)req.block_table.size(), 2);
    EXPECT_EQ((int)req.slot_mapping.size(), 17);

    int pb0 = req.block_table[0];
    int pb1 = req.block_table[1];
    EXPECT_TRUE(pb0 != pb1);
    for(int t = 0; t < 16; ++t)
        EXPECT_EQ(req.slot_mapping[t], pb0 * BLOCK_SIZE + t);
    EXPECT_EQ(req.slot_mapping[16], pb1 * BLOCK_SIZE + 0);
}

void test_prefill_100_tokens()
{
    BlockAllocator alloc(16);
    BlockManager mgr(&alloc);
    Request req;
    EXPECT_TRUE(prefill(std::vector<int>(100, 1), req, mgr));
    EXPECT_EQ((int)req.block_table.size(), 7); // ceil(100/16)
    EXPECT_EQ((int)req.slot_mapping.size(), 100);
    EXPECT_EQ(alloc.num_free_blocks(), 16 - 7);
}

void test_prefill_out_of_memory_rollback()
{
    // 需要 3 块，但只有 2 块 → 应失败且不泄漏
    BlockAllocator alloc(2);
    BlockManager mgr(&alloc);
    Request req;
    EXPECT_FALSE(prefill(std::vector<int>(48, 1), req, mgr));
    // 关键：失败时已分配的块必须回滚
    EXPECT_EQ(alloc.num_free_blocks(), 2);
}

void test_prefill_reused_request_no_leak()
{
    // 同一个 Request 对象连续 prefill 两次，slot_mapping 不应累加
    BlockAllocator alloc(4);
    BlockManager mgr(&alloc);
    Request req;
    EXPECT_TRUE(prefill(std::vector<int>(5, 1), req, mgr));
    EXPECT_TRUE(prefill(std::vector<int>(3, 1), req, mgr));
    EXPECT_EQ((int)req.block_table.size(), 1);
    EXPECT_EQ((int)req.slot_mapping.size(), 3); // 不能是 8
    EXPECT_EQ(req.sequence_length, 3);
}

void test_decode_same_block()
{
    BlockAllocator alloc(4);
    BlockManager mgr(&alloc);
    Request req;
    EXPECT_TRUE(prefill(std::vector<int>(5, 1), req, mgr));
    int pb = req.block_table[0];

    int sid = decode(mgr, req);
    EXPECT_EQ(sid, pb * BLOCK_SIZE + 5);
    EXPECT_EQ((int)req.block_table.size(), 1);
    EXPECT_EQ(req.sequence_length, 6);

    sid = decode(mgr, req);
    EXPECT_EQ(sid, pb * BLOCK_SIZE + 6);
    EXPECT_EQ(req.sequence_length, 7);
}

void test_decode_new_block_at_boundary()
{
    BlockAllocator alloc(4);
    BlockManager mgr(&alloc);
    Request req;
    EXPECT_TRUE(prefill(std::vector<int>(16, 1), req, mgr));
    int pb0 = req.block_table[0];

    int sid = decode(mgr, req);
    EXPECT_EQ((int)req.block_table.size(), 2);
    int pb1 = req.block_table[1];
    EXPECT_TRUE(pb1 != pb0);
    EXPECT_EQ(sid, pb1 * BLOCK_SIZE + 0);
    EXPECT_EQ(req.sequence_length, 17);
    EXPECT_EQ(alloc.get_block(pb0).ref_count, 1);
    EXPECT_EQ(alloc.get_block(pb1).ref_count, 1);
}

void test_decode_long_sequence()
{
    BlockAllocator alloc(16);
    BlockManager mgr(&alloc);
    Request req;
    EXPECT_TRUE(prefill({1, 2, 3}, req, mgr)); // 1 块

    for(int i = 0; i < 100; ++i)
        EXPECT_TRUE(decode(mgr, req) >= 0);

    EXPECT_EQ(req.sequence_length, 103);
    EXPECT_EQ((int)req.block_table.size(), 7); // ceil(103/16)
}

void test_release_restores_free()
{
    BlockAllocator alloc(4);
    BlockManager mgr(&alloc);
    Request req;
    EXPECT_TRUE(prefill(std::vector<int>(20, 1), req, mgr)); // 2 块
    EXPECT_EQ(alloc.num_free_blocks(), 2);
    release_request(req, alloc);
    EXPECT_EQ(alloc.num_free_blocks(), 4);
    EXPECT_EQ(req.sequence_length, 0);
    EXPECT_EQ((int)req.block_table.size(), 0);
}

void test_release_after_decode()
{
    BlockAllocator alloc(4);
    BlockManager mgr(&alloc);
    Request req;
    EXPECT_TRUE(prefill(std::vector<int>(5, 1), req, mgr));
    for(int i = 0; i < 30; ++i)
        decode(mgr, req);
    // 长度 35 → 3 块
    EXPECT_EQ((int)req.block_table.size(), 3);
    release_request(req, alloc);
    EXPECT_EQ(alloc.num_free_blocks(), 4);
}


// =========================================================
// 阶段二：Fork 与 COW
// =========================================================

void test_fork_inc_ref()
{
    BlockAllocator alloc(4);
    BlockManager mgr(&alloc);
    Request father;
    EXPECT_TRUE(prefill(std::vector<int>(20, 1), father, mgr));

    int pb0 = father.block_table[0];
    int pb1 = father.block_table[1];
    EXPECT_EQ(alloc.get_block(pb0).ref_count, 1);
    EXPECT_EQ(alloc.get_block(pb1).ref_count, 1);

    Request son = fork_sequence(father, alloc);

    EXPECT_EQ(alloc.get_block(pb0).ref_count, 2);
    EXPECT_EQ(alloc.get_block(pb1).ref_count, 2);
    EXPECT_EQ(son.sequence_length, father.sequence_length);
    EXPECT_EQ(son.block_table, father.block_table);
}

void test_fork_release_one()
{
    BlockAllocator alloc(4);
    BlockManager mgr(&alloc);
    Request father;
    EXPECT_TRUE(prefill(std::vector<int>(20, 1), father, mgr));
    Request son = fork_sequence(father, alloc);

    int pb0 = father.block_table[0];
    int pb1 = father.block_table[1];
    EXPECT_EQ(alloc.num_free_blocks(), 2);

    release_request(father, alloc);
    EXPECT_EQ(alloc.num_free_blocks(), 2); // 没变，son 还持有
    EXPECT_EQ(alloc.get_block(pb0).ref_count, 1);
    EXPECT_EQ(alloc.get_block(pb1).ref_count, 1);

    release_request(son, alloc);
    EXPECT_EQ(alloc.num_free_blocks(), 4);
}

void test_cow_full_tail_no_copy()
{
    // 尾块已满（N=16），decode 直接分配新块，不 COW
    BlockAllocator alloc(4);
    BlockManager mgr(&alloc);
    Request father;
    EXPECT_TRUE(prefill(std::vector<int>(16, 1), father, mgr));
    int pb0 = father.block_table[0];

    Request son = fork_sequence(father, alloc);
    EXPECT_EQ(alloc.get_block(pb0).ref_count, 2);

    int sid_f = decode(mgr, father);
    int sid_s = decode(mgr, son);

    EXPECT_EQ((int)father.block_table.size(), 2);
    EXPECT_EQ((int)son.block_table.size(), 2);
    int pb_f = father.block_table[1];
    int pb_s = son.block_table[1];
    EXPECT_TRUE(pb_f != pb_s); // 各自新块
    EXPECT_EQ(sid_f, pb_f * BLOCK_SIZE + 0);
    EXPECT_EQ(sid_s, pb_s * BLOCK_SIZE + 0);

    // 历史块仍共享
    EXPECT_EQ(alloc.get_block(pb0).ref_count, 2);
}

void test_cow_partial_tail_trigger()
{
    // 尾块未满且 ref_count>1 → COW
    BlockAllocator alloc(4);
    BlockManager mgr(&alloc);
    Request father;
    EXPECT_TRUE(prefill(std::vector<int>(10, 1), father, mgr));
    int pb_old = father.block_table[0];

    Request son = fork_sequence(father, alloc);
    EXPECT_EQ(alloc.get_block(pb_old).ref_count, 2);
    int free_before = alloc.num_free_blocks(); // 3

    // father 先 decode → COW
    int sid_f = decode(mgr, father);
    int pb_new = father.block_table[0];
    EXPECT_TRUE(pb_new != pb_old); // 换了新块
    EXPECT_EQ(alloc.get_block(pb_new).ref_count, 1);
    EXPECT_EQ(alloc.get_block(pb_old).ref_count, 1); // 只剩 son 持有
    EXPECT_EQ(alloc.num_free_blocks(), free_before - 1);
    EXPECT_EQ(sid_f, pb_new * BLOCK_SIZE + 10);

    // son 再 decode → 尾块 ref=1，不 COW
    EXPECT_EQ(son.block_table[0], pb_old);
    int sid_s = decode(mgr, son);
    EXPECT_EQ(son.block_table[0], pb_old); // 没换
    EXPECT_EQ(sid_s, pb_old * BLOCK_SIZE + 10);
    EXPECT_EQ(alloc.get_block(pb_old).ref_count, 1);
}

void test_cow_only_first_writer_copies()
{
    // 两个序列共享尾块，只有第一个写的人触发 COW
    BlockAllocator alloc(4);
    BlockManager mgr(&alloc);
    Request father;
    EXPECT_TRUE(prefill(std::vector<int>(10, 1), father, mgr));
    int pb_old = father.block_table[0];

    Request son = fork_sequence(father, alloc);
    EXPECT_EQ(alloc.get_block(pb_old).ref_count, 2);

    int sid_f = decode(mgr, father); // 触发 COW
    int sid_s = decode(mgr, son);    // 不触发
    (void)sid_f;
    (void)sid_s;

    EXPECT_TRUE(father.block_table[0] != pb_old);
    EXPECT_EQ(son.block_table[0], pb_old);
    EXPECT_EQ(alloc.get_block(pb_old).ref_count, 1);
    EXPECT_EQ(alloc.get_block(father.block_table[0]).ref_count, 1);
}

void test_cow_after_which_release()
{
    // COW 之后释放原持有者，被拷贝的块也应正确回收
    BlockAllocator alloc(4);
    BlockManager mgr(&alloc);
    Request father;
    EXPECT_TRUE(prefill(std::vector<int>(10, 1), father, mgr));
    int pb_old = father.block_table[0];

    Request son = fork_sequence(father, alloc);

    decode(mgr, father); // father COW → father 用新块，son 还持 pb_old
    release_request(father, alloc);
    EXPECT_EQ(alloc.get_block(pb_old).ref_count, 1); // son 还持
    EXPECT_EQ(alloc.num_free_blocks(), 3);           // 4 - 1 (son 的 pb_old)

    release_request(son, alloc);
    EXPECT_EQ(alloc.num_free_blocks(), 4);
}

void test_fork_chain_stress()
{
    // 压力：1 父 + 3 子，各 decode 20 次，验证不变量
    BlockAllocator alloc(64);
    BlockManager mgr(&alloc);
    Request father;
    EXPECT_TRUE(prefill(std::vector<int>(5, 1), father, mgr));

    std::vector<Request> children;
    for(int i = 0; i < 3; ++i)
        children.push_back(fork_sequence(father, alloc));

    std::vector<Request*> all = {&father, &children[0], &children[1], &children[2]};
    for(int step = 0; step < 20; ++step)
        for(Request* r : all)
            EXPECT_TRUE(decode(mgr, *r) >= 0);

    // 每个序列长度 25 → ceil(25/16) = 2 块
    for(Request* r : all)
    {
        EXPECT_EQ(r->sequence_length, 25);
        EXPECT_EQ((int)r->block_table.size(), 2);
    }

    release_request(father, alloc);
    for(auto& c : children)
        release_request(c, alloc);
    EXPECT_EQ(alloc.num_free_blocks(), 64); // 全部归还
}

void test_fork_independent_growth()
{
    // fork 后两个序列独立增长，block_table 后续应完全不同
    BlockAllocator alloc(16);
    BlockManager mgr(&alloc);
    Request father;
    // 首块填满（16 个 token），之后的 decode 都会走“新增块”分支，不再触碰首块
    EXPECT_TRUE(prefill(std::vector<int>(16, 1), father, mgr));
    Request son = fork_sequence(father, alloc);

    for(int i = 0; i < 40; ++i)
    {
        decode(mgr, father);
        decode(mgr, son);
    }

    // 两个序列长度相同，块数应相同，但除共享历史块外，其余块 id 不应相同
    EXPECT_EQ(father.sequence_length, son.sequence_length);
    EXPECT_EQ(father.block_table.size(), son.block_table.size());

    // 首块（共享历史，已填满，不会再被 COW）应相同
    EXPECT_EQ(father.block_table[0], son.block_table[0]);
    // 后续块必须不同（各自独立分配）
    for(size_t i = 1; i < father.block_table.size(); ++i)
        EXPECT_TRUE(father.block_table[i] != son.block_table[i]);

    release_request(father, alloc);
    release_request(son, alloc);
    EXPECT_EQ(alloc.num_free_blocks(), 16);
}


// =========================================================
// 阶段三：Automatic Prefix Caching
// =========================================================

// 生成 n 个从 base 开始递增的 token，便于构造内容各异的 prompt
static std::vector<int> make_tokens(int n, int base = 0)
{
    std::vector<int> v(n);
    for(int i = 0; i < n; ++i)
        v[i] = base + i;
    return v;
}

void test_prefix_first_request_no_hit()
{
    BlockAllocator alloc(8);
    BlockManager mgr(&alloc);
    auto prompt = make_tokens(32);
    EXPECT_EQ(mgr.match_prefix(prompt).num_matched_tokens, 0);

    Request req;
    EXPECT_TRUE(prefill(prompt, req, mgr));
    EXPECT_EQ((int)req.block_table.size(), 2);
    EXPECT_EQ(alloc.get_block(req.block_table[0]).ref_count, 1);
    EXPECT_EQ(alloc.get_block(req.block_table[1]).ref_count, 1);
    EXPECT_EQ(alloc.num_free_blocks(), 6);
}

void test_prefix_full_hit_shares_blocks()
{
    BlockAllocator alloc(8);
    BlockManager mgr(&alloc);
    auto prompt = make_tokens(32);
    Request a, b;
    EXPECT_TRUE(prefill(prompt, a, mgr));
    EXPECT_TRUE(prefill(prompt, b, mgr));

    EXPECT_EQ(b.block_table, a.block_table); // 完全复用
    EXPECT_EQ(b.slot_mapping, a.slot_mapping);
    EXPECT_EQ(b.sequence_length, 32);
    EXPECT_EQ(alloc.get_block(a.block_table[0]).ref_count, 2);
    EXPECT_EQ(alloc.get_block(a.block_table[1]).ref_count, 2);
    EXPECT_EQ(alloc.num_free_blocks(), 6); // 没有新增占用
}

void test_prefix_match_prefix_is_readonly()
{
    BlockAllocator alloc(8);
    BlockManager mgr(&alloc);
    auto prompt = make_tokens(32);
    Request a;
    EXPECT_TRUE(prefill(prompt, a, mgr));

    auto m = mgr.match_prefix(prompt);
    EXPECT_EQ(m.num_matched_tokens, 32);
    EXPECT_EQ(m.cached_blocks, a.block_table);
    // match_prefix 只查询，不应修改引用计数
    EXPECT_EQ(alloc.get_block(a.block_table[0]).ref_count, 1);
    EXPECT_EQ(alloc.get_block(a.block_table[1]).ref_count, 1);
}

void test_prefix_partial_hit()
{
    BlockAllocator alloc(8);
    BlockManager mgr(&alloc);
    auto pa = make_tokens(32);
    auto pb = make_tokens(16); // 第 1 块与 A 相同
    for(int i = 0; i < 16; ++i)
        pb.push_back(1000 + i); // 第 2 块不同

    Request a, b;
    EXPECT_TRUE(prefill(pa, a, mgr));
    EXPECT_TRUE(prefill(pb, b, mgr));

    EXPECT_EQ(b.block_table[0], a.block_table[0]);     // 共享第 1 块
    EXPECT_TRUE(b.block_table[1] != a.block_table[1]); // 第 2 块独立
    EXPECT_EQ(alloc.get_block(a.block_table[0]).ref_count, 2);
    EXPECT_EQ(alloc.get_block(a.block_table[1]).ref_count, 1);
    EXPECT_EQ(alloc.get_block(b.block_table[1]).ref_count, 1);
    EXPECT_EQ(alloc.num_free_blocks(), 5);
}

void test_prefix_miss_on_first_block_diff()
{
    // 首块不同，即使后续块内容相同也不能命中（哈希链依赖整个前缀）
    BlockAllocator alloc(8);
    BlockManager mgr(&alloc);
    auto pa = make_tokens(32);
    auto pb = pa;
    pb[0] = 9999;

    Request a, b;
    EXPECT_TRUE(prefill(pa, a, mgr));
    EXPECT_EQ(mgr.match_prefix(pb).num_matched_tokens, 0);
    EXPECT_TRUE(prefill(pb, b, mgr));
    EXPECT_TRUE(b.block_table[0] != a.block_table[0]);
    EXPECT_TRUE(b.block_table[1] != a.block_table[1]);
    EXPECT_EQ(alloc.num_free_blocks(), 4);
}

void test_prefix_block_order_matters()
{
    BlockAllocator alloc(8);
    BlockManager mgr(&alloc);
    auto x = make_tokens(16, 0);
    auto y = make_tokens(16, 500);
    std::vector<int> xy = x;
    xy.insert(xy.end(), y.begin(), y.end());
    std::vector<int> yx = y;
    yx.insert(yx.end(), x.begin(), x.end());

    Request a;
    EXPECT_TRUE(prefill(xy, a, mgr));
    EXPECT_EQ(mgr.match_prefix(yx).num_matched_tokens, 0); // 顺序不同不能命中
}

void test_prefix_partial_tail_not_cached()
{
    // 未写满的尾块不会被缓存/复用
    BlockAllocator alloc(8);
    BlockManager mgr(&alloc);
    auto prompt = make_tokens(20); // 1 个满块 + 4 个 token
    Request a, b;
    EXPECT_TRUE(prefill(prompt, a, mgr));
    EXPECT_TRUE(prefill(prompt, b, mgr));

    EXPECT_EQ(b.block_table[0], a.block_table[0]);
    EXPECT_TRUE(b.block_table[1] != a.block_table[1]);
    EXPECT_EQ(alloc.get_block(a.block_table[1]).ref_count, 1);
    EXPECT_EQ(alloc.get_block(b.block_table[1]).ref_count, 1);
    EXPECT_EQ(b.slot_mapping[16], b.block_table[1] * BLOCK_SIZE + 0);
}

void test_prefix_short_prompt_never_hits()
{
    BlockAllocator alloc(8);
    BlockManager mgr(&alloc);
    auto prompt = make_tokens(10);
    Request a, b;
    EXPECT_TRUE(prefill(prompt, a, mgr));
    EXPECT_TRUE(prefill(prompt, b, mgr));
    EXPECT_TRUE(a.block_table[0] != b.block_table[0]);
    EXPECT_EQ(alloc.num_free_blocks(), 6);
}

void test_prefix_hit_after_owner_released()
{
    // A 释放后其满块仍保留缓存内容，B 可以复用并把块从空闲链表中取回
    BlockAllocator alloc(4);
    BlockManager mgr(&alloc);
    auto prompt = make_tokens(32);
    Request a, b;
    EXPECT_TRUE(prefill(prompt, a, mgr));
    std::vector<int> old_table = a.block_table;
    mgr.release(a);
    EXPECT_EQ(alloc.num_free_blocks(), 4);
    EXPECT_EQ(alloc.get_block(old_table[0]).ref_count, 0);

    EXPECT_TRUE(prefill(prompt, b, mgr));
    EXPECT_EQ(b.block_table, old_table);
    EXPECT_EQ(alloc.get_block(old_table[0]).ref_count, 1);
    EXPECT_EQ(alloc.get_block(old_table[1]).ref_count, 1);
    EXPECT_EQ(alloc.num_free_blocks(), 2);

    // 被复用的块不能再被 allocate 重复分配出去
    int x = alloc.allocate();
    int y = alloc.allocate();
    EXPECT_TRUE(x >= 0 && y >= 0);
    EXPECT_TRUE(x != old_table[0] && x != old_table[1]);
    EXPECT_TRUE(y != old_table[0] && y != old_table[1]);
}

void test_prefix_stale_after_realloc()
{
    // 缓存块被回收并重新分配给别的内容后，旧缓存项必须失效
    BlockAllocator alloc(2);
    BlockManager mgr(&alloc);
    auto pa = make_tokens(32);
    auto pc = make_tokens(32, 7000);
    Request a, c, b;
    EXPECT_TRUE(prefill(pa, a, mgr));
    mgr.release(a);

    EXPECT_TRUE(prefill(pc, c, mgr)); // 占用并覆盖 A 的块
    mgr.release(c);

    EXPECT_EQ(mgr.match_prefix(pa).num_matched_tokens, 0);
    EXPECT_EQ(mgr.match_prefix(pc).num_matched_tokens, 32);
    EXPECT_TRUE(prefill(pa, b, mgr)); // A 的内容需要重新分配
    EXPECT_EQ((int)b.block_table.size(), 2);
}

void test_prefix_oom_rollback_keeps_shared_refs()
{
    // 命中 2 块后，新块不足导致失败：不能泄漏，也不能破坏被共享块的引用计数
    BlockAllocator alloc(3);
    BlockManager mgr(&alloc);
    auto pa = make_tokens(32);
    auto pb = pa;
    auto tail = make_tokens(32, 3000);
    pb.insert(pb.end(), tail.begin(), tail.end()); // 64 token，需 4 块

    Request a, b;
    EXPECT_TRUE(prefill(pa, a, mgr));
    EXPECT_EQ(alloc.num_free_blocks(), 1);
    EXPECT_FALSE(prefill(pb, b, mgr));

    EXPECT_EQ(alloc.get_block(a.block_table[0]).ref_count, 1);
    EXPECT_EQ(alloc.get_block(a.block_table[1]).ref_count, 1);
    EXPECT_EQ(alloc.num_free_blocks(), 1);
    EXPECT_EQ((int)b.block_table.size(), 0);
    EXPECT_EQ(b.sequence_length, 0);
}

void test_prefix_decode_fills_block_and_registers()
{
    // decode 把块写满后，该块应被注册，后续相同前缀的请求可命中
    BlockAllocator alloc(8);
    BlockManager mgr(&alloc);
    auto prompt = make_tokens(10);
    Request a;
    EXPECT_TRUE(prefill(prompt, a, mgr));
    EXPECT_EQ(mgr.match_prefix(make_tokens(16)).num_matched_tokens, 0);

    for(int t = 10; t < 16; ++t)
        EXPECT_TRUE(decode(mgr, a, t) >= 0);
    EXPECT_EQ(a.sequence_length, 16);
    EXPECT_EQ(a.num_cached_blocks, 1);

    Request b;
    EXPECT_TRUE(prefill(make_tokens(16), b, mgr));
    EXPECT_EQ(b.block_table[0], a.block_table[0]);
    EXPECT_EQ(alloc.get_block(a.block_table[0]).ref_count, 2);
}

void test_prefix_decode_without_token_no_register()
{
    // 不提供 token 时无法得知块内容，不应注册错误的缓存
    BlockAllocator alloc(8);
    BlockManager mgr(&alloc);
    Request a;
    EXPECT_TRUE(prefill(make_tokens(10), a, mgr));
    for(int i = 0; i < 6; ++i)
        EXPECT_TRUE(decode(mgr, a) >= 0);
    EXPECT_EQ(a.sequence_length, 16);
    EXPECT_EQ(mgr.match_prefix(make_tokens(16)).num_matched_tokens, 0);
}

void test_prefix_hit_then_decode_new_block()
{
    // 命中的满块被共享，decode 从新块开始写，不影响共享块
    BlockAllocator alloc(8);
    BlockManager mgr(&alloc);
    auto prompt = make_tokens(16);
    Request a, b;
    EXPECT_TRUE(prefill(prompt, a, mgr));
    EXPECT_TRUE(prefill(prompt, b, mgr));
    int shared = a.block_table[0];

    int sid = decode(mgr, b, 777);
    EXPECT_EQ((int)b.block_table.size(), 2);
    EXPECT_TRUE(b.block_table[1] != shared);
    EXPECT_EQ(sid, b.block_table[1] * BLOCK_SIZE + 0);
    EXPECT_EQ(alloc.get_block(shared).ref_count, 2);
    EXPECT_EQ((int)a.block_table.size(), 1);
}

void test_prefix_hit_partial_tail_decode_no_cow()
{
    // 命中前缀后，尾块是各自独占的，decode 不触发 COW
    BlockAllocator alloc(8);
    BlockManager mgr(&alloc);
    auto prompt = make_tokens(20);
    Request a, b;
    EXPECT_TRUE(prefill(prompt, a, mgr));
    EXPECT_TRUE(prefill(prompt, b, mgr));
    int tail_b = b.block_table[1];

    int sid = decode(mgr, b, 888);
    EXPECT_EQ(b.block_table[1], tail_b);
    EXPECT_EQ(sid, tail_b * BLOCK_SIZE + 4);
    EXPECT_EQ(b.block_table[0], a.block_table[0]);
}

void test_prefix_release_balance()
{
    BlockAllocator alloc(8);
    BlockManager mgr(&alloc);
    auto prompt = make_tokens(48);
    Request a, b, c;
    EXPECT_TRUE(prefill(prompt, a, mgr));
    EXPECT_TRUE(prefill(prompt, b, mgr));
    EXPECT_TRUE(prefill(prompt, c, mgr));
    EXPECT_EQ(alloc.get_block(a.block_table[0]).ref_count, 3);
    EXPECT_EQ(alloc.num_free_blocks(), 5);

    mgr.release(a);
    mgr.release(b);
    EXPECT_EQ(alloc.get_block(c.block_table[0]).ref_count, 1);
    EXPECT_EQ(alloc.num_free_blocks(), 5);

    mgr.release(c);
    EXPECT_EQ(alloc.num_free_blocks(), 8);
    EXPECT_EQ(c.sequence_length, 0);
    EXPECT_EQ((int)c.block_table.size(), 0);
}

void test_prefix_reprefill_same_request()
{
    // 同一个 Request 重复 prefill：旧资源被归还，且不会和自己的旧块重复计数
    BlockAllocator alloc(8);
    BlockManager mgr(&alloc);
    auto prompt = make_tokens(32);
    Request a;
    EXPECT_TRUE(prefill(prompt, a, mgr));
    EXPECT_TRUE(prefill(prompt, a, mgr));
    EXPECT_EQ((int)a.block_table.size(), 2);
    EXPECT_EQ(alloc.get_block(a.block_table[0]).ref_count, 1);
    EXPECT_EQ(alloc.get_block(a.block_table[1]).ref_count, 1);
    EXPECT_EQ(alloc.num_free_blocks(), 6);
}

void test_prefix_fork_then_decode_with_cache()
{
    // 前缀缓存与 Fork/COW 同时使用：引用计数保持平衡
    BlockAllocator alloc(8);
    BlockManager mgr(&alloc);
    auto prompt = make_tokens(20);
    Request a, b;
    EXPECT_TRUE(prefill(prompt, a, mgr));
    EXPECT_TRUE(prefill(prompt, b, mgr)); // 共享首块
    Request son = fork_sequence(a, alloc);
    EXPECT_EQ(alloc.get_block(a.block_table[0]).ref_count, 3);

    EXPECT_TRUE(decode(mgr, a, 11) >= 0); // 尾块被 son 共享 → COW
    EXPECT_TRUE(a.block_table[1] != son.block_table[1]);
    EXPECT_EQ(alloc.get_block(son.block_table[1]).ref_count, 1);

    mgr.release(a);
    mgr.release(b);
    mgr.release(son);
    EXPECT_EQ(alloc.num_free_blocks(), 8);
}

// =========================================================
// v0.2.0：运行期配置、增量哈希与 CoW 计数
// =========================================================

void test_config_custom_block_size()
{
    constexpr int kBlk = 4;
    BlockAllocator alloc(8, kBlk);
    BlockManager mgr(&alloc);
    EXPECT_EQ(alloc.block_size(), kBlk);
    EXPECT_EQ(mgr.block_size(), kBlk);

    // 10 token = 2 个满块(8) + 1 个半块(2)
    auto prompt = make_tokens(10, 100);
    Request a, b;
    EXPECT_TRUE(prefill(prompt, a, mgr));
    EXPECT_EQ((int)a.block_table.size(), 3);
    EXPECT_EQ(a.num_cached_blocks, 2);
    EXPECT_EQ(a.slot_mapping[9], a.block_table[2] * kBlk + 1);

    // 第二个请求应复用前 2 个满块
    EXPECT_TRUE(prefill(prompt, b, mgr));
    EXPECT_EQ(b.block_table[0], a.block_table[0]);
    EXPECT_EQ(b.block_table[1], a.block_table[1]);
    EXPECT_TRUE(b.block_table[2] != a.block_table[2]);

    // decode 填满第 3 块并跨到第 4 块
    EXPECT_EQ(decode(mgr, a, 110), a.block_table[2] * kBlk + 2);
    EXPECT_EQ(decode(mgr, a, 111), a.block_table[2] * kBlk + 3);
    EXPECT_EQ(a.num_cached_blocks, 3);
    EXPECT_EQ(mgr.match_prefix(a.all_tokens).num_matched_tokens, 12);

    int sid = decode(mgr, a, 112);
    EXPECT_EQ((int)a.block_table.size(), 4);
    EXPECT_EQ(sid, a.block_table[3] * kBlk + 0);

    mgr.release(a);
    mgr.release(b);
    EXPECT_EQ(alloc.num_free_blocks(), 8);
}

void test_config_disable_prefix_caching()
{
    BlockAllocator alloc(8);
    BlockManager mgr(&alloc, BlockManagerConfig{/*enable_prefix_caching=*/false});
    EXPECT_FALSE(mgr.prefix_caching_enabled());

    auto prompt = make_tokens(32, 200);
    Request a, b;
    EXPECT_TRUE(prefill(prompt, a, mgr));
    EXPECT_EQ(a.num_cached_blocks, 0);
    EXPECT_EQ(mgr.match_prefix(prompt).num_matched_tokens, 0);

    EXPECT_TRUE(prefill(prompt, b, mgr));
    EXPECT_TRUE(a.block_table[0] != b.block_table[0]);
    EXPECT_TRUE(a.block_table[1] != b.block_table[1]);
    EXPECT_EQ(alloc.num_free_blocks(), 4);

    mgr.release(a);
    mgr.release(b);
    EXPECT_EQ(alloc.num_free_blocks(), 8);
}

void test_cow_counter_increments()
{
    BlockAllocator alloc(8);
    BlockManager mgr(&alloc);
    Request father;
    EXPECT_TRUE(prefill(make_tokens(10, 50), father, mgr));
    EXPECT_EQ(mgr.num_cow_copies(), 0);

    Request son = fork_sequence(father, alloc);
    EXPECT_TRUE(decode(mgr, father, 60) >= 0); // 共享尾块 → 触发 CoW
    EXPECT_EQ(mgr.num_cow_copies(), 1);

    EXPECT_TRUE(decode(mgr, son, 61) >= 0); // 尾块已独占 → 不触发 CoW
    EXPECT_EQ(mgr.num_cow_copies(), 1);

    mgr.release(father);
    mgr.release(son);
    EXPECT_EQ(alloc.num_free_blocks(), 8);
}

void test_incremental_register_block_matches_full()
{
    // 验证 O(1) 增量 register_block 连续跨多块登记后，哈希链与完整前缀一致
    BlockAllocator alloc(16, 4);
    BlockManager mgr(&alloc);

    Request a;
    EXPECT_TRUE(prefill(make_tokens(6, 300), a, mgr)); // 1 满块 + 2 token
    EXPECT_EQ(a.num_cached_blocks, 1);

    // 连续 decode 10 个 token，总长 16 = 4 个满块
    for(int t = 6; t < 16; ++t)
        EXPECT_TRUE(decode(mgr, a, 300 + t) >= 0);
    EXPECT_EQ(a.num_cached_blocks, 4);

    auto full_prompt = make_tokens(16, 300);
    auto m = mgr.match_prefix(full_prompt);
    EXPECT_EQ(m.num_matched_tokens, 16);
    EXPECT_EQ(m.cached_blocks, a.block_table);

    mgr.release(a);
    EXPECT_EQ(alloc.num_free_blocks(), 16);
}

// =========================================================
int main()
{
    std::cout << "========== 阶段一：分页与 Slot Mapping ==========\n";
    RUN_TEST(test_allocator_allocate_unique);
    RUN_TEST(test_allocator_exhaust);
    RUN_TEST(test_allocator_reclaim);
    RUN_TEST(test_allocator_inc_dec_balance);
    RUN_TEST(test_prefill_empty);
    RUN_TEST(test_prefill_one_token);
    RUN_TEST(test_prefill_exactly_one_block);
    RUN_TEST(test_prefill_spans_two_blocks);
    RUN_TEST(test_prefill_100_tokens);
    RUN_TEST(test_prefill_out_of_memory_rollback);
    RUN_TEST(test_prefill_reused_request_no_leak);
    RUN_TEST(test_decode_same_block);
    RUN_TEST(test_decode_new_block_at_boundary);
    RUN_TEST(test_decode_long_sequence);
    RUN_TEST(test_release_restores_free);
    RUN_TEST(test_release_after_decode);

    std::cout << "\n========== 阶段二：Fork 与 COW ==========\n";
    RUN_TEST(test_fork_inc_ref);
    RUN_TEST(test_fork_release_one);
    RUN_TEST(test_cow_full_tail_no_copy);
    RUN_TEST(test_cow_partial_tail_trigger);
    RUN_TEST(test_cow_only_first_writer_copies);
    RUN_TEST(test_cow_after_which_release);
    RUN_TEST(test_fork_chain_stress);
    RUN_TEST(test_fork_independent_growth);

    std::cout << "\n========== 阶段三：Automatic Prefix Caching ==========\n";
    RUN_TEST(test_prefix_first_request_no_hit);
    RUN_TEST(test_prefix_full_hit_shares_blocks);
    RUN_TEST(test_prefix_match_prefix_is_readonly);
    RUN_TEST(test_prefix_partial_hit);
    RUN_TEST(test_prefix_miss_on_first_block_diff);
    RUN_TEST(test_prefix_block_order_matters);
    RUN_TEST(test_prefix_partial_tail_not_cached);
    RUN_TEST(test_prefix_short_prompt_never_hits);
    RUN_TEST(test_prefix_hit_after_owner_released);
    RUN_TEST(test_prefix_stale_after_realloc);
    RUN_TEST(test_prefix_oom_rollback_keeps_shared_refs);
    RUN_TEST(test_prefix_decode_fills_block_and_registers);
    RUN_TEST(test_prefix_decode_without_token_no_register);
    RUN_TEST(test_prefix_hit_then_decode_new_block);
    RUN_TEST(test_prefix_hit_partial_tail_decode_no_cow);
    RUN_TEST(test_prefix_release_balance);
    RUN_TEST(test_prefix_reprefill_same_request);
    RUN_TEST(test_prefix_fork_then_decode_with_cache);

    std::cout << "\n========== v0.2.0：配置化与增量哈希 ==========\n";
    RUN_TEST(test_config_custom_block_size);
    RUN_TEST(test_config_disable_prefix_caching);
    RUN_TEST(test_cow_counter_increments);
    RUN_TEST(test_incremental_register_block_matches_full);

    if(g_fail_count == 0)
    {
        std::cout << "\n所有测试通过 ✅\n";
        return 0;
    }
    std::cout << "\n失败 " << g_fail_count << " 项 ❌\n";
    return 1;
}