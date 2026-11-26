#include "kvcache.h"
#include <algorithm>
#include <memory>
#include <string>
#include <cassert>
#include <functional>
#include <stdexcept>

// ====== Token生成器 ======
int fake_sample(const Request& r)
{
    return (r.all_tokens.back() * 31 + r.sequence_length) % 50000;
}

// ====== 其他函数 ======
// 注：若 prefill 中途因空间不足回滚，已分配阶段触发的缓存淘汰不会恢复；
//     Scheduler 阶段三的准入预检保证不会在空间不足时调用 prefill。
bool prefill(const std::vector<int>& prompt_tokens, Request& request, BlockManager& manager,
             const BlockManager::PrefixMatch* precomputed_match)
{
    auto allocator = manager.get_allocator();
    const int blk_size = manager.block_size();
    manager.release(request); // 同一个 Request 重复 prefill 时先归还旧资源

    int N = static_cast<int>(prompt_tokens.size());
    if(N == 0)
    {
        return true;
    }

    int block_nums = (N + blk_size - 1) / blk_size; // 需要的块数
    request.block_table.reserve(block_nums);

    // 复用调度器阶段三已算好的匹配结果，避免同一步内重复哈希与重复记账
    BlockManager::PrefixMatch local_match;
    const BlockManager::PrefixMatch& matched =
        precomputed_match ? *precomputed_match : (local_match = manager.match_prefix(prompt_tokens));
    int matched_blocks_num = static_cast<int>(matched.cached_blocks.size());
    int full_query_blocks = N / blk_size;
    manager.record_prefill_stats(N, full_query_blocks, matched_blocks_num);

    // 先将复用块加入block_table，并inc_ref
    for(auto id : matched.cached_blocks)
    {
        allocator->inc_ref(id);
        request.block_table.push_back(id);
    }

    for(int i = matched_blocks_num; i < block_nums; i++)
    {
        int temp_id = manager.allocate_block(); // 统一经 BlockManager 分配，确保淘汰同步清哈希
        if(temp_id < 0)                         // 失败时回滚：复用块和新分配块都只释放本请求持有的引用
        {
            manager.release(request);
            return false;
        }
        request.block_table.push_back(temp_id);
    }

    request.all_tokens = prompt_tokens;
    for(int t = 0; t < N; t++)
    {
        int logical_block_idx = t / blk_size; // 计算每个token属于哪个逻辑块
        int offset = t % blk_size;            // 计算块内偏移
        int physical_block_id = request.block_table[logical_block_idx];
        int slot_id = physical_block_id * blk_size + offset;
        request.slot_mapping.push_back(slot_id);
    }
    request.sequence_length = N;
    request.num_computed_tokens = N;
    request.num_cached_blocks = matched_blocks_num;

    // 把未命中的满块注册进缓存
    manager.register_block(request, matched_blocks_num);
    return true;
}

bool prefill_chunk(Request& request, int num_new_tokens, BlockManager& manager,
                   const BlockManager::PrefixMatch* precomputed_match,
                   int* out_cached_tokens, int* out_computed_tokens)
{
    auto allocator = manager.get_allocator();
    const int blk_size = manager.block_size();
    const int total_tokens = static_cast<int>(request.all_tokens.size());
    if(out_cached_tokens) *out_cached_tokens = 0;
    if(out_computed_tokens) *out_computed_tokens = 0;

    if(total_tokens == 0) return true;

    // 情况 1：首次分块（num_computed_tokens == 0），先做前缀缓存匹配与复用
    if(request.num_computed_tokens == 0)
    {
        manager.release(request);

        BlockManager::PrefixMatch local_match;
        const BlockManager::PrefixMatch& matched =
            precomputed_match ? *precomputed_match : (local_match = manager.match_prefix(request.all_tokens));
        const int matched_blocks_num = static_cast<int>(matched.cached_blocks.size());
        const int cached_tokens = matched.num_matched_tokens;
        const int uncached_tokens = std::max(0, total_tokens - cached_tokens);
        const int chunk_new = (num_new_tokens > 0) ? std::min(uncached_tokens, num_new_tokens) : uncached_tokens;
        const int new_len = cached_tokens + chunk_new;
        const int target_blocks = (new_len + blk_size - 1) / blk_size;

        manager.record_prefill_stats(total_tokens, total_tokens / blk_size, matched_blocks_num);

        request.block_table.reserve(target_blocks);
        for(int id : matched.cached_blocks)
        {
            allocator->inc_ref(id);
            request.block_table.push_back(id);
        }
        for(int i = matched_blocks_num; i < target_blocks; ++i)
        {
            int new_id = manager.allocate_block();
            if(new_id < 0)
            {
                manager.release(request);
                return false;
            }
            request.block_table.push_back(new_id);
        }

        request.slot_mapping.clear();
        request.slot_mapping.reserve(static_cast<size_t>(chunk_new));
        for(int t = cached_tokens; t < new_len; ++t)
        {
            int pb = request.block_table[t / blk_size];
            request.slot_mapping.push_back(pb * blk_size + (t % blk_size));
        }
        request.sequence_length = new_len;
        request.num_computed_tokens = new_len;
        request.num_cached_blocks = matched_blocks_num;

        // 把本 chunk 刚刚写满的块登记进前缀缓存
        manager.register_block(request, matched_blocks_num);
        if(out_cached_tokens) *out_cached_tokens = cached_tokens;
        if(out_computed_tokens) *out_computed_tokens = chunk_new;
        return true;
    }

    // 情况 2：续算分块（0 < num_computed_tokens < total_tokens）
    const int old_len = request.num_computed_tokens;
    const int remaining = std::max(0, total_tokens - old_len);
    const int chunk_new = (num_new_tokens > 0) ? std::min(remaining, num_new_tokens) : remaining;
    if(chunk_new == 0) return true;

    const int new_len = old_len + chunk_new;
    const int old_blocks = static_cast<int>(request.block_table.size());
    const int target_blocks = (new_len + blk_size - 1) / blk_size;

    // 若尾块未满且被共享（防御性 CoW），先分裂尾块
    const int tail_offset = old_len % blk_size;
    if(tail_offset != 0 && !request.block_table.empty())
    {
        int tail_phys = request.block_table.back();
        if(allocator->get_block(tail_phys).ref_count > 1)
        {
            int cow_id = manager.allocate_block();
            if(cow_id < 0) return false;
            request.block_table.back() = cow_id;
            manager.record_cow_copy(tail_offset);
            allocator->dec_ref(tail_phys);
        }
    }

    for(int i = old_blocks; i < target_blocks; ++i)
    {
        int new_id = manager.allocate_block();
        if(new_id < 0)
        {
            // 回滚本次续算新分配的块，保持 old_len 状态一致
            while(static_cast<int>(request.block_table.size()) > old_blocks)
            {
                allocator->dec_ref(request.block_table.back());
                request.block_table.pop_back();
            }
            return false;
        }
        request.block_table.push_back(new_id);
    }

    request.slot_mapping.clear();
    request.slot_mapping.reserve(static_cast<size_t>(chunk_new));
    for(int t = old_len; t < new_len; ++t)
    {
        int pb = request.block_table[t / blk_size];
        request.slot_mapping.push_back(pb * blk_size + (t % blk_size));
    }
    request.sequence_length = new_len;
    request.num_computed_tokens = new_len;

    // 增量注册本 chunk 新写满的块
    manager.register_block(request, request.num_cached_blocks);
    if(out_computed_tokens) *out_computed_tokens = chunk_new;
    return true;
}

int append_slot(BlockManager& manager, Request& request)
{
    auto allocator = manager.get_allocator();
    const int blk_size = manager.block_size();
    int N = request.sequence_length;
    int offset = N % blk_size;
    int tail_block_logical_id = N / blk_size;
    int slot_id = -1;

    // 情况A：需要新增块
    if(offset == 0)
    {
        int new_block_id = manager.allocate_block();
        if(new_block_id < 0) return -1;
        request.block_table.push_back(new_block_id);
        slot_id = new_block_id * blk_size + offset;
    }
    else
    {
        int tail_block_physical_id = request.block_table[tail_block_logical_id];
        PhysicalBlock& tail_block = allocator->get_block(tail_block_physical_id);

        // 情况B：不需要新增且该块被该序列独占
        if(tail_block.ref_count == 1)
        {
            slot_id = tail_block_physical_id * blk_size + offset;
        }
        // 情况C：不需要新增但处于共享状态，需要写时复制
        else
        {
            int new_block_id = manager.allocate_block();
            if(new_block_id < 0) return -1;
            request.block_table[tail_block_logical_id] = new_block_id; // 尾块更新

            // 拷贝原尾块内的有效token进新的块（当前为元信息模拟，记录次数与搬运 token 数）
            manager.record_cow_copy(offset);
            allocator->dec_ref(tail_block_physical_id);
            slot_id = new_block_id * blk_size + offset;
        }
    }

    request.slot_mapping.clear(); // 清空旧的 slot_mapping
    request.slot_mapping.push_back(slot_id);
    assert(slot_id >= 0 && "append_slot: slot_id should be non-negative");
    return slot_id;
}

void append_token(BlockManager& manager, Request& request, int token)
{
    const int blk_size = manager.block_size();
    int N = request.sequence_length;

    // 仅当 token 序列与长度保持同步时才记录，以保证哈希基于真实内容
    if(token >= 0 && static_cast<int>(request.all_tokens.size()) == N)
    {
        request.all_tokens.push_back(token);
    }

    request.sequence_length += 1;
    request.num_computed_tokens += 1;

    // 块刚好写满，注册进前缀缓存
    if(request.sequence_length % blk_size == 0)
    {
        manager.register_block(request, request.num_cached_blocks);
    }
}

Request fork_sequence(Request& father_request, BlockAllocator& allocator)
{
    Request son = father_request;
    son.slot_mapping.clear(); // 清空旧的 slot_mapping
    for(auto& id : son.block_table)
    {
        allocator.inc_ref(id);
    }
    return son;
}
