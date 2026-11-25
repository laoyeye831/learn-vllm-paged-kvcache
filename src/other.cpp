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
bool prefill(const std::vector<int>& prompt_tokens, Request& request, BlockManager& manager)
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

    // 进行复用判断与界定
    auto matched = manager.match_prefix(prompt_tokens);
    int matched_blocks_num = static_cast<int>(matched.cached_blocks.size());

    // 先将复用块加入block_table，并inc_ref
    for(auto id : matched.cached_blocks)
    {
        allocator->inc_ref(id);
        request.block_table.push_back(id);
    }

    for(int i = matched_blocks_num; i < block_nums; i++)
    {
        int temp_id = allocator->allocate(); // 获取空闲物理块id
        if(temp_id < 0)                      // 失败时回滚：复用块和新分配块都只释放本请求持有的引用
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
    request.num_cached_blocks = matched_blocks_num;

    // 把未命中的满块注册进缓存
    manager.register_block(request, matched_blocks_num);
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
        int new_block_id = allocator->allocate();
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
            int new_block_id = allocator->allocate();
            if(new_block_id < 0) return -1;
            request.block_table[tail_block_logical_id] = new_block_id; // 尾块更新

            // 拷贝原尾块内的有效token进新的块（当前为元信息模拟，仅计数不搬运真实 KV）
            manager.record_cow_copy();
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
