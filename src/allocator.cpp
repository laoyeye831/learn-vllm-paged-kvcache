#include "kvcache.h"
#include <algorithm>
#include <memory>
#include <string>
#include <cassert>
#include <functional>
#include <stdexcept>

// ====== 分配器成员实现 ======
BlockAllocator::BlockAllocator(int num_blocks)
{
    blocks_.reserve(num_blocks);
    for(int i = 0; i < num_blocks; i++)
    {
        blocks_.push_back(PhysicalBlock{i, 0, 0});
        free_list_.push_back(i);
    }
}

int BlockAllocator::allocate()
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    if(free_list_.empty()) return -1; // 无空闲块
    int temp_id = free_list_.front();
    free_list_.pop_front();

    auto& block = blocks_[temp_id];
    block.ref_count += 1;
    block.has_hash = false; // 块被重新分配，旧的缓存内容失效
    return temp_id;
}

void BlockAllocator::free(int block_id)
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    auto& block = blocks_[block_id];
    if(block.ref_count == 0) return; // 已经是空闲块
    block.ref_count = 0;
    free_list_.push_back(block_id);
}

void BlockAllocator::inc_ref(int block_id)
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    auto& block = blocks_[block_id];
    if(block.ref_count == 0) // 复用已释放但仍保留缓存内容的块，需从空闲链表摘除
    {
        free_list_.remove(block_id);
    }
    block.ref_count += 1;
}

void BlockAllocator::dec_ref(int block_id)
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    auto& block = blocks_[block_id];
    if(block.ref_count <= 0) return;
    block.ref_count -= 1;
    if(block.ref_count == 0) free_list_.push_back(block_id); // 归零才回收
}

int BlockAllocator::num_blocks() const
{
    return static_cast<int>(blocks_.size());
}

int BlockAllocator::num_free_blocks() const
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    return static_cast<int>(free_list_.size());
}

PhysicalBlock& BlockAllocator::get_block(int block_id)
{
    PhysicalBlock& target_block = blocks_[block_id];
    return target_block;
}

int BlockAllocator::ref_count_of(int block_id) const
{
    assert(block_id >= 0 && block_id < static_cast<int>(blocks_.size()));
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    return blocks_[block_id].ref_count;
}