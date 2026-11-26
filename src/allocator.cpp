#include "kvcache.h"
#include <cassert>

// ====== 分配器成员实现 ======
BlockAllocator::BlockAllocator(int num_blocks, int block_size, FreePolicy policy)
    : block_size_(block_size), policy_(policy)
{
    assert(num_blocks >= 0 && "BlockAllocator: num_blocks must be non-negative");
    assert(block_size_ > 0 && "BlockAllocator: block_size must be positive");
    blocks_.reserve(num_blocks);
    for(int i = 0; i < num_blocks; i++)
    {
        blocks_.push_back(PhysicalBlock{i, 0, 0});
        // 初始块全部无缓存（CLEAN），按编号顺序入队尾
        blocks_[i].tier = FreeTier::CLEAN;
        link_back_locked(i);
        ++num_clean_free_;
    }
}

void BlockAllocator::link_front_locked(int id)
{
    PhysicalBlock& b = blocks_[id];
    b.free_prev = -1;
    b.free_next = free_head_;
    if(free_head_ >= 0)
        blocks_[free_head_].free_prev = id;
    else
        free_tail_ = id;
    free_head_ = id;
}

void BlockAllocator::link_back_locked(int id)
{
    PhysicalBlock& b = blocks_[id];
    b.free_next = -1;
    b.free_prev = free_tail_;
    if(free_tail_ >= 0)
        blocks_[free_tail_].free_next = id;
    else
        free_head_ = id;
    free_tail_ = id;
}

void BlockAllocator::unlink_locked(int id)
{
    PhysicalBlock& b = blocks_[id];
    if(b.free_prev >= 0)
        blocks_[b.free_prev].free_next = b.free_next;
    else
        free_head_ = b.free_next;
    if(b.free_next >= 0)
        blocks_[b.free_next].free_prev = b.free_prev;
    else
        free_tail_ = b.free_prev;
    b.free_prev = -1;
    b.free_next = -1;
}

void BlockAllocator::push_free_locked(int id)
{
    PhysicalBlock& b = blocks_[id];
    assert(b.ref_count == 0 && b.tier == FreeTier::NONE);

    const bool cached = b.has_hash;
    b.tier = cached ? FreeTier::CACHED : FreeTier::CLEAN;
    if(cached)
    {
        ++num_cached_free_;
        // 归还为 CACHED 空闲时打时间戳，保证 LRU 缓存后缀的 last_accessed 单调递增
        b.last_accessed = ++access_clock_;
    }
    else
    {
        ++num_clean_free_;
    }

    // LRU：CLEAN 块进队首（优先复用），CACHED 块进队尾（队首即为最久未用的淘汰受害者）
    // FIFO：所有空闲块一律进队尾
    if(policy_ == FreePolicy::LRU && !cached)
        link_front_locked(id);
    else
        link_back_locked(id);
}

BlockAllocator::AllocOutcome BlockAllocator::allocate_ex()
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    AllocOutcome out;
    if(free_head_ < 0) return out; // 无空闲块

    const int id = free_head_;
    PhysicalBlock& b = blocks_[id];
    out.block_id = id;
    out.evicted = b.has_hash; // 队首是缓存块说明本次分配触发淘汰
    if(out.evicted) out.evicted_hash = b.hash;

    unlink_locked(id);
    if(b.tier == FreeTier::CACHED)
        --num_cached_free_;
    else
        --num_clean_free_;
    if(out.evicted) ++num_evictions_;

    b.tier = FreeTier::NONE;
    b.ref_count = 1;
    b.has_hash = false; // 先读出 evicted_hash 再清空，防止幽灵条目
    b.hash = 0;
    b.last_accessed = ++access_clock_;
    ++num_allocations_;
    return out;
}

void BlockAllocator::inc_ref(int block_id)
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    PhysicalBlock& b = blocks_[block_id];
    if(b.ref_count == 0) // 复活空闲缓存块，O(1) 摘链
    {
        assert(b.tier != FreeTier::NONE);
        unlink_locked(block_id);
        if(b.tier == FreeTier::CACHED)
            --num_cached_free_;
        else
            --num_clean_free_;
        b.tier = FreeTier::NONE;
        ++num_revivals_;
        b.last_accessed = ++access_clock_;
    }
    b.ref_count += 1;
}

void BlockAllocator::dec_ref(int block_id)
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    PhysicalBlock& b = blocks_[block_id];
    if(b.ref_count <= 0) return;
    b.ref_count -= 1;
    if(b.ref_count == 0) push_free_locked(block_id); // 归零才回收，按 has_hash 路由
}

int BlockAllocator::num_blocks() const
{
    return static_cast<int>(blocks_.size());
}

int BlockAllocator::block_size() const
{
    return block_size_;
}

int BlockAllocator::num_free_blocks() const
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    return num_clean_free_ + num_cached_free_;
}

int BlockAllocator::num_free_clean() const
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    return num_clean_free_;
}

int BlockAllocator::num_free_cached() const
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    return num_cached_free_;
}

PhysicalBlock& BlockAllocator::get_block(int block_id)
{
    return blocks_[block_id];
}

const PhysicalBlock& BlockAllocator::get_block(int block_id) const
{
    return blocks_[block_id];
}

int BlockAllocator::ref_count_of(int block_id) const
{
    assert(block_id >= 0 && block_id < static_cast<int>(blocks_.size()));
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    return blocks_[block_id].ref_count;
}

int BlockAllocator::free_head() const
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    return free_head_;
}

int BlockAllocator::free_tail() const
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    return free_tail_;
}

uint64_t BlockAllocator::num_allocations() const
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    return num_allocations_;
}

uint64_t BlockAllocator::num_evictions() const
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    return num_evictions_;
}

uint64_t BlockAllocator::num_revivals() const
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    return num_revivals_;
}


int BlockAllocator::free_head() const
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    return free_head_;
}

int BlockAllocator::free_tail() const
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    return free_tail_;
}

uint64_t BlockAllocator::num_allocations() const
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    return num_allocations_;
}

uint64_t BlockAllocator::num_evictions() const
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    return num_evictions_;
}

uint64_t BlockAllocator::num_revivals() const
{
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    return num_revivals_;
}