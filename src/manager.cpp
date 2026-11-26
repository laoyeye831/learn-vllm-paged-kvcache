#include "kvcache.h"
#include <algorithm>
#include <cassert>

constexpr uint64_t PREFIX_HASH_SEED = 123456789ULL;

// 多项式滚动哈希：直接读取连续区间 [data, data + len)，避免临时 vector 分配
static uint64_t compute_hash(uint64_t seed, const int* data, size_t len)
{
    constexpr uint64_t BASE = 911382629ULL;
    uint64_t h = seed;
    for(size_t i = 0; i < len; ++i)
    {
        h = h * BASE + static_cast<uint64_t>(data[i]);
    }
    return h;
}

// ====== 管理器成员实现 ======
BlockManager::BlockManager(BlockAllocator* allocator, BlockManagerConfig config)
    : allocator_(allocator), config_(config)
{
    assert(allocator_ != nullptr && "BlockManager: allocator must not be null");
}

BlockAllocator* BlockManager::get_allocator()
{
    return allocator_;
}

const BlockAllocator* BlockManager::get_allocator() const
{
    return allocator_;
}

int BlockManager::block_size() const
{
    return allocator_->block_size();
}

int BlockManager::allocate_block()
{
    BlockAllocator::AllocOutcome out = allocator_->allocate_ex();
    if(out.block_id < 0) return -1;
    if(out.evicted) evict_hash_entry(out.block_id, out.evicted_hash);
    return out.block_id;
}

void BlockManager::evict_hash_entry(int block_id, uint64_t hash)
{
    auto it = hash_to_block_.find(hash);
    // 只有映射确实指向被淘汰的这个物理块时才删，避免误删同内容另一块的缓存条目
    if(it != hash_to_block_.end() && it->second == block_id)
    {
        hash_to_block_.erase(it);
    }
}

void BlockManager::register_block(Request& request, int begin)
{
    if(!config_.enable_prefix_caching) return;

    const int blk_size = allocator_->block_size();
    const auto& tokens = request.all_tokens;
    int valid_tokens = std::min<int>(request.sequence_length, static_cast<int>(tokens.size()));
    int full_block_nums = std::min<int>(valid_tokens / blk_size, static_cast<int>(request.block_table.size()));
    if(begin < 0) begin = 0;
    if(begin >= full_block_nums)
    {
        request.num_cached_blocks = std::max(request.num_cached_blocks, full_block_nums);
        return;
    }

    // O(1) 增量哈希：前一块若已登记有效哈希，直接以它为父种子；否则退回从第 0 块推算
    uint64_t seed = PREFIX_HASH_SEED;
    int start_block = 0;
    if(begin > 0)
    {
        const auto& prev_block = allocator_->get_block(request.block_table[begin - 1]);
        if(prev_block.has_hash)
        {
            seed = prev_block.hash;
            start_block = begin;
        }
    }

    for(int i = start_block; i < full_block_nums; i++)
    {
        size_t start = static_cast<size_t>(i) * static_cast<size_t>(blk_size);
        seed = compute_hash(seed, tokens.data() + start, static_cast<size_t>(blk_size)); // 链式哈希，依赖整个前缀
        if(i < begin) continue;                                                          // 回退路径：前面已缓存的块只需推进哈希链

        int physical_block_id = request.block_table[i];
        hash_to_block_[seed] = physical_block_id;
        auto& block = allocator_->get_block(physical_block_id);
        block.hash = seed;
        block.has_hash = true;
        ++num_registrations_;
    }
    request.num_cached_blocks = std::max(request.num_cached_blocks, full_block_nums);
}

BlockManager::PrefixMatch BlockManager::match_prefix(const std::vector<int>& prompt_tokens)
{
    PrefixMatch matched;
    if(!config_.enable_prefix_caching) return matched;

    const int blk_size = allocator_->block_size();
    size_t full_block_nums = prompt_tokens.size() / static_cast<size_t>(blk_size);
    uint64_t seed = PREFIX_HASH_SEED;
    for(size_t i = 0; i < full_block_nums; i++)
    {
        size_t start = i * static_cast<size_t>(blk_size);
        seed = compute_hash(seed, prompt_tokens.data() + start, static_cast<size_t>(blk_size));
        auto it = hash_to_block_.find(seed);
        if(it == hash_to_block_.end()) break;

        const auto& block = allocator_->get_block(it->second);
        if(!block.has_hash || block.hash != seed) // 兜底校验：块已失效则清理
        {
            hash_to_block_.erase(it);
            break;
        }
        matched.cached_blocks.push_back(it->second);
        matched.num_matched_tokens += blk_size;
    }
    return matched;
}

void BlockManager::release(Request& req)
{
    for(int id : req.block_table)
    {
        allocator_->dec_ref(id);
    }
    req.block_table.clear();
    req.slot_mapping.clear();
    req.sequence_length = 0;
    req.num_computed_tokens = 0;
    req.num_cached_blocks = 0;
}

void BlockManager::record_prefill_stats(int prompt_token_count, int query_blocks, int hit_blocks)
{
    if(prompt_token_count > 0) prompt_tokens_ += static_cast<uint64_t>(prompt_token_count);
    if(query_blocks > 0) prefix_cache_queries_ += static_cast<uint64_t>(query_blocks);
    if(hit_blocks > 0) prefix_cache_hits_ += static_cast<uint64_t>(hit_blocks);
}

void BlockManager::record_cow_copy(int copied_tokens)
{
    ++num_cow_copies_;
    if(copied_tokens > 0) num_cow_tokens_ += static_cast<uint64_t>(copied_tokens);
}