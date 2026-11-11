#include "kvcache.h"
#include <algorithm>
#include <memory>
#include <string>
#include <cassert>
#include <functional>
#include <stdexcept>

constexpr uint64_t PREFIX_HASH_SEED = 123456789ULL;

// 多项式滚动哈希，seed作为初始值
static uint64_t compute_hash(uint64_t seed, const std::vector<int>& tokens)
{
    constexpr uint64_t BASE = 911382629ULL;
    uint64_t h = seed;
    for(int t : tokens)
    {
        h = h * BASE + static_cast<uint64_t>(t);
    }
    return h;
}

// vector切片函数，区间为 [start, end)
static std::vector<int> slice_vec(const std::vector<int>& src, size_t start, size_t end)
{
    if(end > src.size()) end = src.size(); // 防止越界，类似python自动截断
    if(start > end) start = end;
    return std::vector<int>(src.begin() + start, src.begin() + end);
}


// ====== 管理器成员实现 ======
BlockManager::BlockManager(BlockAllocator* allocator) : allocator_(allocator) {}

BlockAllocator* BlockManager::get_allocator()
{
    return allocator_;
}

void BlockManager::register_block(Request& request, int begin)
{
    const auto& tokens = request.all_tokens;
    int valid_tokens = std::min<int>(request.sequence_length, static_cast<int>(tokens.size()));
    int full_block_nums = std::min<int>(valid_tokens / BLOCK_SIZE, static_cast<int>(request.block_table.size()));
    uint64_t seed = PREFIX_HASH_SEED;
    for(int i = 0; i < full_block_nums; i++)
    {
        size_t start = static_cast<size_t>(i) * BLOCK_SIZE;
        seed = compute_hash(seed, slice_vec(tokens, start, start + BLOCK_SIZE)); // 链式哈希，依赖整个前缀
        if(i < begin) continue;                                                  // 前面已缓存的块只需推进哈希链

        int physical_block_id = request.block_table[i];
        hash_to_block_[seed] = physical_block_id;
        auto& block = allocator_->get_block(physical_block_id);
        block.hash = seed;
        block.has_hash = true;
    }
    request.num_cached_blocks = std::max(request.num_cached_blocks, full_block_nums);
}

BlockManager::PrefixMatch BlockManager::match_prefix(const std::vector<int>& prompt_tokens)
{
    size_t full_block_nums = prompt_tokens.size() / BLOCK_SIZE;
    uint64_t seed = PREFIX_HASH_SEED;
    PrefixMatch matched;
    for(size_t i = 0; i < full_block_nums; i++)
    {
        size_t start = i * BLOCK_SIZE;
        seed = compute_hash(seed, slice_vec(prompt_tokens, start, start + BLOCK_SIZE));
        auto it = hash_to_block_.find(seed);
        if(it == hash_to_block_.end()) break;

        const auto& block = allocator_->get_block(it->second);
        if(!block.has_hash || block.hash != seed) // 块已被重新分配，缓存项失效
        {
            hash_to_block_.erase(it); // 移除失效的缓存信息
            break;
        }
        matched.cached_blocks.push_back(it->second);
        matched.num_matched_tokens += BLOCK_SIZE;
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
    req.num_cached_blocks = 0;
}