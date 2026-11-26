#pragma once
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <thread>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <future>

constexpr int DEFAULT_BLOCK_SIZE = 16; // 默认每个 block 存放 16 个 token 的 KV
constexpr int DEFAULT_WATER_MARK = 4;  // 默认安全水位（块数）

// 兼容旧代码默认常量
constexpr int BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
constexpr int WATER_MARK = DEFAULT_WATER_MARK;

// BlockManager 运行期配置
struct BlockManagerConfig
{
    bool enable_prefix_caching{true}; // 是否启用前缀缓存匹配与注册
};

// Scheduler 运行期配置
struct SchedulerConfig
{
    int num_workers{1};                       // Worker 采样线程数
    int watermark_blocks{DEFAULT_WATER_MARK}; // 安全水位（绝对块数）；watermark_ratio >= 0 时忽略
    double watermark_ratio{-1.0};             // 安全水位比例（如 0.01）；< 0 表示使用 watermark_blocks
    int max_num_batched_tokens{0};            // 单步 token 预算（<= 0 表示不限制）
    int max_num_seqs{0};                      // 同时运行请求数上限（<= 0 表示不限制）
    bool enable_chunked_prefill{true};        // 是否启用分块预填充（仅在 max_num_batched_tokens > 0 时生效）
    int waiting_token_reserve{0};             // 可选：为 waiting 队列预留的最小 token 预算，防止饥饿
};

// 状态枚举（已移除 SWAPPED，遵循 vLLM v1 仅保留 Recompute 路径）
enum class ReqState : int
{
    WAITING = 0, // 等待/被抢占
    RUNNING,     // 块在显存中
    FINISHED,    // 请求已完成
    ABORTED      // 请求被中止
};

// 空闲块层级：在用为 NONE，空闲按是否带有前缀缓存分为 CLEAN 与 CACHED
enum class FreeTier : uint8_t
{
    NONE = 0,
    CLEAN,
    CACHED
};

// 空闲链表淘汰策略
enum class FreePolicy
{
    FIFO = 0, // 所有空闲块一律 push_back（旧版行为，用于对照实验）
    LRU       // CLEAN 块 push_front 优先分配，CACHED 块 push_back 按释放时间 LRU 淘汰
};

// 物理块内容信息
struct PhysicalBlock
{
    int block_id;
    int ref_count{0};          // 引用计数，表示当前 block 被多少个 KV 使用
    uint64_t last_accessed{0}; // 分配/复活/归还为缓存块时的逻辑时钟戳
    uint64_t hash{0};          // 前缀缓存中该块对应的链式哈希
    bool has_hash{false};      // 该块当前是否登记在前缀缓存中（重新分配时清除）

    // 侵入式双向链表指针（替代 std::list 节点，-1 表示无）
    int free_prev{-1};
    int free_next{-1};
    FreeTier tier{FreeTier::NONE}; // 维护空闲层级计数与不变量检查

    bool is_full(int num_tokens, int block_size = DEFAULT_BLOCK_SIZE) const
    {
        return num_tokens % block_size == 0;
    }
};

class BlockManager;

// 抽象出的单设备内存块分配器，负责物理块的分配、释放、引用计数管理与 O(1) 空闲队列
class BlockAllocator
{
public:
    explicit BlockAllocator(int num_blocks, int block_size = DEFAULT_BLOCK_SIZE,
                            FreePolicy policy = FreePolicy::LRU);

    void inc_ref(int block_id); // O(1) 摘链复活
    void dec_ref(int block_id); // 归零时按 has_hash 路由到 CLEAN / CACHED

    int num_free_blocks() const; // = num_clean_free_ + num_cached_free_
    int num_free_clean() const;
    int num_free_cached() const;
    int num_blocks() const;
    int block_size() const;
    FreePolicy policy() const { return policy_; }
    PhysicalBlock& get_block(int block_id);
    const PhysicalBlock& get_block(int block_id) const;

    // 只读检查接口，供测试 / 断言使用；不修改任何状态
    int ref_count_of(int block_id) const;
    int free_head() const;
    int free_tail() const;

    // 分配器统计指标（锁内维护）
    uint64_t num_allocations() const;
    uint64_t num_evictions() const; // 分配到了 CACHED 块（= 淘汰一条缓存）
    uint64_t num_revivals() const;  // inc_ref 把空闲缓存块复活

private:
    friend class BlockManager; // 仅允许 BlockManager 走“带淘汰语义”的分配，保证哈希表同步清理

    struct AllocOutcome
    {
        int block_id{-1};
        bool evicted{false};
        uint64_t evicted_hash{0};
    };

    AllocOutcome allocate_ex();

    void link_front_locked(int id);
    void link_back_locked(int id);
    void unlink_locked(int id);
    void push_free_locked(int id);

    int block_size_{DEFAULT_BLOCK_SIZE};
    FreePolicy policy_{FreePolicy::LRU};
    std::vector<PhysicalBlock> blocks_;
    int free_head_{-1};
    int free_tail_{-1};
    int num_clean_free_{0};
    int num_cached_free_{0};
    uint64_t access_clock_{0};
    uint64_t num_allocations_{0};
    uint64_t num_evictions_{0};
    uint64_t num_revivals_{0};
    mutable std::mutex alloc_mutex_;
};


struct Request
{
    std::vector<int> block_table;   // 逻辑块到物理块的映射表
    int sequence_length{0};         // 请求已落盘 KV 的序列长度（与 num_computed_tokens 同步）
    int num_computed_tokens{0};     // 已完成 prefill/decode KV 计算的 token 数（支持分块预填充续算）
    std::vector<int> slot_mapping;  // 逻辑槽位到物理块的映射表
    std::vector<int> all_tokens;    // 存储tokens（完整 prompt + 已生成的 token）
    int num_cached_blocks{0};       // block_table 开头已命中/已注册进前缀缓存的块数
    ReqState s = ReqState::WAITING; // 请求状态
    int arrival_id{0};              // 请求到达顺序编号
    int max_new_tokens{0};          // 该请求最多可生成的 token 数量（用于调度策略）
    int num_generated_tokens{0};    // 该请求已生成的 token 数量（用于调度策略）
    int admit_step{-1};             // 首次被准入的步号（抢占重入不覆盖）
    int first_token_step{-1};       // 产出第一个 token 的步号
    int finish_step{-1};            // 完成步号

    bool is_prefill_done() const
    {
        return num_computed_tokens >= static_cast<int>(all_tokens.size());
    }
};

// LLMtoken生成模拟器
int fake_sample(const Request& r);


// 抽象出单设备内存块管理器，负责 KV 缓存的分配、注册、匹配、淘汰清理与释放
class BlockManager
{
private:
    BlockAllocator* allocator_;
    BlockManagerConfig config_;
    std::unordered_map<uint64_t, int> hash_to_block_;

    // 以下指标由调度线程更新，供 step_hook 或 stop() 后读取
    uint64_t prefix_cache_queries_{0}; // 实际 prefill 查询的满块数
    uint64_t prefix_cache_hits_{0};    // 实际 prefill 命中的满块数
    uint64_t prompt_tokens_{0};        // 实际 prefill 的 token 数（含抢占重算）
    uint64_t num_registrations_{0};    // 新登记进哈希表的块数
    uint64_t num_cow_copies_{0};       // 累计写时复制次数
    uint64_t num_cow_tokens_{0};       // 累计写时复制搬运的 token 数

    void evict_hash_entry(int block_id, uint64_t hash);

public:
    explicit BlockManager(BlockAllocator* allocator, BlockManagerConfig config = {});

    // 物理块分配唯一出口：若触发淘汰，同步清理 hash_to_block_ 中对应条目
    int allocate_block();

    // 新请求进来时调用（只查询，不修改引用计数，也不计入 prefill 指标）
    // 返回：命中缓存的物理块列表 + 命中的 token 数
    struct PrefixMatch
    {
        std::vector<int> cached_blocks; // 命中的物理块（调用方需 inc_ref）
        int num_matched_tokens{0};      // 命中的 token 数
    };
    PrefixMatch match_prefix(const std::vector<int>& prompt_tokens);

    // 把 request 中从第 begin 个块起、已写满的块注册进缓存（优先复用前一块哈希做到 O(1) 增量计算）
    void register_block(Request& request, int begin);

    // 释放请求
    void release(Request& req);

    // 记录一次真实 prefill 的前缀缓存查询/命中与 token 统计
    void record_prefill_stats(int prompt_token_count, int query_blocks, int hit_blocks);

    // 记录一次 CoW（含需拷贝的槽位数 offset）
    void record_cow_copy(int copied_tokens = 0);

    BlockAllocator* get_allocator();
    const BlockAllocator* get_allocator() const;
    int block_size() const;
    bool prefix_caching_enabled() const { return config_.enable_prefix_caching; }
    size_t num_hash_entries() const { return hash_to_block_.size(); }

    uint64_t prefix_cache_queries() const { return prefix_cache_queries_; }
    uint64_t prefix_cache_hits() const { return prefix_cache_hits_; }
    uint64_t prompt_tokens() const { return prompt_tokens_; }
    uint64_t num_registrations() const { return num_registrations_; }
    uint64_t num_cow_copies() const { return num_cow_copies_; }
    uint64_t num_cow_tokens() const { return num_cow_tokens_; }
};

// 单步调度决策输出：解耦 Scheduler::schedule()（决策）与 Scheduler::execute()（执行）
struct SchedulerOutput
{
    struct ScheduledPrefill
    {
        std::shared_ptr<Request> req;
        int num_cached_tokens{0};   // 本步前缀缓存复用的 token 数
        int num_computed_tokens{0}; // 本步实际计算的 prefill token 数（含分块）
        bool is_new_admission{false};
    };

    std::vector<ScheduledPrefill> prefill_reqs;        // 本步执行 prefill / chunked prefill 的请求
    std::vector<std::shared_ptr<Request>> decode_reqs; // 本步执行 decode 采样的请求
    std::vector<std::promise<void>> aborted_promises;  // 阶段一准入拒绝的 promise（由 execute 锁外兑现）

    int num_prefill_tokens{0}; // 本步实际计算的 prefill token 总数
    int num_decode_tokens{0};  // 本步 decode token 总数（= decode_reqs.size()）
    int num_preemptions{0};    // 本步发生的抢占次数
    bool preempted{false};

    int total_scheduled_tokens() const { return num_prefill_tokens + num_decode_tokens; }
    bool empty() const { return prefill_reqs.empty() && decode_reqs.empty() && aborted_promises.empty(); }
};

// 抽象出单设备调度策略，负责请求的调度、抢占与两阶段执行（schedule + execute）
class Scheduler
{
private:
    BlockManager& manager_;
    std::deque<std::shared_ptr<Request>> inbox_;         // 收件箱（加锁）
    std::deque<std::shared_ptr<Request>> waiting_queue_; // 等待队列
    std::deque<std::shared_ptr<Request>> running_queue_; // 运行队列
    std::mutex sched_mutex_;                             // 调度器互斥锁（inbox_ + next_arrival_id_）
    std::mutex pending_mutex_;                           // promise操作互斥锁
    mutable std::mutex waiting_mutex_;                   // 等待队列互斥锁
    mutable std::mutex running_mutex_;                   // 运行队列互斥锁
    std::mutex step_mutex_;                              // 保证同一时刻只有一个线程在 step()
    std::condition_variable sched_cv_;                   // 调度器条件变量
    std::atomic<bool> stop_flag_{false};                 // 停止标志
    std::thread sched_thread_;                           // run() 所在的线程
    int next_arrival_id_{0};                             // 下一个请求的到达顺序编号
    std::atomic<int> num_preemptions_{0};                // 累计抢占次数
    std::function<int(const Request&)> sample_func_;     // token 采样器函数，外部设置，内部调用
    std::function<void(const Scheduler&)> step_hook_;    // 每次 step() 结束后在调度线程调用（调试/不变量检查用）
    struct PendingCompletion
    {
        std::promise<void> done;
    };
    std::unordered_map<int, PendingCompletion> pending_; // key = arrival_id
    std::atomic<int> unfinished{0};
    std::atomic<long long> num_steps_{0};                             // 已执行的 step() 次数（调试用）
    std::atomic<int> last_step_prefill_tokens_{0};                    // 最近一步计算的 prefill token 数
    std::atomic<int> last_step_decode_tokens_{0};                     // 最近一步计算的 decode token 数
    std::vector<std::thread> workers_;                                // Worker 线程池（只负责采样）
    std::mutex pool_mutex_;                                           // 保护下面所有池状态
    std::condition_variable task_cv_;                                 // 派活：主线程通知 Worker 开始新一轮
    std::condition_variable done_cv_;                                 // 收工：Worker 通知主线程本轮结束
    uint64_t round_{0};                                               // 轮次号，Worker 用它判断"是否是新的一轮"
    int remaining_{0};                                                // 本轮还剩几个 Worker 没干完
    const std::vector<std::shared_ptr<Request>>* cur_batch_{nullptr}; // 本轮任务
    std::vector<int>* cur_out_{nullptr};                              // 本轮结果写到哪
    size_t cur_batch_size_{0};                                        // 本轮批次大小
    std::atomic<size_t> next_task_idx_{0};                            // 动态分发共享游标
    std::exception_ptr worker_error_;                                 // Worker 里抛出的异常，交回主线程
    SchedulerConfig config_;                                          // 运行期配置
    int num_workers_{1};                                              // worker 线程数
    bool pool_running_{false};                                        // Worker 池是否已启动
    std::atomic<bool> workers_stop_{false};
    std::atomic<bool> started_{false}; // start() 是否已调用（采样器不允许之后再改）

    // 抢占一个请求（调用方需已持有 running_mutex_ -> waiting_mutex_）
    void preempt();

    // 两阶段内部实现（调用方需已持有 step_mutex_）
    SchedulerOutput schedule_locked();
    void execute_locked(SchedulerOutput& output);

    // 立刻中止所有未完成的请求：释放块、标记 ABORTED、唤醒等待者
    void abort_all_pending();

public:
    void set_sampler(std::function<int(const Request&)> sampler);
    int run_sample(const Request& req);

    // 注册 step 结束回调：在调度线程里、Worker 全部收工之后调用
    // 此时没有其他线程在改队列/块表，可以安全地做不变量检查
    // 与采样器一样，必须在 start() 之前设置
    void set_step_hook(std::function<void(const Scheduler&)> hook);

    explicit Scheduler(BlockManager& manager, int num_workers = 1);
    Scheduler(BlockManager& manager, SchedulerConfig config);

    int watermark_blocks() const;
    const SchedulerConfig& config() const { return config_; }

    // 加入新的请求至收件箱
    std::future<void> add_request(std::shared_ptr<Request> req);

    // 是否可以新增槽位
    bool can_append(std::shared_ptr<Request> req);

    // 新请求准入队列判断查询辅助
    int blocks_needed_for_prefill(std::shared_ptr<Request> req);

    ~Scheduler();

    // 调度器主循环，负责调度、抢占、换出等操作
    void run();

    // drain=true：等所有已提交请求跑完再退出；drain=false：丢弃未完成请求
    // 返回前会 join 后台线程，此后 pending_ 一定为空
    void stop(bool drain = true);

    // 两阶段调度接口：schedule() 产出决策，execute() 执行采样与后处理
    SchedulerOutput schedule();
    void execute(SchedulerOutput& output);

    // 单步驱动：等价于在同一 step_mutex_ 下执行 schedule() + execute()
    void step();

    void start(); // 内部持有线程，调用 run()，外部无需调用

    // 只启动 / 停止 Worker 池，不启动调度线程；供单线程 step() 驱动的测试使用
    void start_pool();
    void stop_pool();

    // 收件箱、等待队列、运行队列任一非空即还有未完成的工作
    bool has_work()
    {
        return (unfinished.load() != 0);
    }

    int num_preemptions() const { return num_preemptions_; }
    long long num_steps() const { return num_steps_; }
    int last_step_prefill_tokens() const { return last_step_prefill_tokens_.load(); }
    int last_step_decode_tokens() const { return last_step_decode_tokens_.load(); }
    int last_step_scheduled_tokens() const { return last_step_prefill_tokens_.load() + last_step_decode_tokens_.load(); }

    void worker_loop(int id);
    void sample_batch_parallel(const std::vector<std::shared_ptr<Request>>& batch, std::vector<int>& out);

    // 只读访问器，供测试观察调度状态
    const std::deque<std::shared_ptr<Request>> running() const
    {
        std::lock_guard<std::mutex> lock(running_mutex_);
        return running_queue_;
    }
    const std::deque<std::shared_ptr<Request>> waiting() const
    {
        std::lock_guard<std::mutex> lock(waiting_mutex_);
        return waiting_queue_;
    }
};

bool prefill(const std::vector<int>& prompt_tokens, Request& request, BlockManager& manager,
             const BlockManager::PrefixMatch* precomputed_match = nullptr);

// 分块预填充原语：从 request.num_computed_tokens 处继续推进最多 num_new_tokens 个未缓存 token
// 首次调用（num_computed_tokens == 0）会先复用前缀缓存块，再按块分配并推进 chunk；
// 续算调用（num_computed_tokens > 0）仅追加本次 chunk 所需的新物理块并增量注册满块。
bool prefill_chunk(Request& request, int num_new_tokens, BlockManager& manager,
                   const BlockManager::PrefixMatch* precomputed_match = nullptr,
                   int* out_cached_tokens = nullptr, int* out_computed_tokens = nullptr);

int append_slot(BlockManager& manager, Request& request);

void append_token(BlockManager& manager, Request& request, int token);

Request fork_sequence(Request& father_request, BlockAllocator& allocator);