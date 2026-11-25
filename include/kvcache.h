#pragma once
#include <vector>
#include <unordered_map>
#include <list>
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
};

// 状态枚举
enum class ReqState : int
{
    WAITING = 0, // 等待/被抢占
    RUNNING,     // 块在GPU
    SWAPPED,     // 块在CPU
    FINISHED,    // 请求已完成
    ABORTED      // 请求被中止
};

// 物理块内容信息
struct PhysicalBlock
{
    int block_id;
    int ref_count{0};          // 引用计数，表示当前 block 被多少个 KV 使用
    uint64_t last_accessed{0}; // 用于 LRU 淘汰/换出
    uint64_t hash{0};          // 前缀缓存中该块对应的链式哈希
    bool has_hash{false};      // 该块当前是否登记在前缀缓存中（重新分配时清除）

    bool is_full(int num_tokens, int block_size = DEFAULT_BLOCK_SIZE) const
    {
        return num_tokens % block_size == 0;
    }
};

// 抽象出的单设备内存块分配器，负责物理块的分配、释放、引用计数管理等
class BlockAllocator
{
public:
    explicit BlockAllocator(int num_blocks, int block_size = DEFAULT_BLOCK_SIZE);

    int allocate();
    void free(int block_id);
    void inc_ref(int block_id);
    void dec_ref(int block_id);

    int num_free_blocks() const;
    int num_blocks() const;
    int block_size() const;
    PhysicalBlock& get_block(int block_id);

    // 只读检查接口，供测试 / 断言使用；不修改任何状态
    int ref_count_of(int block_id) const;

private:
    int block_size_{DEFAULT_BLOCK_SIZE};
    std::vector<PhysicalBlock> blocks_;
    std::list<int> free_list_; // 采用双向链表便于 LRU 扩展
    mutable std::mutex alloc_mutex_;
};


struct Request
{
    std::vector<int> block_table;   // 逻辑块到物理块的映射表
    int sequence_length{0};         // 请求的序列长度
    std::vector<int> slot_mapping;  // 逻辑槽位到物理块的映射表
    std::vector<int> all_tokens;    // 存储tokens
    int num_cached_blocks{0};       // block_table 开头已命中/已注册进前缀缓存的块数
    ReqState s = ReqState::WAITING; // 请求状态
    int arrival_id{0};              // 请求到达顺序编号
    int max_new_tokens{0};          // 该请求最多可生成的 token 数量（用于调度策略）
    int num_generated_tokens{0};    // 该请求已生成的 token 数量（用于调度策略）
};

// LLMtoken生成模拟器
int fake_sample(const Request& r);


// 抽象出单设备内存块管理器，负责 KV 缓存的注册、匹配、释放等操作
class BlockManager
{
private:
    BlockAllocator* allocator_;
    BlockManagerConfig config_;
    std::unordered_map<uint64_t, int> hash_to_block_;
    std::atomic<long long> num_cow_copies_{0}; // 累计写时复制次数
    //+LRU 相关数据结构

public:
    explicit BlockManager(BlockAllocator* allocator, BlockManagerConfig config = {});
    // 新请求进来时调用（只查询，不修改引用计数）
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

    BlockAllocator* get_allocator();
    int block_size() const;
    bool prefix_caching_enabled() const { return config_.enable_prefix_caching; }
    long long num_cow_copies() const { return num_cow_copies_.load(); }
    void record_cow_copy() { num_cow_copies_.fetch_add(1, std::memory_order_relaxed); }
};

// 抽象出单设备调度策略，负责请求的调度、抢占、换出等操作
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

    // 抢占一个请求
    void preempt();

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

bool prefill(const std::vector<int>& prompt_tokens, Request& request, BlockManager& manager);

int append_slot(BlockManager& manager, Request& request);

void append_token(BlockManager& manager, Request& request, int token);

Request fork_sequence(Request& father_request, BlockAllocator& allocator);