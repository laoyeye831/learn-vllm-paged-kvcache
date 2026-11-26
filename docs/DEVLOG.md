# 开发日志

按天记录的开发过程：当天目标、实际产出、遇到的问题、遗留项。保留原始记录的语气与措辞，只修正错别字并补充了少数事后回看才明白的结论。

日志的价值不在「完成了什么」，而在**当时为什么会走错、后来怎么想通的**。这些判断散落在代码注释里不好找，集中在这里。

| 阶段 | 对应内容 | 状态 |
|---|---|---|
| Day 1–2 | 阶段一/二：prefill / decode 原语、分配器、CoW | ✅ 完成 |
| Day 3–5 | 阶段三/四：前缀缓存复用、抢占与重算 | ✅ 完成 |
| Day 6–7 | M2–M5：调度线程、promise 完成通知、Worker 池 | ✅ 完成 |
| — | swap（CPU 换出） | ❌ 放弃，改用重算 |

---

## Day 1

**目标**：完成阶段一的 `prefill()` / decode 过程，并完善 `BlockAllocator` 中的相关函数。

**产出**：基本完成目标。

**遇到的问题**：

- 在 `allocate()` 里错误地挨个 block 判断 `ref_count`，其实直接维护 `free_list_` 更省事，后续也做了相应修改和完善。
- 一开始对 `Request` 结构体内 `slot_mapping` 的理解有误，采取了**历史累积**的策略；后续改为只计算、存储当前待写入 token 的槽号。

**遗留**：需要进一步完成阶段二，修改 decode 逻辑，实现 CoW。

> 事后补充：`slot_mapping` 的语义现在是「本步要写的物理槽」，所以 `append_slot()` 每次都先 `clear()` 再压入唯一槽号。这个坑在 `docs/ARCHITECTURE.md` 的数据结构一节也标了出来。

---

## Day 2

**目标**：完成阶段二的 COW。

**产出**：基本完成目标，并修缮了一些 bug，超额进行了阶段三的探索。

**遇到的问题**：

- 复用过程中的缓存写入与判断尚未完成；decode 中还没有加入「块满后写入缓存」的功能，有待完善。
- 在 `BlockManager` 具体成员函数的设计上有些混乱，还需要对 `Request` 做成员更新。**以后动手之前最好先做相应规划。**

**遗留**：如上所述。

---

## Day 3

**目标**：完成阶段三的缓存复用。

**产出**：基本完成阶段三的任务，对阶段四进行初步理解和探索。

**遇到的问题**：缓存的计算和复用时机出现了构建错误。

**遗留**：对阶段四的抢占逻辑还需加深理解。

---

## Day 4

**目标**：完成阶段四抢占逻辑的初步实现。

**产出**：基本完成了抢占逻辑四小阶段的实现。

**遇到的问题**：

- 小阶段三需要判断「该不该让新请求进来」。在所需块不够时，必须**直接 `break` 而不是 `continue`**。原因在于：如果放过前面装不下的大请求去塞后面能装下的小请求，那个大请求极有可能被一直卡住、永远得不到运行。
- 小阶段四需要执行 `append_token`，要考虑达到 `max_new_tokens` 后退出的问题。

**遗留**：小阶段四还没完善；在一些共享变量上与 AI 给出的方案存在出入；以及所谓的多线程功能是否真的有必要实现。

> 事后补充：这两条都成了代码里的关键设计。`break` 的理由原样写进了 `step()` 阶段三的注释；多线程最终证明是值得的（见 Day 7），因为「批次内并行采样」才让 Worker 池有真实的用武之地。

---

## Day 5

**目标**：完成小阶段四，并构思 GPU 与 CPU 的 SWAP 逻辑。

**产出**：基本完成了全部抢占逻辑以及相关测试，并补充了压力测试，后续可以上高压。

**遇到的问题**：存在一些细小的瑕疵，以及一些边界问题。

**遗留**：swap 逻辑尚未实现；实现后再考虑多线程。

---

## Day 6

**目标**：完成多线程的开发配置，swap 功能可以去除。

**产出**：初步完成了 `run()` 以及相应锁结构的设计。

**遇到的问题**：需要用 promise 以及相关原子变量来判断完成情况，而不是去读队列内容。

**遗留**：

- `run()` 的析构还存在问题：`pending_` 并未被排空，造成线程卡死。
- 还得给 `Scheduler` 自己的线程，不能把生命周期放在用户手里。
- `set_value` 最好在锁外执行，通过向量统一处理。

> 事后补充：这三点分别对应现在的 `abort_all_pending()`、`Scheduler::start()`/`stop()` 内部持有线程、以及 `step()` 里「先收集 `finished`，出锁后统一 `set_value()`」。原因是被唤醒的调用方可能立刻回头调 `add_request()`，在锁内兑现会自己堵自己。

---

## Day 7

**目标**：完成 M2 阶段多线程的主体部分，保证 `run()` 的启动和析构。

**产出**：基本完成 M2、M3、M4、M5，多线程并行圆满完成。

**遇到的问题**：

- Worker 的任务分发与相关变量传输必须通过共享对象实现，为此增加了不少成员对象。
- 主线程和 Worker 线程池的启停顺序也值得思考。

**遗留**：目前给 Worker 线程们实现的还是**静态分配**。一旦某个线程被分到耗时很长的任务，其他线程都会空等，严重浪费计算资源；或者某批次任务量不足时大量 Worker 闲置，同样是浪费。需要改为**动态分发**，采用共享的原子游标。

> 这条已经写进 README 的 Roadmap（「Worker 池改为动态分发」），是下一步最值得做的改动。

---

## Day 8（v0.2.0 → v0.3.0）

**目标**：先完成 v0.2.0 工程清理（运行期配置、动态 Worker 游标、零拷贝 $O(1)$ 增量哈希、`running_mutex_` 锁补齐），再落地 v0.3.0（$O(1)$ 侵入式空闲队列、三态 LRU 淘汰、主动哈希清理与可复现 benchmark）。

**关键设计决策与踩坑记录**：

- **为什么 `blocks_needed_for_prefill` 必须只扣 `ref_count > 0` 的命中块**：初看似乎所有命中块都不消耗空间，但 `ref_count == 0` 的 `CACHED` 空闲块位于空闲链表中，复活时会 `unlink` 并使 `num_free_blocks()` 减 1。如果把它算作 0 成本，准入检查会误判可用空间，导致击穿 watermark 甚至 `prefill` OOM。
- **消除阶段三的双重 `match_prefix`**：阶段三预检与 `prefill()` 原本各算一次前缀哈希，不仅浪费计算还会让指标翻倍；改为阶段三只算一次并通过 `precomputed_match` 传入 `prefill()`。
- **淘汰清理哈希表的 `it->second == block_id` 守卫**：同一段内容若曾被两个物理块登记，淘汰旧块时不能无条件 `erase(hash)`，必须确认当前映射确实指向被淘汰的块。
- **`last_accessed` 的打点时机**：为了保证 LRU 空闲链表中 `CACHED` 后缀的 `last_accessed` 单调不减，必须在块归还为空闲（`push_free_locked`）时刷新时间戳，而不仅仅在分配/复活时。
- **TTFT / TPOT 用步数而非墙钟**：同 seed 下步数指标完全确定，不会被 OS 线程调度噪声淹没，适合做 FIFO vs LRU 的严格 A/B 对照。

---

## Day 9（v0.4.0）

**目标**：移除死枚举 `SWAPPED`、解耦 `schedule()` / `execute()`（`SchedulerOutput`）、落地 Chunked Prefill + Token Budget，并用 benchmark 量化改造前后的 TTFT / ITL p50 / ITL p99 / 吞吐量。

**关键设计决策与踩坑记录**：

- **移除 `SWAPPED`**：vLLM v1 已经在生产架构中移除 CPU Swap，保留死枚举只会增加状态机心智负担；聚焦 Recompute + Chunked Prefill 更贴近现代推理引擎。
- **抢占前置条件修复**：旧版 `preempt()` 断言 `all_tokens.size() == sequence_length`，上线 chunked prefill 后“prefill 到一半”的请求成为常态，该断言必然崩溃。改为断言 `num_computed_tokens <= all_tokens.size()`，抢占时释放物理块并清零 `num_computed_tokens`，已写满登记的块留在 `CACHED` 链表中，恢复时自动通过 `match_prefix()` 续算。
- **防止 `waiting` 队列饥饿**：遵循 vLLM 的 running-first 原则，同时把 `max_num_seqs` 与 `max_num_batched_tokens` 分开控制，并增加可选的 `waiting_token_reserve`（钳制在 `max_num_batched_tokens - 1` 以内，保证 running 队列每步至少推进 1 token 不会死锁）。
- **ITL 模拟基准**：利用 `Scheduler::last_step_scheduled_tokens()` 驱动确定性 step-cost 时钟（`5ms + 0.25ms * tokens`），在 `mixed` 长短混合负载下验证：`budget=128` 时 ITL p99 从 `139.5ms` 降至 `37.0ms`（**-73.5%**），TTFT 仅增 `1.7ms`，吞吐保持在 `99.3%`。

---

## Day 10（v0.4.1）

**目标**：修复 v0.4.0 引入的 TSan 报错（`lock-order-inversion`），让 `Linux / ThreadSanitizer` 任务重新变绿。

**产出**：`Scheduler` 的四把队列锁合并为单把 `state_mutex_`，CI 全绿。

**遇到的坑**：

- **报错本身很反常**：TSan 指认的两条环边都落在 `schedule_locked()` 阶段一的同一处代码（`sched_mutex_` → `pending_mutex_` → `waiting_mutex_`），而它们明明是同一个方向获取的，字面上不可能成环。
- **本地 GCC 11 复现不出来**，GCC 15 才报。换 clang（zig 自带 LLVM 18）后一次复现，并打印互斥量地址才确认：报错环是 `sched_mutex_`（`0x…d2b8`）与 `waiting_mutex_`（`0x…d308`），但 TSan 记录的“先前获取点”指向 `schedule()` 里的 `step_mutex_`——同一个调用点被算成了不同互斥量。

**根因**：`std::mutex` 在 libstdc++ / libc++ 里是「静态初始化 + 平凡析构」，不会调用 `pthread_mutex_init/destroy`。TSan 只能按地址记账，于是**不同测试里栈上先后出现的不同 `Scheduler` 实例**，只要某个成员互斥量恰好落在同一地址，就会被当成同一把锁，把两个各自合法的边拼成一个不存在的环。细粒度锁越多，这种地址别名命中率越高。

**决策**：不靠「全局锁顺序」的纪律，直接把队列锁从 4 把收敛成 1 把（`state_mutex_`）。合并后 `step_mutex_ → state_mutex_ → alloc_mutex_` 是唯一路径，不存在反向边；即使 TSan 把不同实例的锁按地址混为一谈，`state_mutex_` 撞上的也只是另一把 `state_mutex_`，自环不会报警。顺带把 `preempt()` / `add_request()` / `abort_all_pending()` 的嵌套解锁全部摊平成一次加锁，读起来也更直白。

**验证**：Windows/MSVC 与 WSL GCC 全量 7 个用例通过；WSL clang + TSan 重编 6 个测试二进制，`lock-order-inversion` 告警数为 0（改造前 `scheduler_test` 稳定复现并 `rc=66` 退出）。

