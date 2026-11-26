# 与 vLLM 的对应关系

本项目是 vLLM 内存管理与调度层的 C++ 复现。下表逐条列出「本项目的符号 ↔ vLLM 中的位置 ↔ 差异」。

**参考版本**：早期 v0.1.0 对照 vLLM **v0 架构**（`vllm/core/`），从 v0.3.0 / v0.4.0 起已全面向 **vLLM v1 架构**（`vllm/v1/`）靠拢：包括 `FreeKVCacheBlockQueue` 式的 $O(1)$ 侵入式 LRU 空闲队列、移除 `SWAPPED`、将调度拆分为 `schedule() -> SchedulerOutput` 与 `execute()`、以及 **Chunked Prefill + Token Budget** 混排调度。表中同时保留 v0/v1 对应位置供对照参考。

> 本项目不含 `PagedAttention` 的 CUDA kernel，也不含模型前向。对照的是「块管理 + 调度」这一层。

## 1. 数据结构

| 本项目 | vLLM (v0) | vLLM (v1) | 差异 |
|---|---|---|---|
| `PhysicalBlock` | `vllm/core/block/interfaces.py: PhysicalTokenBlock` | `vllm/v1/core/block_pool.py: KVCacheBlock` | 只保留元信息（ref/hash/时间戳），没有真实 KV tensor 与 block_id 的指针 |
| `Request` | `vllm/sequence.py: Sequence` / `SequenceGroup` | `vllm/v1/request.py: Request` | 去掉了 sampling params、LoRA、多序列组等，只留 block_table / slot_mapping / 生成长度 |
| `ReqState` | `vllm/sequence.py: SequenceStatus` | `vllm/v1/request.py: RequestStatus` | v0.4.0 已移除 `SWAPPED`，与 vLLM v1 一致仅保留 Recompute 路径 |
| `BlockAllocator` | `vllm/core/block/common.py: BlockAllocator`<br/>`vllm/core/block/naive_block.py: NaiveBlockAllocator` | `vllm/v1/core/block_pool.py: BlockPool` | 只做单设备、单池；没有 `device=CPU/GPU` 双池，也没有 `cached_block_hash_to_block` |
| `free_head_` / `free_tail_` + 侵入式链表 | `NaiveBlockAllocator._free_blocks`（`deque`） | `BlockPool.free_block_queue`（`FreeKVCacheBlockQueue` 侵入式双向链表 + LRU） | v0.3.0 已升级为 $O(1)$ 侵入式双向链表，支持 `FreePolicy::LRU`（CLEAN 前缀 + CACHED 后缀）与 `FreePolicy::FIFO` 对照 |
| `ref_count` | `PhysicalTokenBlock.ref_count` | `KVCacheBlock.ref_cnt` | 语义一致（含 `<=1` 表示不会被共享） |
| `block_size`（默认 16） | `block_size` 参数，默认 16 | 同 | v0.2.0 起运行期可配置 |
| `slot_mapping` | `vllm/worker/model_runner.py: _compute_slot_mapping`<br/>`vllm/attention/backends/flash_attn.py` | `vllm/v1/worker/block_table.py` + attention metadata | 语义一致：本步 token 在展平 KV cache 中的物理槽号；**每步重建，不是历史累积** |
| `block_table` | `vllm/worker/model_runner.py` 的 block tables | `vllm/v1/worker/block_table.py: BlockTable` | 本项目是 `std::vector<int>` 逻辑→物理映射 |
| `SchedulerConfig::watermark_*` | `Scheduler` 的 `watermark`（默认 **0.01**，按块数比例） | 同 | v0.2.0 起同时支持绝对块数与比例模式 |

## 2. 块管理（`src/manager.cpp`）

| 本项目 | vLLM (v0) | vLLM (v1) | 差异 |
|---|---|---|---|
| `BlockManager` | `vllm/core/block_manager_v1.py: BlockSpaceManagerV1` | `vllm/v1/core/kv_cache_manager.py: KVCacheManager` | 只保留块侧逻辑，**不含** `can_allocate` / `allocate` 的门控判断（那部分在本项目里被拆进 `Scheduler`） |
| `hash_to_block_` | `prefix_caching_block.py: PrefixCachingBlockAllocator.hash_to_block_id` | `BlockPool.cached_block_hash_to_block` | 键都是「块级链式哈希」，值是物理块号 |
| `compute_hash`（多项式滚动哈希） | `prefix_caching_block.py: hash_block_tokens` / `hash_request_tokens` | `vllm/v1/core/kv_cache_utils.py: get_block_hash` | vLLM 用 `hash((parent_hash, tuple(token_ids)))`（Python 内置哈希 + `NONE_HASH`）；本项目用 `h = h*BASE + token` 的 64 位多项式。**两者都是「父哈希 + 本块 token」的链式结构** |
| `register_block(req, begin)` | `PrefixCachingBlockAllocator.allocate_immutable_block` 内部登记哈希 | `KVCacheManager.cache_full_blocks` | 本项目的 `begin` 参数用于「跳过已缓存的块，只推进哈希链」；vLLM 通过 `num_cached_blocks` 区分 |
| `match_prefix()` | `PrefixCachingBlockAllocator.get_num_common_prefix_blocks` / `find_longest_cache_hit` | `KVCacheManager.get_computed_blocks` | 本项目**同时**返回块列表与命中 token 数，且命中后要复核 `has_hash && hash == seed` |
| `evict_hash_entry()` + `allocate_ex()` | vLLM 依赖「块被回收/淘汰时从哈希表移除」保证一致性 | v1 `BlockPool._maybe_evict_cached_block` | v0.3.0 通过 `friend class BlockManager` 将「淘汰 CACHED 块 + 清理 `hash_to_block_`」绑定为原子路径，带 `it->second == block_id` 守卫，保证哈希表有界 |
| `BlockManager` / `BlockAllocator` 统计指标 | `PrefixCacheStats` / `SchedulerStats` | `vllm/v1/metrics/stats.py: PrefixCacheStats` | 提供 `prefix_cache_queries`、`prefix_cache_hits`、`num_evictions`、`num_revivals`、`num_cow_copies` 与步数级 TTFT/TPOT |
| `release(req)` | `BlockSpaceManagerV1.free` + `PrefixCachingBlockAllocator.free` | `KVCacheManager.free` | 语义一致：逐个 `dec_ref`，并清空 `block_table` / `slot_mapping` |

## 3. 请求原语（`src/other.cpp`）

| 本项目 | vLLM (v0) | 差异 |
|---|---|---|
| `prefill(prompt, req)` | `BlockSpaceManagerV1.allocate` + `append_slots`（prefill 阶段） | 本项目把「前缀命中 → 引用复用块 → 分配剩余块 → 生成 slot_mapping → 注册满块」串成一个函数，**失败时整体回滚**（`release` 已拿到的引用），vLLM 由 Python 异常 + 上层重试处理 |
| `append_slot(req)` | `BlockSpaceManagerV1.append_slots` / `_can_append_token_to_parent` | vLLM 把「取最后一块 → 判断能否原地写」封装在 `BlockList.append_token_ids` 里；本项目展开成 `offset == 0` / `ref_count == 1` / 否则 CoW 三个分支 |
| CoW 分支 | `PrefixCachingBlockAllocator.allocate_mutable_block` + `BlockList` 的 COW 逻辑 | **本项目不搬运 KV 数据**（只打印 `copy valid tokens to new block`），vLLM 会调用 `copy_blocks()` kernel 真正拷贝 |
| `append_token(req, token)` | `Scheduler._append_slots` 之后的 `seq.append_token_id` + 满块 cache | 块写满（`seq_len % BLOCK_SIZE == 0`）时注册前缀缓存，对应 vLLM 在 append 阶段的 `cache_blocks` |
| `fork_sequence()` | `BlockSpaceManagerV1.fork` / `PrefixCachingBlockAllocator` 的 `_incr_refcount` | 用于 beam search / parallel sampling 共享前缀；本项目仅用于演示与测试共享语义 |
| `fake_sample()` | `vllm/model_executor/layers/sampler.py` | 本项目是确定性哈希 `(last_token*31 + seq_len) % 50000`，**目的是让「小池 vs 大池」的结果可以直接逐个对拍**，与真实采样无关 |

## 4. 调度（`src/scheduler.cpp`）

| 本项目 | vLLM (v0) | 差异 |
|---|---|---|
| `Scheduler::step()` 阶段一 | `Scheduler._schedule_default` 中的 `_schedule_prefills` 前半 + 准入判断 | 本项目用 `ceil((prompt+max_new)/BLOCK_SIZE) > num_blocks - WATER_MARK` 一次判死；vLLM 用 `_get_num_new_uncached_and_cached_tokens` + `block_manager.can_allocate`，并支持 **chunked prefill**（本项目不支持分块 prefill） |
| `step()` 阶段二（抢占） | `Scheduler._schedule_running` | 一致：遍历 running 队列，`can_append_slots` 失败就抢占 |
| `preempt()` | `Scheduler._preempt` | 一致：抢占**最后到达**的请求（`policy = RECOMPUTE`），释放块、放回等待队列队首。Recovery 顺序（`_get_swapped_and_blocked_queue` / `waiting` 的 TTL 排序）在本项目简化为「队首优先」 |
| `step()` 阶段三（准入） | `Scheduler._schedule_prefills` | 关键行为一致：**空间不足时 `break`**，不跳过队首去塞后面的小请求（vLLM 源码注释也是这个理由：避免大请求饥饿） |
| 「没抢占才准入新请求」 | vLLM 用 `preempted` 标志位做同样的短路 | 本项目显式记录 `bool preempted` |
| `Scheduler::schedule()` / `execute()` + `SchedulerOutput` | `Scheduler._schedule_*` → `SchedulerOutputs` → `execute_model` | `vllm/v1/core/sched/scheduler.py: Scheduler.schedule()` → `SchedulerOutput` | v0.4.0 已解耦决策与执行，并支持 `max_num_batched_tokens`、`max_num_seqs`、`enable_chunked_prefill` 与 `waiting_token_reserve` |
| `num_preemptions_` / `num_steps_` | vLLM 有 metrics（`vllm/engine/metrics.py`） | 只保留两个计数器，无 Prometheus 暴露 |
| `abort_all_pending()` | `Scheduler.abort_seq_group` + 引擎停机流程 | 本项目在停机时统一中止，并保证所有 `future` 就绪 |

## 5. 本项目独有、vLLM 无直接对应的部分

| 本项目 | 说明 |
|---|---|
| **Worker 采样池**（`worker_loop` / `sample_batch_parallel` / `round_` 翻牌协议） | vLLM 的 worker 指张量并行 / 流水并行的**模型副本**，由 `ExecutorBase` 管理，语义完全不同。本项目的池只是为了「在不写 GPU 代码的前提下演示批次内并行采样与线程同步」，属于自创的教学装置 |
| **`std::future<void>` 完成通知** | vLLM 是异步引擎（`AsyncLLMEngine` + asyncio），本项目用 `promise/future` + `pending_` 表做等价的「请求级完成回调」 |
| **`check_invariants()` / 串行-并行对拍** | vLLM 靠单测 + 端到端评测；本项目在每一步做引用计数不变量检查，并用同一负载在大小池下对拍输出 |
| **`set_step_hook()`** | 调试钩子，vLLM 无对应 |

## 6. 刻意未实现

| vLLM 特性 | 本项目状态 |
|---|---|
| CPU ↔ GPU swap（`_swap_out` / `_swap_in`） | ❌ 已移除（vLLM v1 同样移除了 CPU Swap，仅保留 Recompute） |
| Chunked prefill + Token Budget | ✅ v0.4.0 已实现 `prefill_chunk()`、running-first 预算分配、`max_num_seqs` 与 `waiting_token_reserve` |
| LRU 空闲块淘汰 | ✅ v0.3.0 已实现 $O(1)$ 侵入式链表 LRU（CLEAN 优先 + CACHED LRU）及 FIFO 对照 |
| 真实 PagedAttention kernel / `copy_blocks` | ❌ KV 内容用整数模拟 |
| 多设备 / TP / PP | ❌ 单设备单池 |
| Beam search / parallel sampling 的真实用法 | ⚠️ 只实现了 `fork_sequence` 的引用计数语义，没有调度器层的分支管理 |
| Prefix caching 的 LRU 驱逐与命中率统计 | ✅ v0.3.0 已实现主动驱逐、指标统计与 `bench/kvcache_bench.cpp` 对照基准 |
| CUDA Graph / 多进程 worker | ❌ |

## 7. 想深入读 vLLM 时建议的阅读顺序

1. `vllm/core/block/common.py` → 看 `BlockAllocator` / `BlockTable` / `BlockList` 的三层抽象（本项目把它们压成了一个 `BlockAllocator` + `Request.block_table`）
2. `vllm/core/block/prefix_caching_block.py` → 对比 `hash_block_tokens` 与本项目的 `compute_hash`
3. `vllm/core/block_manager_v1.py` → `can_allocate` / `allocate` / `append_slots` / `fork` / `free`
4. `vllm/core/scheduler.py` → `_schedule_default` / `_schedule_prefills` / `_schedule_running` / `_preempt`
5. `vllm/worker/model_runner.py` → `_compute_slot_mapping`，理解 `slot_mapping` 到底喂给谁
6. （v1）`vllm/v1/core/kv_cache_manager.py` + `vllm/v1/core/sched/scheduler.py` → 看重构后 `KVCacheManager` 与 `RequestStatus` 的演进
