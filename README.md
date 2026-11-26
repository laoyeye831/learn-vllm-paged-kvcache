**English** | [简体中文](README.zh-CN.md)

# KV_Cache — Reproducing vLLM's Paged KV Cache and Scheduler in C++17

> A study project: reimplementing the core of vLLM from scratch in modern C++ — **paged KV cache, per-block reference counting with copy-on-write, chained prefix-hash reuse, continuous batching, and preemption**.
> No GPU kernels and no real model inference. The focus is a readable, testable implementation of the memory-management and scheduling layer.

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![CMake](https://img.shields.io/badge/CMake-%E2%89%A53.20-informational)
![Tests](https://img.shields.io/badge/tests-61%20unit%20%2B%202%20stress-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
[![CI](https://github.com/laoyeye831/learn-vllm-paged-kvcache/actions/workflows/ci.yml/badge.svg)](https://github.com/laoyeye831/learn-vllm-paged-kvcache/actions/workflows/ci.yml)

> **Note on language:** source comments, the `docs/` guides, and this project's development log are written in **Chinese**. This English README and the Chinese one are kept in sync on facts; the code is the source of truth.

## What this is

vLLM's PagedAttention splits the KV cache into fixed-size physical blocks and keeps a `block_table` mapping logical blocks to physical ones, which eliminates fragmentation and makes prefix sharing cheap. This project rebuilds that **data structure and scheduling policy** in dependency-free C++17, with token ids standing in for real KV tensors.

It is useful for understanding how vLLM manages memory, studying reference-counted resource lifetimes, and practising the design and testing of a multithreaded scheduler.

## Features

| Feature | Description |
|---|---|
| **Paged KV cache** | `BLOCK_SIZE = 16`; `block_table` and `slot_mapping` handle logical-block and slot addressing |
| **Reference-counted allocator** | Free list + `ref_count`, with `allocate` / `inc_ref` / `dec_ref` / `free` |
| **Copy-on-write** | Writing to a shared tail block splits it; an exclusively owned tail block is written in place |
| **Prefix cache reuse** | Chained polynomial hashing; a block is registered as soon as it fills, and lazily invalidated when the block is reallocated |
| **Continuous batching** | A `step()` with four phases: admission → preemption → new-request admission → batched sampling |
| **Preemption with recompute** | When space runs out, the tail of the running queue is preempted, its blocks freed, and the request pushed to the front of the waiting queue |
| **Async engine** | A scheduler thread plus a worker sampling pool; `add_request()` returns `std::future<void>` that fires on completion |
| **Invariant checking** | Every step validates block ref-counts, queue lengths and free-block counts; stress tests cross-check serial vs. parallel runs |

## Architecture

```mermaid
flowchart LR
    subgraph API["Public API"]
        AR["add_request(req)<br/>returns a future"]
    end

    subgraph Scheduler["Scheduler"]
        IN["inbox_"] -->|"1. admission check<br/>too large for memory &rarr; ABORT"| WT["waiting_queue_"]
        WT -->|"3. prefill: match / allocate"| RUN["running_queue_"]
        RUN -->|"2. can_append? no"| PRE["preempt()<br/>free blocks, mark WAITING"]
        PRE -->|"push to front"| WT
    end

    RUN -->|"4. batched sampling"| POOL["Worker pool<br/>sample_batch_parallel()"]
    POOL --> AT["append_token()"]
    AT --> RUN
    AT -->|"block full"| REG["register_block()<br/>chained hash inserted"]

    WT --> PRE2["prefill()"]
    PRE2 --> MG["BlockManager"]
    REG --> MG
    MG --> AL["BlockAllocator<br/>free_list_ / ref_count"]
    AT --> AL
```

Full lifecycle of a single request:

```mermaid
sequenceDiagram
    participant C as Caller
    participant S as Scheduler
    participant M as BlockManager
    participant A as BlockAllocator

    C->>S: add_request(req)
    S->>S: does prompt + max_new_tokens fit?
    Note over S: If it does not fit, mark ABORTED and resolve the future immediately
    S->>M: match_prefix(prompt)
    M->>A: read ref_count of candidate blocks
    M-->>S: matched physical blocks
    S->>A: inc_ref(matched) / allocate(missing)
    S->>M: register_block() for newly filled blocks
    loop every step
        S->>A: can_append? if not, preempt()
        S->>S: batched sampling of new tokens
        S->>A: append_slot() (copy-on-write if needed)
        S->>M: append_token() (register hash when a block fills)
    end
    S-->>C: future resolves (FINISHED / ABORTED)
```

## Quick start

Requirements: CMake >= 3.20 and a C++17 compiler (MSVC 2019+, GCC 9+, Clang 10+). **No third-party dependencies.**

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

- **Windows + MSVC**: run this from *Developer PowerShell for VS* (it provides `cl.exe` and `ninja`), or use the VS Code CMake Tools extension (the Ninja generator is already configured in `.vscode/settings.json`).
- If `ninja` is not installed, drop `-G Ninja` and use the default generator (with a multi-config generator, add `-C Debug` to `ctest`).

### Build options

| Option | Default | Description |
|---|---|---|
| `KV_CACHE_ENABLE_TSAN` | `OFF` | Build with ThreadSanitizer (GCC/Clang only; MSVC fails fast). Concurrency tests are tagged `concurrency` |
| `KV_CACHE_BUILD_EXAMPLES` | `ON` | Build `examples/demo.cpp` |

```bash
# Run the concurrency suite under ThreadSanitizer
cmake -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DKV_CACHE_ENABLE_TSAN=ON
cmake --build build-tsan --parallel
ctest --test-dir build-tsan --output-on-failure
```

> On newer kernels, high-entropy ASLR makes TSan abort with `unexpected memory mapping`. Run `sudo sysctl vm.mmap_rnd_bits=28` first (CI does this).

### Run the demo

![KV cache demo](docs/demo.gif)

```bash
./build/kvcache_demo            # 2-second run, no pauses (this is what ctest executes)
./build/kvcache_demo --paced    # narrated pacing + colors, used to record the GIF
```

`--paced` is purely presentational: it inserts a `sleep` after each step (default 900 ms, `--beat-ms=N` to change) and colorizes hits, ref-count changes and copy-on-write splits. Without the flag the same text is printed with no pauses and no escape sequences, so the demo still finishes in ~2 s and the CI test stays fast.

To re-record the GIF:

```bash
# one-time: winget install charmbracelet.vhs tsl0922.ttyd Gyan.FFmpeg.Essentials
vhs docs/demo.tape              # writes docs/demo.gif
```

The tape ([docs/demo.tape](docs/demo.tape)) controls the canvas, theme and capture pacing; `--paced` controls how long each step stays on screen. The canvas is sized so the whole run fits without scrolling (76 lines × 88 columns at 13 pt). Tune `--beat-ms` if you change the tape's `Sleep`.

The demo runs three scenarios: prefix reuse with copy-on-write, cross-batch prefix-cache hits, and preemption with recompute. Key output from the cross-batch scenario:

```text
common prefix: 48 tokens (= 3 full blocks)

[wave 1] 3 requests, all with the identical 48-token prompt
  peak usage 6 blocks = 3 shared prefix + 3 private decode (12 without sharing)
  free=32/32 (all finished: refs hit zero, blocks are back on the free list)

[wave 2] 3 requests: 48-token common prefix + 16 private tokens (4 blocks each)
  req#0 pre-check before submit: 3 of 4 blocks already cached
  peak usage 9 blocks (without sharing, 3 concurrent requests need 4 each = 12)
  steps=6, preemptions=0, free=32/32
```

And from the copy-on-write scenario, the reference count of a shared block across the whole flow:

```text
prefill(r1)        BLK #0 ref=1     # allocated
prefill(r2)        BLK #0 ref=2     # prefix hit, reused
fork_sequence(r2)  BLK #0 ref=3     # forked
decode(son)        BLK #0 ref=3     # read-only, no copy
                   BLK #3 (ref=2) splits -> BLK #4 (ref=1)
after release(all) free = 16/16
```

## Tests

| Test binary | Coverage | Cases |
|---|---|---|
| `kvcache_phase1_3_test` | Allocator, prefill, decode, copy-on-write, fork, prefix hit/miss/invalidate/rollback | 42 |
| `scheduler_test` | Admission, preemption, resume order, output consistency, oversized-request abort | 9 |
| `scheduler_thread_test` | Background-thread lifecycle, concurrent producers, worker-pool parallelism and shutdown | 10 |
| `random_pressure_test` | Random workloads: small pool (frequent preemption) vs. large pool (none), outputs compared token by token | randomized |
| `thread_stress_test` | Multiple producers + scheduler thread + worker pool, invariants checked every step | randomized |
| `kvcache_demo` | Smoke test: the demo must run to completion, covering scheduler termination and shutdown | 1 |

All of the above pass on **GCC 11 (Linux)** and **MSVC 19.51 (Windows)**, are **clean under ThreadSanitizer**, and the library builds without warnings under `-Wall -Wextra`.

Tests use the bundled `tests/test_util.h` (`CHECK` / `EXPECT_EQ` / `RUN_TEST`). `check_invariants()` verifies on every step that:

- each physical block's `ref_count` equals the number of active requests referencing it
- the number of blocks with `ref_count == 0` equals the free-list length
- a `RUNNING` request's `block_table` length matches its `sequence_length`
- a `WAITING` request (after preemption) has an empty `block_table`

## Repository layout

```
KV_Cache/
├── include/
│   └── kvcache.h          # Data structures and interfaces: PhysicalBlock / Request / BlockAllocator / BlockManager / Scheduler
├── src/
│   ├── allocator.cpp      # Physical block allocation and reference counting
│   ├── manager.cpp        # Chained prefix hashing, cache registration and matching, request release
│   ├── other.cpp          # prefill / append_slot (CoW) / append_token / fork_sequence / fake_sample
│   └── scheduler.cpp      # Four-phase step(), preemption, scheduler thread and worker pool
├── examples/
│   └── demo.cpp           # Executable demo: prefix reuse, CoW, cross-batch cache hits, preemption
├── tests/                 # Unit and stress tests; test_util.h provides assertions and invariant checks
├── docs/
│   ├── ARCHITECTURE.md    # Layers, data structures, four-phase scheduling, locking rules, invariants (Chinese)
│   ├── VLLM_MAPPING.md    # Symbol-by-symbol mapping from this project to vLLM's source (Chinese)
│   ├── DEVLOG.md          # Development log: daily goals, dead ends, leftovers (Chinese)
│   ├── demo.gif           # Recorded demo, embedded above
│   └── demo.tape          # VHS script that produces demo.gif
├── .clang-format
├── .github/workflows/ci.yml
└── CMakeLists.txt
```

## Documentation

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module layers, data structures, request lifecycle, four-phase `step()`, reference counting and CoW, concurrency and locking rules, the four invariants **(Chinese)** |
| [docs/VLLM_MAPPING.md](docs/VLLM_MAPPING.md) | Side-by-side comparison with vLLM's `BlockAllocator` / `PrefixCachingBlockAllocator` / `Scheduler`, plus what is deliberately not implemented **(Chinese)** |
| [docs/DEVLOG.md](docs/DEVLOG.md) | Seven-day development log explaining decisions such as "why `break` instead of `continue`" and "why `set_value` must happen outside the lock" **(Chinese)** |

## Code style

- **Comments are in Chinese.** Interfaces live in `include/kvcache.h`; implementations are split by responsibility under `src/`.
- Formatting uses the repository's [.clang-format](.clang-format), tuned to the existing style (4-space indent, Allman braces, no space before `(` in control statements, includes not reordered):

```bash
clang-format -i include/kvcache.h src/*.cpp tests/*.cpp tests/*.h examples/*.cpp
```

- CI enforces it with `clang-format --dry-run --Werror`, pinned to version `23.1.5`.

## Known limitations

This is a study project; the following are **deliberate simplifications**, not bugs waiting to be fixed:

- **No real KV data.** The copy-on-write branch of `append_slot()` only increments `BlockManager::num_cow_copies()` / `num_cow_tokens()`; no tensors are moved. Blocks carry metadata only.
- **Recompute-only preemption.** `SWAPPED` has been removed (matching vLLM v1); preemption releases blocks and lets `match_prefix()` revive any surviving cached blocks on resume.
- **Placeholder sampler.** `fake_sample()` is a deterministic hash so results can be compared token by token; no model is involved.
- **No hash-collision fallback.** The 64-bit polynomial hash makes collisions vanishingly unlikely, but they are not otherwise guarded.

## Benchmark

### 1. Prefix Caching & Eviction (`./build/kvcache_bench --sweep`)

Default `seed=42`, `block_size=16`, supports `--workload sysprompt|chat|zipf|mixed`, `--blocks N`, `--policy fifo|lru`, `--prefix on|off`, `--csv`:

| workload | blocks | policy / prefix | hit% | evictions | revivals | preempt | steps | tok/step |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `sysprompt` | 24 | `prefix off` | 0.0% | 0 | 0 | 1 | 510 | 2.24 |
| `sysprompt` | 24 | `fifo` + prefix | 71.7% | 187 | 30 | 0 | 307 | 3.72 |
| `sysprompt` | 24 | `lru` + prefix | **73.0%** | **178** | **36** | 0 | 307 | 3.72 |
| `chat` | 64 | `prefix off` | 0.0% | 0 | 0 | 0 | 174 | 3.80 |
| `chat` | 64 | `fifo` + prefix | 15.0% | 344 | 64 | 0 | 174 | 3.80 |
| `chat` | 64 | `lru` + prefix | **17.6%** | **332** | **75** | 0 | 174 | 3.80 |
| `zipf` | 32 | `prefix off` | 0.0% | 0 | 0 | 1 | 252 | 3.70 |
| `zipf` | 32 | `fifo` + prefix | 51.1% | 207 | 28 | 0 | 250 | 3.73 |
| `zipf` | 32 | `lru` + prefix | **53.3%** | **195** | **36** | 0 | 250 | 3.73 |
| `zipf` | 48 | `fifo` + prefix | 61.1% | 158 | 64 | 0 | 250 | 3.73 |
| `zipf` | 48 | `lru` + prefix | **64.4%** | **139** | **76** | 0 | 250 | 3.73 |

### 2. Chunked Prefill + Token Budget (`./build/kvcache_bench --chunked-sweep`)

`mixed` workload (80% short-prompt decode-heavy + 20% 384–512 token long prompts, `blocks=256`, `max_num_seqs=16`, deterministic step-cost model `5ms + 0.25ms/token`):

| chunked | token budget | avg TTFT (ms) | p99 TTFT (ms) | ITL p50 (ms) | ITL p99 (ms) | Throughput (tok/s) |
|---|---:|---:|---:|---:|---:|---:|
| `off` | unlimited | 76.7 | 148.5 | 8.0 | 139.5 | 639.6 |
| `on` | 128 | 78.4 | 180.8 | 8.2 | **37.0 (-73.5%)** | 635.3 |
| `on` | 64 | 122.1 | 250.5 | 21.0 | **21.0 (-84.9%)** | 630.0 |
| `on` | 32 | 719.9 | 1383.8 | 13.0 | **13.0 (-90.7%)** | 542.8 |

## Roadmap

- [x] `examples/demo.cpp`: demonstrate prefix reuse, copy-on-write and preemption
- [x] `docs/ARCHITECTURE.md` / `docs/VLLM_MAPPING.md` / `docs/DEVLOG.md`
- [x] GitHub Actions CI (Linux/GCC + Windows/MSVC + ThreadSanitizer + clang-format)
- [x] `include/` + `src/` layout with a unified clang-format style
- [x] Dynamic worker dispatch (shared atomic cursor) to remove tail idle time
- [x] Runtime-configurable `block_size`, `watermark` (absolute or ratio), and `enable_prefix_caching`
- [x] Zero-copy prefix hashing + O(1) incremental `register_block` and full `running_mutex_` lock coverage
- [x] O(1) intrusive free list and 3-state LRU eviction (CLEAN-first + CACHED LRU + eager hash cleanup)
- [x] Cache/eviction metrics and reproducible `bench/kvcache_bench.cpp` (FIFO vs LRU vs Prefix Off)
- [x] Decoupled `schedule()` / `execute()` (`SchedulerOutput`), removed `SWAPPED`, Chunked Prefill + Token Budget with TTFT/ITL benchmark

## Disclaimer

This is a personal study project, unaffiliated with the [vLLM](https://github.com/vllm-project/vllm) project. It contains no vLLM source code.

## License

[MIT](LICENSE)
