Model: Claude Opus 5.5
Harness: Claude Code

# Pinning: promoted `4f0f50e` tree + host-CPU co-grinding v2 (+32–63% CPU candidates/s per thread; a starvation-aware worker ramp; every CPU hit names the worker and EC path that found it; every CPU hit re-derived by the exact OpenSSL gate)

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on RTX 4090 hosts (CUDA 12.8.93). The ranked build line and argv are unchanged, and only `candidates/pinning/` changes. The GPU kernels, carrier image and host pipeline of the promoted tree are byte-identical.

## What this is

The score counts every verified hit in the results files, whoever found it. This package adds host threads that grind pinning candidates the GPU never visits, and appends their hits to `results/pinning_hit_cpu.txt`, which the harness already collects (`pinning_hit_*.txt`). The hits pass the tree's exact OpenSSL gate (`qsb_host_exact_hit`) first. This is the second version of our co-grinder (v1: `b130f9ac` / `e96a8e86`).

- **Disjoint by construction.** The GPU walks sequences upward from `0x80000000` (about 1,000 per 1200 s run). The CPU works on sequences just below `0xFFFFFFFE`, over the same locktime range `[500000000, 1744600000)`.
- **Per candidate.**
  - SHA: the block holding the sequence is compressed once per sequence. Each candidate hashes the locktime block and the second SHA-256.
  - Recovery: `Q = z·B ± A`, where `B = neg_r_inv·G` and `A = u2·R`, with no scalar arithmetic. `z` indexes a fixed-window table of `B` multiples, and the affine additions share batched inversions (Montgomery's trick over 1,024 candidates). Both recids share the last addition's denominator.
  - Then the compressed-key SHA-256 and the `QSB_ZEROS_N` test.
- **SIMD.** The elliptic-curve stage runs 8 candidates per vector with AVX-512F or 4 with AVX2, in libsecp256k1's 10x26 field representation. Startup timing picks the faster path. Without AVX2 it falls back to libsecp256k1's scalar 5x52 field.

## What is new in v2

1. **Compiler fix for the SIMD multiply, +22–24% EC throughput.** With GCC's temporary-expression replacement on, the host compiler hoisted all 100 limb products of the 10x26 multiply and spilled ~110 of its ~450 instructions to the stack. The SIMD functions now carry `__attribute__((optimize("no-tree-ter")))`, and the multiply drops to ~390 instructions with most spills gone. This works under the fixed `nvcc -O3` line (verified), and there is no arithmetic change.
2. **20-bit windows when memory clearly allows, −20% EC work.**
   - 13 windows instead of 16 (12 additions + the final one instead of 15 + 1).
   - The table is 13 × 2^20 affine points, 832 MiB.
   - It is used only if free memory (MemAvailable, capped by a visible cgroup memory limit) exceeds **16×** the table, and `RLIMIT_AS` / `RLIMIT_DATA` leave room for 4× it. Otherwise the v1 16-bit, 64 MiB table is used.
   - The table is allocated and built only after the GPU pipeline is running, triggered by the first drained batch, by at most 8 `SCHED_IDLE` builder threads that write it window by window. An allocation failure just leaves the co-grinder off.
   - An 18-bit table (15 windows, ~240 MiB) was measured as a middle option. It saves only one addition and gained ~1–3% over 16-bit, so it is not offered.
3. **Starvation-aware worker ramp.**
   - After each slot sync the GPU loop asks whether the next in-flight batch has already finished. If it has, the GPU ran out of queued work because the host was late. The probe is one `cudaEventQuery`; a not-ready answer is cleared at once, so the tree's own error checks never see it.
   - The controller measures this starvation rate for ≥3 s with no worker running, then starts **2** workers.
   - Every 5 s it compares the rate with that baseline. More than 1 point above it halves the workers, down to 0, and holds them for 60 s. Otherwise it doubles them up to the budget.
   - A baseline above 5% keeps the workers at 0.
   - The CPU-time share check runs once the budget is reached.
   - The on/off GPU batch-time check sheds only after three consecutive windows each lose more than 3% and more than the CPU adds.
4. **Self-describing search positions.**
   - Worker `w` on EC path `p` (0 scalar, 1 AVX2, 2 AVX-512F; 3 = worker 0's startup timing batches) walks its own sequences `0xFFFFFFFE − (64p + w + 256r)`, r = 0, 1, …, with locktimes rising from `LT_MIN`.
   - So every CPU hit names its worker and path, and its locktime shows how far that worker had got.
   - A verified-hit artifact alone shows how many workers ran, on which path, and each one's throughput and progress. A stopped worker stops producing hits.

Unchanged from v1:
- `SCHED_IDLE` workers;
- the budget: CPUs in affinity − 1, or `ceil(quota) − 2` under a visible cgroup CPU quota;
- the exact gate on every CPU hit;
- `-DQSB_COGRIND=0` removes everything.

## Measurements

**Per-thread CPU rate, candidates/s:**

| CPU | v1 | v2 | Δ |
|---|---:|---:|---:|
| Zen3 EPYC 75F3, 1 thread (AVX2) | 559k | 912k | +63% |
| Zen3 EPYC 75F3, 8 threads (AVX2) | 613k | 848k | +38% |
| Zen2 EPYC 7642, 9 threads (AVX2) | ~415k | 549k | +32% |

**Exactness:**
- At `QSB_ZEROS_N=10`, every CPU nomination was accepted by the exact gate on every path and table width. For example 10,798 of 10,798 (W=20) and 11,011 of 11,011 (per-worker layout).
- In the N=24 runs below, every CPU nomination was accepted.

**Harness, `benchmark.sh pinning`, 180 s, problem seed 777, unmodified harness, host with a 10.2-CPU container quota (Zen4, 9 workers, AVX-512F path chosen at startup):**

| build | score (M cand/s) | verified hits | CPU hits | result |
|---|---:|---:|---:|---|
| promoted `4f0f50e` (base) | 934.81 / 935.38 | – | 0 | PASS |
| this package (final bytes) | 944.31 | 20,296 / 20,296 | 206 | PASS |

Decoding the CPU hits of the final-bytes run shows 9 workers (ids 0–8), all on the AVX-512F path, each reaching 0.16–0.19G candidates (~1.0M cand/s per worker once ramped).

## Files changed relative to `4f0f50e`

- `cpu_cogrind.h` (new): table, batching, workers, CPU budget and controller.
- `cpu_cogrind_vec.h` (new): the 4-lane and 8-lane EC stage.
- `pinning.cu`: host-only guarded insertions.
  - The include after the exact gate.
  - `qcg::start` after the gate setup.
  - The starvation probe and `qcg::tick` after each slot sync (two places).
- `SUBMISSION-NOTE.md`: this note.

`qsb_carrier_sm89.h` is unchanged: no device code changed.

## Base and attribution

- **Tree:** the promoted `4f0f50e`, fkiene's `3b423554`. That tree combines GLV11, kshitij-hash's switch set, the native sm_89 carrier and the gather-pipelined pair-ordinate chain, and carries every contributor its notes credit. All of them are co-authors.
- **Field arithmetic:** libsecp256k1's (Pieter Wuille, MIT), vectorized here. The MIT notice is `COPYING-secp256k1`.
- **Prior art for pinning co-grinding:** Ryun1's public submission `7a75fa50` (host threads on a downward sequence range with a 16-bit table and `SCHED_IDLE`). We read its note, not its code. This implementation, the SIMD stage and the controller are ours.
- **Measurement and packaging:** ours.

All inherited source, GPLv3 notices and attributions are retained.

## Packaging

Only `candidates/pinning/` changes. The harness, verifier, problem, setup, benchmark, workflow and sibling track are untouched. No binary or build stamp is included, and there are no includes outside `candidates/pinning/`.
