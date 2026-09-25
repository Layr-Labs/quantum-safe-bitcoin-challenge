# Subset: back to GLV12 (our GLV11 draw shows two more cold DRAM records cost the ranked runner as much as the saved addition gains) — GLV12 native carrier + no-JIT + gate FMA adds + i34-9's lean GLV split + explicit 64 B L2 fetch granularity + host-CPU co-grinding on a disjoint candidate set

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on a dedicated RTX 4090 (driver 595.71, CUDA 12.8.93 toolchain). The ranked build line and argv are unchanged.

## The ranked result that motivates this package

Our `7ee5c52a` ported i34-9's GLV11 geometry onto the GLV12 native-carrier tree: 10 additions per candidate instead of 11, and 6 cold DRAM records instead of 4, all fetched with the one-access `.L2::64B` hint. Locally it was +2.96% over GLV12.

On the runner, against the GLV12 draws of the same afternoon:

| draw (UTC) | tree | score | peak self | score/self |
|---|---|---:|---:|---:|
| 11:37 `3c886977` | GLV12 + hint (`ef1b37e9` bytes) | 626.63 | 791.7 | 0.7915 |
| 12:04 `a65340f6` | GLV12 + hint (`d1ddefca` + host change) | 624.36 | 788.9 | 0.7914 |
| 12:58 `adfa8aaa` | GLV11, no hint (i34-9) | 615.63 | 812.4 | 0.7578 |
| 13:25 `7ee5c52a` | GLV11 + hint (ours) | 624.49 | 818.0 | 0.7634 |

- **Peak self:** it followed the local gain exactly. We predicted 819 and the draw gave 818.0.
- **Late-window ratio:** it lost about 3.5% against GLV12 with the hint. On this card, two more random 64 B DRAM records per candidate cost about as much as one point addition saves.
- **The hint:** it lifted GLV11's ratio by only about 0.7% (0.7578 → 0.7634), not the ~2% a per-activation model predicted.

**The ranked runner rewards fewer DRAM records more than fewer instructions.** GLV12's four cold records are the better trade on that host, so this package goes back to GLV12. It adds the two changes that cost no DRAM traffic.

## What this is

This package is our `de0d4f55` ("GLVfa"): newjordan's `d1ddefca` GLV12 native-carrier tree with our no-JIT startup, the warp root inverse and `QSB_SHA_FMA_ADD=1`. On top of that:

1. **i34-9's lean GLV scalar split** (`GLVScalar.cuh` from `adfa8aaa`): `QSB_GLV_LEAN`, `QSB_GLV_ROUND_CC` and `QSB_GLV_HIGH15_HI`.
   - Explicit `mul.wide` / `mad.wide` products, carry-flag rounding, and a bounded high diagonal whose fallback band is widened to stay exact.
   - Measured +0.213% ± 0.066 on the GLV11 tree, with identical 2,828-hit sets.
   - On this GLV12 tree the fixed-problem hit set is a superset of `de0d4f55`'s: 2,800 common, 16 extra from searching further in 30 s.
2. **Explicit 64 B L2 fetch granularity** (`QSB_L2_FETCH64=1`, host-only): `cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 64)` right after `cudaSetDevice`.
   - fkiene's `b864a72c` set this limit, but the `d1ddefca` lineage does not.
   - Ryun1's measurements (pinning `7a75fa50`): at the digest kernel's occupancy, an untouched limit lets a 64 B record's two sectors reach DRAM as two accesses (4.4 G records/s); set explicitly, as one (7.8 G/s).
   - On hot, clock-limited 4090s, Ryun1 measured the `.L2::64B` hint alone at +1.1%, and the hint plus this limit at about +1.9%.
   - On our cool card it is neutral: −0.016% ± 0.125.
   - It targets exactly the late-window DRAM term that the GLV11 draw exposed.

The native image was regenerated with `build_carrier.sh` and CUDA 12.8.93:
- cubin sha256 `d0e9509f85c60663…`, 476,704 B;
- 2 `LTC64B` loads in the digest kernel;
- 0 spills in every function;
- the three GLV-split knobs are added to the build fingerprint;
- default-build PTX sha256 prefix `b825610e66fe`, only JIT-compiled if the carrier is off.

## Host-CPU co-grinding (`CpuGrindSubset.h`, `QSB_CPU_GRIND=1`)

During a ranked run the host CPU is idle except for the exact publication gate. Ryun1 showed on pinning (`7a75fa50`) that its cores can grind a disjoint part of the search and publish through the same exact gate. This package ports that to subset.

**Candidate space.** The GPU grinds every epoch (the 6 early omissions below the cut, 137) with its 128 window-omission patterns. The CPU threads grind epochs t, t+T, t+2T, … (T threads) with the **other 158** of the C(13,3) = 286 patterns. The two sets are disjoint by construction: in a fixed-problem run the hit window triples overlapped in 0 of 128 × 158 patterns, and no hit appeared twice.

**Per candidate:**
- SHA-256d through OpenSSL. Each epoch's fixed prefix is hashed once from the committed midstate; per candidate, only the 10 kept window pushes, the tail and the suffix are hashed, and the second compression runs on a pre-padded block.
- z·A with A = neg_r_inv·G, using sixteen 16-bit windows over a 64 MiB host table.
- The chain uses batch-affine additions, with one Fermat inversion per step shared across a 4,096-candidate batch through Montgomery's trick.
- Both recids from a shared x_C − x_P denominator.
- The compressed-key SHA-256 with the 24-bit test.

The field arithmetic, the table builder and the batch-affine additions are Ryun1's `CpuGrind.h` (GPL-3), unchanged.

**Safety.**
- Every CPU candidate hit is re-derived by the tree's exact OpenSSL gate `qsb_hv_check` under a mutex before it is appended to `results/digest_hit_cpu.txt`. The harness collects that file with the GPU's `digest_hit_0.txt`. A CPU bug can only lose hits, never publish a wrong one.
- The workers run at `SCHED_IDLE`, so they never delay the GPU host thread. The local GPU rate was unchanged: 832–853 M/s with or without them.
- At the stop signal the process holds the CPU output lock, flushes, and calls `_exit(0)`, so no static destructor runs under a live worker.
- The thread count is the affinity/cgroup CPU quota minus two.

**Checks.**
- At `QSB_ZEROS_N=12`, a 25 s run published 1,743 CPU hits, all of which had passed the exact gate. That is in line with 2·2^-12 per candidate.
- The unmodified harness at N = 24 (90 s, fresh seed) verified every GPU and CPU hit: see the validation line below.

**Rate.** About 120k candidates/s per fully available core on a Ryzen 9 7900X. Our development host is shared with other tenants, so we only saw 0.2–0.3 M/s in total there.
- On a dedicated runner with 14–30 free threads, this would add roughly 1.5–3.5 M candidates/s: +0.25–0.5% of the ~640 M/s late-window GPU rate.
- It adds no GPU or DRAM traffic, which is exactly what the GLV11 draw says the runner punishes.

## Measurements behind the pieces (same local protocol as our earlier notes)

| step | Δ local throughput | note |
|---|---:|---|
| no-JIT startup (`ef1b37e9`) | +0.35% of the window | wall clock: the search starts at 0.70 s instead of 4.86 s |
| warp root inverse | +0.404% ± 0.204 | the ranked peak self rose 789.4 → 792.5, as predicted |
| `QSB_SHA_FMA_ADD=1` (`de0d4f55`) | +0.268% ± 0.018 | FMA-pipe adds pay only in the pubkey gate |
| lean GLV split | +0.213% ± 0.066 | measured on GLV11 |
| L2 fetch 64 B | −0.016% ± 0.125 locally | hot-card effect per Ryun1 |

**Dead this session** (paired ABBA on the GLV12 or GLV11 carrier trees):
- SHA loop re-unrolling: −0.24% / −0.47%;
- window roll: −0.36%;
- `QSB_CHAIN_MUL_LEAN=2`: −0.17%;
- `QSB_TABLE_L2_WINDOW=0`: −0.22%;
- σ shifts on IMAD.HI: −0.21%;
- constant-folded SHA256d: −0.05%;
- the same with FMA adds: −0.69%;
- `QSB_K2S_PARITY_HIGH`: −0.13%;
- `.L1::no_allocate` hot loads: −1.0%;
- `QSB_R_CBANK`, the carry sentinel and FMA rotations: spill.

GLV10 (both components on the P18 layout: 9 additions, 8 cold records) is +0.58% locally over GLV11. After the GLV11 draw, it would be worse still on the runner.

## Correctness

- **Exact.** On the fixed problem, the hit set is a superset of `de0d4f55`'s.
- **Unmodified harness** (`benchmark.sh subset`, 90 s, fresh problem seed): see the validation line below.
- **Host gate.** Every published hit is still recomputed by the exact host (OpenSSL) publication gate.

## Base and attribution

- **Tree:** newjordan's `d1ddefca` (GLV12 four-bank geometry, native sm_89 carrier with the `.L2::64B` cold-record fetch, two-slot pipeline, persisting L2 window, exact host gate, `sha_gate_fma.cuh`). Co-author.
- **Lean GLV split, and the GLV11 geometry whose draw taught the lesson above:** i34-9's `adfa8aaa` and `14675ab0`. Co-author.
- **GLV12 four-bank port:** ercumentyildirim's `933abead`. Co-author.
- **fk-lean tree, the L2 fetch-granularity limit and FMA-pipe gate adds:** fkiene (`eaba5205`, `b864a72c`). Co-author.
- **Carrier design, the 64 B fetch measurements and the host-CPU co-grinder design and code** (`CpuGrind.h`): Ryun1 (pinning `25bd990a`, `7a75fa50`). Co-author.
- **Warp root inverse:** newjordan's `5b198ddf`, file set via i34-9's `78208a18`.
- **Ours:**
  - the no-JIT startup (`ef1b37e9`);
  - `QSB_SHA_FMA_ADD` with its census (`de0d4f55`);
  - the GLV11 port and draw (`7ee5c52a`);
  - this composition.
- **Promoted crown:** `7aef224a` (Akashneelesh) and every contributor it credits.

All inherited source, GPLv3 notices (`COPYING`, `COPYING-secp256k1`, VanitySearch headers) and attributions are retained.

## Packaging

Only `candidates/subset/` changes. The harness, verifier, problem, setup, benchmark, workflow and sibling track are untouched, and no binary or build stamp is included. There are no includes outside `candidates/subset/`.

**Validation of the same package without the CPU co-grinder** (our `90fd91e3`): 8,899 of 8,899 hits verified, `RESULT: PASS`.

**Validation of this exact package (with the CPU co-grinder):** the unmodified harness (`benchmark.sh subset`, 90 s, fresh problem seed) verified 8,744 of 8,744 hits, 2 of them from the CPU file: `RESULT: PASS`.
