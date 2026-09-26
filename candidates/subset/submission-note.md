# Subset: `de5739c9` + a faster host-CPU co-grinder (memory-sized wide host table, precomputed SHA-256 schedules, vector canonicalization) — GPU unchanged

Effort: max. Prepared with Claude Opus 5.5 in Claude Code, on an RTX 4090 + i7-13700K host (CUDA 12.8.93). We used Intel SDE 9.48 (`-spr`) to execute and count the AVX-512 IFMA path, which our host cannot run natively.

## Starting point

The base is the promoted `de5739c9` (terrapinelf, 634.72 M/s), reached with `yukon sync`. It adds a host-CPU co-grinder (`CpuGrindSubset.h`) to the GLV12 GPU tree:
- the CPU threads grind the 158 window-omission patterns per epoch that the GPU's 128 do not use;
- every CPU hit passes the exact OpenSSL gate before it is written to `results/digest_hit_cpu.txt`.

**The GPU side is untouched.** Every `.cu`/`.cuh` and every device-visible header is byte-identical to `de5739c9`. `build_carrier.sh` with CUDA 12.8.93 regenerates the native sm_89 image byte-identically: cubin sha256 `29739128a257ba62…`, 476,704 B, 0 spills. Only the header's `source sha256` comment changes, because `CpuGrindSubset.h` is part of the hashed source set. All changes are host-only, in `CpuGrindSubset.h`.

## Why the co-grinder

- The ranked metrics include `candidates_self_reported`, the GPU grinder's own peak-rate estimate. It excludes CPU candidates.
- Across 16 GPU-only draws of the GLV12 tree class (peak self rate 788–797 M/s), the ratio `verified_hits · 2^23 / self_reported` is **0.7842 ± 0.0055**.
- `de5739c9` drew 0.7957, so its co-grinder added about **1.5% ± 0.7%** of hits, roughly 9–10 M/s on the 32-thread runner.
- That is the only part of the score with headroom left: each extra CPU M/s is about one point.

## What costs time in the co-grinder (measured)

Dynamic instructions per candidate on the **IFMA path** (what the ranked runner uses when it has AVX-512 IFMA). Counts are from SDE `-mix`, one worker, 65k–90k candidates:

| part | `de5739c9` | this package |
|---|---:|---:|
| window additions (`ec8_window`) | 2,745 (15 additions) | 1,895 (10 additions) |
| SHA-256 (`qsha_x4` / new `qsha_sched` + `qsha_x4_run`) | 1,989 (≈ 816 are spill moves) | 1,048 (no spills) |
| per-candidate glue in `worker` (message assembly, digits, key bytes) | 1,952 | 724 |
| final step, both recids (`ec8_final`) | 503 | 241 |
| rest (inversion, batch setup, OpenSSL epoch prefix) | ~490 | ~311 |
| **total** | **7,678** | **4,219** (−45%) |

We use these counts only to find waste; SDE says nothing about Zen 4 timing. The rates we can measure are further below.

## Changes (all in `CpuGrindSubset.h`)

### 1. A wide host table, sized to the host's memory at run time
z·A used sixteen 16-bit windows: 64 MiB of table, 15 batch-affine additions per candidate. Host RAM is not the GPU's 24 GiB, so the table now uses **L lookups of ⌊256/L⌋ or ⌈256/L⌉ bits** (L − 1 additions):

| L | widths | table | additions |
|---:|---|---:|---:|
| 16 | 16 | 64 MiB | 15 |
| 12 | 21–22 | 2.0 GiB | 11 |
| 11 | 23–24 | 7.0 GiB | 10 |

**How L is chosen at start-up:**
- The budget is **a quarter** of the smaller of `MemAvailable` and the headroom of every memory cgroup on this process's path. It reads `/proc/self/cgroup`, then walks up v2 `memory.max`/`memory.high` against `memory.current`, and v1 `limit_in_bytes`/`usage_in_bytes`.
- The budget is capped at 8 GiB, so L = 11 needs at least 28 GiB free.
- The table is one `mmap` (2 MiB aligned, `MADV_HUGEPAGE`). If it fails, L grows by one until the 64 MiB layout.
- If thread creation or any allocation fails, the co-grinder degrades (fewer threads, smaller table) or switches off. It never throws out of a thread. The promoted code could abort the whole process on a failed `std::thread` construction.
- The start line prints the chosen layout and budget, e.g. `CPU co-grind: table 11 lookups (23..24 bits), 7167 MiB (budget 8192 MiB), built in 1.27s`.

**How the table is built:** all windows advance together in doubling rounds (`T[have+k] = T[k] + T[have−1]`), in batches of ≤ 4,096 that the threads take from a shared counter. It takes 1.3 s for 7 GiB on 22 threads, while the GPU starts. We checked random entries and the first and last entry of every window against OpenSSL: 0 mismatches at 64 MiB, 2 GiB and 7 GiB, and under `ulimit -v` limits that force each fallback.

### 2. The SHA-256 message schedule is precomputed
- Every epoch leaves the same number of bytes after its midstate (`remlen` = 8), because all epochs omit exactly six early pushes.
- So for each of the 158 CPU patterns, **blocks 1..5 of the candidate's preimage are fixed**. Block 1 holds the pattern's last pushes and the tail start; blocks 2..5 are identical for every pattern.
- Their W+K schedules are computed once at start-up and deduplicated.
- Block 0 is the epoch's 8 bytes plus the pattern's first pushes. The 158 patterns have only **77 distinct first blocks**, so each epoch computes 77 block-0 states instead of 158.
- A candidate then needs only SHA rounds for blocks 1..5 (`qsha_x4_run`): only the chaining states live in registers, so four lanes interleave without the ~800 spill moves per candidate of the old 4-lane routine.
- The second SHA-256 of SHA-256d and the compressed-key hashes build their message words directly from the state words and field elements (`pk_words`), instead of byte loops.

### 3. The final step canonicalizes in vectors
`fe8_canon_words` folds the bits at and above 2^256 twice, conditionally subtracts p, and emits the four 64-bit words and the y parity for 8 lanes at once. It replaces the per-lane scalar loop. It matches the scalar `fe8_lane_canon` on 3.2 M values (0 mismatches), including p, p..p+4, just below p, 2^256 − 1, 2^256 + small and 2^257 − 1.

### 4. Smaller items
- Explicit vector moves where struct copies compiled to `rep movsq`.
- Table rows are prefetched in the backward pass of every window too.
- The zero-digit flag is set while the digits are extracted.

## Exactness

The CPU path still can only lose hits, never publish a wrong one: every CPU hit is re-derived by `qsb_hv_check` (OpenSSL, exact) before it is written. We also compared hit sets directly:

| check | result |
|---|---|
| new SHA primitives vs OpenSSL `SHA256_Transform`, 1.1 M random lane-blocks (single and chained) | 0 mismatches |
| `fe8_canon_words` vs scalar reference, 3.2 M values incl. edge cases | 0 mismatches |
| table entries vs OpenSSL at 64 MiB / 2 GiB / 7 GiB and under forced fallbacks | 0 mismatches |
| hit set, `QSB_ZEROS_N=12`, one worker, first 131,072 candidates, scalar path (native) vs 8-lane IFMA path (SDE), 2 GiB and 7 GiB tables | identical (66 = 66) |
| same hit set vs the pre-change code | identical |
| unmodified harness (`benchmark.sh subset`, `QSB_GRINDER=cmd:… gpu_wrap.py`), N = 24, fresh seeds | 180 s: 17,647 / 17,647 verified, 180 from the CPU file (promoted code in the same test: 129); this exact package in a clean clone, 150 s: 14,608 / 14,608 verified, 122 from the CPU file — `RESULT: PASS` both |

## Rates we can measure (local, no AVX-512, so the scalar EC path)

Same host, same problem, grinder run directly for 60 s with the GPU grinding alongside (the co-grinder's rate as the binary reports it, table build included):

| build | CPU co-grinder | GPU |
|---|---:|---:|
| `de5739c9` as promoted | 5.32 M/s | 811.2 M/s |
| this package (7 GiB table, 11 lookups) | **7.36 M/s (+38%)** | 809.0 M/s |

CPU-only dev bench (`QSB_CPU_DEVBENCH`, 22 workers, 15 s after the table is ready): 5.94 M/s (16 lookups, old hashing) → 7.44 (12 lookups) → 7.85 (11 lookups), and the precomputed schedule adds ~4% on top on this host. On the scalar path the window additions dominate (75–83% of the worker's time), so the table width is most of the gain here. On the IFMA path, hashing and glue were a larger share (see the instruction table), so we expect the relative gain there to be larger. The GPU difference in the table (−0.27%) is within the run-to-run spread of our card; we are measuring it further.

## Expected effect on the ranked runner

The runner prints `nproc` = 32. terrapinelf's `de5739c9` note infers a Zen 4 EPYC 9174F from the hit-verification speed, which would run the IFMA path. On that path the per-candidate instruction count falls by 45%. The window additions (the IFMA-heavy part) fall from 15 to 10 when the host has ≥ 28 GiB available, or to 11 with ≥ 8 GiB. We expect roughly 1.5–1.8× the co-grinder rate, i.e. about +5 to +8 M/s on top of the GPU. If the runner has no IFMA, the scalar path gains as measured above.

## Caveats

- The runner's memory is not published. If less than 28 GiB is available, the table drops to 2 GiB (11 additions); below 8 GiB it drops further. The start line reports the choice.
- Our instruction counts come from SDE on the IFMA path; real Zen 4 timing is not measured.
- `QSB_CPU_DEVBENCH` / `QSB_CPU_DEVCAND` are dev-only hooks (a CPU-only rate and a deterministic hit-set run). They are compiled out of the ranked build.

## Base and attribution

- **Base:** the promoted `de5739c9` by terrapinelf (GLV12xc GPU tree, 8-lane IFMA co-grinder path, shared inversion, `fe8_sqr`, 4-lane SHA-NI, runner-CPU inference). We build on this promotion.
- **Credited through that base:**
  - newjordan: the `d1ddefca` native-carrier GLV12 tree and the warp root inverse;
  - i34-9: the lean GLV split;
  - fkiene: fk-lean, the L2 fetch granularity and `QSB_S3_HALF_WALK`;
  - Ryun1: the carrier design and the `CpuGrind.h` co-grinder design, table and batch-affine code;
  - our own `933abead` GLV12 port;
  - Akashneelesh's crown `7aef224a` and every contributor it credits.
- **Ours in this package:** the memory-sized wide table and its parallel builder; the budget and cgroup logic; the precomputed-schedule SHA path (`qsha_sched`, `qsha_x4_run`, block-0 variants, `pk_words`); `fe8_canon_words`; exception-safe threading; the SDE/hit-set test method.

All inherited source, GPLv3 notices and attributions are kept.
