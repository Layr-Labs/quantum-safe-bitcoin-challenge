Model: Claude Opus 5.5
Harness: Claude Code

# Subset: b539d6dc's tree (82d8493f host producers + warp-uniform root + `QSB_SHA_FMA_ADD=0`) with a faster host co-grinder: a fully vectorized 16-lane AVX-512 pipeline with no per-candidate scalar work

Effort: max. Prepared with Claude Opus 5.5 in Claude Code. The co-grinder was measured on an Intel Xeon Platinum 8488C (Sapphire Rapids) VM and on an AMD EPYC (Zen 4) + RTX 4090 host (CUDA 12.8.93). The ranked build line and argv are unchanged, and only `candidates/subset/` changes.

## What this is

The GPU side is byte-identical to ercumentyildirim's queued `b539d6dc`: terrapinelf's `82d8493f` (host-built epoch producers, warp-uniform root inverse, on the promoted `de5739c9` GLV12 tree; `889742ab` changed only the co-grinder) with `QSB_SHA_FMA_ADD=0`. Every `.cu`/`.cuh` file and every device-visible header is unchanged from `b539d6dc`. `build_carrier.sh` with CUDA 12.8.93 regenerates the same native sm_89 image: cubin sha256 `070afc8c84113a07…` (462,496 B), 0 spills. Only the source-hash comment changes, because it covers `CpuGrindSubset.h`.

The one change is in `CpuGrindSubset.h`. It keeps `b539d6dc`'s co-grinder (signed-digit memory-sized table, ±C folded into the last window, shorter IFMA reduction, no table re-read, precomputed SHA schedules, 16-lane AVX-512 SHA-256) and adds a new worker loop, `worker16`, for hosts that have AVX-512 IFMA and the SHA extensions. It uses the same candidates, table, field arithmetic and exact gate. We moved everything between the field operations and the SHA rounds into vector code, and dropped all per-candidate scalar bookkeeping.

## Why

On a Sapphire Rapids core, the co-grinder is bound by the two 512-bit vector ports, p0 and p5. Two SMT threads per core already keep them busy. Scalar work also issues on p0/p5, and `b539d6dc` does a lot of it for each candidate:
- digit recoding;
- row-address arithmetic for every table row, done twice (load and prefetch);
- building the compressed-key message byte by byte and transposing it through memory;
- the 9 skip indices.

Perf counters on the 8488C (one thread, 11-lookup table) measure p0+p5 uops per candidate at 2,332 for `b539d6dc` and 1,960 for this package.

## Changes (all in `CpuGrindSubset.h`)

1. **Batch bookkeeping.** For each candidate the batch keeps only its SHA-256d digest (a 32-byte copy from the epoch's 16-lane digest array), its epoch slot and its pattern number. The 9 indices are rebuilt only when a key hash passes the prefilter.
2. **Vector signed-digit recoding** (`f16_digits`), 16 candidates per vector: the same branch-free recoding as `put_digits`, with the top window never negative and a zero digit marking the candidate bad. For every window it writes the table-row offsets (window-major, `u32`, in 8-byte units) and the per-group sign masks. For the C-folded last window it writes the offsets of both recids, relative to T+C, with T−C = T+C + `went`. The row loads and the prefetches then read the offsets directly, with no address arithmetic.
3. **Sign folded into the forward subtraction** (`fe8_subsgn`): TY = ±ty − Y becomes t + 4p − y, or 8p − t − y in the negated lanes, with one carry pass. This replaces a negation, a blend and a second carry pass.
4. **Key hashes inside the last window** (`f16_final`). The canonical x and y words of each 8-lane group give the compressed-key message words directly: 9 words, shifts and `vpmovqd`, no per-lane bytes.
   - The recid-0 and recid-1 groups of the same 8 candidates form one 16-lane SHA-256.
   - The prefilter (top N bits of word 0 zero) is a `vptestnmd` mask. The mask is almost always 0.
   - A set bit records (candidate, recid). The exact OpenSSL gate `qsb_hv_check` then runs recid 0 before recid 1, one recid per candidate, as before.
5. **Worker placement:** as in `b539d6dc` without the hybrid (workers on every CPU but the GPU host thread's core, `SCHED_IDLE`). The run-time SHA-NI/16-lane/hybrid calibration is skipped when this pipeline runs, since it always uses 16-lane hashing and all-IFMA.

The previous per-batch path is unchanged and still runs on hosts without AVX-512 IFMA or SHA, and with `-DQSB_CPU_F16=0`.

## Measurements (local)

**Co-grinder alone** (standalone bench around the header, synthetic problem of the ranked shape, g++ -O3 with no -march, `QSB_CPU_DEVBENCH`, 15 s after the table is built). Xeon Platinum 8488C, 2 cores / 4 threads, 11-lookup table (4,352 MiB) for both builds:

| co-grinder | 1 thread | 1 core, 2 SMT threads | 2 cores, 4 threads |
|---|---:|---:|---:|
| `b539d6dc` (16-lane SHA, all-IFMA: its calibrated choice on this CPU) | 1.71 M/s | 1.85 M/s | 3.70 M/s |
| this package | 1.89 M/s (+10%) | 2.21 M/s (+19%) | 4.43 M/s (+20%) |

For reference, on the same VM:
- `889742ab`'s co-grinder does 1.32 M/s per core (2 SMT threads, its own 13-lookup table).
- `b539d6dc` with its 4-lane SHA-NI mode does 1.64 M/s.
- `b539d6dc` with its SMT hybrid does 1.50 M/s.

The 1-thread and 1-core rates were each measured twice, with ≤ 2% spread.

**Perf counters** (8488C, one thread): 1,960 vs 2,332 p0+p5 uops per candidate. By function:
- this package: windows 43%, SHA-256 28%, last window with key hashes 11%;
- `b539d6dc`: windows 44%, SHA-256 28%, and 10% in the per-candidate glue (`worker_body` + `vec_batch`).

**EPYC (Zen 4) host**, standalone, 9 threads: 19.3 M/s (2.14 per thread).

## Exactness

A CPU hit is still written only after `qsb_hv_check` (OpenSSL, exact) re-derives it. Checks:

| check | result |
|---|---|
| prefilter passes (candidate indices + recid) of this pipeline vs `b539d6dc`'s 16-lane path, `QSB_ZEROS_N=12`, one worker, the same first 2,002,944 candidates (8488C) | identical sets, 955 = 955 |
| `QSB_ZEROS_N=16`, 45 s, whole package on the Zen 4 + 4090 host (producers on, this pipeline) | 2,920 CPU hits, all 2,920 pass the harness's `verify_artifact` |
| `QSB_ZEROS_N=16`, 60 s, on an EPYC 7542 (no AVX-512: previous per-batch scalar path) + 4090 host | 4,474 CPU hits, all 4,474 verified |
| unmodified harness (`benchmark.sh subset`), N = 24, 1200 s, fresh seed | 117,251 of 117,251 hits verified, 2,252 of them from the CPU file; score 818.54 M/s, `RESULT: PASS` (Zen 4 + 4090 host, seed 1195927112, 9 co-grinder workers beside the producers) |

## Base and attribution

- **Base: ercumentyildirim's `b539d6dc`** (co-author), the GPU tree and the co-grinder this package extends:
  - signed-digit memory-sized table and builder;
  - C fold;
  - precomputed and 16-lane SHA-256 paths;
  - the 5×52 scalar path, placement and calibration;
  - exception-safe threading;
  - its `889742ab`.
- **terrapinelf** (co-author): `82d8493f` (host-built epoch producers, warp-uniform root inverse) and the promoted `de5739c9` (GLV12xc GPU tree, 8-lane IFMA co-grinder path, `fe8_sqr`, the shared inversion, 4-lane SHA-NI).
- **Through that base:**
  - newjordan (`d1ddefca`, and the Zen 4 prefetch-distance measurement in `2a1f43c5`);
  - i34-9 (`adfa8aaa`, `14675ab0`, `78208a18`);
  - fkiene (`eaba5205`, `b864a72c`, `73224391`);
  - Ryun1 (`25bd990a`, `7a75fa50`: the carrier and the `CpuGrind.h` co-grinder design);
  - ercumentyildirim's `933abead`.
- **libsecp256k1** (MIT, `COPYING-secp256k1`): the inversion addition chain and the 5×52 scalar field code.
- **Ours:**
  - `QSB_SHA_FMA_ADD=0`, the shorter IFMA reduction and the forward-pass `ty − Y` (from our queued `bb2a3eb7`, as credited in `b539d6dc`);
  - here, `worker16`: the batch bookkeeping, vector recoding and row offsets, `fe8_subsgn`, the in-window key messages, 16-lane key hashes and mask prefilter, and the measurements.

All inherited source, GPLv3 notices and attributions are kept.

## Packaging

Only `candidates/subset/` changes. The harness, verifier, problem, setup, benchmark, workflow and the pinning track are untouched. No binary or build stamp is included, and there are no includes outside `candidates/subset/`. Kill switches: `-DQSB_CPU_F16=0` (previous per-batch co-grinder path), `-DQSB_HOST_PRODUCERS=0`, `-DQSB_CPU_GRIND=0`.
