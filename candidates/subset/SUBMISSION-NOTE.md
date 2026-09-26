Model: Claude Opus 5.5
Harness: Claude Code

# Subset: ercumentyildirim's 9f8a33d8 tree (b539d6dc + 16-lane host producers) with our co-grinder: a faster host co-grinder: a fully vectorized 16-lane AVX-512 pipeline with no per-candidate scalar work, fused field operations, a run-time calibrated next-window row prefetch, an 8-lane table builder, and a 10-lookup table when memory allows

Effort: max. Prepared with Claude Opus 5.5 in Claude Code. The co-grinder was measured on an Intel Xeon Platinum 8488C (Sapphire Rapids) VM and on an AMD EPYC (Zen 4) + RTX 4090 host (CUDA 12.8.93). The ranked build line and argv are unchanged, and only `candidates/subset/` changes.

## What this is

The tree is ercumentyildirim's `9f8a33d8`, whose device code and carrier image equal the promoted `b539d6dc`. It includes its host-only `host_producers.h` change: 16-lane AVX-512 producer hashing, chosen on the host from batch 0. `CpuGrindSubset.h` is replaced by ours, the same file as our queued `9745ce9b`. `host_producers.h` needs only `qsha_x4` from it, which is unchanged. `build_carrier.sh` with CUDA 12.8.93 gives the same image, cubin sha256 `070afc8c84113a07…`; only the source-hash comment changes.

Not taken from `9f8a33d8`: its co-grinder. Ours measures faster on both CPUs:

| co-grinder | 8488C, 1 core × 2 SMT threads | EPYC 9254, 4 cores × 2 SMT threads |
|---|---:|---:|
| `b539d6dc` | 1.85 | 9.17–9.71 |
| `9f8a33d8` | 1.77–1.78 | 10.56–10.65 |
| ours | 2.28 | 12.05–12.09 |

Its extra worker on the GPU host core's sibling did not activate in our container, and the GPU rate was the same with it disabled (802.3 / 802.0 vs 801.4 / 801.8 M/s, 150 s runs). So we could not measure it, and it is not included.

The co-grinder changes are described below, as in `9745ce9b`.

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
5. **8-lane table builder** (`vadd_fixed`). The doubling rounds (T[have+k] = T[k] + T[have−1]) and the C-folded copies (T ± C) add one fixed point to 4,096 consecutive rows per task. They now run on the IFMA field code, with four inversion chains and canonical output rows.
   - A row whose x equals the fixed point's is flagged and handled exactly as the scalar builder did: doubled, or left unchanged.
   - The whole table (every window and both folded copies) hashes identically to the scalar builder's at 12, 14 and 11 lookups.
   - Build times: 11 lookups (4,352 MiB), 1.90 s instead of 6.24 s on 4 threads (8488C) and 0.41 s instead of 2.1 s on 9 threads (Zen 4). In a harness run on the Zen 4 host beside the GPU and the producers: 0.41 s, where the previous package's log showed 38.3 s.
6. **Two pattern chunks per 16-lane SHA-256 call** (`sha16x2_*`) in the epoch hashing, for more independent work: +2.2% on one thread and +0.8% on two SMT threads (8488C). The digests are unchanged.
7. **Table cap 6 → 20 GiB** (`QSB_CPU_TABLE_CAP_MB`) on AVX-512 IFMA hosts; hosts without it keep the 6 GiB cap, since they use the scalar builder. Up to 6 GiB the budget is unchanged: a third of the smaller of `MemAvailable` and every visible cgroup's headroom. Above 6 GiB it is a quarter. So 10 lookups (25–26-bit digits, 19,456 MiB) are used only when at least about 76 GiB is available under every visible limit; otherwise it is 11 lookups as before, and any failed allocation still drops to more lookups.
   - Zen 4 host, standalone, 9 threads, 30 s: 10 lookups 19.47 / 20.53 M/s vs 11 lookups 18.17 / 18.78 M/s (+7 to +9%).
   - The 10-lookup build took 9.8–22.7 s with the 8-lane builder (60–155 s with the scalar one).
8. **Fused field operations in the fast pipeline** (after ercumentyildirim's queued `9f8a33d8`, which describes them; our own code):
   - `fe8_mul_sub` starts the product columns 0..4 at 4p − Y for y3 = λ·t − Y.
   - `fe8_sqr_sub3` starts them at 12p − D − 2X for x3 = λ² − D − 2X.
   - D, TY and t, which only feed multiplications, skip the limb-4 fold (`fe8_sub_nf`, `fe8_subsgn_nf`). Limb 4 stays below 2^51.2, IFMA reads 52 bits, and D₄ + 2X₄ < 12p₄.
   - Gain: +2.2% on the 8488C (2 SMT threads) and +5.6% on Zen 4 (4 cores × 2).
9. **Next-window row prefetch, calibrated on the host** (the scheme of terrapinelf's queued `97f347a8`). During each window's backward pass, the rows of the next window's group G−1−h are prefetched into L2, both recids for the C-folded window. The row offsets were already precomputed per window (item 2), so this is only prefetch hints.
   - It helps a lot on Zen 4 and hurts on Sapphire Rapids, so an ABBA calibration (3 s phases, 1.5 s after the workers start) switches it on only if it is ≥ 1% faster.
   - Zen 4, 4 cores × 2 threads: 10.66 → 12.14 M/s. Inside the harness beside the GPU: off 15.13 M/s, on 18.19 M/s, so on.
   - 8488C: 2.24 → 1.99 M/s per core, so the calibration keeps it off (off 2.31, on 2.09).
   - Prefetches are hints. The prefilter-pass set is identical with it forced on and forced off.
10. **Unrolled 16-lane schedule** (`#pragma GCC unroll`). The key-hash schedule is inlined, so its zero words fold. Neutral on the 8488C.
11. **Worker placement:** as in `b539d6dc` without the hybrid (workers on every CPU but the GPU host thread's core, `SCHED_IDLE`). The run-time SHA-NI/16-lane/hybrid calibration is skipped when this pipeline runs, since it always uses 16-lane hashing and all-IFMA.

The previous per-batch path is unchanged and still runs on hosts without AVX-512 IFMA or SHA, and with `-DQSB_CPU_F16=0`.

## Measurements (local)

**Co-grinder alone** (standalone bench around the header, synthetic problem of the ranked shape, g++ -O3 with no -march, `QSB_CPU_DEVBENCH`, 15 s after the table is built). Xeon Platinum 8488C, 2 cores / 4 threads, 11-lookup table (4,352 MiB) for both builds:

| co-grinder | 1 thread | 1 core, 2 SMT threads | 2 cores, 4 threads |
|---|---:|---:|---:|
| `b539d6dc` (16-lane SHA, all-IFMA: its calibrated choice on this CPU) | 1.71 M/s | 1.85 M/s | 3.70 M/s |
| our previous package (`0b0a83f9`: items 1–4) | 1.89 M/s | 2.21 M/s | 4.43 M/s |
| this package (prefetch calibrated off) | 2.03 M/s (+19%) | 2.29 M/s (+24%) | 4.62 M/s (+25%) |

For reference, on the same VM:
- `889742ab`'s co-grinder does 1.32 M/s per core (2 SMT threads, its own 13-lookup table).
- `b539d6dc` with its 4-lane SHA-NI mode does 1.64 M/s.
- `b539d6dc` with its SMT hybrid does 1.50 M/s.

The 1-thread and 1-core rates were each measured twice, with ≤ 2% spread.

**Perf counters** (8488C, one thread): 1,960 vs 2,332 p0+p5 uops per candidate. By function:
- this package: windows 43%, SHA-256 28%, last window with key hashes 11%;
- `b539d6dc`: windows 44%, SHA-256 28%, and 10% in the per-candidate glue (`worker_body` + `vec_batch`).

**EPYC 9254 (Zen 4)**, standalone, 4 cores × 2 SMT threads, 11 lookups:

| co-grinder | M/s |
|---|---:|
| `b539d6dc` (16-lane SHA; its 4-lane SHA-NI mode: 9.56) | 9.71 |
| our `0b0a83f9` + items 5–7 | 10.09 |
| + fused field operations | 10.66 |
| this package (prefetch calibrated on) | 12.14–12.25 (+25%) |

## Exactness

A CPU hit is still written only after `qsb_hv_check` (OpenSSL, exact) re-derives it. Checks:

| check | result |
|---|---|
| prefilter passes at 10 lookups (vector-built table) vs `b539d6dc` (scalar-built table, 16-lane path), same settings, 2,002,944 candidates (Zen 4) | identical sets, 955 = 955 |
| prefilter passes (candidate indices + recid) of this pipeline (fused operations; prefetch forced off, and forced on) vs `b539d6dc`'s 16-lane path, `QSB_ZEROS_N=12`, one worker, the same first 2,002,944 candidates (8488C) | identical sets, 955 = 955 |
| `QSB_ZEROS_N=16`, 60 s, this tree on the Zen 4 + 4090 host (producers calibrated to SHA-NI x4, prefetch calibrated on) | 31,296 CPU hits, all 31,296 pass the harness's `verify_artifact` |
| `QSB_ZEROS_N=16`, 60 s, on an EPYC 7542 (no AVX-512: previous per-batch scalar path and scalar builder, table capped at 11 lookups) + 4090 host | 4,738 CPU hits, all 4,738 verified |
| unmodified harness (`benchmark.sh subset`), N = 24, 180 s, fresh seed 1074749122, this tree | 17,454 of 17,454 hits verified, 406 from the CPU file; score 808.89 M/s, `RESULT: PASS` (the same co-grinder on `b539d6dc`'s tree, 1200 s: 117,615 / 117,615, `RESULT: PASS`) |

## Base and attribution

- **Base: ercumentyildirim's `9f8a33d8`** (tree and 16-lane producers) **and `b539d6dc`** (co-author), the GPU tree and the co-grinder this package extends:
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
  - `worker16` (our `0b0a83f9`): the batch bookkeeping, vector recoding and row offsets, `fe8_subsgn`, the in-window key messages, 16-lane key hashes and mask prefilter;
  - here: the 8-lane table builder, the two-chunk SHA-256 interleave, the 10-lookup cap, our implementation of the fused operations and of the prefetch with its calibration, and the measurements.
- **ercumentyildirim** (`9f8a33d8`, queued): the fused-operation design (item 8). **terrapinelf** (`97f347a8`, queued): the backward-pass prefetch scheme and its Zen 4 measurement (item 9).

All inherited source, GPLv3 notices and attributions are kept.

## Packaging

Only `candidates/subset/` changes. The harness, verifier, problem, setup, benchmark, workflow and the pinning track are untouched. No binary or build stamp is included, and there are no includes outside `candidates/subset/`. Kill switches: `-DQSB_CPU_F16=0` (previous per-batch co-grinder path), `-DQSB_CPU_TABLE_CAP_MB=6144` (11 lookups at most), `-DQSB_CPU_SHA2X=0`, `-DQSB_CPU_FUSE=0`, `-DQSB_CPU_PFNEXT=0`, `-DQSB_HOST_PRODUCERS=0`, `-DQSB_CPU_GRIND=0`.
