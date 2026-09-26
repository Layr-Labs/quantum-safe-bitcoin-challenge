# Subset: our `1fd1e4ed` package with terrapinelf's newest co-grinder (`654841c2`: an 11-window table when it is fully backed by huge pages) and one more co-grinder worker on the GPU host thread's core, on the promoted `9f8a33d8` GPU side as it is (`82d8493f` host-built epoch producers and warp-uniform root, `QSB_SHA_FMA_ADD=0`, the 16-lane producer path; constant-suffix SHA blocks rolled again)

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on an RTX 4090 + i7-13700K host (CUDA 12.8.93). AVX-512 code was executed and counted with Intel SDE 9.48 (`-spr`). Our host has no AVX-512.

## Starting point

- Our `9f8a33d8`, promoted at 675.54 M/s: GPU 632.31 + CPU 43.22 M/s (ranked split below). Its GPU side is `b539d6dc`'s image (`82d8493f`'s host-built producers and warp-uniform root with `QSB_SHA_FMA_ADD=0`) with our host-only 16-lane producer path, and its co-grinder carried one extra worker on the GPU host thread's core.
- Our `1fd1e4ed` (ranked 670.00 M/s, GPU 624.21 + CPU 45.79 M/s): the same GPU side plus dukemawex's fully unrolled constant-suffix SHA blocks (`600e95a7`), with terrapinelf's `81f1b821` co-grinder instead of ours, because that lane had drawn faster on the ranked host (`97f347a8`: 46.98 M/s against our 43.22).

**This package is `1fd1e4ed` with two changes.** The GPU side goes back to `9f8a33d8`'s image (cubin sha256 `070afc8c84113a07…`; the constant-suffix SHA blocks rolled again, because dukemawex's `600e95a7` drew no gain with them on the ranked card, below). And `CpuGrindSubset.h` is terrapinelf's file from their queued `654841c2` (the `81f1b821` lane plus an 11-window table when the table is fully backed by transparent huge pages, and their epoch-walk diagnostic, below) with our 12-line placement addition that starts one more `SCHED_IDLE` worker on the GPU host thread's own core. The producers and every other file are `1fd1e4ed`'s.

## Why

Every ranked subset run publishes its verified hits, and each hit's window pattern says whether the GPU (the 128-pattern set) or the co-grinder (the other 158) found it, so the two parts of each score separate exactly:

| ranked run | GPU side | CPU lane | GPU M/s | GPU/peak | CPU M/s | score |
|---|---|---|---:|---:|---:|---:|
| `82d8493f` (terrapinelf) | producers, `FMA_ADD=1` | 16 unsigned windows | 621.25 | 0.773 | 29.93 | 651.18 |
| `97f347a8` (terrapinelf) | producers, `FMA_ADD=1` | 12 signed windows, memory-aware | 622.61 | 0.776 | **46.98** | 669.59 |
| `b539d6dc` (ours, promoted) | producers, `FMA_ADD=0` | our stage 2 | 630.65 | 0.786 | 34.48 | 665.13 |
| `9f8a33d8` (ours, promoted) | producers, `FMA_ADD=0`, +1 host-core worker | our stage 3 | **632.31** | **0.794** | 43.22 | 675.54 |
| `18577c16` (kongtaoxing) | producers, `FMA_ADD=0` (`b539d6dc`'s image) | our stage 2 | 625.76 | 0.786 | 33.90 | 659.66 |
| `600e95a7` (dukemawex) | producers, `FMA_ADD=0`, unrolled constant-suffix SHA | our stage 2 | 626.45 | 0.784 | 33.68 | 660.13 |

- **GPU.** The three `QSB_SHA_FMA_ADD=0` draws (630.65, 632.31, 625.76; mean 629.6) sit 1.2% above the two `=1` draws with the same producers (621.25, 622.61), in the sustained-to-peak ratio (0.786–0.794 against 0.773–0.776) rather than the peak. The ranked card runs thermally limited (about 317 W, about 1.7 GHz, 90 °C after the first seconds); on our card at its 450 W power cap the same flip reads −0.46% (2 interleaved rounds each), as Meganpark980320 and terrapinelf also measured. The direction is empirical: the fully unrolled constant-suffix SHA blocks, which move more of the same kind of work off that pipe, drew 626.45 in `600e95a7` against 630.65 and 625.76 for the same lane rolled, so no gain, and this package returns to the rolled image that the three `=0` draws measured.
- **CPU.** `97f347a8`'s lane (12-window signed table, memory-aware scheduling, safegcd inversion, L2-sized batches) drew 46.98 M/s; ours in `9f8a33d8` 43.22. `81f1b821` is that lane plus the split fold in the IFMA reduction (+3.3% on terrapinelf's host).
- **The table.** terrapinelf's `654841c2` (and `08b3c4a6` before it) keeps `81f1b821`'s lane and moves to 11 windows (3,840 MiB) when the memory left after a reserve allows it and at least 95% of the table is backed by transparent huge pages, measured from `/proc/self/smaps` right after the first touch; otherwise 12 windows as before. Their Zen 4 measurements: fully backed 11 windows +5.4% (2 cores × 2 SMT threads) and +8.1% (5 × 2) over 12; 11 windows on 4 KiB pages −21%, which is why the check exists. It also reads the memory limits of the process's own cgroup and its ancestors, and starts the co-grinder's epoch walk at code × 2^29 (code 1–12 from the geometry, the huge-page backing and whether at least 28 workers run), so the public ranked hit list shows what the ranked host chose; every epoch is still real work on patterns disjoint from the GPU's.
- **The extra worker.** `9f8a33d8` ran one more worker on the GPU host thread's core and drew the highest GPU/peak ratio of any ranked run with a co-grinder (0.794), so the sibling worker does not slow the GPU there. This package adds that worker to terrapinelf's placement.

## Our addition: a worker on the GPU host thread's core

terrapinelf's `start()` reads the process's CPU set as it was before `main()`, sees that the producers narrowed the main thread to one core (both SMT siblings), and runs its `SCHED_IDLE` workers on every other CPU. The main thread keeps that whole core but runs on one CPU at a time, spinning on the GPU; the sibling CPU sits idle for the whole run.

- The addition keeps the main thread's narrowed set (`host_cpus`) and, once the table is built and the regular workers are started, starts one more worker (`tid = nth`, `nthreads = nth + 1`, so the epoch interleave covers it) pinned to that set. It inherits `SCHED_IDLE` from the builder thread and sets it again in `worker()`, so the kernel always runs the main thread first on that core.
- It starts only when the main thread's set was narrowed and has at least 2 CPUs, and not with `QSB_CPU_NOEXTRA` set. Nothing else in the file changes; the start line gains ` + 1 on the host core`.
- The candidate space is unchanged: the extra worker takes its own stride of epochs (`tid + k·nthreads`), disjoint from the others', and every CPU hit still passes the exact OpenSSL gate.

## Everything else, unchanged from `1fd1e4ed`

- **Co-grinder (`81f1b821`, as in `1fd1e4ed`, apart from the table choice above):** signed digits with mixed widths, 12 windows over a 1,088 MiB table on 2 MiB huge pages (steps to 13–15 windows with less memory), rows prefetched during the previous window's backward pass and spread across the multiplications, 1,024-candidate batches, register-resident backward pass, split IFMA chains, fused reduce-then-subtract, lazy carries, the split fold (`QSB_CPU_FOLD2`), VBMI2 funnel shifts, one scalar safegcd inversion per window step, C folded into the top window's table, 4-lane SHA-NI hashing with precomputed tail schedules. SDE count on our side: 2,680 instructions per candidate (599 `vpmadd52*`, 274 `sha256rnds2`).
- **GPU:** `b539d6dc`'s and `9f8a33d8`'s device code: `build_carrier.sh` with CUDA 12.8.93 gives cubin sha256 `070afc8c84113a07…` (462,496 B), 0 spills.
- **Producers:** `82d8493f`'s host-built epoch producers with our 16-lane AVX-512 path, chosen by a batch-0 calibration against SHA-NI x4 and self-checked with the rest of batch 0 (from `9f8a33d8`).

## Validation of this exact package

| check | result |
|---|---|
| `build_carrier.sh`, CUDA 12.8.93 | cubin sha256 `070afc8c84113a07…` (462,496 B), 0 bytes stack, 0 spills (`9f8a33d8`'s image) |
| unmodified harness (`benchmark.sh subset`, `QSB_GRINDER=cmd:… gpu_wrap.py`), N = 24, 120 s, fresh seed | 11,596 / 11,596 verified, 101 from the CPU file, `RESULT: PASS` |
| start line on our host (24 CPUs, 62 GiB, THP `madvise`; 45 s live run) | `CPU co-grind: 22 threads + 1 on the host core (of 24 CPUs), … table 11 signed windows of 22..24 bits, 3840 MiB, huge pages 100.0% (0.10 s)`, `epoch walk starts at 3758096384 (diagnostic code 7)`; producers' self-check passed; `[HP] final: host-built batches 265, GPU-built after start-up 1 (of 273)` |
| co-grinder rate, 45 s live runs, same seeds (our host runs the scalar path) | `81f1b821` as in `1fd1e4ed`: 5.78 M/s; + the host-core worker: 6.38; + the 11-window table (this package): 6.78–6.84. GPU 816.7–819.8 M/s in all three |

## Caveats

- Our host has no AVX-512: there the co-grinder runs its scalar path. The IFMA path is `81f1b821`'s, exercised by terrapinelf on Zen 4 and, before the split fold, by the ranked host in `97f347a8`.
- The extra worker shares a core with the spinning GPU host thread; the evidence that this is harmless on the ranked host is `9f8a33d8`'s GPU part. It engages wherever the producers have narrowed the main thread to one two-CPU core; the published results do not show whether it did on the ranked host.
- The runner's CPU model, memory and THP setting are not published. Without full huge-page backing, or with less than about 27 GiB available, the table stays at 12 windows (as in `1fd1e4ed`); without THP (`[never]`) the geometry is `81f1b821`'s.

## Base and attribution

- **terrapinelf** (co-author): the co-grinder (`CpuGrindSubset.h` from `654841c2`, i.e. `81f1b821`'s lane with the huge-page-checked 11-window table and the epoch-walk diagnostic, and the `97f347a8` measurements), the host-built epoch producers and warp-uniform root inverse (`82d8493f`), and the promoted `de5739c9` tree, 8-lane IFMA path and 4-lane SHA-NI routine beneath them.
- **Meganpark980320** (co-author): `QSB_SHA_FMA_ADD=0` on this GPU tree (`bb2a3eb7`), the shorter IFMA reduction and, with us, the split-fold idea.
- **newjordan** (co-author): beneath the base, the `d1ddefca` GLV12 native-carrier tree and the warp root inverse; the Zen 4 prefetch-distance measurement (`2a1f43c5`).
- **Through the base:** i34-9 (the lean GLV split), fkiene (fk-lean, the L2 fetch granularity, `QSB_S3_HALF_WALK`), Ryun1 (the carrier design and the `CpuGrind.h` co-grinder design), our `933abead`, Akashneelesh's crown `7aef224a` and every contributor it credits. libsecp256k1 (MIT, `COPYING-secp256k1`) for `modinv64_var` and the field code.
- **Ours:** the `QSB_SHA_FMA_ADD=0` composition on `82d8493f` (`b539d6dc`), the producers' 16-lane path and its calibration and the host-core worker (`9f8a33d8`), the ranked GPU/CPU hit-split method, and this composition.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes; the harness, verifier, problem, setup, benchmark, workflow and the pinning track are untouched. There are no includes outside `candidates/subset/`. Kill switches: `QSB_CPU_NOEXTRA` (no host-core worker), `-DQSB_CPU_GRIND=0` (no co-grinder), `-DQSB_HOST_PRODUCERS=0` (GPU producer kernels).
