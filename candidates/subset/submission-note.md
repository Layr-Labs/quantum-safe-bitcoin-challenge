# Subset: our promoted `b539d6dc` GPU side (terrapinelf's `82d8493f` host-built epoch producers and warp-uniform root with `QSB_SHA_FMA_ADD=0`) carrying terrapinelf's `81f1b821` co-grinder unchanged (12-window signed table, memory-aware scheduling, safegcd inversion, split-fold reduction): the best ranked GPU part and the best ranked CPU lane in one build, with the constant-suffix SHA blocks fully unrolled on the native carrier (after dukemawex's `600e95a7`)

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on an RTX 4090 + i7-13700K host (CUDA 12.8.93). AVX-512 code was executed and counted with Intel SDE 9.48 (`-spr`). Our host has no AVX-512.

## Starting point

- Our `b539d6dc`, promoted at 665.13 M/s: terrapinelf's `82d8493f` tree (host-built epoch producers, warp-uniform root inverse, on the promoted `de5739c9` GLV12 tree) with `QSB_SHA_FMA_ADD=0` and our second co-grinder stage. Ranked split: GPU 630.65 + CPU 34.48 M/s.
- Our `9f8a33d8`, the third stage of our own co-grinder on the same GPU side (promoted at 675.54 M/s, GPU 632.31 + CPU 43.22 M/s).
- terrapinelf's `97f347a8`, its de5739c9-line GPU side (`QSB_SHA_FMA_ADD=1`) with a new co-grinder: ranked 669.59, GPU 622.61 + CPU **46.98** M/s, the fastest CPU lane any ranked run has shown.
- terrapinelf's `81f1b821` (queued): the same package plus a split fold in the IFMA field reduction (`QSB_CPU_FOLD2`, +3.3% lane rate on their host).

**This package is `b539d6dc` with two changes:** `CpuGrindSubset.h` is terrapinelf's file from `81f1b821`, byte for byte (sha256 `9b8cea727b1d7ca9…`), and `subset.cu` turns on the fully unrolled constant-suffix SHA blocks (dukemawex's `600e95a7`, below), with the carrier image regenerated for it (cubin sha256 `d67b51d1934c1460…`, byte-identical to `600e95a7`'s). Everything else, including our host-only 16-lane producer path in `tests/gpu_epochs/host_producers.h`, is `b539d6dc`'s.

## Why this composition

Every ranked subset run publishes its verified hits, and each hit's window pattern says whether the GPU (the 128-pattern set) or the co-grinder (the other 158) found it. That separates the two parts of every score exactly:

| ranked run | GPU side | CPU lane | GPU M/s | CPU M/s | score |
|---|---|---|---:|---:|---:|
| `82d8493f` (terrapinelf) | producers, `FMA_ADD=1` | 16 unsigned windows | 621.25 | 29.93 | 651.18 |
| `b539d6dc` (ours, promoted) | producers, `FMA_ADD=0` | our stage 2 (11 signed lookups) | **630.65** | 34.48 | 665.13 |
| `e560d9b3` (newjordan) | GLV10 async, producers, `FMA_ADD=1` | 22-bit windows | 620.41 | 35.09 | 655.51 |
| `97f347a8` (terrapinelf) | producers, `FMA_ADD=1` | 12 signed windows, memory-aware | 622.61 | **46.98** | 669.59 |

- **The GPU part.** The three `QSB_SHA_FMA_ADD=1` draws with host producers sit at 620.4–622.6 M/s; the one `QSB_SHA_FMA_ADD=0` draw, ours, at 630.65. The ranked card runs throttled (about 317 W, about 1.7 GHz, 90 °C after the first seconds), so energy per candidate sets its sustained rate, and the `FMA_ADD=1` form spends 3 more instructions per round of the key-hash compression on the FMA-heavy pipe. One draw is not proof, but it is the only evidence there is, and it points one way.
- **The CPU part.** Lanes of very different instruction counts sat near one ceiling (29.9–35.1 M/s) until `97f347a8` attacked table-row stalls: rows prefetched during the previous window's backward pass and spread across the multiplications, 1,024-candidate batches sized to the core's L2, and a 12-window table (1.06 GiB) that terrapinelf measured faster than 11 windows on their Zen 4 host. It drew 46.98 M/s. Every other lane with producers, ours included, has drawn at most 35.1.
- i34-9's `1fc6f8aa` (queued ahead of this ticket) composes the same GPU side with `97f347a8`'s lane; this package differs from it by `81f1b821`'s split fold in the co-grinder's reduction and our producers' 16-lane path, and by the unrolled SHA blocks.
- Each part has been measured on the ranked host in a build that differs from this one only in the other part. Neither part changes what the other computes: the GPU keeps its 128 window patterns and the co-grinder its disjoint 158.

## The constant-suffix SHA blocks, fully unrolled (dukemawex's `600e95a7`)

- `subset.cu` sets `QSB_PAIR_SHA_UNROLL_CONST 1` and `QSB_PAIR_SHA_UNROLL_CONST_INNER 1`, both existing, documented switches of `window_schedule_shared.cuh`. The four constant-suffix SHA-256 blocks of each paired window hash were a rolled loop of 8 iterations of 8 paired rounds, a choice that cut the driver-JIT time on the PTX route (terrapinelf's `a75cf15a`). The native carrier is loaded without any JIT, so the unrolled form costs nothing at start-up: the loop-carried `IMAD.IADD`/`IMAD.MOV` work and the loop-indexed `LDC.64` constant loads become `IADD3`s with the precomputed W+K words as immediates. That moves work off the FMA-heavy pipe, the same direction as `QSB_SHA_FMA_ADD=0`. dukemawex's queued `600e95a7` makes this change on `b539d6dc` and gives the static census; we took it as is.
- Same rounds, same message words, same order and feed-forward; only the loop structure changes. `build_carrier.sh` with CUDA 12.8.93 regenerates the native image: cubin sha256 `d67b51d1934c1460…` (573,728 B), 0 bytes stack, 0 spills, byte-identical to `600e95a7`'s image. Static `kernel_digest` size: 14,528 → 21,480 instructions.
- On our RTX 4090, interleaved 60 s runs on a fixed problem: at the card's 450 W power cap the unrolled image ran 0.36% slower (811.0 vs 813.9 M/s, 4 interleaved rounds of 60 s, both near 2,385 MHz). A power cap is not the ranked card's regime: it runs thermally limited near 317 W. There, `QSB_SHA_FMA_ADD=0`, which Meganpark980320 also measured slightly slower at a 450 W cap (−0.41%), drew a GPU/peak ratio of 0.786 against 0.773–0.776 for the `=1` draws with producers, at the same ~802 M/s peak (the peak comes from the first, power-capped seconds). The unroll moves more of the same kind of work (loop-carried `IMAD.IADD`/`IMAD.MOV`) off that pipe, and our previous ticket's GPU part (above) is the second draw of the `=0` knob, on a GPU side that also carries one extra co-grinder worker on the host thread's core. This ticket's draw is the unroll's test.

## The co-grinder (terrapinelf's `81f1b821` file, unchanged)

Summarized from its note and code; we did not modify a line:
- **Table:** signed digits with mixed widths, so 12 windows cover the 256-bit scalar with a 1,088 MiB table (11 additions per candidate). Sized at run time from a quarter of min(MemAvailable, cgroup limit − usage), never fewer than 12 windows, stepping to 13–15 with less memory; built on 2 MiB huge pages and spot-checked against OpenSSL before any worker starts.
- **Memory-aware scheduling:** each window's rows are prefetched during the previous window's backward pass, spread 4 before and 4 after the first multiplications of each group; rows read in the forward pass are kept, not re-read; batches of 1,024 candidates keep both SMT threads' EC state inside the 1 MB L2.
- **Arithmetic:** 8-lane AVX-512 IFMA field code with a register-resident backward pass per group, split IFMA accumulator chains, fused reduce-then-subtract, lazy carries for multiplication-only inputs, the split fold in the reduction (`QSB_CPU_FOLD2`; the idea is credited in their file to Meganpark980320's `e5b67ed2` and our `b539d6dc`), VBMI2 funnel shifts, and one scalar safegcd inversion per window step (libsecp256k1's `modinv64_var`) instead of an 8-lane exponentiation.
- **C folded into the top window's table**, so both recovery keys come from one final addition.
- **Hashing:** 4-lane SHA-NI with precomputed tail schedules; the pattern-shared first tail block is hashed once per group per epoch.
- **Placement:** workers run at `SCHED_IDLE` on every CPU of the process's pre-`main()` CPU set except the GPU host thread's core, below the 3 producer threads. It sizes itself from that saved set, so the one-core mask the producers put on the main thread is not inherited by the workers.
- **Exactness:** a CPU hit is still written only after the exact OpenSSL gate re-derives it. The candidate set, gate and record format are those of every co-grinder since `de5739c9`.

Their own measurements (untouched here): instructions per candidate 7,269 → 2,658–2,692 from `de5739c9`'s lane; +14.6% (2 cores × 2 SMT threads) for the memory-aware step and +3.3% for the split fold on their host; a 1,200 s unmodified-harness run of `81f1b821` verified 120,287 of 120,287 hits, 591 from the CPU file.

Our own count of this exact file under SDE (`-spr -mix`, one worker, `QSB_ZEROS_N=24`, 65,536 candidates): **2,680 instructions per candidate** (599 `vpmadd52*`, 274 `sha256rnds2`), against 3,163 for our own newest lane in its 16-lane hashing mode and 3,753 for our `b539d6dc` lane. By function: window additions 1,361, final step 121, inversions 148, SHA-NI hashing 787, per-candidate glue about 210.

## Integration with this tree

- `82d8493f`'s host producers use the co-grinder header's SHA-NI routines (`qsha_x4`, `qsha_iv`, `qsha_k`, `qsha_supported`). `81f1b821`'s header provides them unchanged, and our 16-lane producer path (`flush16`/`produce16`, chosen by a batch-0 ABAB calibration against SHA-NI x4 and self-checked with the rest of batch 0) is host-only code in `host_producers.h` that does not touch the co-grinder.
- The producers pin the GPU host thread to one core before the co-grinder starts. `81f1b821`'s header reads the CPU set saved before `main()` and places its workers on every other CPU; we checked the start line on our host (below).
- The device code is `b539d6dc`'s plus the unroll switch pair: `#define QSB_SHA_FMA_ADD 0` and the two unroll switches in `subset.cu` are the only device-code settings that differ from `82d8493f`. `build_carrier.sh` with CUDA 12.8.93 regenerates the native sm_89 image: cubin sha256 `d67b51d1934c1460…` (573,728 B), 0 spills, byte-identical to `600e95a7`'s.

## Validation of this exact package

| check | result |
|---|---|
| `build_carrier.sh`, CUDA 12.8.93 | cubin sha256 `d67b51d1934c1460…` (573,728 B), 0 bytes stack, 0 spills; byte-identical to `600e95a7`'s image |
| `CpuGrindSubset.h` against `81f1b821`'s branch | byte-identical (sha256 `9b8cea727b1d7ca9…`) |
| unmodified harness (`benchmark.sh subset`, `QSB_GRINDER=cmd:… gpu_wrap.py`), N = 24, 120 s, fresh seed | 11,535 / 11,535 verified, 85 from the CPU file, `RESULT: PASS` (without the unroll: 11,824 / 11,824, 84) |
| start line on our host | `Native sm_89 carrier: on (573728-byte image, sha256 d67b51d1934c1460…)`; `CPU co-grind: 22 threads (of 24 CPUs), scalar, 4-lane SHA-NI, 158 window patterns per epoch disjoint from the GPU's 128; table 12 signed windows of 20..22 bits, 1088 MiB` (every CPU but the host thread's core); producers' self-check passed, then `[HP] final: host-built batches 353, GPU-built after start-up 2 (of 361)` in a 60 s run |

Our host has no AVX-512, so there the co-grinder runs its scalar 4×64 path and the producers their SHA-NI x4 path. The IFMA path is `81f1b821`'s, exercised by terrapinelf on their Zen 4 host and, in its `97f347a8` form without the split fold, by the ranked host itself.

## Caveats

- One ranked `QSB_SHA_FMA_ADD=0` GPU draw (630.65) against three `=1` draws (620.4–622.6). If the difference is a lucky draw, this package is `81f1b821` with a neutral GPU knob.
- The unrolled SHA blocks have no ranked draw yet (`600e95a7` is queued ahead of this ticket). Their argument is the one that made `QSB_SHA_FMA_ADD=0` pay: less FMA-pipe work on a power-limited card.
- `97f347a8`'s 46.98 M/s is one ranked draw; its hit count gives about ±1.3% Poisson noise on the CPU part alone.
- The co-grinder's load costs the thermally limited GPU a roughly fixed ~2% on the ranked host whatever the lane's speed, so the faster lane is pure gain over the slower one.
- The runner's CPU model and memory are not published. The 12-window table needs about 4.4 GiB available (a quarter rule) and steps down to 13–15 windows with less.

## Base and attribution

- **terrapinelf** (co-author): the whole co-grinder (`CpuGrindSubset.h` from `81f1b821`, and its `97f347a8` measurements), the host-built epoch producers and warp-uniform root inverse (`82d8493f`), and the promoted `de5739c9` GLV12xc tree, 8-lane IFMA path and 4-lane SHA-NI routine beneath them.
- **Meganpark980320** (co-author): `QSB_SHA_FMA_ADD=0` on this GPU tree (`bb2a3eb7`), the shorter IFMA reduction, and with us the split-fold idea `81f1b821` re-implemented.
- **dukemawex** (co-author): the fully unrolled constant-suffix SHA blocks on the carrier route (`600e95a7`) and their static census.
- **newjordan** (co-author): beneath the base, the `d1ddefca` GLV12 native-carrier tree and the warp root inverse; the Zen 4 prefetch-distance measurement (`2a1f43c5`).
- **Through the base:** i34-9 (the lean GLV split), fkiene (fk-lean, the L2 fetch granularity, `QSB_S3_HALF_WALK`), Ryun1 (the carrier design and the `CpuGrind.h` co-grinder design), our `933abead`, Akashneelesh's crown `7aef224a` and every contributor it credits. libsecp256k1 (MIT, `COPYING-secp256k1`) for `modinv64_var` and the field code.
- **Ours:** the `QSB_SHA_FMA_ADD=0` composition on `82d8493f` and its regenerated carrier (`b539d6dc`), the producers' 16-lane AVX-512 path with its batch-0 calibration, the ranked GPU/CPU hit-split method, and this composition.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes; the harness, verifier, problem, setup, benchmark, workflow and the pinning track are untouched. There are no includes outside `candidates/subset/`. Kill switches as in the base: `-DQSB_CPU_GRIND=0` (no co-grinder), `-DQSB_HOST_PRODUCERS=0` (GPU producer kernels).
