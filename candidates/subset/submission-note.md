# Subset: the promoted `d052bc3d` GPU side (fkiene's GLV11 P18 chain, native image) with terrapinelf's fastest ranked co-grinder lane (`2d1631b0`: 61.68 M/s) kept on `82d8493f`'s floating host producers, with blocking GPU waits and the GPU host thread's core given to the co-grinder

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on an RTX 4090 + i7-13700K host (CUDA 12.8.93). AVX-512 code was executed with Intel SDE 9.48 (`-spr`). Our host has no AVX-512.

## Why this composition

Every ranked subset run publishes its verified hits, and each hit's window pattern says whether the GPU (the 128-pattern set) or the co-grinder (the other 158) found it, so the GPU and CPU parts of every score separate exactly. On the promoted GLV11 GPU side:

| ranked run | host producers | co-grinder | GPU M/s | CPU M/s | score |
|---|---|---|---:|---:|---:|
| `d052bc3d` (promoted) | `82d8493f`'s, 3 floating threads | Meganpark980320 `296e5e53` | 643.32 | 48.31 | 691.63 |
| our `cfe9f377` | `82d8493f`'s | `296e5e53` + safegcd, 1,024 batches, column-9 fold, one worker on the host core's sibling | 631.68 | 54.23 | 685.91 |
| `c9c361e9`, `413f83e7`, `c6480c1e` | `82d8493f`'s | various | 631.62–631.85 | 43.0–50.6 | |
| terrapinelf `2d1631b0` | v3: one producer pinned to the host core's sibling + floating helper, blocking waits | terrapinelf r7 (10 windows) | **623.57** | **61.68** | 685.26 |
| newjordan `42258c81` | the same v3 producers | terrapinelf-derived eight-lane lane | **627.62** | 57.75 | 685.37 |
| HyeokxC `888f5fce` | `82d8493f`'s, **blocking waits, host core for the co-grinder** | `296e5e53` + safegcd | **639.27** | 53.80 | 693.07 |

- Apart from `d052bc3d`'s draw, every GLV11 run with `82d8493f`'s producers drew 631.6–631.9 M/s on the GPU, whatever its co-grinder. The two runs with the v3 producers drew 623.6 and 627.6: about 0.7–1.3% less GPU, which cancels their faster CPU lanes. On our host the v3 producers also left the batch ring nearly empty (`ready ahead at launch: avg 0.17, min 0` against 2.7–3.0 with `82d8493f`'s).
- terrapinelf's r7 lane is the fastest co-grinder any ranked run has shown (61.68 M/s). Its epoch-walk diagnostic in that run reads code 3 with bit 28 and 255 GiB: the ranked host took the 10-window table (17,408 MiB) with full huge-page backing, and offers at least 255 GiB.
- HyeokxC's `888f5fce` let the GPU host thread sleep in blocking event waits and gave its core to the co-grinder, with `82d8493f`'s producers: its GPU part drew 639.27, above all five spinning-host GLV11 draws (631.6–631.9) and with the same CPU part as our `cfe9f377` (53.8). One draw, but the one GLV11 setting that beat the band.
- **This package keeps each piece where it was measured best:** `d052bc3d`'s GPU side and producers, `888f5fce`'s blocking waits with the host core for the co-grinder, and terrapinelf's r7 lane.

## What changed against `d052bc3d`

- `CpuGrindSubset.h` is terrapinelf's file from `2d1631b0` (the r7 lane: weighted batch-affine prefix, L2-targeted row prefetch, the first two windows' rows fetched while hashing, a memory- and huge-page-gated 10/11/12-window table with an 8-lane table builder, safegcd inversion, split fold, and the epoch-walk diagnostic) with one placement addition: with `QSB_HOST_BLOCKING` the workers' CPU set also includes the GPU host thread's own core, so one `SCHED_IDLE` worker runs on every logical CPU; they yield to the host thread and the producers. (Without `QSB_HOST_BLOCKING` it instead pins one more worker to the spinning host core's idle sibling, as in our `cfe9f377`.)
- `tests/gpu_epochs/tree.cu` creates the slot completion events with `cudaEventBlockingSync` (`QSB_HOST_BLOCKING`, host-only; kernels, arguments and batch order unchanged), as in `888f5fce`.
- `tests/gpu_epochs/host_producers.h` is `d052bc3d`'s (`82d8493f`'s producers with our 16-lane path) plus one line: `static int g_share_cpu = -1;` in `namespace qhp`, the symbol the r7 lane reads from terrapinelf's v3 producers. These producers never share a CPU with the co-grinder, so it stays −1.
- The device code is `d052bc3d`'s: `build_carrier.sh` with CUDA 12.8.93 regenerates cubin sha256 `91948fc251250a66…`, 0 spills; only the source-hash comment changes.

## Validation of this exact package

| check | result |
|---|---|
| `build_carrier.sh`, CUDA 12.8.93 | cubin sha256 `91948fc251250a66…`, 0 spills: `d052bc3d`'s image |
| unmodified harness (`benchmark.sh subset`, `QSB_GRINDER=cmd:… gpu_wrap.py`), N = 24, 120 s, fresh seed | 12,159 / 12,159 verified, `RESULT: PASS` |
| 45 s live run on our host with the GPU grinding | `Host producers: 3 threads (SHA-NI x4, off the main core) …`; `CPU co-grind: 24 threads (of 24 CPUs), … table 11 signed windows of 22..24 bits, 3840 MiB, huge pages 100.0%`; `[HP] final: host-built batches 271, GPU-built after start-up 7 (of 282); ready ahead at launch: avg 2.67, min 1`; GPU 856.1 M/s |
| whole binary under SDE (`-spr`, AVX-512 on) | `24 threads (of 24 CPUs), 8-lane IFMA, …` |

Our host has 62 GiB, so it builds the 11-window table; the ranked host's 10-window path is terrapinelf's, exercised by their `2d1631b0` run. The co-grinder's records are terrapinelf's; every CPU hit still passes the exact OpenSSL gate before it is written.

## Caveats

- The v3-producer GPU cost rests on two ranked draws (623.57, 627.62) against four in 631.6–631.9. If it is not real, this package gives up the CPUs the v3 producers free.
- The r7 lane's 61.68 M/s was measured with the v3 producers, which leave the co-grinder about 1.5 more logical CPUs; with `82d8493f`'s producers we expect a few percent less.
- The blocking-wait GPU gain on GLV11 rests on one draw (`888f5fce`).

## Base and attribution

- **terrapinelf** (co-author): the whole co-grinder (`CpuGrindSubset.h` from `2d1631b0`), the host-built epoch producers and warp-uniform root (`82d8493f`), the promoted `de5739c9` tree.
- **kshitij-hash** (co-author): the promoted `d052bc3d` composition and its rebuilt native image.
- **fkiene** (co-author): the GLV11 P18 five-term chain with the per-warp Q-layout mix (`413f83e7`).
- **HyeokxC** (co-author): blocking GPU waits with the host core given to the co-grinder (`0735233a`, `888f5fce`).
- **Meganpark980320** (co-author): `QSB_SHA_FMA_ADD=0` on this GPU tree (`bb2a3eb7`), the shorter IFMA reduction.
- **newjordan** (co-author): beneath the base, the `d1ddefca` GLV12 native-carrier tree and the warp root inverse.
- **Through the base:** i34-9, Ryun1, our `933abead`, Akashneelesh's crown `7aef224a` and every contributor it credits. libsecp256k1 (MIT, `COPYING-secp256k1`).
- **Ours:** the promoted `9f8a33d8` tree beneath `d052bc3d` (`QSB_SHA_FMA_ADD=0` composition, the producers' 16-lane path), the host-core worker, the ranked GPU/CPU hit-split analysis behind this composition, and the composition.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes. Kill switches: `-DQSB_HOST_BLOCKING=0` (spin wait; then one extra worker on the host core's sibling, `QSB_CPU_NOEXTRA` off), `-DQSB_CPU_GRIND=0`, `-DQSB_HOST_PRODUCERS=0`.
