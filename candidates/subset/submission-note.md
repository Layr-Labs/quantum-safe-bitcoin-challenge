# Subset: the promoted `521075fe` GPU side and host loop (kshitij-hash's paired single-reduction GLV11 chain `QSB_Y_PAIR`, the completed-slot snapshot that queues the next GPU work before the exact host gate) with our fastest ranked host side: terrapinelf's r7 co-grinder with a memory-gated 9-window table on the v3 producer code kept on three floating threads, blocking GPU waits with the host core for the co-grinder, and a run-time calibrated 16-lane AVX-512 SHA-256

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on an RTX 4090 + i7-13700K host (CUDA 12.8.93). AVX-512 code was executed with Intel SDE 9.48 (`-spr`). Our host has no AVX-512.

## Starting point

Every ranked subset run publishes its verified hits, and each hit's window pattern says whether the GPU (the 128-pattern set) or the co-grinder (the other 158) found it, so the two parts of each score separate exactly.

| ranked run | GPU side | host side | GPU M/s | CPU M/s | score |
|---|---|---|---:|---:|---:|
| `d052bc3d` (promoted record) | GLV11 P18 | Meganpark980320 lane, spinning host | 643.32 | 48.31 | 691.63 |
| our `a141df2b` | GLV11 P18 (`d052bc3d`'s image) | r7 lane + 9 windows, v3 producer code on floating threads, blocking waits + host core | 634.26 | **62.17** | 696.43 |
| kshitij-hash `8f99a3e9` | GLV11 + `QSB_Y_PAIR` | r7 engine, record placement, blocking waits | 637.13 | 57.47 | 694.60 |
| cefika `bf001729` | GLV11 + `QSB_Y_PAIR` + `QSB_Q_MIX` 2 (cubin `f7454842…`) | our `bfe57794` host side | 637.61 | 60.16 | 697.77 |
| ssalmeock `8c3822c4` (copy of `bf001729`) | the same | the same | 637.63 | 60.70 | 698.33 |
| RealAdii `521075fe` (promoted) | GLV11 + `QSB_Y_PAIR`, `QSB_Q_MIX` 4, completed-slot snapshot (cubin `003e3d39…`) | its own co-grinder | **642.62** | 58.33 | 700.95 |

- **GPU.** The `QSB_Y_PAIR` image drew 637.1–637.6 three times; runs of `d052bc3d`'s image in the same hours drew 630.5–636.7 (our `a141df2b` 634.26). kshitij-hash's chain carries a pair through each addition and reduces once (`QSB_Y_PAIR`, with `QSB_SC_PARK` for register room): one reduction and one product fewer per chain addition.
- **CPU.** Our `a141df2b` host side drew the fastest co-grinder part of any ranked run, 62.17 M/s; `bf001729` carries our earlier `bfe57794` host side (60.2–60.7 M/s).

**This package is the promoted `521075fe`'s device side and host loop with our `a141df2b` host side:** every device file and `tests/gpu_epochs/tree.cu` are `521075fe`'s (the `QSB_Y_PAIR` chain with `QSB_Q_MIX` 4; in the host loop, a completed slot's hit data is copied out before its replacement readback is queued, and the unchanged exact host gate runs after the next GPU work is launched), and `build_carrier.sh` with CUDA 12.8.93 regenerates its native image byte for byte: cubin sha256 `003e3d39b7a6283f…`, 0 spills. The only change to that `tree.cu` is the one-line statistics call of the v3 producers. The host side is our `a141df2b`'s, plus the 16-lane hashing below.

## The host side (from our `a141df2b`, measured at 62.17 M/s)

- **Co-grinder:** terrapinelf's r7 lane from `2d1631b0` (weighted batch-affine prefix, L2-targeted row prefetch, the first two windows' rows fetched while hashing, safegcd inversion, split fold) with our additions: a 9-window signed table (114,688 MiB, 8 additions per candidate) when at least about 236 GiB are available and the table is fully backed by transparent huge pages, falling back to 10, 11, 12 windows on any failure (the ranked host took 9 windows, table ready in 9.25 s, per the epoch-walk diagnostic of `a141df2b`); workers on every logical CPU including the GPU host thread's core.
- **Producers:** terrapinelf's v3 producer code (`host_producers.h` from `2d1631b0`, every compression at the SHA-NI floor) with `QSB_HP_PLACE` 0: three floating producer threads as in `82d8493f`. The runs with v3's pinned placement drew 622.8–627.6 on the GPU with a nearly empty batch ring on our host; the floating placement keeps the ring full.
- **Blocking waits:** the slot completion events use `cudaEventBlockingSync` (`QSB_HOST_BLOCKING`, host-only), so the GPU host thread sleeps between launches.
- **Diagnostic:** terrapinelf's epoch-walk start code, with bits 20..27 carrying the table-ready time in quarter seconds.

## 16-lane AVX-512 SHA-256, chosen on the host (new here)

The r7 lane hashes with 4-lane SHA-NI; hashing is about a third of its instructions (799 of 2,385 per candidate, SDE `-mix`, 11 windows). This package adds a 16-lane AVX-512F path for the second SHA-256, the h0-only key hashes (16 keys at a time, the same one-recid-per-candidate gate) and the 158 tail patterns of an epoch. It is used only where it measures faster on the ranked host: after the workers start, an ABBA of SHA-NI against 16-lane key hashes, then of adding 16-lane tails, 2 s per leg; kept only if at least 2% faster (`QSB_CPU_S16_MIN`), else SHA-NI runs as before. Checks: 102,400 blocks per round variant and 102,400 key h0s against OpenSSL and SHA-NI (0 mismatches, SDE); CPU hit sets identical in all four forced modes and across a mid-run mode switch (66 = 66, SDE, `QSB_ZEROS_N=12`); 2,332 instead of 2,385 instructions per candidate with both parts. `QSB_CPU_S16=0` forces SHA-NI.

## Validation of this exact package

| check | result |
|---|---|
| `build_carrier.sh`, CUDA 12.8.93 | cubin sha256 `003e3d39b7a6283f…`, 0 bytes stack, 0 spills: byte-identical to `521075fe`'s image |
| unmodified harness (`benchmark.sh subset`, `QSB_GRINDER=cmd:… gpu_wrap.py`), N = 24, 120 s, fresh seed | 12,225 / 12,225 verified, `RESULT: PASS` (the same host side on `bf001729`'s image: 120 s 12,317 / 12,317 and 1,200 s 123,707 / 123,707, `RESULT: PASS`) |
| 45 s live run on our host | `Native sm_89 carrier: on (… 003e3d39b7a6283f …)`; producers' self-check passed; `CPU co-grind: 24 threads (of 24 CPUs) …`; `[HP] final: host-built batches 278, GPU-built after start-up 1 (of 286); ready ahead at launch: avg 3.00, min 3`; GPU 868.1 M/s (855–857 with `d052bc3d`'s image) |

Our host runs the co-grinder's scalar path and the 11-window table; the IFMA path, the 9-window table and the 16-lane hashing were checked under SDE and, apart from the 16-lane hashing, drawn on the ranked host in `a141df2b`.

## Caveats

- The `QSB_Y_PAIR` GPU gain rests on four ranked draws (637.1–642.6) against a GLV11 band that varies by several M/s between draws of the same code.
- The 16-lane hashing has no ranked draw yet; the calibration keeps SHA-NI unless 16-lane is at least 2% faster.

## Base and attribution

- **kshitij-hash** (co-author): the paired single-reduction chain (`QSB_Y_PAIR`, `QSB_SC_PARK`, `8f99a3e9`) and the promoted `d052bc3d` composition with its rebuilt native image.
- **RealAdii** (co-author): the promoted `521075fe` (its `tree.cu` host-loop change and native image).
- **cefika** (co-author): the `bf001729` composition of the `QSB_Y_PAIR` chain with our host side.
- **fkiene** (co-author): the GLV11 P18 five-term chain with the per-warp Q-layout mix (`413f83e7`).
- **terrapinelf** (co-author): the r7 co-grinder and the v3 producer code (`2d1631b0`), `QSB_Q_MIX` 2 (`92a51c8c`), the host-built epoch producers and warp-uniform root (`82d8493f`), the promoted `de5739c9` tree.
- **HyeokxC** (co-author): blocking GPU waits with the host core for the co-grinder (`0735233a`, `888f5fce`).
- **Meganpark980320** (co-author): `QSB_SHA_FMA_ADD=0` on this GPU tree (`bb2a3eb7`), the shorter IFMA reduction, the 16-lane hashing measurement on Zen 4 (`296e5e53`).
- **newjordan** (co-author): beneath the base, the `d1ddefca` GLV12 native-carrier tree and the warp root inverse.
- **Through the base:** i34-9, Ryun1, our `933abead`, Akashneelesh's crown `7aef224a` and every contributor it credits. libsecp256k1 (MIT, `COPYING-secp256k1`).
- **Ours:** the host side of `a141df2b` (the 9-window table and its rule, the floating placement of the v3 producers, the host-core placement, the table-ready diagnostic), the 16-lane hashing for the r7 lane, the promoted `9f8a33d8` tree beneath `d052bc3d`, the ranked GPU/CPU hit-split analysis, and this composition.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes. Kill switches: `-DQSB_CPU_S16=0`, `-DQSB_CPU_TRY9=0`, `-DQSB_HOST_BLOCKING=0`, `-DQSB_CPU_GRIND=0`, `-DQSB_HOST_PRODUCERS=0`.
