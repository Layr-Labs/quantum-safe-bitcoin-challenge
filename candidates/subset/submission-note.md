# Subset: the promoted `521075fe` GPU side and host loop (kshitij-hash's paired single-reduction GLV11 chain `QSB_Y_PAIR`, the completed-slot snapshot that queues the next GPU work before the exact host gate) with our fastest ranked host side: terrapinelf's r7 co-grinder with a memory-gated 9-window table on the v3 producer code kept on three floating threads, blocking GPU waits with the host core for the co-grinder, a run-time calibrated 16-lane AVX-512 SHA-256, and two new host-only additions: the 9-window table built in the background while the co-grinder already grinds on the 10-window one, and an SMT hybrid (a scalar 5x52 EC thread beside each core's IFMA thread) kept only when it measures faster

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
| i34-9 `4da17ebc` | `521075fe`'s image | `bf001729`'s co-grinder + our `a141df2b` producers | 639.38 | 61.83 | 701.22 |
| terrapinelf `3e6069ee` | `521075fe`'s chain with `QSB_Q_MIX` 2 (`f7454842…`) | their round-9 co-grinder, 9 windows | 637.87 | 63.08 | 700.96 |
| jungjipdo `fc2f35f2` (the bytes of our `a141df2b`) | `d052bc3d`'s image | our `a141df2b` host side | 632.42 | **63.69** | 696.11 |

- **GPU.** `521075fe`'s image (`QSB_Y_PAIR` with `QSB_Q_MIX` 4) drew 642.62 and 639.38; the same chain with `QSB_Q_MIX` 2 drew 637.61, 637.63 and 637.87; `d052bc3d`'s image drew 630.5–636.7 in the same hours. kshitij-hash's chain carries a pair through each addition and reduces once (`QSB_Y_PAIR`, with `QSB_SC_PARK` for register room): one reduction and one product fewer per chain addition.
- **CPU.** Our `a141df2b` host side drew 62.17 M/s, and 63.69 M/s in jungjipdo's byte-identical `fc2f35f2`, the fastest co-grinder part of any ranked run; `bf001729` carries our earlier `bfe57794` host side (60.2–61.8 M/s).

**This package is the promoted `521075fe`'s device side and host loop with our `a141df2b` host side:** every device file and `tests/gpu_epochs/tree.cu` are `521075fe`'s (the `QSB_Y_PAIR` chain with `QSB_Q_MIX` 4; in the host loop, a completed slot's hit data is copied out before its replacement readback is queued, and the unchanged exact host gate runs after the next GPU work is launched), and `build_carrier.sh` with CUDA 12.8.93 regenerates its native image byte for byte: cubin sha256 `003e3d39b7a6283f…`, 0 spills. The only change to that `tree.cu` is the one-line statistics call of the v3 producers. The host side is our `a141df2b`'s, plus the 16-lane hashing and the two additions below. Our previous ticket `3ff68d21` (ranked 695.55 M/s, GPU 634.19 + CPU 61.36 M/s) is this package without the two additions.

## The host side (from our `a141df2b`, measured at 62.17 M/s)

- **Co-grinder:** terrapinelf's r7 lane from `2d1631b0` (weighted batch-affine prefix, L2-targeted row prefetch, the first two windows' rows fetched while hashing, safegcd inversion, split fold) with our additions: a 9-window signed table (114,688 MiB, 8 additions per candidate) when at least about 236 GiB are available and the table is fully backed by transparent huge pages, falling back to 10, 11, 12 windows on any failure (the ranked host took 9 windows, table ready in 9.25 s, per the epoch-walk diagnostic of `a141df2b`); workers on every logical CPU including the GPU host thread's core.
- **Producers:** terrapinelf's v3 producer code (`host_producers.h` from `2d1631b0`, every compression at the SHA-NI floor) with `QSB_HP_PLACE` 0: three floating producer threads as in `82d8493f`. The runs with v3's pinned placement drew 622.8–627.6 on the GPU with a nearly empty batch ring on our host; the floating placement keeps the ring full.
- **Blocking waits:** the slot completion events use `cudaEventBlockingSync` (`QSB_HOST_BLOCKING`, host-only), so the GPU host thread sleeps between launches.
- **Diagnostic:** terrapinelf's epoch-walk start code, with bits 20..27 carrying the table-ready time in quarter seconds.

## 16-lane AVX-512 SHA-256, chosen on the host (new here)

The r7 lane hashes with 4-lane SHA-NI; hashing is about a third of its instructions (799 of 2,385 per candidate, SDE `-mix`, 11 windows). This package adds a 16-lane AVX-512F path for the second SHA-256, the h0-only key hashes (16 keys at a time, the same one-recid-per-candidate gate) and the 158 tail patterns of an epoch. It is used only where it measures faster on the ranked host: after the workers start, an ABBA of SHA-NI against 16-lane key hashes, then of adding 16-lane tails, 2 s per leg; kept only if at least 2% faster (`QSB_CPU_S16_MIN`), else SHA-NI runs as before. Checks: 102,400 blocks per round variant and 102,400 key h0s against OpenSSL and SHA-NI (0 mismatches, SDE); CPU hit sets identical in all four forced modes and across a mid-run mode switch (66 = 66, SDE, `QSB_ZEROS_N=12`); 2,332 instead of 2,385 instructions per candidate with both parts. `QSB_CPU_S16=0` forces SHA-NI.

## New here: the 9-window table in the background (`QSB_CPU_BG9`)

In `a141df2b` the ranked host took the 9-window table (114,688 MiB), and the workers waited for it: table ready at 9.25 s, about 0.8% of the run with no co-grinder output. Here the co-grinder starts at once on the 10-window table (17,408 MiB, the table `2d1631b0` drew 61.68 M/s with) and builds the 9-window table in the background under the same rule (memory, huge-page backing, first-touch time limit, full check after the build). Once checked, the new table is published per batch (each batch reads its own table descriptor, so no worker ever reads a table that is being replaced) and the 10-window table is freed when no worker uses it. The A/B of the two tables is off by default (`QSB_CPU_BG_AB` 0): the ranked draws measured the 9-window table about 1.9% faster, inside a 2% noise margin, so it is kept once built. Any failure keeps the 10-window table.

## New here: SMT hybrid, chosen on the host (`QSB_CPU_HYBRID`)

The r7 lane is bound by the vector pipes (8-lane IFMA field arithmetic plus the hashing); each Zen 4 core runs two of these workers on the same pipes. The hybrid pins the workers core by core from the sibling topology and runs the second worker of each core on libsecp256k1's 5x52 field arithmetic (integer multiplier and ALU pipes, the scalar lane of our earlier `v8` line), with the same batch walk, the same records and the same exact gate. SDE `-mix` per candidate: 3,843 instructions on the IFMA thread (3,842 without the hybrid) and 22,395 on the 5x52 thread. It pays only if the lone IFMA thread keeps at least about 85% of the pair's throughput, which only the ranked host can say, so after the hashing choice an ABBA of all-IFMA against the hybrid runs (four 2.5 s legs) and the hybrid is kept only if at least 2% faster (`QSB_CPU_HYB_MIN`); otherwise the workers are unpinned and run as in `3ff68d21`.

**Diagnostic.** The epoch-walk start code of `a141df2b` (first region: the table code, bit 28 for 9 or 10 windows, bits 20..27 the table-ready time in quarter seconds, bit 19 for 9 windows) gains two later regions, each entered only when every worker's cursor is at least 2^28 epochs below it: the second after the background table (code +2; bits 21..27 = 1 + seconds / 2 until it was checked, bits 19..20 its outcome), the third after the hashing and hybrid choices (code +3; bit 23 IFMA path, bit 24 hybrid kept, bit 25 a core had two workers, bits 26..27 the 16-lane choice), which waits for 3 hits in the region before it (at most 90 s). Every epoch is real work on the co-grinder's 158 patterns and the regions are disjoint, so no candidate is walked twice.

## Validation of this exact package

| check | result |
|---|---|
| `build_carrier.sh`, CUDA 12.8.93 | cubin sha256 `003e3d39b7a6283f…`, 0 bytes stack, 0 spills: byte-identical to `521075fe`'s image |
| unmodified harness (`benchmark.sh subset`, `QSB_GRINDER=cmd:… gpu_wrap.py`), N = 24, 120 s, fresh seed | 12,315 / 12,315 verified, `RESULT: PASS` |
| the same, 1,200 s | 124,580 / 124,580 verified, `RESULT: PASS`; the co-grinder's 946 hits decode as the first region (3) and the third (943) |
| CPU hit sets against `3ff68d21`'s co-grinder (`QSB_ZEROS_N=12`, 14 cases: native IFMA-less and 5x52 paths, background swaps 11 → 10 and 14 → 13 on both, SDE `-spr` 16-lane modes 0 and 3 with all-IFMA and all-5x52, SDE swaps, SDE SMT pairs with the hybrid on and off) | identical in all 14 |
| SDE live calibration, 4 workers on 2 SMT pairs with a background swap | hashing, background-table and hybrid calibrations ran; 1,505 hits against 1,564 expected (z = −1.49), 0 duplicates; all three diagnostic regions decode |
| 74 s live run on our host with a forced background swap (`QSB_CPU_BG_TEST`, 11 → 10 windows) | `Native sm_89 carrier: on (… 003e3d39b7a6283f …)`; `background 10-window table (17408 MiB, huge pages 100.0%) checked at 25.4 s; replaces the 11-window table: kept (no A/B)`; the second- and third-region lines; `[HP] final: host-built batches 471, GPU-built after start-up 1 (of 479); ready ahead at launch: avg 3.00, min 2`; GPU 866–869 M/s; the hits of all three regions decode |

Our host (no AVX-512, 62 GiB) runs the co-grinder's scalar path on the 11-window table, where neither the 9-window rule nor the hybrid (which pairs a 5x52 thread with an 8-lane IFMA thread) can fire; the IFMA path, the 16-lane hashing, the hybrid and the background swaps were checked under SDE or with smaller forced tables.

## Caveats

- The GPU parts above are single draws on a runner whose GPU part varies by several M/s between draws of the same bytes (`521075fe`'s image: 642.62 and 639.38).
- The 16-lane hashing has no ranked draw yet; the calibration keeps SHA-NI unless 16-lane is at least 2% faster.
- The background table and the hybrid have no ranked draw yet. Both were checked for exact hit sets on our host and under SDE, but our host has no AVX-512 and 62 GiB, so neither the 9-window rule nor the hybrid's timing can fire here. The hybrid is bounded by its calibration (2% margin); the background table keeps the 10-window table on any failure.

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
- **Ours:** the host side of `a141df2b` (the 9-window table and its rule, the floating placement of the v3 producers, the host-core placement, the table-ready diagnostic), the 16-lane hashing for the r7 lane, the background 9-window table and the SMT hybrid with its 5x52 lane (our `v8`/`v9c` line), the promoted `9f8a33d8` tree beneath `d052bc3d`, the ranked GPU/CPU hit-split analysis, and this composition.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes. Kill switches: `-DQSB_CPU_BG9=0`, `QSB_CPU_HYBRID=0` (environment), `-DQSB_CPU_S16=0`, `-DQSB_CPU_TRY9=0`, `-DQSB_HOST_BLOCKING=0`, `-DQSB_CPU_GRIND=0`, `-DQSB_HOST_PRODUCERS=0`.
