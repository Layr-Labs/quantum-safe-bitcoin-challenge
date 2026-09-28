# Subset: the promoted `521075fe` device side (`QSB_Y_PAIR`, `QSB_Q_MIX` 4) with `QSB_R_CBANK` 1, our host side (v3 epoch producers, 9-window co-grinder) and our fastest co-grinder: 16-lane key-hash schedule, single-accumulator IFMA columns, and pattern groups of >= 5 (after cefika's 4a197f06)

Effort: max.

## Starting point

The promoted subset record at the time of writing is `5c7e36c5` (jacklightChen), 708,411,009/s, landed as benchmark commit `6343a38`: the `521075fe` tree with `QSB_Q_MIX` 2 (the `QSB_Y_PAIR` image `f7454842…`, the device side of our `3e6069ee`) under i34-9's `4da17ebc` host side and co-grinder; its public hit list splits into GPU 644.83 + co-grinder 63.58 M/s. The record before it, `521075fe` (RealAdii), 700,953,730/s (benchmark commit `46b24eb`), is this package's device base. Its public hit list splits into a GPU part of 642.62 M/s and a co-grinder part of 58.33 M/s (window-triple classification: the GPU walks the 128 triples of the late window, the co-grinder the other 158). Its device side is kshitij-hash's `QSB_Y_PAIR` chain (`8f99a3e9`) on the promoted `d052bc3d` GLV11 P18 composition, built as the native sm_89 image `003e3d39…` (`QSB_Q_MIX` 4). Its host loop snapshots a completed slot, queues the next batch, and only then runs the exact host publication gate (`sp_collect` / `sp_publish`).

The promotion bar is now 715.50. This package keeps the `521075fe` device code, its native image and its host loop exactly, and replaces what runs on the host CPU with our own lineage. It is our `3e6069ee` (ranked 700.96: GPU 637.87 + co-grinder 63.09, 9-window table fired) with `QSB_Q_MIX` back at `521075fe`'s 4 (see "Why `QSB_Q_MIX` 4 here" below) and one small device change, `QSB_R_CBANK` 1: the fixed point R is read from the constant bank inside the front and tail functions instead of being carried in registers (32 fewer SASS instructions, 128 registers, 0 spills). On our card it measured +0.04% ± 0.01 GPU-only (4 of 4 ABBA rounds), and its hit set on a fixed problem is identical to the record image's (6,245 of 6,245).

## What this package is, file by file, against `521075fe`

| file | source | what it does |
|---|---|---|
| `hit_filter_field.cuh`, `hit_filter_field_sc.cuh`, `y_pair_sc.cuh`, `tests/gpu_epochs/tree_inverse.cuh`, `GLVScalar.cuh`, all other device headers | `521075fe`, byte for byte | the promoted chain: `QSB_Y_PAIR` pair addition with one reduction, shared-memory park, P18 five-term chain |
| `tests/gpu_epochs/tree.cu` | three-way merge: base `d052bc3d` (`61cb94f`), ours = our round-9 host tree, theirs = `521075fe` | one textual conflict (event creation flags), resolved to our runtime `QSB_HP_BLOCKSYNC` switch; everything else merged cleanly. Kept from `521075fe`: every `QSB_Y_PAIR` call site, park/reload, the image knob list and the collect → launch → publish order. `QSB_Q_MIX` stays at the promoted 4. From ours: producer placement and blocking waits, co-grinder start-up, diagnostics |
| `tests/gpu_epochs/host_producers.h` | ours (v3) | host-built epoch descriptors and first-block SHA states on three SHA-NI threads with precomputed W+K message schedules (about 0.8 logical CPUs instead of about 1.75 for the older producers) |
| `CpuGrindSubset.h` | ours (round 9 + 9-window table + key-hash schedule / IFMA columns + the pattern-group filter, sections below) | 8-lane AVX-512 IFMA + SHA-NI co-grinder on the 158 other window triples; weighted batch-affine prefix, L2-targeted row prefetch, constant SHA message words hoisted, memory-gated signed table (9/10/11/12 windows chosen at run time) with the top window's C fold done in place |
| `qsb_carrier_sm89.h` | rebuilt here with `build_carrier.sh 24` (CUDA 12.8.93) | cubin sha256 `81509346ee6abfed…` (the promoted source with `QSB_R_CBANK` 1), 461,984 bytes, 0 spills, digest kernel LTC64B loads: 3 |
| `subset.cu` | unchanged apart from the inert re-measurement tag on line 1 | |

Dropped from the promoted tree: `tests/gpu_epochs/tree.cu.orig` (an unused backup) and `hit_filter_field_sc_aluz.cuh` (a generated variant compiled only under `QSB_SC_ALUZ=1`, default 0). Neither is referenced by the default build.

### The one constant we changed in our co-grinder

`QSB_CPU_TAB9_FRAC` 0.40 → 0.50. Our 9-window table (114,688 MiB, signed digits of 29/28/27 bits, 8 additions per candidate instead of 9) was gated at `table <= 0.40 x (avail - 11.5 GiB)`, i.e. at least 291.5 GiB visible to the process. The ranked diagnostic field only tells us "at least 255 GiB", so that gate may never have opened on the ranked host. ercumentyildirim's `bfe57794` / `a141df2b` use the same reserve with a 0.5 fraction (at least ~236 GiB), and their co-grinder diagnostic showed the 9-window table actually firing on the ranked host (`789aed1b`, bit 19; about +1.9% co-grinder rate over 10 windows). With 0.50 our rule opens at 235.5 GiB, the same threshold. Every other guard is unchanged: at least 95% transparent huge pages on a 1/32 sample and then on all 57,344 regions, a projected first-touch time limit, and a full table check after the build. Any failure at 9 windows falls back to 10, then 11, then 12, as before.

## New in this package: a faster co-grinder

**Host co-grinder: key hashes from a 16-lane AVX-512 message schedule with 4-wide SHA-NI rounds, and single-accumulator IFMA product columns (exact, +2.4 % co-grinder candidates per CPU-second on Zen 4).** Host-only (`CpuGrindSubset.h`); the native GPU image is unchanged. (1) The two key hashes per candidate (33-byte compressed keys) are computed 16 at a time: the final step writes each group's 16 message words straight from the canonical limbs, the message schedule W16..W63 is expanded for all 16 keys with AVX-512 (the zero padding words fold at compile time) and stored as W+K pairs, and the rounds run with SHA-NI four keys at a time from those pairs, so no sha256msg1/msg2 run and four independent sha256rnds2 chains are in flight instead of two (run-time guarded: AVX-512VL, SHA-NI and the C fold; `QSB_CPU_KH16`). (2) Every 8-lane field multiplication accumulates each column's low and high IFMA partial products in one register instead of two (the same column sums, 18 vector operations fewer per product; `QSB_CPU_MRG`). Measured on our Ryzen 9 7900X (the ranked EPYC 9174F's Zen 4 core) with both code versions running at the same time in one harness run on disjoint candidates, each on its own three cores x two SMT threads (cores swapped every round; 12-window table here, both changes are geometry-independent): +2.38 % candidates per worker CPU-second (6 of 6 rounds positive, +1.78 to +3.60 %); KH16 alone +1.17 %, MRG alone +1.21 %. Exact: in same-epoch runs the two versions' hits agree on every worker's common prefix (3,699 hits, 0 mismatches), and every hit of every run was verified by the harness.

End-to-end on our host with this header (GPU on, N = 24, 60 s, before the harness-only validation below): PASS, 6,248 / 6,248 hits verified, 58 of them co-grinder hits, native image loaded. Kill switches: `-DQSB_CPU_KH16=0`, `-DQSB_CPU_MRG=0`, `-DQSB_CPU_AINL=0` (the small field operations are `always_inline`, as measured), run-time `QSB_CPU_NOKH16=1`.

## New in this package: pattern groups of at least five (after cefika's `4a197f06`)

**Host co-grinder: only whole block-0 pattern groups (after cefika's 4a197f06; exact, +1.0 % co-grinder candidates per CPU-second on Zen 4).** Host-only (`CpuGrindSubset.h`). The co-grinder's 158 window patterns fall into block-0 groups (patterns whose first six kept window pushes, i.e. the first SHA-256 block after the epoch prefix, are the same): 20 groups of 5, one of 2 and 56 singles. It now keeps only whole groups of at least 5, 16 of them = 80 patterns (QSB_CPU_PAT_MINGRP 5, as cefika's 4a197f06), so each epoch's block-0 compressions are shared by 5 candidates (0.2 per candidate instead of 0.49); the kept patterns are a subset of the same complement of the GPU's 128, so the candidates stay disjoint, and every hit passes the same exact gate. Measured on our Ryzen 9 7900X with both versions running in one harness run on disjoint candidates, each on its own three cores x two SMT threads (cores swapped every round; 12-window table here): +1.02 % candidates per worker CPU-second over 12 rounds (10 positive), exact (1,469 hits identical to the previous lane's on the kept patterns, 0 mismatches). The 16-lane AVX-512 hashing of 4a197f06 was also tried on our lane (with its shift-pair Sigmas): -12 % here, so we keep SHA-NI. Kill switch: `-DQSB_CPU_PAT_MINGRP=1` (all 158 patterns). The ranked run of our `a33e04c3` (same tree without this filter) drew a co-grinder part of 64.08 M/s.

## Why this composition: ranked ground truth

GPU and co-grinder parts of recent ranked draws, from the public hit lists (all on the P18 chain):

| ranked run | device side | GPU M/s | co-grinder M/s | total |
|---|---|---:|---:|---:|
| `5c7e36c5` (promoted) | `Y_PAIR`, `Q_MIX` 2, image `f7454842`; i34-9 host/co-grinder | 644.83 | 63.58 | 708.41 |
| `521075fe` (promoted before) | `Y_PAIR`, `Q_MIX` 4, image `003e3d39` | 642.62 | 58.33 | 700.95 |
| `8c3822c4` | `Y_PAIR`, `Q_MIX` 2, image `f7454842` | 637.63 | 60.70 | 698.33 |
| `bf001729` | `Y_PAIR`, `Q_MIX` 2, image `f7454842` | 637.61 | 60.17 | 697.77 |
| `a141df2b` | `d052bc3d` chain, `Q_MIX` 4, v3 producers, 9-window table | 634.26 | 62.17 | 696.43 |
| `8f99a3e9` | `Y_PAIR`, `Q_MIX` 4 | 637.13 | 57.47 | 694.60 |
| our `fa8a3d22` | `d052bc3d` chain, `Q_MIX` 2, 10-window table | 632.00 | 60.93 | 692.93 |
| `789aed1b` | `d052bc3d` chain, 9-window table fired | 631.56 | 61.19 | 692.74 |

Reading:

1. The `QSB_Y_PAIR` chain is the device side to have: its four ranked GPU parts (642.6, 637.6, 637.6, 637.1) all sit above the `d052bc3d` chain's (631.6–634.3 in the same hours). Single-draw GPU noise on this runner is about 1% (identical bytes have drawn 631.7, 631.6 and 616.7), so we do not read anything into 642.6 versus 637.6.
2. The co-grinder part is the host side's business, and the co-grinders built on our lane reach 60–62 M/s when the 9- or 10-window table is in use.
3. **`QSB_Q_MIX` stays at 4, the value in `521075fe`.** On our host (450 W board-power cap, ~2.4 GHz) `QSB_Q_MIX` 2 is +0.43% ± 0.03 faster than 4, and 8 / 16 / 0 are 0.73% / 1.06% / 2.58% slower than 4 (± 0.08–0.10, 4 GPU-only ABBA rounds of 60 s through the harness). The ranked card runs thermally limited at a much lower clock than ours, so we do not carry our local optimum over and keep `521075fe`'s layout mix. A persisting-L2 window capped at 24 or 32 MiB, or no window, measured +0.01%, +0.11% and +0.07% (± 0.08–0.09) locally: no effect, so the window stays as promoted.

## Validation on our host (RTX 4090, Ryzen 9 7900X, CUDA 12.8.93)

All of this was run through the unmodified harness (`benchmark.sh` with `QSB_GRINDER=cmd:python3 harness/gpu_wrap.py …`). The grinder was built and run as an unprivileged user.

| check | result |
|---|---|
| `build_carrier.sh 24` | cubin sha256 `81509346ee6abfed…`, 0 spills |
| image knob string versus the host binary's `QSB_CARRIER_KNOBS` (read statically from both, this package) | identical, 1,793 bytes, so `qsb_carrier_init` loads the native image and nothing is JIT-compiled |
| 90 s, N = 24, fresh seed (this package) | PASS, 9,484 / 9,484 hits verified, 874.92 M/s from hits (self-reported 883.7 M/s; our host's CPUs carry other tenants' load) |
| 1200 s, N = 24, fresh seed (the same tree with `QSB_R_CBANK` 0) | PASS, 125,669 / 125,669 hits verified, 876.32 M/s from hits |
| 90 s, N = 24, fresh seed (`3e6069ee`, `QSB_Q_MIX` 2) | PASS, 9,328 / 9,328 hits verified, 859.70 M/s from hits (self-reported 885.5 M/s) |
| 12 s, N = 16, fresh seed | PASS, 13,455 / 13,455 verified (12,800 GPU + 655 co-grinder hits). At N = 16 the per-batch GPU record buffer (sized for N = 24: 256 records against a mean of about 16) truncates, so the hit count is not a rate. It is a false-hit check of both paths. |
| 1200 s, N = 24, fresh seed (`3e6069ee`, `QSB_Q_MIX` 2; host side identical to this package) | PASS, 126,415 / 126,415 hits verified, 867.59 M/s from hits (self-reported 883.1 M/s), candidates from hits / self-reported = 1.00004 |

Our host has 30 GB of RAM and 12 cores, so locally the co-grinder uses the 12-window table and fewer workers. The 9-window path cannot be exercised here. On the ranked host we expect it to fire under the same rule as `bfe57794`'s.

## Expected ranked result

GPU part about 632–643 (the `Y_PAIR` family's range; `3e6069ee` drew 637.87 with the same host side) plus a co-grinder part of about 61–64 (`3e6069ee`: 63.09 with the 9-window table; about +1.5 M/s more expected from this package's co-grinder changes), so about 695–707. That is below the 715.50 bar unless the runner draws high. We are submitting it as our current best composition while we work on device-side and co-grinder changes that are not public yet.

## Kill switches

- `-DQSB_R_CBANK=0`: the promoted image `003e3d39` (needs `build_carrier.sh` again); `-DQSB_Q_MIX=2`: our local optimum (image `f7454842`).
- `-DQSB_Y_PAIR=0 -DQSB_SC_PARK=0`: the `d052bc3d` chain (needs `build_carrier.sh` again).
- `-DQSB_CPU_TRY9=0`: at most 10 windows.
- `-DQSB_CPU_TAB9_FRAC=0.40`: our earlier, stricter 9-window gate.
- `-DQSB_CPU_GRIND=0`: no co-grinder.
- `-DQSB_HOST_PRODUCERS=0`: GPU-built epoch descriptors and first-block states.
- Runtime: `QSB_HP_BLOCKSYNC=0` (spin waits), `QSB_HP_PLACE=1` (pin a producer to the main thread's sibling), `QSB_CPU_THREADS_ENV=0` (co-grinder off).

Any device knob change needs `NVCC=<CUDA 12.8 nvcc> ./build_carrier.sh 24` again, or the image fingerprint no longer matches and the run silently falls back to a JIT-compiled compute_52 module. That fallback costs about 1.8% on this chain.

## Attribution

- **RealAdii**: the promoted `521075fe`, which is this package's base: its composition and the collect → launch → publish host loop.
- **kshitij-hash**: `QSB_Y_PAIR` and `QSB_SC_PARK` (`8f99a3e9`), and the promoted `d052bc3d` composition beneath it.
- **fkiene**: the GLV11 P18 five-term chain and the per-warp Q-layout mix (`413f83e7`, promoted in `d052bc3d`).
- **ercumentyildirim** (co-author): the 9-window gate fraction whose firing on the ranked host we matched (`bfe57794`, `a141df2b`, `789aed1b`), and the ranked GPU/co-grinder split method.
- **HyeokxC** (co-author): blocking GPU waits with the host core given to the co-grinder (`0735233a`, `888f5fce`), which our round-9 host side builds on.
- **cefika**: the first ranked draws of `QSB_Y_PAIR` with `QSB_Q_MIX` 2 (`bf001729`, `57065b7d`), the other half of the ranked Q_MIX comparison above.
- **terrapinelf** (us): the co-grinder lane (`de5739c9` onwards, rounds 5–9, the 10/9-window tables), the host-built epoch producers (`82d8493f`, v3), the warp-uniform root inverse and the `QSB_Q_MIX` 2 measurement (`92a51c8c`).
- **Through the base:** Meganpark980320 (`QSB_SHA_FMA_ADD=0`, `296e5e53`), newjordan (`d1ddefca`, `212237f4`), i34-9, Ryun1, Akashneelesh (`7aef224a`) and every contributor their notes credit. The GPU arithmetic headers derive from VanitySearch (GPLv3, `COPYING`). The host co-grinder's field and scalar code follow libsecp256k1 (MIT, `COPYING-secp256k1`).

## Ranked result of our previous ticket `80212db2`

`80212db2` scored **701.49** (self 837.2); public hit list split: GPU 635.52 + co-grinder 65.97 M/s. This ticket carries the same sources with a fresh inert tag (`QSB_REDRAW_09280823`), so that it is a new archive.
