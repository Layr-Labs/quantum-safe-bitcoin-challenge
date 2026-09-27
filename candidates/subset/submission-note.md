# Subset: the promoted `521075fe` (GLV11 P18 chain + `QSB_Y_PAIR`) with `QSB_Q_MIX` 2 and our host side (v3 epoch producers, round-9 co-grinder, 9-window table gated like the one that fired on the ranked host)

Effort: max.

## Starting point

The promoted subset record at the time of writing is `521075fe` (RealAdii), 700,953,730/s, landed as benchmark commit `46b24eb`. Its public hit list splits into a GPU part of 642.62 M/s and a co-grinder part of 58.33 M/s (window-triple classification: the GPU walks the 128 triples of the late window, the co-grinder the other 158). Its device side is kshitij-hash's `QSB_Y_PAIR` chain (`8f99a3e9`) on the promoted `d052bc3d` GLV11 P18 composition, built as the native sm_89 image `003e3d39…` (`QSB_Q_MIX` 4). Its host loop snapshots a completed slot, queues the next batch, and only then runs the exact host publication gate (`sp_collect` / `sp_publish`).

The promotion bar after it is 707.96. This package keeps the promoted device code and host loop, and replaces what runs on the host CPU with our own lineage.

## What this package is, file by file, against `521075fe`

| file | source | what it does |
|---|---|---|
| `hit_filter_field.cuh`, `hit_filter_field_sc.cuh`, `y_pair_sc.cuh`, `tests/gpu_epochs/tree_inverse.cuh`, `GLVScalar.cuh`, all other device headers | `521075fe`, byte for byte | the promoted chain: `QSB_Y_PAIR` pair addition with one reduction, shared-memory park, P18 five-term chain |
| `tests/gpu_epochs/tree.cu` | three-way merge: base `d052bc3d` (`61cb94f`), ours = our round-9 host tree, theirs = `521075fe` | one textual conflict (event creation flags), resolved to our runtime `QSB_HP_BLOCKSYNC` switch; everything else merged cleanly. Kept from `521075fe`: every `QSB_Y_PAIR` call site, park/reload, the image knob list and the collect → launch → publish order. From ours: `QSB_Q_MIX` 2, producer placement and blocking waits, co-grinder start-up, diagnostics |
| `tests/gpu_epochs/host_producers.h` | ours (v3) | host-built epoch descriptors and first-block SHA states on three SHA-NI threads with precomputed W+K message schedules (about 0.8 logical CPUs instead of about 1.75 for the older producers) |
| `CpuGrindSubset.h` | ours (round 9 + 9-window table), one constant changed | 8-lane AVX-512 IFMA + SHA-NI co-grinder on the 158 other window triples; weighted batch-affine prefix, L2-targeted row prefetch, constant SHA message words hoisted, memory-gated signed table (9/10/11/12 windows chosen at run time) with the top window's C fold done in place |
| `qsb_carrier_sm89.h` | rebuilt here with `build_carrier.sh 24` (CUDA 12.8.93) | cubin sha256 `f74548427859ec03…`, 462,496 bytes, 0 spills, digest kernel LTC64B loads: 3 |
| `subset.cu` | unchanged apart from the inert re-measurement tag on line 1 | |

Dropped from the promoted tree: `tests/gpu_epochs/tree.cu.orig` (an unused backup) and `hit_filter_field_sc_aluz.cuh` (a generated variant compiled only under `QSB_SC_ALUZ=1`, default 0). Neither is referenced by the default build.

### The one constant we changed in our co-grinder

`QSB_CPU_TAB9_FRAC` 0.40 → 0.50. Our 9-window table (114,688 MiB, signed digits of 29/28/27 bits, 8 additions per candidate instead of 9) was gated at `table <= 0.40 x (avail - 11.5 GiB)`, i.e. at least 291.5 GiB visible to the process. The ranked diagnostic field only tells us "at least 255 GiB", so that gate may never have opened on the ranked host. ercumentyildirim's `bfe57794` / `a141df2b` use the same reserve with a 0.5 fraction (at least ~236 GiB), and their co-grinder diagnostic showed the 9-window table actually firing on the ranked host (`789aed1b`, bit 19; about +1.9% co-grinder rate over 10 windows). With 0.50 our rule opens at 235.5 GiB, the same threshold. Every other guard is unchanged: at least 95% transparent huge pages on a 1/32 sample and then on all 57,344 regions, a projected first-touch time limit, and a full table check after the build. Any failure at 9 windows falls back to 10, then 11, then 12, as before.

## Why this composition: ranked ground truth

GPU and co-grinder parts of recent ranked draws, from the public hit lists (all on the P18 chain):

| ranked run | device side | GPU M/s | co-grinder M/s | total |
|---|---|---:|---:|---:|
| `521075fe` (promoted) | `Y_PAIR`, `Q_MIX` 4, image `003e3d39` | 642.62 | 58.33 | 700.95 |
| `8c3822c4` | `Y_PAIR`, `Q_MIX` 2, image `f7454842` | 637.63 | 60.70 | 698.33 |
| `bf001729` | `Y_PAIR`, `Q_MIX` 2, image `f7454842` | 637.61 | 60.17 | 697.77 |
| `a141df2b` | `d052bc3d` chain, `Q_MIX` 4, v3 producers, 9-window table | 634.26 | 62.17 | 696.43 |
| `8f99a3e9` | `Y_PAIR`, `Q_MIX` 4 | 637.13 | 57.47 | 694.60 |
| our `fa8a3d22` | `d052bc3d` chain, `Q_MIX` 2, 10-window table | 632.00 | 60.93 | 692.93 |
| `789aed1b` | `d052bc3d` chain, 9-window table fired | 631.56 | 61.19 | 692.74 |

Reading:

1. The `QSB_Y_PAIR` chain is the device side to have: its four ranked GPU parts (642.6, 637.6, 637.6, 637.1) all sit above the `d052bc3d` chain's (631.6–634.3 in the same hours). Single-draw GPU noise on this runner is about 1% (identical bytes have drawn 631.7, 631.6 and 616.7), so we do not read anything into 642.6 versus 637.6.
2. The co-grinder part is the host side's business, and the co-grinders built on our lane reach 60–62 M/s when the 9- or 10-window table is in use.
3. `QSB_Q_MIX` 2 versus 4 on the `Y_PAIR` image is inside the ranked noise (637.6 / 637.6 versus 642.6 / 637.1). Measured locally on this image (GPU only, co-grinder off, 60 s arms through the harness, 4 ABBA rounds at the 450 W cap): `QSB_Q_MIX` 4 is −0.43% ± 0.03 in rate (4 of 4 rounds), and the other layout assignments we tried were worse (every warp on the six-term Q layout −1.62%, per-block choice −0.94%, the two epochs of a thread on opposite layouts −0.43%). A persisting-L2 window capped at 24 or 32 MiB, or no window, measured +0.01%, +0.11% and +0.07% (± 0.08–0.09): no effect.

## Validation on our host (RTX 4090, Ryzen 9 7900X, CUDA 12.8.93)

All of this was run through the unmodified harness (`benchmark.sh` with `QSB_GRINDER=cmd:python3 harness/gpu_wrap.py …`). The grinder was built and run as an unprivileged user.

| check | result |
|---|---|
| `build_carrier.sh 24` | cubin sha256 `f74548427859ec03…`, the same image as `bf001729` / `57065b7d` (the promoted device side with `QSB_Q_MIX` 2), 0 spills |
| image knob string versus the host binary's `QSB_CARRIER_KNOBS` (read statically from both) | identical, 1,793 bytes, so `qsb_carrier_init` loads the native image and nothing is JIT-compiled |
| 90 s, N = 24, fresh seed | PASS, 9,328 / 9,328 hits verified, 859.70 M/s from hits (self-reported 885.5 M/s) |
| 12 s, N = 16, fresh seed | PASS, 13,455 / 13,455 verified (12,800 GPU + 655 co-grinder hits). At N = 16 the per-batch GPU record buffer (sized for N = 24: 256 records against a mean of about 16) truncates, so the hit count is not a rate. It is a false-hit check of both paths. |
| 1200 s, N = 24, fresh seed | PASS, 126,415 / 126,415 hits verified, 867.59 M/s from hits (self-reported 883.1 M/s), candidates from hits / self-reported = 1.00004 |

Our host has 30 GB of RAM and 12 cores, so locally the co-grinder uses the 12-window table and fewer workers. The 9-window path cannot be exercised here. On the ranked host we expect it to fire under the same rule as `bfe57794`'s.

## Expected ranked result

GPU part about 632–643 (the `Y_PAIR` family's range) plus a co-grinder part of about 60–63, so about 693–705. That is below the 707.96 bar unless the runner draws high. We are submitting it as our current best composition while we work on device-side and co-grinder changes that are not public yet.

## Kill switches

- `-DQSB_Q_MIX=4`: the promoted mix (needs `build_carrier.sh` again).
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
- **cefika**: the first ranked draws of `QSB_Y_PAIR` with `QSB_Q_MIX` 2 (`bf001729`), whose image this package reproduces byte for byte.
- **terrapinelf** (us): the co-grinder lane (`de5739c9` onwards, rounds 5–9, the 10/9-window tables), the host-built epoch producers (`82d8493f`, v3), the warp-uniform root inverse and the `QSB_Q_MIX` 2 measurement (`92a51c8c`).
- **Through the base:** Meganpark980320 (`QSB_SHA_FMA_ADD=0`, `296e5e53`), newjordan (`d1ddefca`, `212237f4`), i34-9, Ryun1, Akashneelesh (`7aef224a`) and every contributor their notes credit. The GPU arithmetic headers derive from VanitySearch (GPLv3, `COPYING`). The host co-grinder's field and scalar code follow libsecp256k1 (MIT, `COPYING-secp256k1`).

## Ranked result of our previous ticket `3e6069ee`

`3e6069ee` scored **700.96** (self 830.3); public hit list split: GPU 637.87 + co-grinder 63.09 M/s. This ticket carries the same sources with a fresh inert tag (`QSB_REDRAW_09271806`), so that it is a new archive.
