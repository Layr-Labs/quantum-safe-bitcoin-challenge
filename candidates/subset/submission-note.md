# Subset: our `57065b7d` (the `d052bc3d` GLV11 P18 chain + kshitij-hash's `QSB_Y_PAIR` + `QSB_Q_MIX` 2, ercumentyildirim's host side with terrapinelf's v3 producers; ranked co-grinder part 63.15, the highest so far) with a 24 MiB cap on the persisting-L2 table window (after AvinashNayak27's `cd33f1b6` and fkiene's `8009bfb9`)

Prepared with Claude Opus 5.5 in Claude Code. This host has no GPU and no AVX-512. Everything below that says "measured" was measured by the ranked runner or by the authors named. What I did myself is the composition, the builds, byte comparisons, and an analysis of public ranked artifacts.

## Summary

This is our `57065b7d` with one host-only change. `qsb_table_l2_window` caps the persisting table window at 24 MiB (`QSB_TABLE_L2_WINDOW_MIB`, default 24; 0 restores the full 48 MiB dense window). The persisting set then stays below the dense table, so the per-launch epoch state, group buffers and streamed hit records keep normal L2 capacity. It is a cache policy only: every loaded value is unchanged. The device code and the native sm_89 image are unchanged; the cubin is byte-identical to `bf001729`'s and `57065b7d`'s (`f74548427859ec03…`).

## Starting point

The promoted subset record is `521075fe` (RealAdii), 700,953,730/s, which is kshitij-hash's `8f99a3e9` tree (`QSB_Y_PAIR`, `QSB_Q_MIX` 4, the light co-grinder) with a refill-before-publish host loop. This package replaces the whole editable tree with our `57065b7d` tree plus the cap. Relative to the record, it carries `QSB_Q_MIX` 2, the heavy co-grinder with the 9-window table and host-core placement, terrapinelf's v3 producers and the L2 cap, and it keeps the record's `QSB_Y_PAIR` device files.

## What the ranked runs say (analysis of public artifacts)

Every ranked subset run uploads `run-subset.json` with its verified hits. The GPU grinds 128 of the C(13,3) = 286 window patterns and the co-grinder the other 158. Counting each hit's last three skip indices therefore splits every score exactly into a GPU part and a CPU part. All subset runs execute on the same host, one job at a time.

Draws that bear on this package:

| submission | what it carries | GPU part | CPU part | total |
|---|---|---:|---:|---:|
| `a141df2b` | record device side + `bfe57794` host side + v3 producers | 634.26 | 62.17 | 696.43 |
| our `bf001729` | `QSB_Y_PAIR` + `QSB_Q_MIX` 2 + `bfe57794` host side, old producers | 637.61 | 60.16 | 697.77 |
| `8c3822c4` | public copy of `bf001729` | 637.63 | 60.70 | 698.33 |
| `521075fe` (promoted) | `8f99a3e9` + refill-before-publish | 642.62 | 58.33 | 700.95 |
| `cd33f1b6` | `a141df2b` **+ 24 MiB L2 cap** | 635.60 | 61.89 | 697.49 |
| our `57065b7d` | `bf001729` + v3 producers (this package without the cap) | 629.61 | **63.15** | 692.76 |

### The card's phase dominates the GPU part

I regressed the GPU part of 40 P18 draws from 2026-09-26 15:00 to 2026-09-27 16:15 UTC on code features taken from each submission's source, with the mean GPU part of the two previous and two next runs as a control for the card's state:

| factor | effect on GPU part, M/s |
|---|---:|
| neighbours' mean GPU part (card phase) | coefficient 0.94 ± 0.20 |
| blocking GPU waits | +2.2 ± 1.9 |
| 24 MiB L2 cap | +1.6 ± 3.5 (2 draws) |
| `QSB_Y_PAIR` | +1.3 ± 2.9 |
| refill-before-publish | +0.2 ± 3.0 |
| `QSB_Q_MIX` 2 | +0.0 ± 2.4 |
| co-grinder part, per +1 M/s | −0.28 ± 0.18 |

The residual standard deviation is 4.5 M/s. The card's phase has a standard deviation of about 5.9 M/s, and consecutive runs share it (lag-1 correlation 0.37). No code feature is measured to better than about ±2 M/s from single draws. The one clear signal is on the host side: the v3 producers lifted the co-grinder part from 60.2 to 62.2–63.2 M/s with the same co-grinder file, at about 0.28 M/s of GPU part per M/s. The refill-before-publish loop, which earlier notes counted as a dead end from two slow-phase draws, is neutral once the phase is controlled for.

### Why the L2 cap

- fkiene measured +0.43% locally for the 24 MiB cap on its own in `8009bfb9`, whose only ranked draw bundled it with IMAD-routed SHA additions.
- AvinashNayak27's `cd33f1b6` isolates it on `a141df2b`'s tree. It drew 635.60 GPU after a 632.84 neighbour, while `a141df2b` drew 634.26 between 636.72 and 637.61 neighbours.
- It is a host-side stream attribute, not in the image fingerprint, and costs nothing when the persisting window is not applied.

## What this package contains, against `d052bc3d`

| file | source | change |
|---|---|---|
| `hit_filter_field_sc.cuh`, `y_pair_sc.cuh` (new), `hit_filter_field.cuh`, `tests/gpu_epochs/tree_inverse.cuh` | `8f99a3e9` (as in the record), unchanged | `QSB_Y_PAIR`: the chain trip carries `(Q', R)` and reduces once per addition. `QSB_SC_PARK` parks the first candidate's Z product in the idle product-tree arena |
| `tests/gpu_epochs/tree.cu` | `8f99a3e9`, plus edits | the `QSB_Y_PAIR` call sites, park and reload, knobs, blocking slot events; `QSB_Q_MIX` 4 → 2; the v3 helper-chunk count in the final `[HP]` line; **new here:** `QSB_TABLE_L2_WINDOW_MIB` 24 and its cap in `qsb_table_l2_window` |
| `tests/gpu_epochs/host_producers.h` | `a141df2b`, unchanged | terrapinelf's v3 producer code (precomputed W+K message schedules on the SHA-NI path) on three floating threads (`QSB_HP_PLACE` 0) |
| `CpuGrindSubset.h` | `a141df2b`, unchanged | terrapinelf's `2d1631b0` lane, ercumentyildirim's host-core placement under blocking waits, the memory-gated 9-window table and the table-ready diagnostic |
| `qsb_carrier_sm89.h` | rebuilt here | same cubin as `bf001729` / `57065b7d`; only the source-hash comment changes |
| `SOURCE-MANIFEST.json`, this note | here | regenerated |

## Build and checks done here

All builds ran in an amd64 container with CUDA 12.8.93, the ranked runner's toolkit.

| check | result |
|---|---|
| toolchain reproduces committed images | byte-identical for `bfe57794` (`45f988c8…`) and `8f99a3e9` (`003e3d39…`) |
| `build_carrier.sh 24` on this tree | cubin sha256 `f74548427859ec03273f05e9151c810716c6475596e0213a0db3915e468aa5dc`, 462,496 bytes, unchanged by the cap |
| `kernel_digest` | 127 registers, 0 stack, 0 spill; no kernel in the image spills |
| the harness's build line (`nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`) | builds with no errors |
| image `qsb_carrier_knobs` against host `QSB_CARRIER_KNOBS` | identical, 1,793 bytes, so the native image loads |

## Exactness

- The L2 cap changes only which table lines the stream marks persisting. Every value loaded is the same.
- `QSB_Y_PAIR` files are kshitij-hash's, byte for byte, as in the promoted record. Every GPU tentative is re-derived by the exact host gate.
- `QSB_Q_MIX` 2 only chooses which precomputed layout a warp reads partial sums from. `qsb_s3_selfcheck` replays both descriptor lists at start-up.
- Host producers: terrapinelf's v3 code self-checks batch 0 against the GPU producer kernels; a late or failed host batch falls back to those kernels.
- Co-grinder: every CPU hit passes the exact OpenSSL gate `qsb_hv_check` before it is written; the 9-window table falls back to 10, 11, 12 windows on any failure. In `a141df2b`'s ranked hits, its diagnostic decodes to the 9-window table, ready after about 9.25 s.

## Expected result and caveats

- Expected: a GPU part of about 630 plus the card's phase (±6), plus a co-grinder part of about 62–63, for a total of about 692–700 in an average phase. The promotion bar is 707.96, so this needs a fast phase like the record's (GPU 642.6). I do not claim a measured improvement over the record; the package is the record's device side with the best measured host side and the L2 cap.
- Not run on a GPU by me. The L2 cap has two ranked draws, one of them bundled with other changes.

## Reproducing

```
./setup.sh subset
./benchmark.sh subset
```

To regenerate the native image after any device-code edit: `NVCC=/path/to/cuda-12.8/bin/nvcc ./build_carrier.sh 24` in `candidates/subset`. To get the GPU/CPU split of any ranked run: download its `benchmark-diagnostics-subset-<run>` artifact, count hits per `skip[6:9]`, take the 128 most frequent patterns as the GPU's, and compute rate = hits × 2^23 / `elapsed_s`.

Kill switches: `-DQSB_TABLE_L2_WINDOW_MIB=0` (full dense window), `-DQSB_Y_PAIR=0` (with `QSB_SC_PARK=0`), `-DQSB_Q_MIX=4`, `-DQSB_CPU_TRY9=0`, `-DQSB_HOST_BLOCKING=0`, `-DQSB_CPU_GRIND=0`, `-DQSB_HOST_PRODUCERS=0`. Any device knob change needs `build_carrier.sh` again.

## Base and attribution

- **AvinashNayak27** (co-author): the isolated 24 MiB persisting-L2 cap on the host side (`cd33f1b6`).
- **terrapinelf** (co-author): the v3 producer code, the co-grinder lane (`2d1631b0`), the host-built epoch producers and warp-uniform root (`82d8493f`), the `QSB_Q_MIX` 2 setting and the ranked energy model (`92a51c8c`).
- **ercumentyildirim** (co-author): the `bfe57794` / `a141df2b` host side and the ranked GPU/CPU split method.
- **kshitij-hash** (co-author): `QSB_Y_PAIR` and `QSB_SC_PARK` (`8f99a3e9`, in the promoted record) and the `d052bc3d` composition.
- **HyeokxC** (co-author): blocking GPU waits with the host core given to the co-grinder (`0735233a`, `888f5fce`).
- **fkiene**: the first persisting-L2 cap (`8009bfb9`), the GLV11 P18 chain and the per-warp Q-layout mix (`413f83e7`).
- **RealAdii**: the promoted `521075fe` record.
- **Through the base:** Meganpark980320, newjordan, i34-9, Ryun1, Akashneelesh and every contributor their notes credit. libsecp256k1 (MIT, `COPYING-secp256k1`).
- **Mine:** the ranked split and regression analysis above, the composition, the merges, the image rebuilds and the byte checks.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
