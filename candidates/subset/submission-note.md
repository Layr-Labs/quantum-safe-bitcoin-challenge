# Subset: our `bf001729` (ranked 697.77: GPU 637.61 + CPU 60.16; the promoted `d052bc3d` GLV11 P18 chain + kshitij-hash's `QSB_Y_PAIR` + `QSB_Q_MIX` 2 + ercumentyildirim's `bfe57794` host side) with terrapinelf's cheaper v3 producer code on the three floating producer threads (as in ercumentyildirim's `a141df2b`, ranked CPU part 62.17)

Prepared with Claude Opus 5.5 in Claude Code. This host has no GPU and no AVX-512. Everything below that says "measured" was measured by the ranked runner or by the authors named. What I did myself is the composition, the builds, byte comparisons, and an analysis of public ranked artifacts.

## Summary

This is our `bf001729` (ranked 697.77, 0.11% short of the promotion bar) with one host-side change: `tests/gpu_epochs/host_producers.h` is replaced by terrapinelf's v3 producer code, kept on the three floating producer threads, as in ercumentyildirim's queued `a141df2b`. The co-grinder file also gains `a141df2b`'s table-ready-time diagnostic (host-only). The device code and the native sm_89 image are unchanged. The rebuilt cubin is byte-identical to `bf001729`'s (`f74548427859ec03…`).

Why: the v3 producer code builds each epoch's first-block states from precomputed W+K schedules. terrapinelf measured about 162 ns per epoch instead of 355–365 ns, which is about 0.8 instead of 1.75 logical CPUs for the three producer threads at the ranked batch rate. The freed CPU goes to the co-grinder. Its ranked draws with floating placement read 634.7 + 59.9 (`eb9ee8f3`) and 632.0 + 60.9 (`fa8a3d22`). The two low GPU draws with v3 code (623.6, 627.6) used the pinned placement, which this package does not use.

## Starting point

The promoted subset record is `d052bc3d` (kshitij-hash), commit `61cb94f`, 691,630,437/s. This package's tree is our queued `bf001729` plus the files listed below. The benchmark branch tip `f0e453d` has `61cb94f`'s `candidates/subset/`.

## What the ranked runs say (analysis of public artifacts)

Every ranked subset run uploads `run-subset.json` with its verified hits. The GPU grinds 128 of the C(13,3) = 286 window patterns and the co-grinder the other 158. Counting each hit's last three skip indices therefore splits a score exactly into a GPU part and a CPU part. In every run, the 128th most frequent pattern has 612–656 hits and the 129th has 52–81. All subset runs execute on the same host, one job at a time.

Over 76 runs from 2026-09-25 20:36 UTC to 2026-09-27 08:28 UTC:

| family | runs | GPU part, M/s | CPU part, M/s | total, M/s |
|---|---:|---:|---:|---:|
| GLV12 trees, CPU > 40 | 23 | 628.6 (sd 3.4) | 46.3 | 674.9 |
| GLV11 P18 trees, light co-grinder | 17 | 629.0 (sd 7.6) | 47.6 | 676.6 |
| GLV11 P18 trees, heavy co-grinder | 8 | 628.4 (sd 4.0) | 60.1 | 688.5 (sd 4.4) |

Draws since then (the ranked card has been running faster this morning for every tree on the record's GPU chain):

| submission | what it carries | GPU part | CPU part | total |
|---|---|---:|---:|---:|
| `8f99a3e9` | `QSB_Y_PAIR`, light co-grinder, blocking waits | 637.13 | 57.47 | 694.60 |
| `77f3ad08` | `888f5fce` copy (blocking waits, host core to the co-grinder) | 637.54 | 54.15 | 691.69 |
| `2aa4321a` | frontier redraw | 632.83 | 59.60 | 692.43 |
| `53b54061` | blocking waits, host-core workers | 636.03 | 52.11 | 688.15 |
| `49f1ab79` | r7 CPU lane on the record's producers | 636.72 | 56.53 | 693.25 |
| `a141df2b` | `bfe57794` host side **+ v3 producers**, `QSB_Q_MIX` 4, no `QSB_Y_PAIR` | 634.26 | **62.17** | 696.43 |
| **our `bf001729`** | **`QSB_Y_PAIR` + `QSB_Q_MIX` 2** + `bfe57794` host side (old producers) | **637.61** | 60.16 | **697.77** |

The last two ran back to back (12:09 and 12:37 UTC) with the same co-grinder and the same host-side placement. They differ in exactly the two things this package combines. `bf001729`'s device side (`QSB_Y_PAIR` + `QSB_Q_MIX` 2) drew 3.35 M/s more GPU part than `a141df2b`'s record device side. `a141df2b`'s v3 producers drew 2.0 M/s more co-grinder part than `bf001729`'s old producers. This package is `bf001729` with `a141df2b`'s producer and co-grinder files, so it carries both.

What I take from it:

1. The ranked GPU part moves by about ±4 M/s with the host's conditions, whatever the GPU code. On the thermally limited card (about 317 W, 90 °C), the P18 chain's +4.4% on cool cards shows as about +0.3%.
2. Blocking GPU waits keep appearing next to high GPU parts (639.3, 637.5, 636.0), but so does the record's code without them in the same hours. The effect is not settled.
3. A heavier co-grinder adds its rate almost fully: a regression over 22 P18 draws gives about 0.35 M/s of GPU part lost per +1 M/s of co-grinder. Cutting CPU work per candidate is worth more than adding workers. The cheaper producers do exactly that for the producer threads.
4. The 9-window host table fired on the ranked host in `789aed1b`: its co-grinder diagnostic has bit 19 set, with huge pages at 95% or more and 255 GiB or more visible. It gave +1.9% co-grinder rate over 10 windows.

## What this package contains, against `d052bc3d`

| file | source | change |
|---|---|---|
| `hit_filter_field_sc.cuh`, `y_pair_sc.cuh` (new), `hit_filter_field.cuh`, `tests/gpu_epochs/tree_inverse.cuh` | `8f99a3e9`, unchanged | `QSB_Y_PAIR`: the chain trip carries `(Q', R)` and the next trip forms `R = red(S·ZZZ + Q'·R)`, one reduction for both products. `QSB_SC_PARK` parks the first candidate's Z product in the idle product-tree arena across the second candidate's front call |
| `tests/gpu_epochs/tree.cu` | `8f99a3e9`, plus two edits | the `QSB_Y_PAIR` call sites, park and reload, new knobs in the image fingerprint, and blocking slot events; **`QSB_Q_MIX` 4 → 2**; the final `[HP]` line prints the v3 helper-chunk count (host-only) |
| `tests/gpu_epochs/host_producers.h` | `a141df2b`, unchanged | **new in this package:** terrapinelf's v3 producer code (precomputed W+K message schedules on the SHA-NI path) on three floating threads (`QSB_HP_PLACE` 0, ercumentyildirim's default; the pinned placement with a helper thread stays off), with `g_share_cpu` staying −1 |
| `CpuGrindSubset.h` | `a141df2b`, unchanged | terrapinelf's `2d1631b0` lane with ercumentyildirim's host-core placement under blocking waits, the memory-gated 9-window table, and (**new here**) the table-ready time in the diagnostic's bits 20..27 |
| `qsb_carrier_sm89.h` | rebuilt here | the same cubin as `bf001729`; only the source-hash comment changes |
| `SOURCE-MANIFEST.json`, this note | here | regenerated |

Everything else is `d052bc3d`'s, byte for byte.

## Build and checks done here

All builds ran in an amd64 container with CUDA 12.8.93, the ranked runner's toolkit.

| check | result |
|---|---|
| toolchain reproduces committed images | byte-identical for `bfe57794` (`45f988c8…`) and `8f99a3e9` (`003e3d39…`) |
| `build_carrier.sh 24` on this tree | cubin sha256 `f74548427859ec03273f05e9151c810716c6475596e0213a0db3915e468aa5dc`, 462,496 bytes: identical to `bf001729`'s, so the new producer code is host-only |
| `kernel_digest` | 127 registers, 0 stack, 0 spill; no kernel in the image spills |
| the harness's build line (`nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`) | builds with no errors |
| image `qsb_carrier_knobs` against host `QSB_CARRIER_KNOBS` | identical, 1,793 bytes (`…QSB_Q_MIX=2;QSB_Y_PAIR=1;…QSB_SC_PARK=1…`), so the native image loads |

## Exactness

- `QSB_Y_PAIR` files are kshitij-hash's, byte for byte. Their checks in `8f99a3e9` hold: the four-operand routine matches `red(a·b + c·d)` on 101,452 cases, and point-addition outputs are bitwise identical on 19,800 random states. Every GPU tentative is re-derived by the exact host gate.
- `QSB_Q_MIX` 2 only chooses which precomputed layout a warp reads partial sums from. The rolled chain loop walks either descriptor list, and `8f99a3e9` already exercised both at `QSB_Q_MIX` 4. `qsb_s3_selfcheck` replays both lists at start-up.
- Host producers: terrapinelf's v3 code carries its own start-up self-check against the GPU producer kernels (bit-identical descriptors and first-block states for batch 0). A late or failed host batch falls back to the GPU producer kernels, which are unchanged in the image.
- Co-grinder: every CPU hit passes the exact OpenSSL gate `qsb_hv_check` before it is written. Any failure on the 9-window path falls back to 10, 11 or 12 windows.

## Expected result and caveats

- Expected: a GPU part of about 632–638, depending mostly on the host's conditions at run time (`bf001729` drew 637.61), plus a co-grinder part of about 62 (`a141df2b`'s 62.17 with the same producer and co-grinder files). That gives a total of about 694–700, against the promotion bar of 698.55. The two back-to-back draws above suggest each half of the combination is worth its measured part, but each rests on one draw, and the GPU part moves by about ±5 M/s with the host's conditions.
- Not run on a GPU by me. This exact combination has not run before. Its halves have: the device side in `bf001729` (637.61 GPU), and the producer and co-grinder files in `a141df2b` (62.17 CPU, with the ready-time diagnostic reading 37 quarter seconds: the 9-window table was ready after about 9.25 s).
- If another ticket is promoted before this one runs, the bar moves to about +1% over it.

## Reproducing

```
./setup.sh subset
./benchmark.sh subset
```

To regenerate the native image after any device-code edit: `NVCC=/path/to/cuda-12.8/bin/nvcc ./build_carrier.sh 24` in `candidates/subset`. To get the GPU/CPU split of any ranked run: download its `benchmark-diagnostics-subset-<run>` artifact, count hits per `skip[6:9]`, take the 128 most frequent patterns as the GPU's, and compute rate = hits × 2^23 / `elapsed_s`.

Kill switches: `-DQSB_Y_PAIR=0` (with `QSB_SC_PARK=0`), `-DQSB_Q_MIX=4`, `-DQSB_CPU_TRY9=0`, `-DQSB_HOST_BLOCKING=0`, `-DQSB_CPU_GRIND=0`, `-DQSB_HOST_PRODUCERS=0`. Any device knob change needs `build_carrier.sh` again.

## Base and attribution

- **terrapinelf** (co-author): the v3 producer code, the co-grinder lane (`2d1631b0`), the host-built epoch producers and warp-uniform root (`82d8493f`), the `QSB_Q_MIX` 2 setting and the ranked energy model (`92a51c8c`), and the promoted `de5739c9` tree.
- **ercumentyildirim** (co-author): the `bfe57794` / `a141df2b` host side (floating placement of the v3 producers, host-core placement under blocking waits, the 9-window table, the table-ready diagnostic), and the ranked GPU/CPU split method.
- **kshitij-hash** (co-author): the pair chain `QSB_Y_PAIR` and park `QSB_SC_PARK` (`8f99a3e9`), and the promoted `d052bc3d` composition.
- **HyeokxC** (co-author): blocking GPU waits with the host core given to the co-grinder (`0735233a`, `888f5fce`).
- **fkiene**: the GLV11 P18 five-term chain and the per-warp Q-layout mix (`413f83e7`, promoted in `d052bc3d`).
- **Through the base:** Meganpark980320, newjordan, i34-9, Ryun1, Akashneelesh and every contributor their notes credit. libsecp256k1 (MIT, `COPYING-secp256k1`).
- **Mine:** the ranked split analysis above, the composition, the merges, the image rebuilds and the byte checks.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
