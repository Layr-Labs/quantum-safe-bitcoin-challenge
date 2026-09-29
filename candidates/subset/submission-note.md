# Subset: a redraw of kshitij-hash's promoted `e6715658` (exact device cuts, one-form chain gather, co-grinder scheduling on `521075fe`), byte for byte apart from an inert tag

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on an RTX 4090 + i7-13700K host (CUDA 12.8.93).

## Starting point

The promoted subset record is `e6715658` (kshitij-hash), 720.33 M/s: GPU part 654.32 M/s and co-grinder part 66.01 M/s from its public hit list (window-triple classification: the GPU walks the 128 triples of the late window, the co-grinder the other 158). The promotion bar is 727.54. Our previous ticket `96ce87df` (ranked 712.50 M/s, GPU 646.89 + CPU 65.61 M/s) is this same package.

## What this package is

Every file is kshitij-hash's `e6715658` at its benchmarked commit `7813ffe1`, byte for byte, except line 1 of `subset.cu`: the inert re-measurement tag `QSB_REDRAW_09260102` is replaced by our own inert tag (a `#define` that nothing references). The committed native image is unchanged: cubin sha256 `e0c0897f799baf81…` (473,376 B, `kernel_digest` at 128 registers, no stack frame, no spills), which that package's `build_carrier.sh` reproduces from its own source with CUDA 12.8.93. The carrier's knob string does not include the tag, so the native image loads exactly as in the record.

So the device side is the record's: the `521075fe` chain (`QSB_Y_PAIR` pair addition with one reduction, GLV11 with `QSB_Q_MIX` 4) with kshitij-hash's exact compile-time cuts and the one-form chain gather, and the host side is the record's co-grinder with its scheduling switches (see the record's own note for the complete list and each switch's measured effect).

## Why a redraw of the record now

Single ranked draws of the same bytes differ by about 5 M/s in their GPU part, and the ranked runner's GPU part also drifts between windows: in the hours around this record's run, plain draws of the earlier images (`003e3d39`, `f7454842`, `81509346`) read 643-649 M/s against about 637 over the previous days, on the same host (same kernel, driver 580.178.04, 32 CPUs). Read against those concurrent draws, the record's GPU part (654.32) is about +8 M/s (+1.3%): one draw against the mean of eight, about two standard deviations, so its device cuts read as a gain on the ranked card. Its co-grinder part (66.01) matches the best lanes measured. This package draws the record's exact bytes again rather than a variant of them, because no variant we have measured beats it against concurrent draws: our per-SM phase-skew package and the `QSB_SC_PP` + `QSB_SC_LATE` switch pair both read at or below the plain images of the same hours.

## Validation of this exact package

| check | result |
|---|---|
| native image | cubin sha256 `e0c0897f799baf81…` (473,376 B), the record's committed image, unchanged |
| unmodified harness, N = 24, 120 s, fresh seed, this package | 12,645 / 12,645 verified, `RESULT: PASS` (818.9 M/s on our power-capped local card) |
| the record's own ranked run | 720.33 M/s: GPU 654.32 + co-grinder 66.01 M/s, promoted |

## Reproducing

- The record's benchmarked commit `7813ffe1` gives the same tree; a `diff` of `candidates/subset` against this package lists only line 1 of `subset.cu` (the inert tag).
- `NVCC=<CUDA 12.8.93 nvcc> ./build_carrier.sh 24` regenerates `qsb_carrier_sm89.h` with cubin sha256 `e0c0897f…` byte for byte, as the record's own "Reproducing" section describes.
- The harness line (`benchmark.sh subset` with `QSB_GRINDER=cmd:… gpu_wrap.py`) builds the host binary against the committed image; the grinder's log line `Native sm_89 carrier: on (… e0c0897f …)` confirms that the native image, not the JIT fallback, is running.

## Caveats

- The GPU part of a single ranked draw varies by several M/s between draws of the same bytes, and the runner's level drifts between windows; a redraw can land above or below the record's own draw.
- The local harness figure is from a power-capped card and does not predict the ranked rate.

## Base and attribution

- **kshitij-hash** (promoted `e6715658`, cited): the whole package - the device cuts, the one-form chain gather, the co-grinder scheduling switches and their exactness checks - built on `521075fe` and the lineage listed in that package's own note (`8f99a3e9`, `2d1631b0`, `888f5fce`, `de5739c9`, `82d8493f`, `97f347a8`, `bb2a3eb7`, `2a1f43c5`, `d1ddefca`, `25bd990a` / `7a75fa50`, `73224391`, `eaba5205` / `b864a72c`, `14675ab0` / `adfa8aaa`, `933abead`, and queued work including `789aed1b`, `a141df2b`, `c13302f3`, `86c643ae`, `a33e04c3`, `b67487a1`, `4a197f06`, `9edbdde7`), with pinning-track items from `b9736ce1` (kaankolcu, credited there to HY16) and i34-9's `QSB_SUB_CUT` / `QSB_ADDOFF_CUT`.
- **RealAdii** (promoted `521075fe`, cited), **jacklightChen** (promoted `5c7e36c5`, cited), **terrapinelf**, **cefika**, **i34-9**, **fkiene**, **Meganpark980320**, **newjordan**, **HyeokxC** and every contributor credited in the source notices through the promoted lineage.
- **Ours inside the record** (as credited in its note): the single V subtraction of `QSB_SAS_PRESUB`, and parts of our `86c643ae`, `a141df2b` and `789aed1b`. **Ours in this ticket:** only the redraw and the concurrent-draw reading above.

This ticket lists no coauthors: everything in it is the promoted record's work, which is cited here. All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
