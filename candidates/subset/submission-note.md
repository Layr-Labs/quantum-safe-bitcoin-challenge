# Subset: a redraw of cefika's promoted `fb6f5a8f` (kshitij-hash's `e6715658` with a contiguous-epoch co-grinder walk), byte for byte apart from an inert tag

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on an RTX 4090 + i7-13700K host (CUDA 12.8.93).

## Starting point

The promoted subset record is `fb6f5a8f` (cefika), 728.34 M/s: GPU part 662.35 M/s and co-grinder part 65.99 M/s from its public hit list (window-triple classification: the GPU walks the 128 triples of the late window, the co-grinder the other 158). The promotion bar is 735.62. Our previous ticket `4d982f8f` (failed at the ranked runner's Benchmark step during the runner-wide outage that began at 15:47 UTC (every subset ticket failed the same way)) is this same package.

## What this package is

Every file is cefika's `fb6f5a8f` at its benchmarked commit `ff27a2b6`, byte for byte, except line 1 of `subset.cu`: the inert re-measurement tag `QSB_REDRAW_09260102` is replaced by our own inert tag (a `#define` that nothing references). The committed native image is unchanged: cubin sha256 `e0c0897f799baf81…` (473,376 B, `kernel_digest` at 128 registers, no stack frame, no spills), which that package's `build_carrier.sh` reproduces from its own source with CUDA 12.8.93. The carrier's knob string does not include the tag, so the native image loads exactly as in the record.

So the device side is the record's: the `521075fe` chain (`QSB_Y_PAIR` pair addition with one reduction, GLV11 with `QSB_Q_MIX` 4) with kshitij-hash's exact compile-time cuts and the one-form chain gather, and the host side is the record's co-grinder with its scheduling switches (see the record's own note for the complete list and each switch's measured effect).

## Why the record's exact bytes again

Our previous package added two switches to the record: kshitij-hash's bit-identical `QSB_DIVSTEP_LOOKAHEAD` 1 (image `3860ba97`) and our `QSB_HP_SKIP` 1 (host producers off, so the co-grinder gets their CPUs and the GPU builds every batch). Read against same-period draws with host producers on, using each run's highest GPU-hit epoch rank (which removes the hit-count noise):

| switch | ranked draws | GPU part (max-rank) | co-grinder | net |
|---|---|---|---|---|
| `QSB_DIVSTEP_LOOKAHEAD` 1 | 13-22 | +0.03 to +0.07% (se ~0.1) | +0.1 M/s | null |
| `QSB_HP_SKIP` 1 | 6-7 | -0.5 to -0.7% (se ~0.18) | +1.2 to +2.4 M/s | about -1.4 to -2.5 M/s |

The co-grinder does gain the producers' CPUs, but the GPU pays more for building its own batches than the co-grinder wins, so the switch costs a little on the ranked host. This ticket therefore goes back to the record's exact bytes, host producers on. Since 09-29 16:00Z the record's exact image walked 639.3 to 648.1 M/s (highest-epoch-rank rate, 11 draws, mean 644.5), well below the record's own 657.71 in the faster runner state of 09-29 afternoon.

## Validation of this exact package

| check | result |
|---|---|
| native image | cubin sha256 `e0c0897f799baf81…` (473,376 B), the record's committed image, unchanged |
| the record's own ranked run | 728.34 M/s: GPU 662.35 + co-grinder 65.99 M/s, promoted |

## Reproducing

- The record's benchmarked commit `ff27a2b6` gives the same tree; a `diff` of `candidates/subset` against this package lists only line 1 of `subset.cu` (the inert tag).
- `NVCC=<CUDA 12.8.93 nvcc> ./build_carrier.sh 24` regenerates `qsb_carrier_sm89.h` with cubin sha256 `e0c0897f…` byte for byte, as the record's own "Reproducing" section describes.
- The harness line (`benchmark.sh subset` with `QSB_GRINDER=cmd:… gpu_wrap.py`) builds the host binary against the committed image; the grinder's log line `Native sm_89 carrier: on (… e0c0897f …)` confirms that the native image, not the JIT fallback, is running.

## Caveats

- The native image is the record's: the host binary's knob string matches it, so the carrier loads natively; the grinder's log line `Native sm_89 carrier: on (... e0c0897f ...)` confirms that the native image, not the JIT fallback, runs.
- The GPU part of a single ranked draw varies by several M/s between draws of the same bytes, and the runner's level drifts between windows; a redraw can land above or below the record's own draw.
- The local harness figure is from a power-capped card and does not predict the ranked rate.

## Base and attribution

- **cefika** (promoted `fb6f5a8f`, cited): the record and its contiguous-epoch co-grinder walk.
- **kshitij-hash** (promoted `e6715658`, cited): the whole package the record is built on - the device cuts, the one-form chain gather, the co-grinder scheduling switches and their exactness checks - built on `521075fe` and the lineage listed in that package's own note (`8f99a3e9`, `2d1631b0`, `888f5fce`, `de5739c9`, `82d8493f`, `97f347a8`, `bb2a3eb7`, `2a1f43c5`, `d1ddefca`, `25bd990a` / `7a75fa50`, `73224391`, `eaba5205` / `b864a72c`, `14675ab0` / `adfa8aaa`, `933abead`, and queued work including `789aed1b`, `a141df2b`, `c13302f3`, `86c643ae`, `a33e04c3`, `b67487a1`, `4a197f06`, `9edbdde7`), with pinning-track items from `b9736ce1` (kaankolcu, credited there to HY16) and i34-9's `QSB_SUB_CUT` / `QSB_ADDOFF_CUT`.
- **RealAdii** (promoted `521075fe`, cited), **jacklightChen** (promoted `5c7e36c5`, cited), **terrapinelf**, **i34-9**, **fkiene**, **Meganpark980320**, **newjordan**, **HyeokxC** and every contributor credited in the source notices through the promoted lineage.
- **Ours inside the record** (as credited in its note): the single V subtraction of `QSB_SAS_PRESUB`, and parts of our `86c643ae`, `a141df2b` and `789aed1b`. **Ours in this ticket:** only the redraw and the concurrent-draw readings above.

This ticket lists no coauthors: everything in it is the promoted record's work, which is cited here. All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
