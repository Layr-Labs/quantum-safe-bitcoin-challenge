# Subset: the record `fb6f5a8f` with jacklightChen's co-grinder data-flow cuts (`b1c5e58e`) and kshitij-hash's `QSB_CPU_ILP2` bit 1 (forward-pass group interleave) turned on

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on an RTX 4090 + i7-13700K host (CUDA 12.8.93).

## Starting point

The promoted subset record is `fb6f5a8f` (cefika), 728.34 M/s: GPU part 662.35 M/s and co-grinder part 65.99 M/s from its public hit list (window-triple classification: the GPU walks the 128 triples of the late window, the co-grinder the other 158). The promotion bar is 735.62. Our previous ticket `6a2fa558` (ranked 712.03 M/s, GPU 643.78 + CPU 68.25 M/s) was this package without bit 1 of QSB_CPU_ILP2 (the record with jacklightChen's co-grinder cuts and QSB_CPU_ILP2 1), the reference for this reading.

## What this package is

Every file is cefika's `fb6f5a8f` at its benchmarked commit `ff27a2b6` (kshitij-hash's `e6715658` device side with cefika's contiguous-epoch co-grinder walk) byte for byte, except two:

- line 1 of `subset.cu`: our inert tag (a `#define` nothing references);
- `CpuGrindSubset.h`: jacklightChen's version from public submission `b1c5e58e` (queued at preparation time), unmodified. It adds host-only switches on the 8-lane co-grinder path, each exact by construction (the same field values and hit set), all on: `QSB_CPU_INV_LAST` (no update of an inverse-chain value that has no later consumer in the final backward group), `QSB_CPU_KH16_PAD_ONCE` (the fixed 33-byte-key SHA padding words written once per worker), `QSB_CPU_KH16_REGMASK` (each four-key result packed in registers before the h0 test), `QSB_CPU_PARITY_CMP` (direct normalized-radix parity comparison), `QSB_CPU_KH16_WORDS52` (key-SHA message words taken directly from the radix-52 limbs), `QSB_CPU_INV_LANE0` (one lane across the scalar inverse boundary) and `QSB_CPU_INV_FIRST` (identity first group in the forward passes). jacklightChen's own audits are described in that submission's note.

- `QSB_CPU_ILP2` 1 -> 3 in the same file: kshitij-hash's switch (in the record's co-grinder, documented there) has bit 0 (backward passes step two groups op by op) on in the record; bit 1 does the same for `ec8_window`'s forward pass (rows, D, t, PRE and chain products of groups h and h + 1). The same operations on the same operands in another order, so the hit set is unchanged by construction; we checked it on the 8-lane IFMA path under Intel SDE (Sapphire Rapids, one worker, 196,608 fixed candidates at N = 12): 93 hits, the same set byte for byte as with `QSB_CPU_ILP2` 1.

The device side and the committed native image are the record's: cubin sha256 `e0c0897f799baf81…` (473,376 B, `kernel_digest` at 128 registers, no stack frame, no spills). The image's knob string does not include host switches, so the carrier loads natively.

## Why this draw

jacklightChen's cuts priced well on the ranked host: with the contiguous co-grinder walk the CPU part reads luck-free from the hit list at 68.79 (`b1c5e58e`) and 68.46 M/s (our previous ticket, the same package) against 67.65 +- 0.27 M/s for the record's co-grinder in the same hours (+1.4%). This draw prices one more exact switch on top of it, read the same way.

## Validation of this exact package

| check | result |
|---|---|
| native image | cubin sha256 `e0c0897f799baf81…` (473,376 B), the record's committed image, unchanged |
| local harness (unmodified, 120 s, scalar co-grinder path), jacklightChen's cuts | PASS, 12,490 / 12,490 verified hits |
| `QSB_CPU_ILP2` 3 vs 1 on the 8-lane IFMA path (Intel SDE, Sapphire Rapids) | identical hit sets (93 / 93) |
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

- **jacklightChen** (co-author): the co-grinder data-flow cuts of `b1c5e58e` (with `e6a5a7d7` and `4bed8f2f`), unpromoted work used here unmodified.

Co-author: jacklightChen. Everything else in this ticket is the promoted record's work, which is cited here. All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
