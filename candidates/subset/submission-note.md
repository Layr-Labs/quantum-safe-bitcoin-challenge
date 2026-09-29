# Subset: a ranked multi-arm probe of the record `e6715658` with `QSB_DIVSTEP_LOOKAHEAD` on: five device images time-sliced in one run over disjoint epoch ranges (patternrecognition9-del's probe from `9b2fbb14`)

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on an RTX 4090 + i7-13700K host (CUDA 12.8.93).

## Starting point

The promoted subset record is `e6715658` (kshitij-hash), 720.33 M/s: GPU part 654.32 M/s and co-grinder part 66.01 M/s from its public hit list (window-triple classification: the GPU walks the 128 triples of the late window, the co-grinder the other 158). The promotion bar is 727.54. Our previous ticket `95b8ce12` (this package without the probe: the record with `QSB_DIVSTEP_LOOKAHEAD` on) was cancelled before it ran, in favour of this measurement; the draw before it, `96ce87df` (ranked 712.50 M/s, GPU 646.89 + CPU 65.61 M/s), was a redraw of the record byte for byte.

## This ticket is a measurement

Single ranked draws cannot resolve device effects below about 1%. Read from the highest GPU-hit epoch rank of each run (which removes the hit-count noise), the record's exact image walked 647.7 to 657.0 M/s in consecutive runs today: that spread is the runner's own drift. patternrecognition9-del's public ticket `9b2fbb14` showed how to measure several device images inside one run instead: the images take turns in 700 ms slices over disjoint epoch ranges, and the public hit list alone gives each image's epochs per slice. We decoded that ticket (239 paired rounds, about 0.15-0.2% per arm, duplicate control consistent at -0.13% +- 0.21%): `QSB_Q_MIX` 2 read -1.09% +- 0.18% on this tree and four other switches read within +-0.3%.

This ticket runs the same probe on our package (the record with its own `QSB_DIVSTEP_LOOKAHEAD` switch on) with the arms below, to decide what our next draws carry. Its own score is expected below the package's: in probe mode every batch is built by the GPU producers (the host producers' build-ahead cannot follow a time-driven schedule; our local A/B puts that at about -1% of the GPU rate), each slice ends with both slots drained, and the arms that lose pull the total down. Every candidate is still exact and searched once, so the run scores normally.

## What this package is

- **Base:** kshitij-hash's `e6715658` at its benchmarked commit `7813ffe1`, with `QSB_DIVSTEP_LOOKAHEAD` 1 (kshitij-hash's bit-identical lookahead root inverse, shipped at 0 in the record) set in `subset.cu` for every arm unless an arm's flags say otherwise.
- **Probe (patternrecognition9-del's code from `9b2fbb14`, GPL-3, unchanged apart from the lines named here):** `QsbCarrier.h` (the arm loader, the knob check that lets arms differ only in device-only switches, the round-robin slice schedule), `build_carrier.sh` (one cubin per row of its `ARMS` table), and their search-loop hooks in `tests/gpu_epochs/tree.cu`. Our changes: the `ARMS` table, `QSB_ARMS` 5, `QSB_PROBE_SLOTS` 384 (so 5 arms x 384 slices x ~0.715 s outlast the 1,200 s run; with the original 256 a five-arm schedule would end the search at about 915 s), and `QSB_DIVSTEP_LOOKAHEAD` added to the probe's list of device-only switches an arm may change.
- **Arms** (each image built from this tree by `build_carrier.sh 24` with CUDA 12.8.93; 128 registers, no spills; each only changes `kernel_digest`):

| arm | flags added to this tree | image (cubin sha256) | why |
|---|---|---|---|
| 0 | none (control) | `e2a5d42aa026caf2…` | the package itself |
| 1 | `QSB_DIVSTEP_LOOKAHEAD=0` | `cbf23484ff825e79…` | the record's own device code: is the lookahead a gain on the ranked card? (+0.21% on ours; terrapinelf +0.33%) |
| 2 | `QSB_ROOT_LUT_SMEM=1 QSB_SHA_CONST_PEEL=1` | `b192eadb27ce585d…` | read +0.16% +- 0.18% (+0.24% +- 0.20% against the second control) on the record in `9b2fbb14`; is it a gain on top of the lookahead? |
| 3 | `QSB_PAIR_SHA_UNROLL_CONST=1` | `a0b056bfed6d69d4…` | the paired constant SHA blocks unrolled (constant-bank operands, no loop counter), which the record rolls; never measured on the native image |
| 4 | none (control again) | `e2a5d42aa026caf2…` | A/A check |

Arm k, slice j searches epoch ranks [k·A + j·S, +w) with A = C(137,6) / 5 and S = A / 384, round robin, one arm per 700 ms slice (a per-round dither shared by all arms). The producer kernels and the start-up tables always come from arm 0, so every arm consumes the same epoch descriptors.

## How to read it from the public hit list

For each GPU hit (window triple in the GPU's 128), the rank of its six early omissions gives the arm k and slot j. Per slot, w = the highest hit offset + 1 + one mean inter-hit gap; per round, each arm's w over the mean w of the two control arms; mean and standard error over the complete rounds after the first. The two controls also give an A/A check of the method.

## Validation of this exact package

| check | result |
|---|---|
| arm images | `build_carrier.sh 24`, CUDA 12.8.93: five arms, 128 registers, 0 spill bytes in `kernel_digest`, header combined sha256 `35d8def8684af0bf…`; log lines `Probe arm k: … loaded OK` |
| unmodified harness, N = 24, 120 s, fresh seed, this package | 12,141 / 12,141 verified, `RESULT: PASS` (784.3 M/s on our power-capped card: lower than the plain package by design, see above) |
| local decode of that run | 37 rounds, 0 hits outside the schedule; duplicate control -0.03% +- 0.24% of the first (our fast local card fills most slots, so the local arm readings are compressed; the ranked card fills about 85% of each) |
| schedule length | 5 x 384 slices x ~0.715 s = 1,373 s > 1,200 s; a slot holds 4.28M epochs against ~3.6M per ranked slice |
| fallback | if an arm fails to load or its knobs differ outside the device-only list, the log reads `Probe arms: off` and the run is the plain package |

## Caveats

- This is a measurement ticket: its score is expected below a plain draw of the package.
- On our local card the first arm read about 0.7% above the duplicate control in a 40-round run (the ranked `9b2fbb14` showed +0.13% +- 0.21% for the same comparison), so each arm is also read against the second control alone.

## Base and attribution

- **patternrecognition9-del** (co-author): the whole multi-arm probe (`9b2fbb14`, not promoted), which this ticket reuses with a new arm table and slot count.
- **terrapinelf** (co-author): the choice of `QSB_DIVSTEP_LOOKAHEAD` on this record and its GPU-only measurement (`9b0c36bb`, not promoted).
- **kshitij-hash** (promoted `e6715658`, cited): the whole package, including `QSB_DIVSTEP_LOOKAHEAD`, `QSB_ROOT_LUT_SMEM`, `QSB_SHA_CONST_PEEL` and their exactness checks, built on `521075fe` and the lineage listed in that package's own note.
- **RealAdii** (promoted `521075fe`, cited), **jacklightChen** (promoted `5c7e36c5`, cited), **cefika**, **i34-9**, **fkiene**, **Meganpark980320**, **newjordan**, **HyeokxC** and every contributor credited in the source notices through the promoted lineage.
- **Ours inside the record** (as credited in its note): the single V subtraction of `QSB_SAS_PRESUB`, and parts of our `86c643ae`, `a141df2b` and `789aed1b`. **Ours in this ticket:** the arm table, the slot count, the decode of `9b2fbb14` and the local checks.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
