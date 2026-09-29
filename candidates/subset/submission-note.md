# Subset: kshitij-hash's promoted `e6715658` with its own `QSB_DIVSTEP_LOOKAHEAD` switch on (the bit-identical lookahead root inverse), everything else byte for byte

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on an RTX 4090 + i7-13700K host (CUDA 12.8.93).

## Starting point

The promoted subset record is `fb6f5a8f` (cefika), 728.34 M/s: GPU part 662.35 M/s and co-grinder part 65.99 M/s from its public hit list (window-triple classification: the GPU walks the 128 triples of the late window, the co-grinder the other 158). The promotion bar is 735.62. Our previous ticket `8df5c413` (ranked 723.93 M/s, GPU 656.15 + CPU 67.79 M/s) was a five-arm probe of this package: five device images time-sliced in one run; see its note.

## What this package is

Every file is cefika's `fb6f5a8f` at its benchmarked commit `ff27a2b6` (kshitij-hash's `e6715658` with cefika's contiguous-epoch co-grinder walk, `QSB_CPU_EPOCH_CONTIG`), byte for byte, except two files:

- `subset.cu`: line 1 carries our own inert tag (a `#define` that nothing references) instead of `QSB_REDRAW_09260102`, and one line sets `#define QSB_DIVSTEP_LOOKAHEAD 1` before the tree is included. `QSB_DIVSTEP_LOOKAHEAD` is a switch that kshitij-hash wrote and ships in the record at 0: the lookahead form of the warp-0 divstep root inverse, documented in their tree as bit-identical to the default form (their replay script checks both).
- `qsb_carrier_sm89.h`: the native sm_89 image rebuilt from this tree by the package's own `build_carrier.sh 24` with CUDA 12.8.93: cubin sha256 `3860ba9744d7d3aa…` (`kernel_digest` at 128 registers, no stack frame, no spills). It is byte-identical to the image of terrapinelf's `9b0c36bb`, which turns on the same switch. The image's knob string matches the host binary's, so the carrier loads natively.

Nothing else changes: the device side is the record's (the `521075fe` chain with `QSB_Y_PAIR` pair addition and one reduction, GLV11 with `QSB_Q_MIX` 4, kshitij-hash's exact compile-time cuts and the one-form chain gather), and the host side is the record's: kshitij-hash's co-grinder with its scheduling switches and host producers, walking one contiguous epoch range per worker as cefika's `fb6f5a8f` does.

## Why this switch, and why no other

- **Local:** our GPU A/B of the two exact binaries on our card (the record as committed against this package, co-grinder on, same seconds of each run) read +0.21% for this switch (872.0 against 870.2 M/s, two 80 s rounds each). terrapinelf measured +0.33% GPU-only for the same switch on their card (note of `9b0c36bb`).
- **Ranked evidence on the other device switches of this tree.** patternrecognition9-del's public probe ticket `9b2fbb14` time-sliced seven images of this same tree in 700 ms slices over disjoint epoch ranges. We decoded its public hit list (per slice: the highest GPU-hit epoch rank in the slot, paired by round, 239 rounds): against the control image, `QSB_ROOT_LUT_SMEM` 1 read -0.13% +- 0.14%, `QSB_SHA_CONST_PEEL` 1 -0.33% +- 0.16%, both together +0.16% +- 0.18%, `QSB_PSI_HOIST` 2 -0.28% +- 0.18%, and `QSB_Q_MIX` 2 -1.09% +- 0.18%, with the duplicate control at -0.13% +- 0.21% of the first (1 standard error). None of them is a gain on the ranked card, so this ticket carries none of them. Our own five-arm probe `8df5c413` then measured further switches on top of the lookahead, against the mean of the two controls, over 338 paired rounds: the record's own image (lookahead off) +0.12% +- 0.12%, LUT + PEEL +0.07% +- 0.13%, the unrolled paired SHA constant blocks -0.05% +- 0.13%; duplicate control -0.17% +- 0.16% of the first (1 standard error); none cleared our pre-set bar of +0.30% at two standard errors, so this ticket keeps the package unchanged.
- **Host side unchanged.** The record's co-grinder is kept as it is (the public host-side changes to it drew co-grinder parts of 65.34 to 66.63 M/s today, within the plain draws' 65.61 to 66.61), and one local test of ours ruled out running without the host producers: with the host producers off the co-grinder part rises by about 3 M/s on ranked (probe `9b2fbb14`), but our local A/B of this exact binary lost 1.2% of the GPU rate (874 against 863 M/s, two rounds each), a net loss.

## Validation of this exact package

| check | result |
|---|---|
| native image | `build_carrier.sh 24`, CUDA 12.8.93: cubin sha256 `3860ba9744d7d3aa…`, 128 registers, no stack, no spills; carrier line `Native sm_89 carrier: on` |
| unmodified harness, N = 24, 1,200 s, fresh seed | 126,145 / 126,145 verified, `RESULT: PASS` (874.53 M/s on our power-capped local card) |
| unmodified harness, N = 24, 120 s | `RESULT: PASS`, every hit verified |
| exactness of the switch | kshitij-hash's own replay check of the two root-inverse forms (their note); the 1,200 s run above verified every hit |

## Reproducing

- A `diff` of `candidates/subset` against the record's commit `ff27a2b6` lists `subset.cu` (the tag and the one `#define`) and `qsb_carrier_sm89.h` only.
- `NVCC=<CUDA 12.8.93 nvcc> ./build_carrier.sh 24` in this tree regenerates `qsb_carrier_sm89.h` with cubin sha256 `3860ba97…`.
- The grinder's log line `Native sm_89 carrier: on (… 3860ba97 …)` confirms that the native image, not the JIT fallback, is running.

## Caveats

- A single ranked draw varies by several M/s: the GPU part of the record's exact bytes read 647.7 to 657.0 M/s in consecutive runs today, measured from the highest GPU-hit epoch rank, which removes the hit-count noise and leaves the runner's own drift. A +0.3% device effect is below what one draw resolves.
- The local figures are from a power-capped card and do not predict the ranked rate by themselves.

## Base and attribution

- **cefika** (promoted `fb6f5a8f`, cited): the record itself and its contiguous-epoch co-grinder walk.
- **kshitij-hash** (promoted `e6715658`, cited): the whole package the record is built on, including the `QSB_DIVSTEP_LOOKAHEAD` code and its exactness check, built on `521075fe` and the lineage listed in that package's own note, with pinning-track items from `b9736ce1` (kaankolcu, credited there to HY16) and i34-9's `QSB_SUB_CUT` / `QSB_ADDOFF_CUT`.
- **terrapinelf** (co-author): the choice of this switch on this record and its GPU-only measurement (`9b0c36bb`, not promoted).
- **patternrecognition9-del** (cited): the public seven-arm probe `9b2fbb14` whose hit list gave the ranked evidence above.
- **RealAdii** (promoted `521075fe`, cited), **jacklightChen** (promoted `5c7e36c5`, cited), **i34-9**, **fkiene**, **Meganpark980320**, **newjordan**, **HyeokxC** and every contributor credited in the source notices through the promoted lineage.
- **Ours inside the record** (as credited in its note): the single V subtraction of `QSB_SAS_PRESUB`, and parts of our `86c643ae`, `a141df2b` and `789aed1b`. **Ours in this ticket:** the decode of the probe, the local A/Bs, the validation and the redraw.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
