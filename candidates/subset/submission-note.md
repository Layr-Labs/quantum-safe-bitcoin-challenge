# Subset: the record `fb6f5a8f` with kshitij-hash's `QSB_DIVSTEP_LOOKAHEAD`, `QSB_PRE3_ROOT` and `QSB_TAIL_STAGGER` switches on plus six exact micro-cuts of ours (all bit-identical)

## Starting point

The promoted subset record is cefika's `fb6f5a8f`, 728.34 M/s (GPU part 662.35 + co-grinder 65.99 M/s from its public hit list), landed as benchmark commit `ff27a2b`; the promotion bar is 735.62. Its device side and native image are kshitij-hash's `e6715658` byte for byte (cubin sha256 `e0c0897f799baf81...`); cefika's change is host-only: the co-grinder walks one contiguous epoch range per worker (`QSB_CPU_EPOCH_CONTIG` 1, summarized below). This package is `fb6f5a8f`'s tree byte for byte except for three device switches the record carries at 0 (turned on here), the `QSB_PARK128` value the second and third require (with `QSB_TAIL_PARK` 2), six exact micro-cut switches of ours (below) and a fresh inert tag on line 1 of `subset.cu`. The stray backup file `tests/gpu_epochs/tree.cu.orig`, which no build reads, is not included.

## New in this package (1): `QSB_DIVSTEP_LOOKAHEAD` 1

One switch that `fb6f5a8f` (and `e6715658` before it) carries in its tree at 0: `QSB_DIVSTEP_LOOKAHEAD` 1 (tests/gpu_epochs/tree.cu), the lookahead form of the warp-0 divstep root inverse that kshitij-hash wrote and documented as bit-identical (their replay script checks both forms). We measured it on top of `e6715658`, whose device side `fb6f5a8f` keeps unchanged, GPU-only through the unmodified harness on our RTX 4090, 60 s arms in ABBA order: +0.20, +0.44, +0.34, then +0.09, +0.33, +0.45, +0.44% (7 of 7 rounds positive, mean +0.33% +/- 0.08) and -0.24% energy per candidate. On a fixed seed its hit set matches the base (all 6,286 base hits among the variant's 6,305; the common epoch prefix identical, 6,251 = 6,251).

## New in this package (2): `QSB_PRE3_ROOT` 1 with `QSB_PARK128` 0

Another switch kshitij-hash wrote and ships at 0 in `e6715658` (tests/gpu_epochs/tree.cu, code in pair_shared.cuh, tree_inverse.cuh and kernel_digest). Warps 1 to 7 move the recovery denominator's pre3 step (one field multiply and two adds per candidate: yb = yR*ZZZ, then yb-Y and yb+Y) out of their two front calls into the tree's root window, where they otherwise wait at the down-sweep barrier while warp 0 runs the wave top and the root inverse. Warp 0 keeps the base's order, so the root's critical path does the same work; every branch is warp-uniform. Their tails see the same 12 words, so the hit set is unchanged by construction. The switch does not build together with `QSB_PARK128` 1 (the record's value): patternrecognition9-del's ticket `6c2a321f` showed that `QSB_PARK128` 0 unlocks it, which is how we set it here. That ticket also moves `QSB_Q_MIX` to 8; we measured `QSB_Q_MIX` 8 at -0.50% on our card and keep the record's 4.

`kernel_digest` stays at 128 registers with no stack frame and no spills; the chain loop is one instruction longer than the record's (1,106 instructions, 25 MOV, 55 SEL, 666 IMAD.WIDE).

Checks of this exact tree on our RTX 4090 through the unmodified harness:

| check | result |
|---|---|
| fixed-seed identity against our previous package (the lookahead plus seven exact micro-cuts, itself identical to the record on the same seed; seed 24681357) | 6,299 = 6,299 hits on the common epoch prefix, identical |
| GPU-only A/B, 60 s arms, ABBA, 4 rounds, against our best previous package (the lookahead plus seven exact micro-cuts) | +0.19, +0.13, +0.22, +0.25% (mean +0.20% +/- 0.02), energy per candidate -0.24% |
| native image rebuilt with `build_carrier.sh 24` (CUDA 12.8.93) | cubin sha256 `0b5933b9231d3a11...`, knob string MATCH (2,533 bytes) |
| 90 s run of this exact tree | PASS, 9,801 / 9,801 hits verified |

The gain comes from moving work into time the seven warps otherwise spend waiting for the root, not from fewer instructions.

## New in this package (3): `QSB_TAIL_STAGGER` 1 with `QSB_TAIL_PARK` 2

A third switch kshitij-hash ships at 0 in `e6715658` (pair_shared.cuh, tree.cu). The two paired tails after the block inverse are split into their halves (finish: the eight x-words and two parities; gate: the two pubkey hashes on those nine values), and warps 4 to 7 run both finishes before both gates while warps 0 to 3 keep finish-gate-finish-gate, so the two warps a block has on one sub-partition put different pipes under load at the same time (the finish is IMAD-heavy, the gate ALU-heavy). Same functions on the same values: bit-identical. Like `QSB_PRE3_ROOT`, it does not build with `QSB_PARK128` 1; with 0 it builds, and the parking form `QSB_TAIL_PARK` 2 keeps `kernel_digest` at 128 registers, 0 stack, 0 spills and the chain loop unchanged (the default form 4 spills 4 B next to `QSB_PRE3_ROOT`; form 3 measured -1.2%).

| check (this exact tree) | result |
|---|---|
| fixed-seed identity against the package with `QSB_PRE3_ROOT` alone (seed 24681357) | 6,321 = 6,321 hits on the common epoch prefix, identical |
| GPU-only A/B, 60 s arms, ABBA, 4 rounds, against the package with `QSB_PRE3_ROOT` alone | +0.04, +0.08, +0.11, +0.20% (mean +0.11% +/- 0.03), energy per candidate -0.22% |
| native image rebuilt with `build_carrier.sh 24` (CUDA 12.8.93) | cubin sha256 `723c57b3369e2279...`, knob string MATCH (2,533 bytes) |
| 90 s run of this exact tree | PASS, 9,813 / 9,813 hits verified |

## New in this package (4): six exact micro-cuts (ours)

New compile-time switches, each in the image's knob list, each computing the same words as the code it replaces (0 = the record's code):

| switch | file | change |
|---|---|---|
| `QSB_PW_QN` 2 | tests/gpu_epochs/parity_window_subset.cuh | the parity-window product drops four adds of words that are uniformly zero at this window size |
| `QSB_TAIL_K32` 1 | tests/gpu_epochs/filter_tail_sc.cuh | the tail's short field subtract/add in the K32 form, bit-identical to the record's `SHORT_CARRY4` forms |
| `QSB_FRONT_K32` 1 | tests/gpu_epochs/pair_shared.cuh | the same K32 short forms in the front's finish/prepare steps |
| `QSB_SEED_K_FOLD` 1 | hit_filter_field_sc.cuh | the seed's small h times K as two 32-bit operations |
| `QSB_GLV_UNI` 1 | GLVScalar.cuh | the GLV rounding fallback taken warp-uniformly with a per-lane select |
| `QSB_GLV_RND_FOLD` 1 | GLVScalar.cuh | the GLV rounding bias folded into the high-product sum |

A seventh cut of the same set, 128-bit shared-memory rows in the block tree (`QSB_TREE128`), stays off: it does not combine with `QSB_PARK128` 0 (-2.1% in our measurement). `kernel_digest` stays at 128 registers, no stack frame, no spills, chain loop unchanged (1,106 instructions, 25 MOV).

| check (this exact tree) | result |
|---|---|
| fixed-seed identity against the package without the six cuts (seed 24681357) | 6,321 = 6,321 hits on the common epoch prefix, identical |
| GPU-only A/B, 60 s arms, ABBA, 6 rounds, against the package without the six cuts | +0.01, +0.10, +0.06, +0.12, +0.18% in five clean rounds (mean +0.09%; a sixth round whose base arm ran disturbed is left out), energy per candidate about -0.1% |
| native image rebuilt with `build_carrier.sh 24` (CUDA 12.8.93) | cubin sha256 `7440fb7826bd438c...`, knob string MATCH (2,690 bytes) |
| 90 s run of this exact tree | PASS, 9,672 / 9,672 hits verified |

## Why this ticket

The record's device side has not changed since `e6715658`. This ticket turns on three of kshitij-hash's own bit-identical switches that the record ships at 0, the lookahead root inverse, the pre3 move into the root window and the staggered tails, each positive in every round we measured on top of the previous one, and six exact micro-cuts of ours, with everything else as in the record and the host producers on.

## The co-grinder change in `fb6f5a8f` (cefika, kept unchanged)

- Each co-grinder worker walks one contiguous range of epochs instead of epochs t, t + T, t + 2T, ...; consecutive epochs then share a longer prefix, so less of it is re-hashed per epoch (about 3.4 SHA-256 blocks per epoch instead of about 6.1, by cefika's replay).
- Each epoch's early omissions follow from the previous epoch's by a next-combination step; the binomial unrank runs only at the start of each range.
- The ranges are disjoint and the co-grinder's patterns stay the complement of the GPU's, so no candidate is walked twice; every hit still passes the exact OpenSSL gate.
- Kill switch: `-DQSB_CPU_EPOCH_CONTIG=0` restores `e6715658`'s stride walk. cefika's note for `fb6f5a8f` documents the change and its emulation checks in full.

## kshitij-hash's note for `e6715658` (unchanged)


This package starts from the subset record `521075fe` (commit `46b24eb`) and is not rebased onto the later record
`5c7e36c5`. Its device code is byte-identical to our earlier
submission `8f99a3e9` (cubin `003e3d39`): the chain that carries a pair and reduces once per addition, GLV11 with
`QSB_Q_MIX` 4, and the native sm_89 image. The package adds compile-time switches at file scope. Setting a switch to 0 gives
back the previous code for that part, and with every switch at 0 the PTX is `521075fe`'s apart from the knob string.

`kernel_digest` runs at 128 registers with no stack frame and no spills, 2 blocks per SM as before. The committed native
image is cubin sha256 `e0c0897f799baf81...` (473,376 B), built from this tree's own source with CUDA 12.8.93;
`build_carrier.sh` reproduces it byte for byte. The ranked build line exits 0.

Written with Claude Fable 5.1 and Claude Opus 5.5 in Claude Code.

## Device switches (on)

Files without a directory sit in `tests/gpu_epochs/`. Every device switch is in the image's knob list, so flipping one needs
`build_carrier.sh`.

| switch, default | file | change |
|---|---|---|
| `QSB_GATHER_LEA2` 1 | tree.cu | the 64-byte record address as LOP3, LEA, LEA.HI.X |
| `QSB_GLV_EO` 1, `QSB_GLV_ZDEC` 1, `QSB_GLV_RND` 2, `QSB_DECODE_CUT` 2 | GLVScalar.cuh, tree.cu | the GLV split's residual products from column-pair accumulators, signed residuals for the walker, the rounding constant folded into a carry, and the residual carries taken straight off the carry flag |
| `QSB_TREE_WAVE_TOP` 1 | tree_inverse.cuh | the top of the block inversion tree as four packed waves on warp 0 |
| `QSB_GT_BATCH` 1 | tree.cu | the table build shares one field inversion across 16 records and checks each inverse |
| `QSB_SHA_SCHED_V4` 1, `QSB_SHA_CONST_IV` 1, `QSB_GATE_W8_LEA` 1 | tree.cu, window_schedule_shared.cuh | 128-bit schedule loads, one induction variable for the constant blocks' loop, gate word 8 as one LEA |
| `QSB_YNEG_FOLD` 1 | tree.cu, pair_shared.cuh | the last chain addition hands the pre-inverse step the negated ordinate |
| `QSB_HIT_NO_COMBO` 1 | tree.cu | the hit record keeps its 4-byte tag; the host rebuilds the rest, so the published hit text is unchanged |
| `QSB_FX3_PRESUB` 1, `QSB_FX3_PRESUB_EARLY` 1 | hit_filter_field_sc.cuh | the fused X3 subtracts V once, before the odd row is merged |
| `QSB_K32_SUBCUT` 7, `QSB_K32_ADDCUT` 1 | hit_filter_field_sc.cuh | the K corrections of the subtractions and the anchor sum on limb 0's 32-bit halves |
| `QSB_S3_DOFF` 1, `QSB_S3_UNIFORM_G` 1 | tree.cu | the chain loop counts a byte offset; a warp vote makes the psi branch warp-uniform |
| `QSB_S3_NM_MASK` 1, `QSB_S3_NM_SEED` 1, `QSB_GATHER_ONE_FORM` 1 | tree.cu | the chain's and the seed's gathers take index and mask as two walker values, and each record is loaded in one form |
| `QSB_XNEG_BRANCH` 1, `QSB_PARK128` 1, `QSB_OK_FOLD` 2, `QSB_TID_UNSIGNED` 1 | tree.cu | a grid-uniform sign branch, 16-byte parked rows, one OR for an unusable lane, unsigned half index |
| `QSB_SC_OPS` 48 | hit_filter_field_sc.cuh | the operand order of the point add's seven products, searched over all 32 legal orders for this switch set |

Other device switches are in the tree at 0, each with its alternative code kept for later images.

## Host switches (on)

None is in the knob list, so they leave the image unchanged.

| switch, default | file | change | source |
|---|---|---|---|
| `QSB_SP_REFILL_FIRST` 1 | tree.cu | the slot loop launches batch k before it gates batch k-2 | `521075fe` after `9edbdde7` |
| `QSB_CPU_TRY9` 1, `QSB_CPU_ALLCPU` 1, `QSB_HP_V3` 1 | CpuGrindSubset.h, host_producers.h | a 9-window table on huge pages when memory allows, workers on every CPU, the v3 host producers | `789aed1b`, `a141df2b` |
| `QSB_CPU_X4PS` 1, `QSB_CPU_SHC` 1 | CpuGrindSubset.h | shared schedule rows for four lanes; padding carried as constants | `c13302f3` |
| `QSB_CPU_KH16` 1, `QSB_CPU_MRG` 1, `QSB_CPU_AINL` 1 | CpuGrindSubset.h | 16-key key-hash schedules, one accumulator per column, inlined small field operations | `a33e04c3` |
| `QSB_CPU_BATCH_AUTO` 1 | CpuGrindSubset.h | batch size chosen from the CPU topology | `86c643ae` |
| `QSB_CPU_PREFIX100` 1 | CpuGrindSubset.h | the co-grinder walks the 100 of its 158 window patterns whose first message block five patterns share | `b67487a1`, after `4a197f06` |
| `QSB_CPU_SHA4` 1, `QSB_CPU_DNF` 1, `QSB_CPU_RECODE_NG` 1, `QSB_CPU_FOLD4` 1 | CpuGrindSubset.h | four interleaved SHA-NI lanes, an unfolded limb in the window step, hash words by transpose instead of gathers, an 11-IFMA reduction | ours |
| `QSB_CPU_ILP2` 1, `QSB_CPU_NCH` 2, `QSB_CPU_PFSPREAD` 3 | CpuGrindSubset.h | two groups interleaved in the backward passes, two inversion chains instead of four, spread row prefetches | ours |
| `QSB_NO_SUMMARY_FSYNC` 1, `QSB_LAZY_MODULES` 1, `QSB_CPU_FOLD_PAR` 1, `QSB_CPU_TOUCH_GUARD` 1, `QSB_FAST_TEARDOWN` 1, `QSB_STARTUP_THREADS` 1 | tree.cu, CpuGrindSubset.h | start-up and exit hardening: no fsync of an unread file, lazy module loading, a parallel fold, a first-touch time limit with a 10-window fallback, table unmap during the drain, threaded start-up ladders | ours |

## Exactness

Most device switches are bit-identical: the same addresses, words, coefficients and field elements by a different route.
The GLV split, the signed-residual walker and the gather forms are replayed by the start-up self-check before each run, which
also catches deliberately broken variants. The point add at this operand order matches a bigint model of the madd on every
tested random state. The paired hash matches OpenSSL's compression on 10^6 cases.

Some cuts use the rare-carry class the chain and tree already have in `521075fe`: the tree top's short-carry products and the
K32 corrections can drop a carry in a small fraction of candidates (about 2^-16 to 2^-21 each). Such a candidate can lose a
hit but can never publish a wrong one, and the loss is a few hits per hundred thousand. The host gate recomputes every GPU
nomination and every co-grinder hit with OpenSSL before publishing it.

The co-grinder switches run the same operations on the same operands (reordered, regrouped with every lazy value carried
before a subtraction, or with prefetches moved). On fixed work (one worker, 16,179,200 candidates, `QSB_ZEROS_N` 14) the
co-grinder gives the same hit file with each host switch on or off; the pattern selection changes which candidates are
walked, stays disjoint from the GPU's patterns, and every one of its hits re-derives with the harness's own
`problem.candidate_hash`. Hit sets matched in every comparison we ran between this package and its predecessors.

## Full run of this exact package

One official-path run of this exact tree, 1,200 s, on a rented Zen 4 EPYC slice (12 CPUs) with an RTX 4090 (CUDA 12.8).
We copied the harness, replaced `candidates/subset`, deleted any binary, then ran `./setup.sh subset` and
`./benchmark.sh subset` through the command grinder with a cold JIT cache and problem seed 20260927. `setup.sh` exited 0 and
the native image loaded.

| result | score | verified hits | elapsed |
|---|---:|---:|---:|
| PASS (scored) | 872.95 M/s | 125,077 of 125,077 | 1,201.92 s |

This host is not the runner (its card held its power limit and the co-grinder had 12 CPUs), so the score does not predict
the ranked one. The run shows that the package builds with the ranked line, loads its image and publishes only verified
hits.

## Reproducing

```
./setup.sh subset
./benchmark.sh subset
```

Off the official runners, the harness's command grinder runs the same build:

```
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/{bench}/{bench}.cu --no-build' ./benchmark.sh subset
```

To isolate a change, set its switch with `-D` or at its `#define` and rebuild. After any device switch flips, regenerate the
image with `NVCC=/path/to/cuda-12.8/bin/nvcc ./build_carrier.sh 24`. After a change to a `QSB_FX3_*`, `QSB_K32_*`,
`QSB_S3_*` or `QSB_GATHER_*` switch, search `QSB_SC_OPS` again. `QSB_S3_NM_MASK` 0 needs `QSB_GATHER_ONE_FORM` 0.
`QSB_CPU_ALLCPU` 0 puts `QSB_CPU_RSV_CORE` back to 1.

## Base and credits

- Base: `521075fe`, itself built on our `8f99a3e9`; the refill-before-publish order after `9edbdde7`.
- Through `8f99a3e9`: `2d1631b0` (the co-grinder engine, with the batch-affine prefix after `55757d4d`), `888f5fce` after
  `212237f4` (the blocking host wait), `de5739c9` (the promoted GLV12xc base, half-walk, the 8-lane IFMA and 4-lane SHA-NI
  co-grinder), `82d8493f` and `97f347a8` (host producers, warp-uniform root, row prefetch), `bb2a3eb7`, `2a1f43c5`,
  `d1ddefca`, `25bd990a` / `7a75fa50` (carrier and co-grinder design), `73224391`, `eaba5205` / `b864a72c`, `14675ab0` /
  `adfa8aaa`, `933abead`; libsecp256k1 (MIT, notice kept).
- From queued work: `789aed1b` and `a141df2b` (also in `4da17ebc`); `c13302f3`; `86c643ae`; `a33e04c3`; `b67487a1` after
  `4a197f06`.
- From pinning work: the pinning record `b9736ce1` (kaankolcu) for `QSB_DECODE_CUT` bit 2 (credited there to HY16) and the
  seed gathers' index and mask form; the single V subtraction of ercumentyildirim's `QSB_SAS_PRESUB`; i34-9's `QSB_SUB_CUT`
  and `QSB_ADDOFF_CUT`, written here as the K32 switches.
- Ours: the remaining device switches, the operand-order search, `QSB_CPU_SHA4`, `QSB_CPU_DNF`, `QSB_CPU_RECODE_NG`,
  `QSB_CPU_FOLD4`, `QSB_CPU_ILP2`, `QSB_CPU_NCH`, `QSB_CPU_PFSPREAD`, the hardening switches, and the ports of the items
  above as switches.

Their authors are credited as coauthors in the submission metadata up to Yukon's limit of ten; cefika (`4a197f06`) and
anamdongparkjinhyeong (`9edbdde7`) are credited here. All inherited source, GPLv3 notices and attributions are kept.


## Attribution

- **cefika**: the promoted record `fb6f5a8f` this ticket starts from: the co-grinder's contiguous epoch walk on top of `e6715658`.
- **kshitij-hash**: the record `e6715658` in full, including the `QSB_DIVSTEP_LOOKAHEAD`, `QSB_PRE3_ROOT` and `QSB_TAIL_STAGGER` switches turned on here.
- **patternrecognition9-del** (co-author): ticket `6c2a321f`, which showed that `QSB_PARK128` 0 makes `QSB_PRE3_ROOT` build (and with it `QSB_TAIL_STAGGER`).
- Everyone credited in kshitij-hash's note above and in cefika's note for `fb6f5a8f`.
- **terrapinelf** (us): the six micro-cut switches, the measurements above and this ticket.

## Ranked result of our previous ticket `3b9b0936`

`3b9b0936` scored **711.22** (self 850.6); public hit list split: GPU 646.04 + co-grinder 65.18 M/s. Line 1 of `subset.cu` carries a fresh inert tag (`QSB_REDRAW_09300447`) so the archive is new.
