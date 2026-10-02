Model: Claude Fable 5.1 and Claude Opus 5.5
Harness: Claude Code

# Subset: exact device cuts, a one-form chain gather and co-grinder scheduling on 521075fe

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

