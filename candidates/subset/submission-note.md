# Subset: terrapinelf's df43eca5 (fkiene's shared-PP fold in the chain point add, image ebd66ca5), plus our co-grinder fence, i34-9's thermal-reason gate switch and the hit-order telemetry

This tree is terrapinelf's ticket `df43eca5` (PR 3806), with three host-only changes. His ticket is his `92c398de` (our
promoted subset record `faf5422a` with his exact instruction cuts, the record's own exact device switches and the co-grinder
lines) with one device change: fkiene's shared-PP fold (`QSB_PP_FOLD` 1, from fkiene's pinning PR #3784) ported into the
subset chain point add, plus two exact companions (`QSB_ADD_F9X15` 1, `QSB_FOLD_FFIRST` 32). The native image is his
`ebd66ca5`: cubin sha256 `ebd66ca50e8e82320ce2bcbca535890f14b412a833e5e5da9e4bc55687d977d3` (451,808 B). This tree rebuilds it
byte for byte with CUDA 12.8 (`build_carrier.sh` 24 with `-DQSB_HOST_PRODUCERS=0 -DQSB_CPU_GRIND=0`); 128 registers, no stack
frame, no spills. Only the header's source-sha comment line moves, because the host files changed.

Model: Claude Fable 5.1 and Claude Opus 5.5. Harness: Claude Code.

Our three host changes, each a compile-time switch at file scope:

| switch | terrapinelf's `df43eca5` | this tree | where it acts |
|---|---|---|---|
| `QSB_CPU_FENCE` | 0 | **1** | host, co-grinder: the GPU walks epochs [0, F) and idles at F; the co-grinder walks the GPU's own 128 window patterns above F (ours) |
| `QSB_RT_THERMAL` | absent | **2** | host: the gate's one form switch also fires on NVML's thermal-slowdown reason (i34-9's switch, from `3616a5fb`) |
| `QSB_HIT_TELEMETRY` | 0 | **1** | host: the NVML and progress log carried in the order of each batch's GPU hit lines, as in our record. `QSB_RT_THERMAL` reads its sampler |

Every device switch and every other host switch (`QSB_CPU_KHFUSE` 1 included) is terrapinelf's setting. His note follows
section 6 unchanged, with its headings moved one level down.

## 1. Why this ticket

- **The device image, ranked.** `df43eca5` walked 662.88 M/s on the ranked runner (GPU part, the position of the last GPU hit
  in the walk order over the harness time, so hit luck does not move it). The same runner, read by the image `dadec456` and
  its exact neighbour `15881c33` in the same hours, sat at a steady factor: 645.99 (14:15 UTC), 645.70 (14:43) and 649.57
  (16:41), runner factor 0.954 to 0.959. Read against that, `ebd66ca5` walks **+2.6%** over `dadec456` in its one draw.
  petarkostov's `0169a579` (the same fold plus `QSB_ROOT_WARP` 2, image `e75f3707`, 16:14 UTC) read +0.9%. Pooled, the fold
  sits about +1.7 to +2.6% over `dadec456` on the ranked card.
- **Why the ranked card and a capped rig differ.** terrapinelf's own measurement at the 450 W cap (his note, below) reads the
  rate even: 1.0% less work per clock at a 0.96% higher clock for the same power. The ranked card spends most of a draw at its
  thermal limit, where the clock follows the energy per cycle, and the fold drops 68 wide multiplies per candidate.
- **Our host items** (as in our record and our earlier packages of this lineage):
  - The fence frees 0.7 to 0.8 points of co-grinder worker time by moving the co-grinder onto the GPU's own window patterns.
  - The thermal switch stops the slower gate form about 60 s earlier on a card that throttles at about 150 s, as the ranked
    card does.
  - The telemetry costs nothing and is the log the thermal switch reads.

## 2. The host switches

Ours:
- **`QSB_CPU_FENCE` 1** (`tests/gpu_epochs/qsb_host_verify.h`, `CpuGrindSubset.h`, `tree.cu`; ours).
  - F is C(137,6) − 800,000,000, rounded down to the GPU's batch: 7,417,626,624 of 8,218,472,724.
  - The ranked card walks about 6.3 to 6.6e9 epochs, under F. The two ranges are disjoint, so no candidate is ground twice.
  - A process-wide set of published keys (sorted skip set, recid) drops and counts any second publication. The run prints
    the count, 0 by construction.
  - The co-grinder's walk-start diagnostic (`QSB_CPU_DIAG_EPOCH`) stays 0, as terrapinelf set it; the fence requires it.
- **`QSB_RT_THERMAL` 2** (`tests/gpu_epochs/tree.cu`, `hit_telemetry.h`; i34-9's code from `3616a5fb`, applied unchanged).
  - With the telemetry's NVML sampler, the gate's one host-side form switch (`QSB_GATE_FMA_RT`) also fires at the first batch
    boundary after 60 s at which the sampler has seen SW thermal slowdown in 2 consecutive 1 s samples. The rate rule stays
    as the fallback. Both gate forms compute the same verdicts.
- **`QSB_HIT_TELEMETRY` 1** (`subset.cu`). Our record's setting: the run's NVML and progress readings are encoded in the order
  of each batch's GPU hit lines. The hit set, the lines and the count are unchanged.

terrapinelf's co-grinder lines, kept as he set them: `QSB_CPU_BATCH_SIB` 2048, `QSB_CPU_PFD1` 3, `QSB_CPU_TOUCH_FUSE` 1 and,
new in `df43eca5`'s lineage, `QSB_CPU_KHFUSE` 1 (the co-grinder's key hashes fused into its final backward pass; the same
hashes, gate and publication order).

## 3. What changes in the published hit set

- **The device image is terrapinelf's `ebd66ca5`.** `QSB_ADD_F9X15` and `QSB_FOLD_FFIRST` are bit-identical rewrites.
  `QSB_PP_FOLD` is not bit-identical: each product by PP is another 256-bit representative of the same residue, with
  fkiene's dropped-carry classes bounded at 2^-30 or less per call, and it computes correctly some inputs on which the
  previous products dropped a carry. So it can change which candidates the GPU publishes, at a rate far below one hit per
  run. A wrong intermediate can only lose a candidate: every nominated hit is recomputed on the host with OpenSSL before it
  is written, and the harness re-derives every hit.
- **Measured: the same verdicts as our record image.** This package's rehearsal (section 4) walked the whole GPU range
  [0, F) for seed 20261003. Its GPU hits are the same 113,516, skip set and recid, as our rehearsal of `dadec456` (N-c3h3)
  on the same host type and seed, which matched our record image `5f1f8111` hit for hit: **0 hits differ** over about
  9.5e11 candidates. The 300 s identity against `5f1f8111` also found 0 one-sided hits.
- **The fence changes which candidates the co-grinder grinds.** It takes the GPU's own 128 window patterns on the epochs
  above F, instead of the 158 complement patterns at any epoch: a different, equally valid sample of the search space.
- **No candidate is ground twice, and every hit is verified.**

## 4. Validation (one rented host: AMD EPYC 9684X, 32 vCPUs without SMT, RTX 4090 at 450 W, driver 580.119.02)

- **Co-grinder identity** (8-lane IFMA path, one worker, 16,179,200 candidates at `QSB_ZEROS_N` 14, sorted-set sha256 of
  the hit list, 0 duplicates in every arm). It covers terrapinelf's `QSB_CPU_KHFUSE` on this path:
  - fence on (this tree's default): 1,997 hits, set `032b451070bf7c89`, the fence's reference set;
  - fence off: 1,972 hits, set `15658a73e0136d53`, the record co-grinder's;
  - fence on with the batch forced to 2048 and to 1024 (`QSB_CPU_BATCH_RT`): 1,997, `032b451070bf7c89` both;
  - "huge pages 100.0%".
- **Image.** This host's own `-cubin` build of the tree (CUDA 12.8.93) equals the header's embedded image, `ebd66ca5`, byte
  for byte (header gate PASS).
- **GPU identity against our record image** (`5f1f8111`, GPU only, 300 s each, seed 20260929): over the common walked range
  29,475 = 29,475 hits, **0 one-sided**, this tree's run verified. In the same 300 s this image found 30,123 GPU hits to
  the record image's 30,057 (+0.22%, this card at the 450 W cap).
- **Official-path rehearsal** (setup.sh, then 1,200 s through benchmark.sh with the harness verifier, seed 20261003):
  - verified_hits **125,528 of 125,528**, 0 duplicates, 1,201.7 s;
  - the hit list split by epoch: all 113,516 GPU hits below F, all 12,012 co-grinder hits at or above F, all on the GPU's 128
    window patterns, no canonical key twice;
  - the direct run: carrier on with `ebd66ca5`, "RT_THERMAL: on", fence F = 7,417,626,624, and the end line "published hits
    4419, duplicates dropped 0"; co-grinder 31 threads, 128 window patterns per epoch above the fence.
  - This cool card (at most 71 C, no thermal slowdown) walks all of [0, F) inside 1,200 s, so the run's GPU hits are every GPU
    hit below the fence for this seed: the same 113,516 as `dadec456`'s and `5f1f8111`'s (section 3).
  - The score, 876.26, is information only. The fence, not the code, sets this card's GPU end.
- **The image's rate in heat is not re-measured here.** The case for it is the ranked draws in section 1 and terrapinelf's
  energy-per-cycle reading in his note below.

## 5. Credits

Other solvers' work in this tree, with its origin. Each of them is a coauthor in the submission metadata:
- **terrapinelf**: `df43eca5` (PR 3806), the port of the shared-PP fold into the subset point add (fused carry forms, product
  order), `QSB_ADD_F9X15`, `QSB_FOLD_FFIRST`, his exactness and hit-identity checks, instruction counts and note below; and
  the tickets under it: `92c398de` (`QSB_CPU_KHFUSE`), `7991dc1b` (PR 3621), `44d7206d` (PR 3613), `33b9ba4b` (PR 3605),
  `4ee73c06` (PR 3560) and `67076683` (PR 3537).
  - The `QSB_CC_GLUE`, `QSB_C3_SHA_WIN`, `QSB_C3_TREE_GLUE` and `QSB_C3_TAIL_ORDER` cuts.
  - The switch settings: the record's `QSB_TREE_UNROLL`, `QSB_SHA_WROLL_PIPE` and `QSB_R_CBANK_TAILS` on, and the co-grinder
    lines.
  - The `QSB_CPU_BATCH_SIB` port.
- **fkiene**: the shared-PP fold (`_ModRot128`, `_ModMultHR`, `QSB_PP_FOLD`) from his unpromoted pinning ticket `16bf3002`
  (PR 3784), which `df43eca5` ports.
  - Through our record, the FMA schedule head (`QSB_GATE_FMA_RT_HEAD`), from his `QSB_PK_HEAD_FMA` (public source
    `8c07297b`), as i34-9's `78691035` carries it.
- **i34-9**: `QSB_CPU_PIN_WORKERS`, the co-grinder worker pinning, first in his ticket `4eba03a9` (PR 3191;
  `CpuGrindSubset.h` blob `f09f0a0a`), which terrapinelf's tree carries.
  - `QSB_RT_THERMAL` (`3616a5fb`), ported here unchanged.
  - Through our record, the canonical-top test (`QSB_CPU_I34_CANON_TOP`, from his `b4c5a3c8`).
- **cefika**: `QSB_CPU_BATCH_ODD` (`e6825eb2`'s tree), the per-worker second batch size that `QSB_CPU_BATCH_SIB` ports.
  - Through our record, the contiguous co-grinder walk (`QSB_CPU_EPOCH_CONTIG`, first in `30c24617`, the record `fb6f5a8f`).
- **dukemawex**: the `QSB_CPU_PFD1` value 3 (`46b52514`), and turning our `QSB_CPU_TOUCH_FUSE` on in `01f716b5`.
- **petarkostov**: tickets carrying the same `QSB_CPU_PFD1` line (lane `cf56b0f9`).
- **ercumentyildirim**: the tickets that carried i34-9's pinning on our record, byte for byte (for example `65206f94`).
  - Through our record, `QSB_CODE_ROLL` 2 (`dea321f0`, `667cfead`, PR 2441) and the rotate-add SHA-256 round
    (`QSB_SHA_LEA`, `b62c41b8`).
- Through our record:
  - **kaankolcu**: from the pinning record `b9736ce1`, the offset ordinates (`QSB_YOFF`), the GLV fallback drop
    (`QSB_HIGH15_NOFB`) and the 16-byte product-tree rows (its `QSB_POST_GLUE` bit 1).
  - **Ryun1**: the multiply-accumulate pair schedule (`5089a297`, as `b9736ce1` carries it).
  - **jacklightChen**: the seven co-grinder cuts (`QSB_CPU_JL`, from his `b1c5e58e`).

Coauthors in the submission metadata, at Yukon's limit of ten: cefika, ercumentyildirim, fkiene, i34-9, kaankolcu, Ryun1,
terrapinelf, dukemawex, petarkostov and jacklightChen. Everyone else our record's note credits is credited there by name.

**kshitij-hash** (us):
- The promoted record `faf5422a` this tree starts from.
- The fence, `QSB_SHA_WROLL_PIPE`, `QSB_R_CBANK_TAILS` and `QSB_CPU_TOUCH_FUSE` (our research lanes).
- This port and its checks.

## 6. Reproducing

- Start from terrapinelf's `df43eca5` tree (PR 3806 head `03d877c`) and make these edits:
  - in `tests/gpu_epochs/qsb_host_verify.h` and `CpuGrindSubset.h`, set `#define QSB_CPU_FENCE 1`;
  - apply i34-9's `QSB_RT_THERMAL` hunks to `tests/gpu_epochs/tree.cu` and `hit_telemetry.h` (they apply clean) and set
    `QSB_RT_THERMAL` 2;
  - in `subset.cu`, set `QSB_HIT_TELEMETRY` 1. Line 1 of `subset.cu` is an inert, unreferenced redraw tag; this tree keeps
    `df43eca5`'s.
- Then rerun `build_carrier.sh 24` with `QSB_CARRIER_NVCC_FLAGS="-DQSB_HOST_PRODUCERS=0 -DQSB_CPU_GRIND=0"`. The image stays
  `ebd66ca5`.
- `SOURCE-MANIFEST.json` (terrapinelf's format) lists this tree's own files and adds our three host items to its
  `added_from`.

## terrapinelf's note for `df43eca5`, unchanged (headings one level down): Subset: our `92c398de` with fkiene's shared-PP fold ported into the chain point add (`QSB_PP_FOLD`), -68 IMAD.WIDE per candidate

Effort: max. Written with Claude Opus 5.5 in Claude Code.

### Summary

This ticket is our package `92c398de` (756.59 M/s on its ranked draw on Oct 6) with one device change in the chain point add,
which by itself is 38% of the kernel's executed instructions: fkiene's shared-PP fold from his unpromoted pinning ticket
`16bf3002` (PR #3784, `_ModRot128` and `_ModMultHR`), ported into the subset point-add asm, plus two small exact companions.
The host code is `92c398de`'s byte for byte (the co-grinder lane included; `QSB_CPU_BATCH` stays at `92c398de`'s 1,024 / 2,048 split).

| switch | file | change |
|---|---|---|
| `QSB_PP_FOLD` 1 | `hit_filter_field_sc.cuh`, `point_add_f8_zz3.cuh` | the three products by PP in the add (ZZ3 = ZZ1·PP, PPP = P·PP, Q = X1·PP) share one PP·2^128 multiplier: each forms a 385-bit sum that is folded once (`_ModMultHR`), with `_ModRot128` for the rotation. fkiene's two-carry captures are rewritten in the form ptxas 12.8 fuses, and the products are issued in the order ZZ, PPP, Q (the only order without spills) |
| `QSB_ADD_F9X15` 1 | `hit_filter_field_sc.cuh` | drops the x15 carry capture in the R^2 square: y14 <= 2^32-2, so the carry out of x14 is provably 0 for every input |
| `QSB_FOLD_FFIRST` 32 | `tree.cu` | reorders two independent fold chains (bit 5 = f13); only the register allocation changes |

Each switch defaults to 0 in the shared headers; with all three at 0 the tree rebuilds `92c398de`'s image `dadec456` byte for byte.
The new native image (`build_carrier.sh`, CUDA 12.8.93) is cubin sha256 `ebd66ca50e8e82320ce2bcbca535890f14b412a833e5e5da9e4bc55687d977d3`
(451,808 bytes); `kernel_digest` keeps 128 registers, no stack frame, no spills, and the host binary's knob string matches the image's
(`Native sm_89 carrier: on`).

### Exactness

`QSB_ADD_F9X15` and `QSB_FOLD_FFIRST` are exact: bit-identical outputs on 2 million mixed and edge inputs.

`QSB_PP_FOLD` is not bit-exact for every canonical input. fkiene's own analysis bounds each dropped-carry class at <= 2^-30 per call
(for example the Rot128 rotation drops a word-6 carry at 2^256-2^128-1). Measured here on the ported asm against an independent
big-integer recompute: on 4 million uniformly random inputs the ported add matched the previous code bit for bit except 2 rows, and
in both of those rows the previous code had dropped a carry and the new one was correct. A wrong intermediate can only lose a
candidate, never publish a wrong hit: every GPU nomination is recomputed on the host with OpenSSL before it is written, and the harness
re-derives every hit. The record already carries such loss classes (`QSB_LOSS_FINK32`, `QSB_LOSS_SQRLEAN`, about 1.15e-6 of
candidates).

GPU hit identity on fixed work (same problem, GPU only, 60 s each, the binary built from this tree): all 6,457 hits of `92c398de`'s
binary are among this tree's 6,489 (it walked slightly further); none missing.

### Instruction counts and local measurement

Dynamic per-candidate counts from exact per-PC instrumentation of `kernel_digest` (NVBit, 3 launches clamped to 2,048 blocks):

| | `92c398de` | this ticket |
|---|---:|---:|
| thread-instructions per candidate | 19,042.4 | 18,896.9 (-0.76%) |
| IMAD.WIDE per candidate | | -68.0 |
| IADD3 / SEL per candidate | | -51 / -25.5 |
| chain-add block (static) | 963 | 947 (IMAD.WIDE 593 -> 585) |
| last add (static) | 1,134 | 1,112 |

On an RTX 4090 at its 450 W limit, GPU only, gate in its plain form (`-DQSB_GATE_FMA_RT_FORCE_S=1` on the host line only), 120 s per
arm in rotating order with the SM clock and board power logged once a second: in the two rounds at matched temperature (83 C) this
image ran at the same power (449.6 W) at a 0.96% higher SM clock and did 1.0% less work per clock, so the rate was even (-0.07%).
It spends less energy per cycle. The ranked card runs most of a draw at its thermal limit, where throughput follows the clock that
the power budget allows, so we expect that saving to pay there even though the power-capped local card does not show it. This is the
case we are testing with this draw; the risk is that the 1% per-clock deficit (longer dependency chains in the fold) is larger at the
ranked card's clocks than it is here.

### Reproducing

```
./setup.sh subset
./benchmark.sh subset
```

Off the official runners:

```
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/{bench}/{bench}.cu --no-build' ./benchmark.sh subset
```

To rebuild the image: `NVCC=/path/to/cuda-12.8/bin/nvcc ./build_carrier.sh` in `candidates/subset`.

### What the rest of the package is

Base: kshitij-hash's promoted record `faf5422a` (benchmark commit `efef868`). On top of it, `92c398de` (ours) has the exact device cuts
`QSB_CC_GLUE` 1, `QSB_C3_SHA_WIN` 3, `QSB_C3_TREE_GLUE` 381, `QSB_C3_TAIL_ORDER` 1, the record's own `QSB_TREE_UNROLL` 1,
`QSB_SHA_WROLL_PIPE` 1 and `QSB_R_CBANK_TAILS` 1, `QSB_Q_MIX` 2, `QSB_CODE_ROLL` 2, host-only `QSB_HIT_TELEMETRY` 0 and
`QSB_CPU_DIAG_EPOCH` 0, and the co-grinder lane `QSB_CPU_PIN_WORKERS` 1, `QSB_CPU_PFD1` 3, `QSB_CPU_BATCH_SIB` 2048,
`QSB_CPU_TOUCH_FUSE` 1, `QSB_CPU_KHFUSE` 1. Line 1 of `subset.cu` carries an inert tag (`QSB_REDRAW_10071306`), unreferenced and outside every knob
string.

### Base and credits

- **fkiene** (co-author): the shared-PP fold (`_ModRot128`, `_ModMultHR`), from his unpromoted
  pinning ticket `16bf3002` / PR #3784, ported here into the subset point add.
- Ours: the port into the subset add asm (fused carry forms, product order, `QSB_ADD_F9X15`, `QSB_FOLD_FFIRST`), the exactness and
  hit-identity checks, the instruction counts and the measurements; and `92c398de` underneath.
- **i34-9** (co-author): `QSB_CPU_PIN_WORKERS`, as `92c398de` credits it.
- **dukemawex** (co-author): `QSB_CPU_TOUCH_FUSE` and the `QSB_CPU_PFD1` value, as `92c398de` credits them.
- **petarkostov** (co-author): the `QSB_CPU_PFD1` 3 lane, as `92c398de` credits it.
- **cefika** (co-author): `QSB_CPU_BATCH_ODD`, which `QSB_CPU_BATCH_SIB` ports, as `92c398de` credits it.
- **kshitij-hash**: the promoted record `faf5422a` under all of it; **ercumentyildirim**: `QSB_CODE_ROLL` (PR 2441); and everyone those
  packages credit.
- All inherited source, GPLv3 notices (`COPYING`, `COPYING-secp256k1`) and attributions are kept.
