# Pinning preparation experiment, 2026-09-24

Status: CPU algebra/source checks and CUDA compilation only. No GPU timing or
device hit-set comparison. Selected for one official Yukon validation;
see SUBMISSION-OVERLAP.md for the final configuration and full limitations.

## Baseline and attribution

The baseline is promoted main `1fe5a8e40008befcd917668ea9b1a23c6ee590c4`,
881,273,403 verified candidates/s. It includes the six-segment GLV12 big table
introduced by odinfree and later replayed by anamdongparkjinhyeong. All
inherited licences are retained.

The lean GLV coefficient header and its test come from Portablelle's public
`1ec3157008734f913e1e95997e758bf45f3b0eaa`, crediting the ItlaStudent lineage.
Only this exact coefficient arithmetic is imported. The later donor's lossy
field multiply change, host verification changes and table sampling are not
part of this candidate. The independent seed-sign rewrite is ported from our
earlier `ac934fd` experiment. These components need separate GPU arms before
a combined candidate is selected.

## Switches

| Switch | Default | Purpose |
| --- | --- | --- |
| `QSB_GLV_LEAN` | 1 | Reduce work in constant coefficient multiplication |
| `QSB_GLV_HIGH10_HI` | 1 | Guarded high-limb coefficient path with exact fallback |
| `QSB_GLV_ROUND_CC` | 1 | Express coefficient rounding with condition-code carry |
| `QSB_SEED_FUSE_X3` | 1 | Reverse both seed differences and reuse fused X arithmetic |
| `QSB_OVERLAP_REG_CAP` | 104 | Preparation-only register cap |
| `QSB_GLV_ANCHOR_PARK` | 1 | Park next anchor in unused digit-arena planes |

The cap uses a separately named kernel with `__maxnreg__`. Both versions
include the same `PinningPipeline.cuh` body. The finishing kernel keeps its
original launch bounds. An early implementation also capped finishing and
introduced spills in the default sm_52 build; that implementation is superseded.

## Compiler evidence

CUDA 12.8.93, organizer-default compute_52 PTX reassembled with ptxas for sm_89:

| Arm | Preparation registers | Spill stores/loads | Static preparation instructions |
| --- | --- | --- | --- |
| Promoted baseline | 122 | 0 / 0 | 6,680 |
| Lean GLV + seed sign, cap 0 | 114 | 0 / 0 | 6,608 |
| Lean GLV + seed sign, cap 112 | 110 | 0 / 0 | 6,616 |

Preparation uses 12,288 bytes of shared memory. Finishing remains at 64
registers and zero spills. A full default executable with cap 112 also builds
with zero spills; its sm_52 register allocation differs from sm_89.

The cap may leave register capacity for a preparation block and a finishing
block to share an SM. NVIDIA's occupancy and stream documentation only shows
that this can be allowed; it does not establish actual overlap or a speedup.
Driver JIT, block scheduling, memory traffic and power limits remain material.
The observed static instruction reduction is not a measured rate prediction.

## Correctness and controls

```bash
python3 -B candidates/pinning/test_glv_coeff.py
python3 -B candidates/pinning/test_seed_fuse.py
```

The GLV test checks 300,818 inputs for each reciprocal (601,636 coefficient
comparisons), including high-limb fallback boundaries. It executes a CPU wrapper
of the source helpers. The seed test checks 4,096 edge tuples and 10,000 random
tuples over the field; degenerate equal-X cases are skipped. It proves the
sign transformation's algebra, not GPU PTX execution or complete point-chain
equivalence. A device comparison is still required.

The promoted control disables all six switches. The final defaults enable
the lean GLV, seed rewrite, anchor parking and cap 104; the original matrix
below documents the earlier screen before anchor parking. For attribution,
GPU runs should compare control, lean GLV only, seed only, their combination,
then combination plus cap 112. Use equal work and independently verified hits,
followed by full-duration fresh-problem runs on RTX 4090. A timeline is needed
to test the specific overlap hypothesis.

```bash
# Proposed capped arm; use CUDA 12.8.
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_OVERLAP_REG_CAP=112 \
  -o /tmp/pinning-overlap candidates/pinning/pinning.cu -lcrypto -lm
# Promoted control adds:
# -DQSB_GLV_LEAN=0 -DQSB_GLV_HIGH10_HI=0 -DQSB_GLV_ROUND_CC=0
# -DQSB_SEED_FUSE_X3=0 -DQSB_GLV_ANCHOR_PARK=0 -DQSB_OVERLAP_REG_CAP=0
```

Detailed logs and a source inventory are in the sibling `qsb-study-20260924`
directory. Historical `SUBMISSION.md` is an inherited
donor record, not a claim about this experiment. This research note
and the refreshed `SOURCE-MANIFEST.json` identify the new work.
