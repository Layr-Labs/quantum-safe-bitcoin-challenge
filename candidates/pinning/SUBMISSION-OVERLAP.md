# Pinning: shorten anchor lifetime for spill-free preparation overlap

Effort: xhigh. Track: pinning. Prepared with GPT 6 Astra in Codex.
No local NVIDIA GPU was available. The official run is the first RTX 4090
performance test of this package. Compiler counts below are not speed claims.

## Baseline and attribution

The production baseline is promoted commit
`1fe5a8e40008befcd917668ea9b1a23c6ee590c4`, submission
`2c7a195e-48d6-4497-8530-ecea28b042df`, at 881,273,403 verified candidates/s.
It inherits odinfree's GLV12 big table and anamdongparkjinhyeong's replay.
This package retains its six table segments, 48 MiB dense prefix, persisting
L2 window, two-slot pipeline, packed cofactor recovery and exact host gate.

Portablelle's public `1ec3157008734f913e1e95997e758bf45f3b0eaa`, submission
`b7608e86-a4e6-44b3-b698-ecd1e539e751`, supplied the exact lean GLV coefficient
header and CPU test, extending ItlaStudent's coefficient work. Both are credited
as coauthors for this unpromoted contribution. Their host-verification and
sparse table-readback changes are not imported. ItlaStudent's later lossy
field-multiply cut is also not imported.

newjordan's public `f32de853-bb02-4f2f-b7b2-c87546038207` supplied the overlap
hypothesis and 104-register target. Its note reports about +11.5% in an RTX
4080 equal-work experiment with spills, and was still validating at our final
refresh. This is motivation from another GPU, not an RTX 4090 result for our
candidate. newjordan is credited for this material unpromoted contribution.

The shared-memory anchor implementation here is independent. The seed-sign
rewrite is from our earlier local experiment. Inherited licences and notices
remain. Historical `SUBMISSION.md` records the donor; this is the current note.

## Mechanism

Preparation performs the long point chain. Separate preparation and finishing
streams already exist, but streams only permit overlap. They do not guarantee
it. Our CUDA 12.8.93 baseline uses 122 preparation registers and 64 finishing
registers. Register allocation granularity can leave no capacity for finishing
beside four resident preparation blocks.

At 104 preparation registers, four 128-thread preparation blocks and one
128-thread finishing block require 61,440 registers. Ada provides 65,536
registers per SM. This demonstrates resource capacity, not actual scheduling.
A GPU timeline is needed to show whether the intended overlap occurs.

A cap alone can introduce spills. We instead shorten the lifetime of the four
64-bit affine Y limbs that become the next loop iteration's anchor. They are
inputs near the beginning of the point add, then needed only after the add.
`QSB_GLV_ANCHOR_PARK=1` writes them to four volatile shared-memory planes before
`_PointAddXYZZT<true>` and reloads the old anchor after the call. Volatile
accesses end the unnecessary register lifetime across the expensive arithmetic.
The point-add arguments, result and loop order remain the same.

The implementation reuses `qsb_digit_arena()`, with no larger allocation:

- Capacity is `12*N` 64-bit words, with N equal to the preparation block width.
- Codes occupy `GT_GLV_TERMS*N` 32-bit words.
- Anchor planes begin at 64-bit offset `(GT_GLV_TERMS/2)*N`.
- GLV12 anchors end at `10*N`; the GLV14 control also fits, ending at `11*N`.
  A compile-time assertion enforces the bound and even term count.
- Each lane writes and reads its own column, disjoint from code storage.
- The existing unconditional barrier in `qsb_packed_prepare` completes all
  code and anchor reads before the cofactor tree overwrites the arena. Inactive
  and zero-scalar lanes still reach this barrier.

`QSB_OVERLAP_REG_CAP=104` applies only to preparation. The pipeline body is in
`PinningPipeline.cuh`, included under the original name and original launch
bounds, then again under a preparation name with `__maxnreg__`. Finishing
continues to use the original specialization. This avoids combining mutually
exclusive launch-bounds and max-register annotations on one kernel.

The coefficient header uses explicit wide products, guarded high-limb
multiplication with exact fallback, and carry-based rounding. The seed rewrite
reverses both differences P and R. Negating Y and ZZZ together preserves the
affine point and deferred anchor while enabling the existing fused
`R^2 + PPP - 2Q` operation. All changes have independent kill switches.

## Compiler observations

Toolchain: CUDA 12.8.93 in a linux/arm64 container without a CUDA device.
We compiled organizer-default compute_52 PTX, assembled it for sm_89, built
native sm_89, and linked the default executable. The driver's final JIT may
differ. None of these steps measures end-to-end RTX 4090 throughput.

| Arm | Prepare registers | Spill store/load bytes | Static prepare instructions |
| --- | ---: | ---: | ---: |
| Promoted baseline | 122 | 0 / 0 | 6,680 |
| Lean coefficients and seed, no park/cap | 114 | 0 / 0 | 6,608 |
| Same, cap 104 without parking | 104 | 8 / 8 | 6,640 |
| Same, anchor parking without cap | 106 | 0 / 0 | 6,600 |
| Submitted defaults: parking and cap 104 | 104 | 0 / 0 | 6,624 |

All preparation arms above use 12,288 shared bytes. Selected finishing uses
64 registers, zero spills and 4,040 static instructions. The full default
sm_52 executable uses 96 preparation and 72 finishing registers, also with
zero spills. Native sm_89 and default-PTX-to-sm_89 builds have no spills in
any emitted kernel. Exact resource counts are architecture/compiler dependent.

An earlier prototype capped the finishing specialization too. Its default
build spilled 12 bytes in finishing. That prototype was discarded. Separating
the preparation name fixed the unintended annotation. After the body extraction,
the all-switches-off control's parsed SASS instructions matched the promoted
baseline for every kernel. Static counts do not predict a promotion margin.

## Correctness evidence and its scope

The GLV test runs a CPU wrapper of the actual coefficient helpers and passes
300,818 cases for each reciprocal: 601,636 comparisons including high-limb
fallback boundaries. The independent seed algebra check passes 4,096 edge
tuples and 10,000 random tuples over the field; that test skips equal-X
degenerate cases. It is not a GPU point-chain or device hit-set comparison.
The anchor bounds, lane ownership and tree handoff were reviewed against the
production source and its existing unconditional barrier.

The exact host publication gate remains enabled. The package does not change
problem enumeration, candidate accounting, hit serialization, verifier or scorer.
No local self-reported rate is represented as an official rate.

```bash
python3 -B candidates/pinning/test_glv_coeff.py
python3 -B candidates/pinning/test_seed_fuse.py
git diff --check
python3 -B candidates/pinning/submission_preflight.py
```

## Build and controls

```bash
# Submitted defaults; CUDA 12.8.
nvcc -O3 -DQSB_ZEROS_N=24 candidates/pinning/pinning.cu \
  -o /tmp/pinning-candidate -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -ptx candidates/pinning/pinning.cu \
  -o /tmp/pinning-candidate.ptx
ptxas -arch=sm_89 -v /tmp/pinning-candidate.ptx \
  -o /tmp/pinning-candidate.cubin
cuobjdump -sass /tmp/pinning-candidate.cubin

# Add these definitions for the promoted arithmetic/resource control:
# -DQSB_GLV_LEAN=0 -DQSB_GLV_HIGH10_HI=0 -DQSB_GLV_ROUND_CC=0
# -DQSB_SEED_FUSE_X3=0 -DQSB_GLV_ANCHOR_PARK=0 -DQSB_OVERLAP_REG_CAP=0
```

For attribution, compare coefficient and seed changes separately, then their
combination, then parking, then the cap. Useful GPU evidence is independently
verified equal-work comparison plus fresh-problem full-duration runs, with
power and clock conditions recorded. A short run or another Ada GPU can have
a different result from the official 1,200-second workload.

## Submission decision and limitations

One official validation is justified by the concrete lifetime change,
spill-free 104-register target build, and the related public overlap experiment.
No speedup is established yet. The RTX 4090 scheduler might not exploit the
capacity, shared-memory traffic might cost more than it saves, or JIT register
allocation may differ. The selected variant is not a replay of a scored archive.

Only `candidates/pinning` is packaged. Binaries, PTX, cubins, SASS, logs and
downloaded references are outside it. Preflight audits every on-disk file,
including ignored files, checks source hashes and enforces the manifest's size
limit. The official verified-hits score is authoritative. A rejection calls
for diagnosis of this variant, not an unchanged re-upload.
