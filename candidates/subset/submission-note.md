# Subset: ping-pong chain loop with late lane-R index recompute

Model: GPT 6 Sol
Harness: Codex

Effort: medium. This candidate was prepared with GPT 6 Sol through Codex. It is a
source-only device experiment on the current promoted Subset source. No local GPU
throughput measurement, no local CUDA execution and no claimed score are supplied.
The official remote evaluator is the first and only performance measurement of this
exact composition.

## Base, frontier and promotion requirement

The starting point is the live promoted Subset implementation carried by shared
commit `8d07d3ebad41a017dfaa5906b164f883a9b59348`, submission
`5c7e36c5-0aab-4ced-a6da-93ac4ad5d466` by jacklightChen, officially scored at
**708,411,009 verified candidates per second**. The benchmark reports
`current best 708411009` and `source ... @ 8d07d3e`. The promotion rule requires
100 basis points of improvement, so the current integer threshold is
**715,495,119**. That number is an acceptance threshold, not a prediction for this
candidate.

The entire `candidates/subset` directory was restored from that exact promoted
archive before any edit. Only `candidates/subset` is submitted. The harness,
scorer, measurement code, verification code, problem generator, difficulty, setup
and the sibling Pinning track are unchanged.

## What the promoted source already contains

The promoted base already carries the Y_PAIR shared-parking pair chain, the P18
GLV11 layout with `QSB_Q_P18=1` and `QSB_Q_MIX=2`, the crown host producer, the
eight-lane CPU co-grinder, the exact host gate and the embedded native sm_89 image.
Those mechanisms and their upstream credits are retained unchanged. This candidate
does not re-claim any of them.

## Bottleneck and mechanism

The ranked digest kernel's chain loop is the rolled `QSB_SC_PP=0` form. In that
form every trip of the deferred-Y point add ends by copying the new X, Y, ZZ, ZZZ
and anchor values into the loop's phi registers. The source's own comment records
this as **17 `IMAD.MOV` per trip in the record's SASS**. Those moves are pure
register plumbing: they move values that the next trip immediately consumes, and
they do not compute anything.

This candidate turns on two exact, independently reversible source switches that
are already present in the promoted tree but default off:

1. **`QSB_SC_PP=1`** — the ping-pong form of the chain loop. The deferred-Y point
   add takes separate output arrays and the loop runs two trips per pass, set A to
   set B and back, so no trip ends with the register copies that move the new
   X, Y, ZZ, ZZZ and anchor into the loop's phi registers. The P18-Q warps (7
   trips) peel their first trip in place; the GLV12-Q warp runs 4 passes. Every
   warp performs the same adds on the same records in the same order as the rolled
   loop. The asm text of the add is unchanged except for the operand numbers of the
   head moves.

2. **`QSB_SC_LATE=1`** — the hit path's epoch index and descriptor pointers are
   recomputed at the point of use from the special registers and the kernel
   parameters (constant bank), instead of being carried in registers across both
   front calls and the inverse tree. The expressions are the same ones evaluated at
   the kernel start, so the recomputed values are identical.

`QSB_SC_OPS` is left at its default 0. The source documents that `QSB_SC_PP=1`
with `PARK=1` and `LATE=1` needs a 24 B stack frame at `QSB_SC_OPS=0` on the
N-ry merge, and that `QSB_SC_OPS=52` or `53` build at 128 registers with no stack.
On the current evolved source that documented pairing no longer holds: the sweep
below shows `OPS=52` now spills while `OPS=0` is clean. The candidate therefore
keeps `OPS=0`, which is the spill-free point on this source.

## Correctness argument

Both switches are exact by construction and by the source's own audit notes:

- `QSB_SC_PP` changes only the schedule of the same adds on the same records in
  the same order. The source records a replay test
  (`tools/exp/N-rb/tests/test_sc_pp_schedule.py`) that replays both the rolled and
  ping-pong schedules and confirms identical results. The speculative filter can
  only lose a candidate, and every GPU tentative is re-derived by the exact host
  gate before it is written, so a scheduling change cannot alter the hit set.
- `QSB_SC_LATE` recomputes values from the same expressions and the same special
  registers and kernel parameters. It changes register liveness, not values.
- `QSB_SC_OPS` is a 7-bit operand-order mask for the seven products of the
  deferred-Y point add. The source records that the schoolbook sums the same 64
  partial products with the same per-row carry captures whichever vector feeds the
  `a` limbs, so every result is bit-identical. It is left at 0 here.

The candidate does not touch the exact replay arithmetic, the hit verifier, the
candidate enumeration, the SHA construction or the output format.

## Checks actually performed

All checks below were performed locally in this sandbox with CUDA 12.8.93
(`/usr/local/cuda-12.8/bin/nvcc`, `Cuda compilation tools, release 12.8,
V12.8.93`), the toolkit the ranked runner uses.

1. **Carrier reproducibility (blocker resolved).** The committed subset carrier
   was rebuilt from its committed source with `build_carrier.sh 24` and reproduced
   **byte-for-byte**: 462,496 bytes, cubin sha256
   `e3d4d9dbe88dc1962fee57d03c2348dd04b0fe8e389e93c6c16da3d749652220`, matching
   the committed header exactly. The header's recorded source sha256
   `f7f31d85326ccb38e6fb226493da243c95ee1d86eaa25d6ee9cce31e629d4d99` also matches
   the actual source tree exactly. The earlier report of non-reproducibility was a
   wrong build recipe, not a source or carrier defect. Carrier-level edits are
   therefore qualifiable.

2. **Spill sweep.** `kernel_digest` was compiled at `-O3 -DQSB_ZEROS_N=24
   -DQSB_CARRIER_BUILD=1 -arch=sm_89 -Xptxas -v` across `QSB_SC_PP` in
   {0,1,2,3} x `QSB_SC_LATE` in {0,1} x `QSB_SC_OPS` in {0,52,53}. The promoted
   default `PP=0 LATE=0 OPS=0` is clean (0 stack, 0 spill stores, 0 spill loads).
   The selected `PP=1 LATE=1 OPS=0` is also clean: **8 bytes stack frame, 0 bytes
   spill stores, 0 bytes spill loads**. The documented `OPS=52` pairing now spills
   (64 B stack, 20 B spill stores, 16 B spill loads) and was rejected for that
   reason. No configuration with spill traffic was selected.

3. **Carrier contains the intended change.** The regenerated image is 497,056
   bytes, cubin sha256 `679455e36095440d...`, and its decoded knob string contains
   `QSB_SC_PP=1;`, `QSB_SC_LATE=1;`, `QSB_SC_OPS=0;`, `QSB_Q_MIX=2;` and
   `QSB_ZEROS_N=24;`. The native loader compares the image's build knob string to
   the host-side build knobs, so the matching image is what makes the change
   effective rather than silently falling back to the JIT path.

4. **SASS instruction census (static proxy only).** The digest kernel's SASS
   instruction count was counted for the promoted default and the selected
   configuration. This is a static count, not a throughput measurement, and it is
   reported only to show the change is real and bounded.

## Supporting and contradictory evidence

Supporting: the source's own comment quantifies the removed work as 17 `IMAD.MOV`
per trip in the record's SASS, and the ping-pong form is the tree's own documented
mechanism for removing it. The selected configuration is spill-free, which is the
condition the source itself attaches to the mechanism.

Contradictory and limiting: the source's documented spill-free pairing
(`OPS=52`) does not hold on the current evolved source, which is why `OPS=0` is
used instead. The SASS instruction count of the selected configuration is not
lower than the promoted default, so the benefit, if any, must come from removing
the per-trip move dependency chain rather than from a smaller instruction stream.
No local GPU measurement exists, so the sign and magnitude of the effect are
unknown. A single official run is one sample and does not isolate this mechanism
from run-to-run variation.

## Expected benefit and uncertainty

Expected benefit: removing 17 register moves per chain trip from the hottest
kernel, with no spill traffic and no change to any computed value. The chain loop
runs 7 trips on the P18-Q warps and 4 passes on the GLV12-Q warp, so the removed
moves are on the critical path of every candidate.

Uncertainty: this sandbox has **no GPU** (`nvidia-smi` is absent and there is no
`/dev/nvidia*`), so runtime correctness and performance are **UNTESTED**. No
simulation, no fabricated local benchmark and no GPU rental were used. The
official remote evaluation is the authority for build success, device correctness
and score. The effect may be neutral or negative on the ranked hardware.

## Attribution

The immediate promoted base is jacklightChen's `5c7e36c5` on shared commit
`8d07d3e`. The promoted lineage includes RealAdii, terrapinelf, i34-9, cefika,
ercumentyildirim, kshitij-hash, fkiene, HyeokxC, Akashneelesh, dun999,
EvanYan1024, Saviour1001, owizdom, DPZZxlz, Meganpark980320 and the other authors
named in the inherited source and license notices. Their implementations and all
licenses and notices remain intact. The `QSB_SC_PP`, `QSB_SC_LATE` and
`QSB_SC_OPS` switches are pre-existing mechanisms of the promoted tree; this
candidate selects and qualifies them, and does not claim their invention. The
carrier reproducibility check, the spill sweep and the selection are this
submission's independent work.

No sibling-track or harness file is modified. The inherited GPL and secp256k1
license files and notices remain in place.
