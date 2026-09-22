# Pinning: isolated raw slope products on the TOP16 and narrow parity source

## What this source changes

This submission candidate starts from our PR #976 executable source, commit
`b84b7bdb08b7da66472046163aba88fcfcff8638`. Its **only new executable
change** is in `PackedRecovery.cuh`: `QSB_FIN_RAWS=1` leaves two slope-product
residues raw in the stage-2 recovery finish, while retaining a boundary check
before the two hashed x-coordinate additions. The fallback branch remains
available as `-DQSB_FIN_RAWS=0`. The signed-digit funnel-shift and doubled-sum
switches from the public PR #993 are deliberately excluded. No stage-0 source,
cofactor traversal, parity-window header, verifier, benchmark harness, input
problem, or scoring code changes with this package.

This is an exploratory source change with **weak and inconsistent local speed
evidence**. On one official-seed, fixed-work RTX 4090 A/B/B/A, RAW measured a
pooled **+0.07143%** completed-work throughput change; the two adjacent pairs
had opposite signs, **-0.1380% and +0.2809%**. We do not claim a reliable
positive gain or predict promotion from this measurement. The organizer's
independent ranked run and verifier determine the score. At package
preparation the latest known pinning record was **805,428,058/s** and its
1%-higher promotion floor was **813,482,339/s**; these values are only a
snapshot and must be checked before any submission decision.

The integration/package model is **GPT-5**, harness **Codex**. This session
ported and isolated a public mechanism; it does not claim invention of the
raw-product idea. PR #993's public archive, ticket
`52a058ef-b8ea-424a-9ee4-c3a07084ed4c`, lists **fkiene** as coauthor and
contains the exact `QSB_FIN_RAWS` branch used here. Earlier public PR #849 and
PR #866 also used raw finish products. We give those sources credit and do not
represent this source as a novel large breakthrough.

## Inherited source and attribution

The underlying promoted parent is public commit
`94abdd0d72847b780c7d4f99da4f367e6f9f0fd1`, Yukon submission
`07009ac3-94a4-428e-b030-1f6ce317ccb7`, official score **797,446,582/s**.
Its code and note were prepared with **GPT 5.6 Sol / Codex** and include
@stffinfcti's PR #827 field schedule, @EvanYan1024's PR #885 bounded parity
window, and the recovery isomorphism integrated in that parent.

Public PR #927, head `caf7f8c00ec87a2af0b0f1c3bcae6569068e9f45`, supplied
the TOP16 cofactor traversal. Its public authorship credits
@ercumentyildirim's implementation with **Claude Opus 5 / Claude Code** and
@EvanYan1024's earlier TOP16 schedule idea. We copied its
`cofactor_checkpoint.h` byte for byte in the prior package. The merged tree
reduces the upper cofactor up-sweep and exclusion chain from seven dependent
waves to four. PR #927's official score was **804,598,773/s**; a second public
TOP16 executable drew **804,457,861/s**. These historical results were below
their then-current promotion floor, and are not scores for the present source.

Public PR #965, head `db3a694bb155ce266213c0499914628abb3363d3`, by
@Portablelle with **GPT 6 Astra / Codex**, supplied the narrow speculative
parity window in `ParityWindow.cuh`. Its header and `NARROW-PARITY.md` retain
the original proof, test method, and source credit. The window removes nine
ordinary-path cross-products while retaining the original full-product
fallback around carry boundaries. PR #970 combined TOP16 with that parity
window and received **786,181,069/s** in its official run. PR #976 repeated
the same executable source, with only its public note/manifest changed, and
received **793,581,236/s** on a different ranked run. These two scores show
material run-to-run variation and do not provide evidence that a +0.071%
local observation will overcome the current floor.

The only runtime file changed relative to PR #976 here is
`PackedRecovery.cuh`. The `QSB_FIN_RAWS` branch follows public PR #993's
finish arithmetic; the old branch is retained for direct A/B and source
review. PR #993's `QSB_FIN_SUM2` and `QSB_DIGIT_SHF` are omitted because our
separate static screen found no compelling whole-source gain from them.

## Raw-product arithmetic and recall boundary

Let `B=2^256`, `p=B-K`, and `K=2^32+977` for secp256k1. The field multiplier's
raw residue `s` lies in `[0,B)`, and a canonical residue represents the same
value modulo `p`. The finish computes two slope products of the form
`s=sum*(l-c)` or `s=sum*(m-c)`, then adds fixed affine abscissa `a` to obtain
the x-coordinate hashed by the downstream gate. It also obtains two y-parity
bits by multiplying `s` by `l` or `m` before adding canonical `b`.

The existing `qsb_parity_product_window` accepts raw 256-bit operand
representatives and retains its bounded fallback; its parity calculation is
congruence invariant. The RAW branch therefore computes each parity before
any normalization of `s`. For the hashed x-coordinate, the existing
`qsb_add_boundary(s,a)` normalizes `s` if the top limb of `a` is all ones. In
the other case `a < B-2^192 < B-2K`, so `s+a < 2p`; the one conditional
subtraction in `_ModAdd256` returns the canonical x-coordinate. The branch
only removes normalization where this boundary argument permits it. The
original `qsb_recovery_mul` path remains available with the switch off.

These bounds address the two changed products only. The inherited speculative
parity window has its own documented fast-path guard and full-product
fallback. The unchanged OpenSSL host gate independently reconstructs and
re-hashes every published tentative hit. That gate rejects false GPU hits;
it cannot recover hits that an erroneous GPU filter misses. The finite hit
comparisons below are therefore evidence for tested inputs, not an
all-input completeness proof.

## Isolated matched-work evidence

The public-source port was built in scratch against the exact PR #976 source.
Before test-only timing edits, the all-off native sm_89 cubin SHA256 was
`75fdf08b2f222e30d3ef0c646254163a68200768c12a4bf274880918aaa6d4e0`,
byte-identical to an independently compiled PR #976 control. With only
`QSB_FIN_RAWS` active, native stage-0 remained **6,088 SASS instructions,
128 registers, 12,288 bytes shared, zero spills**. Native stage-2 changed
from **4,056 SASS / 66 registers** to **4,064 SASS / 64 registers**, still
zero spills. The organizer-default sm_52 builds both reported **101
stage-0 and 72 stage-2 registers**, zero ptxas spill. The full three-switch
PR #993 combination did not retain the 64-register native stage-2 result;
its stage-2 register count returned to 66.

We inserted the same finite diagnostic stop into both **scratch** sources,
compiled `nvcc -O3 -DQSB_ZEROS_N=24` for the organizer-default sm_52/PTX
path, and ran the public problem seed **9,072,764** in `single_hash` mode.
Each arm completed exactly **9,956,800,000 candidates**. Every arm emitted
the same **1,201 sorted, OpenSSL-gated hit lines**, SHA256
`58d9fbc1a44e241bc821a4876dcce4b8e1921592090f2bc58de2c070a214fd5b`.

| Run order | Source | Fixed-work seconds |
|---|---|---:|
| A1 | PR #976 control | 11.971310 |
| B1 | PR #976 + RAW only | 11.987853 |
| B2 | PR #976 + RAW only | 11.987452 |
| A2 | PR #976 control | 12.021120 |

The control mean was **11.996215 s**, RAW mean **11.9876525 s**. The
arithmetic pooled effect is **+0.07143% completed-work throughput**. A1/B1
favored the control; B2/A2 favored RAW. Clock and order drift at this scale
is unresolved. The result is too small to establish a durable gain. An
earlier related raw-finish source family was also near noise in local
experiments; PR #849's full combination was negative on its different
stage-0 base. These historical observations add caution, not a direct
measurement of this executable on the ranked runner.

The scratch report, source-port script, static cubins, four output logs, hit
hashes, and machine summary are retained outside this submission tree at
`/tmp/qsb-pin-pr993-screen/`. They are not submitted as source, binaries,
result files, or problem inputs. The production tree uses the unmodified
contest loop and has no finite stop or diagnostic timing code.

## Reproduction and interpretation

The organizer-default N24 build is
`nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm`
from the pinning candidate directory. `QSB_FIN_RAWS=1` is the source default
for this package; `-DQSB_FIN_RAWS=0` restores the PR #976 finish branch.
`SOURCE-MANIFEST.json` records SHA256 and byte counts for every included
pinning source, license, and note file. Its manifest hash list is recalculated
after note edits and before committing.

The ranked score is based on independently verified hits over elapsed time,
so fixed-work throughput and official score are different measurements. The
runner's timeout `candidate_count` can extrapolate a peak observed progress
rate over elapsed time; it is not a completed-work count. We base the local
+0.07143% figure solely on the matched fixed candidate count above and do
not infer an official score from that number. Public prior source credit and
the prior package's runtime files remain preserved in the Git history and
headers. The score and promotion decision require the contest's own full
run, current floor, and exact verifier.
