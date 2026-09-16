# Subset: external batch inversion and direct XYZZ recovery

Effort: xhigh. Developed by GPT 6 Astra in Codex. This is an architectural
successor to our rejected subset PR27, combining the strongest compatible
public epoch and deferred-coordinate work with a checkpointed inversion
pipeline. No GPU speedup or local score is claimed. The official evaluation
must establish CUDA compilation, verified hits and end-to-end throughput.

## Current result, starting point and attribution

Our earlier submission `c9a85a87-e2b8-4ab8-8d4f-c79d96546d03`, commit
`e5820d032d53123f558ea533546520373eeea2a6`, returned **418504460 verified
candidates/s**. The promoted subset frontier is **433346795**, submission
`873ed724-9815-4e13-a02f-072f21e3f992`, landed at
`cfc0d9cf5dd7cc8c607a8ff53b0dadb90e6424c1`. Thus our returned result is 3.425%
below the relevant epoch baseline. Its improvement over the older 129574439
submission-time comparison does not make it competitive now. In particular,
fewer inverse-tree barriers and fewer shared bytes did not establish higher
throughput. We have no matched GPU profile that identifies the regression.

The prepared composite before this change combined subset PR36 by
jacklightChen, source `472b536106d2e2aae2b927cbf217678a58bebb73`, with subset
PR40 by DPZZxlz, source `08a4b7b44aee9cddc5b8c2852f9ed2e6b4f3d5f8`.
PR36 composes streamed recoding, deferred Y and fused upper inverse-tree
levels. PR40 specializes the ranked epoch path. Their additional mechanisms
are unpromoted at preparation time, so both contributors are coauthors.
The prior composite production/audit fingerprint was
`d0ccab7c66830429f8bc39dc619ad4b369883d0ea9c7929b64a569592c65512e`.

The external product-tree checkpoint hierarchy, vector state transport and
direct XYZZ recovery originate in public pinning work, most directly
alvaroborras PR24, commit `6e76a74fed8e6e5b8439e64ec20f586085f37d52`, and its
nullforest8200 predecessor. The original broader epoch architecture remains
credited to its promoted authors, including odinfree's earlier GPU producer
work. Meganpark980320 contributed the unpromoted upper-level fusion retained
in the generic inverse helper and used in the tiny super-root kernel.
The coauthor list records substantial unpromoted contributions from
jacklightChen, DPZZxlz, nullforest8200, alvaroborras and Meganpark980320.
Existing GPL notices and COPYING are preserved.

During preparation, pinning submission
`6ce23203-c159-4f84-a668-1066d5fde85b` was promoted at
`4d39b5f0a881653d6332a7801dd84bc14175fa61`, with an official score of
**644546620 verified candidates/s**. It uses this family of external inversion
pipelines. This is useful evidence that the architecture can work on the
challenge GPU, but it is a different track and binary, with different hashing
and table geometry. It is not a measurement or prediction of our subset score.

## Why this experiment is substantial

All reviewed pending subset candidates preserve either a per-CTA inverse or
an older recovery architecture. The selected composite already contains the
compatible deferred-coordinate and ranked-hash changes. Another small barrier
or flag patch would not address the failed assumption behind our first result.

A monolithic 256-lane consumer reaches a scalar inversion with most lanes
waiting. Dividing an estimated inverse latency by 256 would miss the CTA's
critical path. This experiment ends the producer before that inversion,
checkpoints its product tree, and inverts the complete batch through two
additional levels. A normal batch of 8388608 candidates now performs one
scalar inversion instead of 32768 scalar inversions. The number of saved
inversions alone is not a speedup estimate: global transport, extra launches,
register allocation and the existing arithmetic still matter.

The direct recovery path also avoids converting raw XYZZ coordinates to
homogeneous coordinates. Excluding product-tree arithmetic, the old
conversion-plus-recovery costs 13 field multiplications and 3 squarings;
the new preparation-plus-recovery costs 10 multiplications and 4 squarings.
The inherited deferred 16-point chain costs 102 multiplications and 30
squarings rather than the promoted 116 and 30. Together the chain plus
recovery changes from 129M+33S to 112M+34S by source operation count.
The 14M chain saving is inherited; the extra recovery change and subset
pipeline integration are this experiment. These counts are not timings.

## Ranked pipeline and runtime dependence

The existing host guard selects only the eligible single-GPU, single-hash,
non-easy, non-calibration short-epoch problem shape. It retains the existing
runtime problem parsing, table construction, constant schedules, epoch
producer and candidate enumeration. Unsupported shapes continue through the
generic `kernel_digest<false>` implementation.

After `kernel_build_epochs`, five ordered launches on the default stream do:

1. `qsb_ranked_prepare` consumes each runtime epoch descriptor, performs the
   scheduled first SHA-256 and second SHA-256, recodes the actual digest and
   computes the deferred 16-term fixed-base sum. It saves four field elements,
   checkpoints the non-root products and publishes one root per search CTA.
2. `qsb_root_group_prepare` groups at most 256 search roots, pads an incomplete
   group with multiplicative identities and checkpoints each group tree.
3. `qsb_invert_super_roots` uses one 256-lane block inverse to invert at most
   256 group roots, again padding with identities.
4. `qsb_root_group_finish` restores the group trees and replaces search roots
   with their inverses.
5. `qsb_ranked_finish` restores the search trees, obtains each candidate's
   inverse, computes both recovery flags, hashes compressed public keys and
   applies the unchanged ranked leading-zero gate.

The port retains subset's two planar tables, sixteen signed digits and its
own seven-argument `_FixedBaseSignedXYZZ`. It does not substitute pinning's
fifteen-digit/interleaved table geometry. The final deferred point addition
resolves the exact ordinate with `DEFER_Y=false` before recovery.

State consists of C, Y, W and ZZZ, in eight aligned `ulonglong2` planes.
Their stride is the current launch's candidate count, shared by save and
restore. A smaller last launch therefore remains within the maximum scratch
allocation. Each search block is one complete 256-candidate epoch; there is
no early-return lane inside the checkpoint collectives. The launch helper
requires 1 through 65536 blocks and exactly 256 candidates per block.
Compile-time assertions bound the production batch to that hierarchy.

The immutable product tree uses 254 non-root internal nodes, stored in four
limb-major planes of stride 256. The root is published separately. Both halves
use identical packed numbering and every dependency is separated by a CTA
barrier. The root-group inverse helper uses its own independent tree layout;
its layout is not mixed with checkpoint indexing.

Every launch error is checked before continuing, and the final synchronization
result is checked directly. Scratch allocation failures are reported. There
are no host transfers of checkpoint data and no overlapping streams. Existing
hit harvesting remains per launch: indices carry the local candidate number
and recid bit, while the six early and three window omissions reconstruct the
same verifier-visible storage indices. Single-hash mode leaves hash_choice=0.

## Direct recovery identities

Let an exact XYZZ point P have xP=X/ZZ and yP=Y/ZZZ, with ZZZ^2=ZZ^3.
For the fixed runtime affine point R=(xR,yR), set:

```
d = xR*ZZ - X
W = ZZ^2*d
C = ZZ*d^2
inv = 1/W
h = ZZZ*inv
delta = C*inv = xR-xP
xs = 2*xR-delta = xP+xR
```

The two slope magnitudes are `(yR*ZZZ-Y)*h` and `(yR*ZZZ+Y)*h`.
Squaring and subtracting xs gives the two recovered x coordinates. The y
formulas are anchored at R, so no separate affine yP reconstruction is needed.
The second slope is negated relative to the ordinary addition slope; its
parity uses field negation, preserving zero rather than unconditionally
flipping the low bit. The checked multiplication/square primitives produce
canonical values, and subtraction of canonical operands remains canonical.

The saved W retains a singular lane's zero. Its product-tree factor is one,
and the finish stage makes the same substitution before the collective,
then omits that unusable lane after the collective. This prevents one singular
lane from poisoning unrelated inverses. The inherited fixed-base mixed-add
formulas are not complete formulas for exceptional/infinite intermediate
points; this experiment does not claim to resolve that inherited limitation.

## Arithmetic repair, separate from performance

The imported hot 8x32 multiply and square had dropped the final carry from
bit 255. This is an actual arithmetic defect, independently reproducible with
canonical a=b=p-65537: the old product gives 130096 instead of 4295098369.
Random field tests alone almost never sample this boundary.

Both primitives now retain the carry with `addc.cc.u32 z7,z7,0`, capture it,
and add its residue `2^32+977`. If the second fold overflows, its low remainder
is less than `(2^32+977)^2`; the third fold remains below 2^96. Carry propagation
therefore need not traverse all eight 32-bit limbs. Five additional arithmetic
instructions implement this bounded fold, followed by a shared C++ conditional
subtraction of p to make the output canonical. Both host and device paths
contain the repair. An explicit mutation test rejects a patch that captures
stale carry without changing the high-limb addition to `.cc`.

The inverse-tree field primitive remains its separate canonical implementation.
The new GPU audit exercises both implementations instead of treating an
inverse-tree arithmetic pass as evidence about the distinct hot curve core.
See NVIDIA's primary documentation for carry semantics:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-addc
The local PTX interpreter checks this straight-line assembly's residue; it is
not a CUDA assembler, compiler, emulator of the whole GPU, or timing model.

## Local validation and reproducibility

This host is an Apple Silicon Mac without nvcc or an NVIDIA GPU. Setup and
the baseline run were attempted earlier: the CPU verifier smoke passed; the
ranked GPU run could not start its Linux/CUDA bridge. We did not rent a GPU,
change trusted scoring files or invent a local score. The existing user
policy permits a well-supported submission for remote validation.

Run from the benchmark root:

```sh
python3 candidates/subset/check_candidate.py
python3 candidates/subset/check_deferred_source.py
python3 candidates/subset/audit_integrated.py
python3 candidates/subset/check_pipeline.py
python3 candidates/subset/research/check_production_field.py
python3 candidates/subset/research/check_host_syntax.py
python3 candidates/subset/preflight.py --note candidates/subset/submission-ranked-pipeline.md
```

Observed CPU validation:

- 3840 inverse outputs, 6144 complete SHA256d values, 10262 signed recodings,
  and 8086 combinadic unrankings from extracted source.
- 31920 intermediate deferred-coordinate states and 2128 raw-producer chains,
  including 128 curve chains, using independent OpenSSL field operations.
- 1282 hierarchical inverse outputs with 1, 255, 256, 257 and 513 roots;
  2000 direct recovery cases; three complete simulated producer/finish CTAs,
  containing 768 candidates and three deliberately singular lanes.
- 237 emitted hit records checked against independent compressed-key hashes,
  recovery flags and omission indices. This integration test uses controlled
  point-producer/first-hash inputs and an easier gate to exercise hits; actual
  scheduled hashes and point helpers have separate source checks.
- 60540 actual host multiplies and 40360 independent host squares, with
  canonical equality and aliasing checks. 2180 multiply and 2180 square PTX
  interpretations match big-integer residues. The stale-carry mutant fails.
- Full production and audit C++ projections pass syntax/type checking. This
  deliberately erases launch syntax and assembly and declares CUDA stubs;
  it catches ordinary integration errors but is explicitly not a CUDA build.

Reports include the production/audit source fingerprint. The included GPU
audit now covers targeted carry-boundary pairs, hot multiply/square aliasing,
block inverses and checkpoint hierarchy groups with incomplete tails. That
GPU audit has not been executed on this host. On a CUDA host, compile in a
fresh tree with `preflight.py --cuda`, execute the resulting arithmetic audit,
and run the unchanged official benchmark. The wrapper's existing cache checks
only subset.cu's timestamp; after header edits remove its generated binary and
build stamp before local CUDA setup/run. Never reuse an older binary as proof.

Actual signed-in Grok 4.6 and Gemini 3.8 Flash (High) provided read-only reviews.
Their initial reports used stale pinning signatures and unrepaired arithmetic;
those claims were checked against the exact source and rejected. Their memory
traffic and register-allocation concerns remain legitimate unmeasured risks.
Model agreement is not validation, and neither reviewer ran a GPU benchmark.

## Costs, limits and result interpretation

For an 8M batch, vector state uses 1 GiB and search-tree scratch uses 256 MiB.
Roots and root-tree storage add about 2 MiB. State and tree transport costs
about 320 bytes per candidate, excluding table traffic and other work. At
433M candidates/s this alone would require roughly 139 GB/s. Fitting the
allocation into VRAM does not prove cache residency or available bandwidth.

The prepare kernel combines subset scheduled hashing with a checkpoint
upsweep; its shared-memory demand differs from pinning. The finish kernel
retains the public three-CTA launch bound. Without ptxas we cannot know its
register allocation, spills, achieved occupancy or whether code duplication
in the two public-key hashes changes that tradeoff. Five dependent launches
also replace a single consumer launch. We preserve those source-backed choices
for one substantial architectural evaluation rather than claiming a locally
tuned optimum.

The final scan includes pending PR33, PR36, PR39, PR40, PR42, PR57 and PR59.
PR28 returned rejected at 433564114, only 0.05015% over the promotion.
Compatible leading mechanisms from PR36/40 are retained; older-base
ports and hardware-specific unsupported hints are not stacked indiscriminately.

The last two arrivals were inspected before upload. PR57
(`3b1bba4b-5095-4dc5-a072-1589c5589107`) uses direct XYZZ recovery at 10M+3S
and retains the per-CTA inverse. It saves one square relative to our direct
checkpoint recovery but uses a different saved denominator/state relationship.
Our selected four-field transport avoids adding another saved field or
reconstructing W for the finish collective; we retain that already checked
port rather than assume the monolithic formula's saving transfers unchanged.
PR59 (`5605ad84-dd72-4aed-9e6d-60a6bb64a61b`), validation head
`da066cac70e48ca97ce70942f1f60ea17539ec0c`, combines two-stream epoch overlap,
constant recovery coordinates and paired compressed-key SHA. Its source still
uses the homogeneous recovery/per-CTA inverse. The author reports matched
full-duration relative gains of 1.67% and 2.03% on two power-limited RTX 4090 hosts.
Those are author results, not ours or an official new frontier.
Paired SHA holds two schedules live; its reported monolithic-kernel benefit
is not established under our separate three-CTA finish register budget.
Multi-stream overlap would also change buffer/event lifetimes. Both remain
recorded follow-up options pending exact-pipeline GPU evidence, rather than
being treated as automatically composable wins. Neither new entry supersedes
the external-inversion architecture selected for this submission.

The basis for expecting a competitive successor is the combined reduced point
chain, ranked-only consumer, direct recovery and already successful related
external-inversion architecture. This remains a prediction for this exact
binary, with meaningful risk from memory traffic and arithmetic repair costs.

Compare the returned verified score against the live subset promotion and any
newly evaluated pending candidate, not against the obsolete 129M frontier.
If this loses, inspect complete-kernel timing, ptxas resources and transport
cost before attributing the result to inversion count. If validation fails,
use the new hot-arithmetic and checkpoint audits first. Preserve the submitted
source identity and remote diagnostics so the next iteration addresses the
actual failure rather than repeating a static-count assumption.

Final production/audit include fingerprint:
`44f9d5808add05e204b169194af8893dc91a9c01d31448362b0e01c667c9fd06`. Ten source files are covered by this identity.
