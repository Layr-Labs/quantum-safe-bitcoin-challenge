# Pinning candidate: corrected three-field recovery on 128-leaf trees

## Status and scope

This package is a bounded native CUDA performance candidate for the QSB pinning
track. It composes three changes that are each source-bound and reviewable:
the per-candidate product and inverse trees use 128 threads, the existing
fixed-base table uses the shifted persisting-L2 access-policy window, and the
kernel boundary carries three field values for symmetric recovery. The
candidate preserves the current search domain, transaction suffix handling,
sequence and locktime enumeration, signed fixed-base table construction, hit
cap, hit-index encoding, both public-key records, and both hash checks.

The implementation is based on the public source at commit
372a3251e707616011ff6dc0c961cf944d565149:

https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/372a3251e707616011ff6dc0c961cf944d565149

Only the pinning candidate surface is changed. The required license notice,
COPYING file, and existing hash code remain in the package. This note is a
server-validation handoff. It reports source and CPU evidence and makes no
claim of a CUDA runtime score, a promotion, credited points, or measured
throughput for this candidate.

The implementation attribution is GPT-5.6-Luna (Luna Max), max effort, in the
HarnessYukon workflow. Architecture and host-review input came from an Astra
High review, with Astra XHigh parent planning and review. These are development
roles and provenance; they do not represent a GPU result or an independent
peer measurement.

## Three-field Y/V/W recovery

The fixed-base multiplication produces XYZZ coordinates with
xP=X/ZZ and yP=Y/ZZZ. For the fixed point R=(xR,yR), stage 0 computes
d=xR*ZZ-X and the shared denominator

    W=ZZ^2*d.

The block product tree inverts W for every usable lane. The kernel boundary
retains Y, V=ZZZ, and W. X and the original ZZ die after the prepare
calculation. Stage 2 receives I=1/W and derives

    h=V*I
    t=V*h
    u=yR*t
    v=Y*h.

The two slopes are represented by u-v and -(u+v). A fixed host-computed
constant K=3*xR^2 is uploaded once. The shared x base and cross term are

    F=2*u^2-K*t+xR
    H=2*u*v,

which give x_plus=F-H and x_minus=F+H. The two y values are then anchored at R
with the corresponding slope formulas. This is the symmetric 10M+2S recovery
path. It removes the old C field, removes the extra C square and
multiplication, and computes both x results from one square and shared
products.

The corrected GPUMath header establishes the canonical coordinate invariant
that every field result consumed by this path is in [0,p). The finish helper
therefore squares u directly and does not perform a per-finish normalization or
an upper-half representative branch. This relies on the canonical invariant
implemented by the header's repaired carry path; it is a source-bound
dependency of this candidate rather than an assumption about GPU lowering.

The state layout is exactly six ulonglong2 planes per candidate: two for Y,
two for V=ZZZ, and two for W. The planes retain the batch-size
structure-of-arrays stride and 16-byte alignment. At the configured
16,777,216-candidate batch this is 96 bytes per candidate per direction and
1.5 GiB for the state allocation. W is the same denominator leaf consumed by
the unchanged product and inverse tree recurrence. The allocation, stage 0
stores, stage 2 loads, and usability reconstruction use the same order.

## Complete exceptional recovery

The shared inverse is undefined for an inactive lane, an infinity point, or a
point whose x coordinate equals xR. The source maps inactive lanes to the
multiplicative identity before every collective and returns from stage 2 only
after the inverse collective has completed. Active cold lanes use a validity
mask so an exceptional point cannot cause an uninitialized record or a
divergent barrier.

For P at infinity, the two output records are R and -R and both validity bits
are set. For P=R, the finite +2R record is emitted and the infinity P-R record
is suppressed. For P=-R, the infinity P+R record is suppressed and the finite
-2R record is emitted. The comparison uses the scaled Y value
Y versus yR*ZZZ, so it remains valid for the projective representation. This
mapping preserves both-recId semantics at the cold boundary. The exact
canonical affine doubling dispatch in the corrected header also handles the
fixed-base mixed-add edge cases, including finite doubling and infinity
outputs.

## Candidate tree and root hierarchy

The candidate tree has N=128 leaves, with leaf nodes 0 through 127,
checkpointed non-root nodes 128 through 253, and root node 254. The generic
checkpoint helper uses a stride of 4*N 64-bit words per candidate block. The
inverse helper restores all 126 internal non-root nodes, expands the supplied
root inverse through counts 2, 4, 8, 16, 32, and 64, and writes the 128
canonical leaf inverses. Partial final candidate blocks use identity leaves.

The root-group hierarchy remains 256 lanes. Candidate blocks are grouped in
sets of 256 roots. The root-group prepare, multi-CTA super-root inversion,
root-group finish, and stage 2 launches remain ordered. A full 16,777,216
candidate batch has 131,072 candidate blocks and 512 root groups, so the
super-root inverse launch covers all groups with two 256-lane CTAs. The source
does not truncate at one CTA, and partial root groups use identity lanes.

The fixed-base table remains the existing 15-term, 64 MiB table with the
current mixed signed windows. The shifted access-policy window starts after
the first chunk only when the table is larger than that cold chunk plus the
device-reported persistence budget. The code keeps the existing capability
queries, byte caps, maximum-window guard, default-stream ordering, and
diagnostic error path. With a 50 MiB persistence budget, the geometry would
skip the first 8 MiB chunk and cover a 50 MiB later range, approximately 12.5
of the 15 regions by byte range. That is a conditional layout calculation,
not a measured GPU result.

The tree and cache effects may overlap or interfere. Candidate-tree barriers,
global checkpoint traffic, launch overhead, register and occupancy changes,
the three-field recovery arithmetic, table locality, and driver policy can
change the result in either direction. Measured performance remains unavailable. The conditional analytical forecast
below is a source-based planning estimate, not a CUDA result. The public three-field
peer result of 657,885,190 versus the then-base 644,546,620 is retained as
motivation and provenance only; it is not this candidate's score and is not
added to the 128-tree or shifted-cache effects.

## Focused evidence

The source-bound 128-tree audit passed 52 partial or mixed batches, 353
candidate checkpoints, and 52 root checkpoints. It checked zero and inactive
identity leaves, raw and noncanonical field representatives, candidate and
root launch boundaries, all multi-CTA super-root geometry, and the shifted-L2
guard cases.

The three-field external audit passed 210 split-tree cases and 4,018 on-curve
baseline, affine, and compressed-encoding recovery comparisons. It covered
random affine points, projective z values 1, 2, p-1 and random nonzero z,
near-p denominators with deltas 65,537, 2^20 and 2^32, and the cold
infinity/P=+R/P=-R valid-mask branches. The symmetric path matched the
independent affine formulas and the production compressed-byte construction
in every tested usable case.

The two-level root audit passed 137,492 roots over 12 boundary sizes,
including counts that require more than one 256-root super-tree. The
six-plane vector-layout audit passed all tested batch sizes and confirmed the
exact 1.5 GiB allocation. The fast-tail audit passed 20 host selection cases,
and the SHA-tail audit passed 2,320 synthetic SHA256d comparisons across
sequence and locktime boundaries. Python bytecode compilation and git
whitespace checks also passed.

A compatible actual-source host arithmetic receipt is retained as supporting
evidence: HOST_PASS cases=656, squares=276, aliases=3, doubles=5; 106
extracted PTX vectors for each multiply and square were congruent, the known
(p-65537)^2 case matched 0x100020001 exactly, and the host wrapper executed
the canonicalizer. The same receipt compiled the extracted production
qsb_xyzz_finish_symmetric body against those exact host arithmetic branches
and matched the independent affine reference for 82 usable projective cases:
64 deterministic random cases and 18 near-p denominator boundary cases,
including both output x coordinates and both recovery parities. The complete
infinity/P=+R/P=-R cold mapping was checked separately in three source-bound
cases. The final stdout was
HOST_PASS cases=656 squares=276 aliases=3 doubles=5 symmetric=82. The
receipt validates the reviewed arithmetic boundary, alias semantics, and
the narrow recovery extraction. It does not validate a compiled CUDA kernel,
target driver behavior, resource counts, throughput, or hit production.

The static corrected-call ledger is 105 field multiplies plus 30 field
squares, 135 field calls per three-field candidate (M denotes multiply, not a
unit prefix). The inline carry tail has five static instructions, with two
normally executed and five executed on the carry path; predicated issue
consumption and the host canonical predicate remain target-dependent. This is
an instruction ledger and a conditional resource discussion, not a throughput
measurement.

The focused checks can be reproduced from the candidate package with:

    python3 candidates/pinning/audit_tree128_pipeline.py
    python3 candidates/pinning/audit_external_pipeline.py
    python3 candidates/pinning/audit_superbatch_roots.py
    python3 candidates/pinning/audit_vector_state_layout.py
    python3 candidates/pinning/audit_fast_tail_contract.py
    python3 candidates/pinning/check_tail_words.py
    python3 candidates/pinning/audit_field_source.py

No CUDA compiler or NVIDIA device was available for this development run.
The official server build, verifier, ranked workload, and current frontier
comparison remain authoritative. The next validation step is the authorized
server submission path under the current rules.

## Public provenance and coauthor attribution

The three-field Y/V/W formulation follows the public pinning work in PR129,
commit 3bccced9abdbf4fe89f8f8218782ec8c0d112f39:

https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/3bccced9abdbf4fe89f8f8218782ec8c0d112f39

The public PR's Y/V/W recovery and fixed K computation are used here with the
corrected header and the complete cold mapping. The public tree geometry
reference is the 0xCramJam work at commit
5c85ae053bc0effa27db4df76fcbf09ee4aaa1b2:

https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/5c85ae053bc0effa27db4df76fcbf09ee4aaa1b2

Only the compatible 128-lane tree and launch geometry are retained from that
reference; speculative lazy arithmetic, prefetch, shared parking, and
offloaded-tree variants are outside this candidate. The shifted cache policy
is attributed to the public ercumentyildirim pinning PR142:

https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/142

The public three-field work is attributed to xlib. The required unpromoted
coauthors for this composition are xlib, 0xCramJam, and ercumentyildirim.
Their public mechanisms remain separately identified so the record does not
turn a bundled peer result into an isolated causal measurement.

## Server decision rule

A successful hosted result requires a clean official build, normal
termination, preservation of all-input correctness, complete hit grammar and
both recIds, and a verified improvement against the current frontier
653,505,529 candidates per second under the current rules. A build failure,
race, hang, arithmetic mismatch, lost cold record, lower result, or failed
promotion margin falsifies the performance hypothesis. A neutral or negative
run still closes useful learning: three-field state traffic and recovery
arithmetic are source-compatible with the 128-tree and shifted-cache geometry,
while their combined runtime effect remains an empirical question.


## Conditional analytical performance forecast

The exact final source above has no local NVIDIA runtime measurement. A bounded
Astra High performance review combines relevant public observations with an
explicit source-based instruction-throughput model to decide whether this is a
credible potentially leading server attempt. This forecast is not a claimed
score sent to the verifier, a confidence interval, or a promised promotion.

The reference is 653,505,529 * (1.019/1.007) * (667.83/659.78) *
(657,885,190/644,546,620) = 683,213,646 candidates/s. The first ratio is a
provisional isolation of the public paired tree-plus-lean result from its
reported lean-on-tree component; it is not a clean factorial measurement.
The second is the public shifted-cache paired result. The last is the official
three-field result against its historical base. These transfers may overlap,
interact, or fail on this exact corrected candidate.

The repair penalty is calculated from an explicit cost ledger rather than an
arbitrary percentage. Units represent an operation with a theoretical rate of
64 results per clock per SM. NVIDIA's native-arithmetic throughput table motivates
relative weights: wide/extended multiplication 2, an extended 64-bit add modeled
as two extended 32-bit operations 2, extended 32-bit add 1, ordinary 32-bit add
0.5, bitwise/shift/predicate 1, and register-pair alias moves 0. These are lowering
assumptions, not observed SASS. Source:
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#throughput-of-native-arithmetic-instructions

The model counts 105 field multiplications at 316 units and 30 field squares at
219 units, totaling 39,750. The normal single-hash path performs four SHA-256
transforms; source rounds, schedules, and feedforward contribute 5,296 modeled
units. The point/recovery path adds 31 field additions and 80 subtractions,
modeled at 22 and 20 units respectively, totaling 2,282. The raw inversion trees
contribute approximately 978 amortized units. Total counterfactual work is
48,306 units. Actual constant folding, native instruction fusion, memory overlap,
SIMT utilization and synchronization are not measured by this ledger.

The exact cold-carry header adds five static PTX instructions per coordinate
multiply/square against the public unrepaired header. All five are charged even
when three correction instructions are predicated off. The central assumption
for the C++ canonical predicate is seven 32-bit equivalent logic/predicate/control
units per call; this is a plausible lowering model, not a compiler receipt.
Across 135 calls that gives 1,620 added units. The finite exceptional guard adds
16 modeled units. Thus the derived fractional overhead is 1,636 / 48,306 =
3.3867%, and the conditional forecast is 683,213,646 / (1 + 1,636/48,306) =
660,832,934 candidates/s. This is 1.121% above the observed frontier and only
0.120% above its current one-percent promotion threshold of 660,040,584.

This is fragile: changing the canonical-predicate assumption from seven to nine
units predicts 657,279,503, below promotion. Reducing modeled hashing work through
compiler folding can likewise remove the margin. New spills, occupancy loss,
longer dependency chains, cache/state-traffic overlap, or failure of the public
ratios to transfer can invalidate the forecast. Sensitivity values are alternative
assumptions, not statistical bounds. The obsolete signed-square workaround has
been removed under the corrected arithmetic invariant; no extra gain is assigned
to that removal. The official complete run is the decisive measurement.
