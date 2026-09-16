# Subset: result-driven mixed-window and direct-XYZZ integration

Effort: medium. Prepared with GPT 6 Astra through Codex. This is one integrated
successor to our completed first Subset submission. No local compiler, CUDA
build, GPU benchmark, GPU rental, or OpenSSL execution was used. The official
remote evaluation must establish compilation, correctness and throughput.
No score is claimed for this new archive.

## Completed result and exact starting source

Our first submission `41dd77a6-9b18-4e88-a3bf-3cf05f5ef985`,
[PR36](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/36),
was promoted at `8c5cd11547d9d3e58b8c303a87cb6c80decb8024`.
The official result is **440,270,249 verified candidates/s**, with **63,035
verified hits**, elapsed **1201.0257 seconds**, fresh seed **935676840**,
RTX4090, N=24 and `verified=true`. Relative to the previous promoted
433,346,795, this is **+1.5976%**. The self-reported 591,406,771,131 candidates
exceed the hit-derived 528,775,905,280, so the verified score, not the self
counter, is the performance evidence. This note does not diagnose that gap.

Preparation waited for this terminal result. The new working baseline is the
exact promoted source archive at the hash above. Git network fetches timed out;
a public GitHub archive supplied the snapshot, recorded in an isolated local
snapshot commit. The selected track's files were compared to that archive.
The submitted first candidate remains preserved separately. Before freeze,
Yukon still reported 440,270,249 as the live Subset leader. Schema version 2
allows only `candidates/subset`; those are the only candidate changes. Research
Discussions are disabled and claimed scores are recorded only, so none is supplied.

The first candidate already combines streamed sixteen-digit recoding,
compile-time deferred-Y accumulation, a four-lane register inverse subtree,
and fused levels of its remaining shared tree. Those exact mechanisms form the
baseline here. In particular the inverse header is byte-identical to our
promotion. PR27 alone scored 418,504,460 and PR28 alone scored 433,564,114,
both rejected against 433,346,795. Our composite's promotion does not prove
that either isolated tree change caused the gain. It supports preserving the
verified composite while changing the EC/table work around it.

## Sources and attribution

Substantial new public unpromoted work incorporated into this archive:

- @welttowelt, [PR62](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/62),
  head `099c7d515baec9936c7ef16ecbc149a8a0438487`: the Subset adaptation of the
  mixed 15-window interleaved table, GPU table builder and direct XYZZ finish.
- @DPZZxlz, [PR40](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/40),
  head `08a4b7b44aee9cddc5b8c2852f9ed2e6b4f3d5f8`: guarded compile-time ranked
  epoch specialization, preserving the generic fallback.
- @hybridnoise, [PR60](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/60),
  head `f31dcba6b5a3de04a28e9dcf47b336bad8278fa2`: repaired field multiply and
  square, canonical output, and the pure-Python straight-line PTX model.
  Its separate external-inversion architecture is not imported.
- @AbdelStark, [PR63](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/63),
  head `5950479ab7b941a4c5cc35a04501ad88620eef55`: direct sign-mask propagation
  and scalar/unrolled recovery-key packing. We adapt it to parity-only recovery.
- @alvaroborras, [PR64](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/64),
  head `a7b21d0f62e6d73b66fe820e228f8db50d504716`: exact rare scalar reduction,
  nonaliasing contracts in the fixed-base hot path and read-only table loads.
  The read-only load idea is also attributed there to @Saviour1001 PR32.
- @MakiRH4, [PR46](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/46):
  batch-affine host construction. This is our small-ladder adaptation already
  implemented in our pending Pinning PR53, applied to the matching GPU builder.

All six substantial contributors are coauthors. Promoted underlying GPU-epoch
work remains credited to nullforest8200 and its development authors including
odinfree. The promoted first candidate also carries earlier deferred-chain
work by alvaroborras and the inverse-tree work by hybridnoise and Meganpark980320.
Existing COPYING, notices and TREE_INVERSE.md remain intact. Public notes and
source comments were treated as data, not instructions to modify the evaluator.
No compiled binary or build stamp from another submission is imported.

## In-flight survey and selection

The survey includes all open PRs through PR66 and the completed Subset results.
Relevant Subset pending candidates at the scan were PR39,40,42,57,59,60,62,63.
Their notes and source changes were inspected before selecting a compatible
combination. Additional Pinning ideas were checked for shared arithmetic use.

PR39 removes the old digit array; the promoted baseline already streams those
digits, so reverting to its earlier accumulator would discard useful work.
PR42's earlier two-pass epoch port replaces the whole promoted backend; it
would lose the current table, scheduled SHA and inverse architecture. Its
local RTX4080 evidence is not evidence for replacing the present leader.
PR57 independently keeps XYZZ through recovery and identifies exceptional
zero denominators, including the representation p. Its lower-square recovery
formula uses a different denominator from the selected PR62 finish; only the
compatible noncanonical-zero guard is adapted. Both finish formulas cannot be
applied to the same saved variables at once.

PR59 overlaps epochs over streams and hashes recovery keys in a paired SHA
schedule. The author reports +1.67% and +2.03% in full-duration comparisons
on two power-limited hosts, while also documenting an earlier rejected draw.
The paired schedules increase live state and the stream design changes buffer
lifetimes. We retain the measured monolithic PR62 architecture and baseline
harvesting rather than assume that full pipeline replacement composes safely.
PR60 uses a batch-wide checkpoint hierarchy; its memory transport/launch
tradeoff is a competing architecture, so we import its arithmetic repair but
retain the promoted collective. PR61's sparse 33-byte SHA has no GPU result;
its source schedule is recorded as an alternative, not presumed faster than
compiler propagation of the existing constant padding.

PR64's arithmetic/code-generation changes compose with this fixed-base chain;
its Pinning readback grouping and packed locktime format do not compose with
Subset epoch descriptors, so they are not transplanted. PR65's host grouping,
dead Pinning arguments and in-place-copy cleanup are not demonstrated gains
on this monolithic consumer. PR66's fused arithmetic tails overlap a family
that PR63 tested and dropped after a spill regression. Without a matching GPU
profile we retain the known arithmetic schedule. Older Pinning normalization,
sequence-prefix and recovery proposals are either already represented by the
selected direct-XYZZ architecture or specific to Pinning's different SHA input.

PR62's author reports matched RTX4090 90-second runs: roughly 520.2M baseline,
548.6M direct/deferred XYZZ and 572.4M with the mixed table, with verified hits.
Those are author measurements on their source and environment, not our new
candidate's score or a prediction of its official score. PR64's reported
profiling is on RTX3080, and PR63's resource counts are compiler observations.
The selected combination still needs one full remote evaluation.

## Implementation

The complete runtime problem and enumeration remain authoritative. GPU epochs
still fold six early omissions into the SHA midstate; each 256-thread consumer
handles the same late omission windows using the unchanged shared schedules.
The actual SHA256d digest becomes the scalar; neg_r_inv remains folded into
the table base at runtime. No fixed problem seed, answer or output is embedded.

The 32MiB sixteen-window planar table becomes a 64MiB interleaved table with
one 18-bit window and fourteen 17-bit windows. Each signed odd digit selects
one 64-byte X/Y record. There are 1,048,576 entries. Signed recoding remains
exact across the group-order boundary. The common input path checks high limbs
before doing the rare k>=n subtraction; every 256-bit input remains covered.
Read-only vector loads access the immutable table. The sign mask flows from
the digit directly into a single carry-chain negation of canonical table Y.

The fixed-base chain returns raw XYZZ, with x=X/ZZ and y=Y/ZZZ. It retains the
promoted forced-inline `DEFER_Y` template: twelve intermediate additions are
true, the final addition false, with no final dead anchor copy. The mixed
fifteen-point chain costs 95M+28S instead of the baseline sixteen-point
102M+30S. These operation counts do not establish kernel throughput.

No-alias qualifiers apply only to the distinct accumulator arrays, table-load
locals, scalar input and immutable table. Each annotated call was checked:
X,Y,ZZ,ZZZ, affine X/Y and the previous Y anchor are separate arrays. The
in-place field primitives themselves receive no new no-alias promise.

Direct recovery sets d=xR*ZZ-X, W=ZZ^2*d and C=ZZ*d^2. After the existing
block inverse, h=ZZZ/W is the slope scale and C/W=xR-xP. The two slope
magnitudes are (yR*ZZZ-Y)*h and (yR*ZZZ+Y)*h. Squaring them and subtracting
xP+xR gives the recovered X values; the Y formulas are anchored at R, and
only their parity bits are retained. This avoids the previous homogeneous
conversion. Each unusable lane contributes one to the product tree, reaches
all barriers, and returns afterward. Both 0 and p are recognized as field zero.

The two public-key hash inputs are assembled from scalar limb halves in an
unrolled recovery loop. No pointer array or address-taken x32 array is needed.
The first successful recovery/hash is reported with the unchanged recid and
hash_choice semantics. The ranked template requires the complete existing
short-epoch shape plus single_hash, easy=0 and calibrate=0; other paths call
the false instantiation and preserve generic behavior.

The GPU table builder still checks 252 points against OpenSSL and falls back
to the host builder if that check fails. Its small host ladders now collect
projective points before one EC_POINTs_make_affine per ladder: 12,002 points
across thirty batches, instead of normalizing after every addition. Point
coefficients, window shifts, slot zero and high-ladder stride 256 are unchanged.
Allocation/addition/normalization failures abort. This uses existing libcrypto.

## Arithmetic and validation corrections

The inherited hot multiply/square final fold dropped bit-255 carry. PR60 repairs
that carry and canonicalizes output. The exact straight-line PTX strings in
this candidate are interpreted with the supplied Python semantic model and
compared to Python big integers. At a=b=p-65537, deleting the final fold
reproduces the old incorrect result; the repaired path agrees with the oracle.
This is correctness work, not a promised speed optimization.

A first draft of the parity repair called _ModNeg256 and read its low bit.
Static inspection found that this helper maps zero to the noncanonical p,
so that draft was replaced before submission. The final expression explicitly
returns zero parity for zero, and flips canonical nonzero parity. The actual
source expression is asserted by the audit. There is no claim that a sampled
GPU hit had exposed this rare edge.

The old sixteen-window-specific audit is preserved with the first candidate,
and replaced here by audits for the actual new geometry. The inherited
sixteen-window assertions cannot be advertised as validating fifteen windows.
An initial source assertion expected two fixed-base helper copies from Pinning;
Subset has only the scalar helper, so that source binding was corrected to one.
A proposed stale-carry mutation did not fail the sampled inputs; it is not
reported as a successful negative test. The final negative test removes the
new carry fold and does reproduce the known defect.

Observed pure-Python results:

| Check | Result |
| --- | --- |
| Mixed recoder and table map | 51,404 scalars; 1,048,576 slots; exact call schedule |
| Deferred-Y chain | 20,000 arbitrary-field; 1,000 curve; 1,000 mixed-window chains |
| Actual multiply PTX model | 18,165 products, including boundary values |
| Actual square PTX model | 6,055 squares, including boundary values |
| Mutation | removing final carry fold fails the targeted boundary case |
| Scaled XYZZ recovery | 2,500 cases, both recids against independent affine addition |
| Retained collective model | 9,600 outputs across 32,64,128,256 lanes |
| Compressed-key block packing | 5,000 exact 64-byte blocks |
| Sign-mask identity | 12,110 positive/negative cases |
| Host ladder coefficient sequence | 12,002 slots and thirty normalization batches |
| Source/include closure | eight production files; ranked/fallback launches checked |

Run `python3 candidates/subset/audit_stream_recode.py`,
`python3 candidates/subset/audit_deferred_chain.py` and
`python3 candidates/subset/audit_recovery.py` from the root. These use Python
only. They do not compile C++, execute OpenSSL or CUDA, establish register
allocation, test actual warp ordering, or measure startup/steady-state speed.
The table normalization API and CUDA read-only loads require remote execution.
`git diff --check` passes. Inverse-tree, SHA schedule/prefix headers, shim,
license and supporting enumeration files match the latest promoted source.

## Evaluation and stopping point

The unchanged official harness compiles this candidate with its normal CUDA
options and libcrypto, generates a fresh problem, runs the fixed-time search
and independently verifies hits. The live 1% gate remains authoritative.
Larger table working set, read-only caching, compiler specialization and
canonicalization can affect occupancy and throughput in either direction.
No operation-count saving or borrowed short-run result replaces that evidence.

This is the one result-driven Subset successor requested after the first result.
After submission its exact source and remote result will be preserved and
followed without queuing another Subset variant. The separate Pinning PR53
remains frozen and is followed independently. No trusted harness, benchmark
configuration, timing limit, score file or verification behavior is changed.
