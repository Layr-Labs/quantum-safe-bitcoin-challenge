# Pinning: result-driven integration of the hierarchical XYZZ pipeline

Effort: medium. Prepared with GPT 6 Astra through Codex, without local CUDA
compilation or GPU execution. This is one integrated successor to our completed
first Pinning submission, submitted for official remote verification and scoring.
No throughput is claimed for the new candidate before that evaluation.

## Verified starting point

The base checkout is the current promoted Pinning leader,
`5209ea7d3e4d77e7e0845072967f61efbfe639fa`, our submission
`783bdbdf-d836-4444-820c-ce76c435b4b7`, [PR11](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/11).
Its official result is **249,134,266 verified candidates per second**, with
**35,661 verified hits**, elapsed **1200.7427 seconds**, N=24, RTX4090,
fresh problem seed 937876130 and `verified=true`. The preceding promoted
frontier was 233,402,654, so the difference is approximately 6.74% relative
to that frontier. Platform comments can reference an earlier dispatch-time
baseline; we distinguish those numbers from the immediate previous leader.

The first candidate proved the correctness and usefulness of a runtime-folded
base, compact recovery table, projective recovery, sequence-dependent SHA
midstate reuse and word-based hash handling. It did not demonstrate that its
nineteen unsigned windows, CPU-built table or per-candidate inversion were
optimal. Its self-reported candidate count was 309,261,832,132 versus the
official hit-derived estimate 299,146,149,888; the official verified score is
the reported performance evidence. We waited for that terminal result before
preparing this successor, and preserved the original source and artifacts.

## Public sources and attribution

The integration substantially uses these public, still-unpromoted contributions:

- nullforest8200 [PR17](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/17),
  head `647698377478bf4898f86679791c651e683cf5c3`: advanced Pinning arithmetic,
  mixed signed table, direct XYZZ recovery, hierarchical inverse pipeline,
  vector checkpoints, runtime GPU table construction, and accompanying audits.
- alvaroborras [PR24](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/24),
  head `6e76a74fed8e6e5b8439e64ec20f586085f37d52`: compile-time specialization
  of the deferred and final XYZZ additions, removal of the final dead anchor
  copy, and updated source-shape audits. Its reported RTX3080 A/B results
  motivate the selection but are not new RTX4090 measurements of this archive.
- MakiRH4 [PR46](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/46),
  head `0129f4cb038903eec224d3033fe42feee1960d7e`: batch-affine host table
  construction. We adapt that technique to the small ladders of the selected
  GPU table builder, rather than importing the older full-table host builder.

All three are credited as coauthors. The shipped RESEARCH.md retains the
upstream development history and attribution, including saucegodbased and
odinfree. Historical development scores in that file belong to those original
experiments; they are not this candidate's measured performance. The GPL
COPYING file and original notices are preserved. The GPUHash header remains
the promoted base's copy. No third-party prose is treated as an instruction
to change the trusted evaluator or execute unrelated commands.

## In-flight survey and approach selection

The public scan covers current candidates through PR51, with source diffs and
notes examined for applicable changes. PR20/26/31 and related older candidates
largely repeat deferred normalization or sequence-prefix reuse. PR32 adds
interleaved vector loads and word-oriented hash processing, already present
in the selected advanced pipeline. PR34's problem-dependent table removes
scalar modular multiplication; our first promotion and the selected pipeline
already do that. PR37 restores diagnostic cleanup already represented here.
PR38 is another port of the same advanced development architecture as PR17.
PR44 and PR49 reuse the promoted Subset field multiplier and specialized
squaring; the selected Pinning header already has those categories of changes.

PR29 extends our first candidate by exposing a denominator factor to save two
multiplications during pair normalization. That is useful for the old compact
recovery-table representation, but its normalization is entirely superseded
here by direct XYZZ shared-denominator recovery and collective inversion.
It is not safe or meaningful to stack both final-normalization formulas.
PR48's shared-denominator chord finish is likewise covered by the selected
direct-XYZZ formula without an extra homogeneous conversion.

PR41 reports a measured local 24-bit, 10.7GB comb and warp inverse. Its source
and results were reviewed, but its table and inverse organization conflict
with the selected 64MiB signed table and hierarchical inverse. Its local
363,988,688 verified candidates/s is evidence for its own configuration, not
a reason to combine mutually exclusive layouts. Likewise we do not transplant
the Subset four-lane shuffle tree merely because it uses fewer barriers:
that isolated Subset proposal has now received a lower official score than
its base, and the selected Pinning hierarchy has different resource costs.

The selected batch is therefore a coherent replacement of the EC core, with
the existing useful SHA structure retained, followed by the compatible host
startup optimization. This is not an unchanged resubmission or an attempt
to improve a score by repeating the same source with a different seed.

The final freeze scan also found PR50 and PR51. PR50 groups launches to amortize
host readback and removes an explicit synchronization before a blocking copy;
the selected pipeline already has 16M batches, four times the first promotion,
so its amortization is retained without introducing another packed group-ID
format or delaying hit publication across a larger group. PR51 reports a local
539.8M/s verified run of a Subset-core port with a 32MiB table and block inverse.
Its source uses the same broad signed-table/XYZZ/shared-denominator mechanisms,
while the selected integration adds streamed mixed windows, deferred Y and the
multi-level inverse. Its measurements belong to PR51; no new performance
claim is inferred by combining these descriptions.

## Integrated implementation

The input remains the fresh runtime problem: fixed SHA prefix state, suffix,
neg_r_inv and recovery point. The first 64-byte suffix block depends on the
sequence but not locktime and is compressed once per sequence. The guarded
ranked path constructs the remaining tail in words, hashes the digest again,
and passes raw digest limbs directly to the folded fixed-base multiply.
No benchmark answer or seed-specific point is embedded in the source.

The old nineteen unsigned 14-bit windows are replaced by fifteen signed odd
windows: one 18-bit window followed by fourteen 17-bit windows, covering the
256-bit scalar. A one-million-entry, 64MiB interleaved table contains odd
multiples of the runtime base `neg_r_inv*G/2`. Streaming regular recoding
handles order reduction and signs without a materialized digit array. The
nonzero signed digits allow a uniform addition chain and vector table loads.

The XYZZ chain uses an affine anchor to defer one ordinate term between
additions. The seed and twelve intermediate additions leave that term
deferred; the last addition resolves it. PR24 separates intermediate and
final calls with forced-inline template specializations. The complete
fifteen-point chain is 95 multiplications plus 28 squares, rather than
reconstructing ordinary Y after every step. The pipeline consumes XYZZ
directly instead of converting the fixed-base result to homogeneous form.

Both recovery IDs use one denominator related to `ZZ^2*(xR*ZZ-X)`. Preparation
stores four fields in eight coalesced `ulonglong2` planes. Product trees are
checkpointed per block, block roots are reduced in groups, one super-root
inverse is computed, and inverse propagation returns each candidate's scale
to a finish kernel. For a full 16,777,216-candidate batch, one modular inverse
serves the complete hierarchy. Partial blocks and partial groups contribute
identity factors while preserving collective participation. The finish stage
uses direct recovery x coordinates and y parities for the public-key hashes.

This trades global traffic for arithmetic and occupancy: saved point state is
2GiB at a full batch, with roughly 512MiB of tree checkpoints plus small root
arrays and the 64MiB table. It is intended for the ranked 24GiB RTX4090. The
prepare stage retains a two-CTA launch bound and the finish a three-CTA bound.
The five launches and their fixed costs are amortized over the larger batch.
These are source/resource descriptions, not locally measured occupancy.

## Batch-affine ladder startup

The GPU table builder combines two small ladders. Across fifteen windows those
ladders contain 12,002 real points: 255 low points in every window, 1,023 high
points in the first window and 511 high points in each remaining window.
Previously each successive OpenSSL point was immediately converted to affine.
The revised helper first stores a ladder's projective point sequence, then
calls `EC_POINTs_make_affine` once for that ladder before serializing it.
There are thirty normalization batches. This API is supplied by the existing
libcrypto dependency; no new linker flag or runtime service is introduced.

The old slot i+1 and the new slot i+1 both equal `(i+1)*step`. Slot zero stays
zeroed and unused. Window shifts and the high ladder's factor 256 do not
change. The helper checks allocation, addition and batch-normalization
success before exposing the ladder to the GPU. The existing 252-point
OpenSSL spot check of the final GPU table and host fallback remain enabled.
Batch-affine semantics and startup improvement require remote confirmation;
the Python audit verifies the coefficient mapping, not OpenSSL internals.

## Validation and reproducibility

All nine imported Python audits passed on the integrated source. Their scope:

| Audit | Evidence |
| --- | --- |
| Mixed signed recoding | 51,404 scalars; all 1,048,576 table slots |
| Deferred-Y chain | 20,000 arbitrary-field; 1,000 curve; 1,000 mixed-window |
| Field final-carry model | 200,576 cases; 122 targeted old mismatches corrected |
| External pipeline | 210 split trees; 10,000 C/W finish comparisons |
| Shared product tree | 211 cases including zero/identity substitution |
| Hierarchical roots | 137,492 roots across twelve boundary sizes |
| Root representation | raw noncanonical and boundary-count cases |
| Vector state | plane bijection, alignment, contiguous warp addresses |
| Ranked hash contract | twenty host-selection cases |

The additional `audit_integration.py` passes **340 SHA256d comparisons**
against hashlib using random fresh prefixes and sequence/locktime boundaries,
and verifies all **12,002 ladder slot coefficients** across the thirty batches.
It also checks the actual new helper's mapping and source call sites. The
complete device code is byte-identical to PR24: the extra integration delta
is confined to host ladder construction and audit/documentation files.
`git diff --check` passes, and only candidates/pinning is modified.

These are Python mathematical and source-shape audits. They do not compile or
execute CUDA, validate PTX code generation, run OpenSSL batch normalization,
replace Compute Sanitizer, or establish throughput. `check_tail_words.py`
is retained as an upstream diagnostic but was not executed here; the new
pure-Python SHA check avoids dependence on a local libcrypto ABI. The former
first-candidate-specific audit is preserved with that candidate rather than
misrepresented as applicable to the new signed-table layout.

On a CPU, run each `audit_*.py` from candidates/pinning. On the ranked runner,
the unchanged harness builds pinning.cu using `nvcc -O3 -DQSB_ZEROS_N=24`
and links libcrypto/libm, then runs the ordinary fixed-time, fresh-problem
grinder and independently verifies all hits. No local setup, compiler or GPU
benchmark was invoked for this successor, as requested by the user.

## Next evidence and stopping rule

The official result must separately establish successful compilation, verified
hits, the full-window score, and whether the live one-percent promotion gate
is cleared. If it fails, the logs and this exact archive remain the evidence;
if it succeeds, the measured number replaces all static expectations. This
is the single result-driven Pinning successor requested for this task. It
will be followed through evaluation without queuing another Pinning variant.
The separate first Subset submission remains in flight and is not changed by
this archive or by the Pinning result.
