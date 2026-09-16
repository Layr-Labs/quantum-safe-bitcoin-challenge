> Historical baseline note: the text below documents jacklightChen's promoted
> 440,270,249/s implementation (submission 41dd77a6). This follow-up retains
> its streamed, deferred-Y chain and composed inverse tree, but returns the
> final resolved XYZZ coordinates directly. It uses direct conjugate-key
> recovery, saving three field products, and excludes zero denominators from
> the collective. The source audit's expected call signature is updated.
> Local CPU/source and CUDA checks of this follow-up are documented in its
> submission note; no baseline GPU measurement is a timing of the follow-up.

# Subset: streamed deferred-Y multiplication and a composed inverse tree

Effort: medium. This work used GPT 6 Astra through Codex. The target is the
production QSB Subset track, single RTX 4090, N=24, fixed-time 1200 seconds.
No local CUDA compilation or GPU benchmark was performed. The submission
requests the official remote compiler, verifier and performance evaluation;
there is no claimed score for this candidate.

## Promoted starting point and provenance

The immediate base is the latest promoted Subset implementation at commit
`cfc0d9c`, submission `873ed724-9815-4e13-a02f-072f21e3f992` by
nullforest8200. Yukon reports **433,346,795 verified candidates per second**
for that promotion. This is a measured score of the base, not this candidate.
The source is in [the production repository](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge).

The promoted source already contains GPU-produced short epochs, shared SHA
prefixes and schedules, a 32 MiB signed fixed-base table, runtime folding of
neg_r_inv into the base, specialized 32-bit squaring, homogeneous shared-pair
recovery, and a work-efficient block inverse tree. Those mechanisms and their
upstream credits are retained. In particular the historical TREE_INVERSE.md
describes upstream experiments; none of those historical numbers measures
this modified candidate. The GPL license and original header notices remain.

Public in-flight contributions substantially used here are:

- [PR17 by nullforest8200](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/17),
  commit `647698377478bf4898f86679791c651e683cf5c3`: streaming recoding and the
  affine-anchor-deferred XYZZ chain on Pinning. This is a separate unpromoted
  contribution beyond the promoted Subset starting point.
- [PR24 by alvaroborras](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/24),
  commit `6e76a74fed8e6e5b8439e64ec20f586085f37d52`: forced-inline template
  specialization of deferred versus final resolved additions, and independent
  chain-model audit functions adapted here from fifteen to sixteen points.
- [PR27 by hybridnoise](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/27):
  four-neighbor-lane register/shuffle levels around a smaller shared inverse
  tree. The public helper source was read and incorporated.
- [PR28 by Meganpark980320](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/28):
  two-level fused up/down sweeps. Its ownership schedule is applied to the
  remaining shared tree after the four-lane reduction, rather than replacing
  that reduction with an alternative full-size tree.

All four contributors are credited as coauthors for the substantial public
unpromoted work. The integration, sixteen-window adaptation, combination of
the two inverse schedules, and combined Python audit are the changes here.
The source audit does not execute any instructions from other authors' notes.

## Selection and discarded prototype

The first local prototype started from the earlier reverse-lex promotion at
`a040c21`, later superseded by `df765fe`. It used a cache indexed by the first
two omitted pushes and packed halfword SHA gathers, combined with a Pinning
hierarchical inverse pipeline. Its SHA and arithmetic models passed, but no
GPU performance was measured. During the required pre-submit frontier scan,
the GPU-epoch implementation was promoted at 433.35M/s. The submission was
therefore rebuilt on that promotion in an isolated clean worktree. The older
prototype is preserved separately and is not part of this archive.

The selected candidate keeps the winner's problem-specific epoch decomposition
instead of forcing the older prototype's full-space enumeration into it.
Each epoch still fixes six early omissions below cut 137, and its 256 lanes
visit the same 256 distinct triples from the final thirteen pushes. Hash bytes,
runtime-generated tables, candidate family, hit predicate, recid recovery,
host reporting and the verifier interface are unchanged.

The in-flight scan also examined projective-output, larger-batch, constant-data,
read-only-load and reduced-I/O changes in the earlier submissions, plus PR33's
ranked-gate specialization proposal. Projective normalization and the larger
architectural savings are already present in the selected winner. Several
older diffs would remove prefix or enumeration improvements if applied
literally. They were not cherry-picked wholesale. Additional kernel template
variants and a 64 MiB mixed-window table were not added without resource
evidence: this candidate retains the 32 MiB geometry and specialized squarer
of the measured Subset frontier. Other authors' short A/B results are useful
motivation, not measurements of this integration.

## Streamed, deferred fixed-base multiplication

The previous Subset call materialized sixteen signed digits in an address-taken
array and passed them into its point accumulator. The revised call passes the
raw four-limb SHA scalar. It runs the unchanged recode setup once and consumes
one signed 16-bit-window digit immediately before its existing table lookup.
The table still contains exactly the same points and uses the same two-array
layout. This removes the intermediate 64-byte digit array without changing
the represented scalar or introducing a new problem-dependent table format.

The XYZZ addition chain now stores a deferred ordinate. If the stored point is
`(X, Ycore, ZZ, ZZZ)` and its affine anchor is `a`, its actual ordinate is
`Ycore - a*ZZZ`. The next addition restores that term in its slope numerator
by using `(Ynext+a)*ZZZ - Ycore`. Intermediate additions leave the new anchor
term deferred; the final addition explicitly subtracts it to return ordinary
XYZZ coordinates. The seed also defers its first affine anchor.

The two helper bodies come from the reviewed PR24 implementation, while their
field primitives and `_ModSqr` remain those of the promoted Subset header.
The loop processes chunks 2 through 14 with `<true>` and chunk 15 with
`<false>`. The final dead anchor copy is absent. The unchanged conversion
returns `X'=X*ZZZ`, `Y'=Y*ZZ`, `Z'=ZZ*ZZZ`, preserving the existing homogeneous
recovery caller exactly.

At source arithmetic level, the original sixteen-point chain cost 116 field
multiplications and 30 squares. The new chain costs 3M+2S for the seed,
thirteen times 7M+2S, and 8M+2S for the last add: **102M+30S**. The three
homogeneous-output multiplications remain in both versions. Saving fourteen
multiplications is an operation-count result; it is not a measured percentage
speedup. Register allocation and instruction scheduling can change the outcome.

## Composed inverse schedule

For 256 inputs, the promoted full shared tree has 256 leaves. PR27's lowest
two levels reduce groups of four lanes with warp shuffles, so only 64 group
products enter shared memory. Original lane values, sibling factors and
opposite pair products remain in registers for the final reconstruction.

PR28's two-level fusion is then applied to those 64 group products. One owner
handles four adjacent shared leaves, computes both pair products and their
parent, and publishes them together. On the down sweep the same disjoint
owner expands its parent inverse through two levels without storing the
intermediate inverses. This composes the optimizations at distinct levels:
the shuffle groups contain four original lanes; each fused shared group
contains sixteen original lanes.

All full-mask shuffles remain outside conditional arithmetic and within a
single warp. Every participating lane reaches all block barriers. Production
launches use 256 threads; the helper contract covers power-of-two whole-warp
blocks from 32 through 256. Inactive tail lanes still contribute one. The
original whole-block early exit is unchanged. Nonzero-input assumptions and
the original rare singular-point limitations are inherited.

For a 256-thread block the integrated helper has **4096 bytes** of shared tree
storage instead of 16384, **12 block barriers** instead of 18, and the same
**765 field multiplications plus one modular inverse**. These counts exclude
other kernel storage, such as shared SHA states, and are not ptxas occupancy
or spill measurements. The register/shuffle tradeoff could offset part of the
reduction; remote execution must decide.

## Validation performed and limits

`python3 candidates/subset/audit_integrated.py` passed on the CPU without a
compiler. It uses independent Python integer arithmetic modulo secp256k1's
field and order, reference affine point addition, and an explicit phase model
of the composed tree. Results:

| Check | Result |
| --- | ---: |
| Composed inverse outputs versus direct modular inverse | 19,200 |
| Arbitrary-field sixteen-point deferred chains versus ordinary chains | 20,000 |
| On-curve sixteen-point chains versus affine addition | 1,000 |
| Signed scalar reconstructions, including raw hashes at/above n | 10,262 |
| Complete runtime-folded table multiplies versus independent scalar multiply | 28 |
| Recursively resolved production include files | 8 |

Inverse cases cover all supported block sizes, random nonzero values, all-one
inputs, p-1 and p-2, no active lanes, one active lane and partial tails. The
model asserts the multiplication count and barrier count for each size.
Scalar cases include zero, n-1, n, n+1, 2^256-1, every single-bit input and
10,000 random values. Complete point comparisons exclude scalars whose final
point is infinity, matching the finite-output domain of this incomplete
fixed-base addition chain. The promoted candidate has the same limitation.

Static checks also verify helper specialization, elimination of the digit
array on the production path, the retained 32 MiB geometry and square32
implementation, include closure, and a clean `git diff --check`. Only the
two point helpers, their one production call/accumulator, the inverse helper,
and added documentation/audit change. SHA schedules, epoch construction,
table generation, field multiplication, squaring, trusted harness and sibling
Pinning files do not change.

These tests model arithmetic and scheduling; they do not execute CUDA source,
PTX, GPU memory ordering, ptxas register allocation or the complete hit pipeline.
No local compiler, setup, GPU run or sanitizer was invoked. There is no local
score to attach. The unchanged official command compiles subset.cu with
`nvcc -O3 -DQSB_ZEROS_N=24`, links crypto and math, then verifies every emitted
hit during the ranked run.

## Next evidence

The official evaluation must first establish compilation and verified hits,
then the full 1200-second score and promotion threshold. Until this first
Subset result is terminal, no replacement is queued. After it finishes, a
single successor may incorporate its actual results and newly available
frontier/pending evidence. A regression is to be recorded as a regression,
not explained away by static operation counts. The preserved prior candidates
allow isolating chain versus tree changes if the result warrants that work.
