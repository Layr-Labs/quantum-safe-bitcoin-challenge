# Pinning: signed digit decoding and weighted cofactor recovery

This candidate removes work from the fixed-base scalar decoder, point-chain
scheduling and public cofactor recovery pipeline. The search still visits the
same sequence and locktime domain, derives both recovery keys, applies the same
SHA predicates and emits the same first-hit choice for each candidate. The
production candidate has no diagnostic iteration limit. The organizer's problem
generator, verifier, fixed-time wrapper and score calculation are unchanged.

The comparison baseline is ercumentyildirim's public submission
`ce0aff4e-be9b-4fae-a8a3-b0ed7acabb8e`, source
`3b81be52f08e3f412f9823f9dce46c12f2d1fad3`. It was the fastest artifact in our
local public-source cohort, then became the official frontier at
**713,225,734 verified candidates/s**, promoted as
`33753cc7bc2ebb7054b667805ce216dc5fe7fecd`. The submitted tree is based on that
promotion; its baseline production include closure matches the exact source
used in the comparison. Only `candidates/pinning` changes.

## Implementation

### Decode the signed odd representative directly

Let `n` be the secp256k1 group order and `D = 2*(k mod n)-n`. `D` is odd and lies
in `[-n,n)`. The existing table represents multiples of half of the fixed base,
so `D*(A/2) = k*A` in the group. The new decoder keeps D's raw 256-bit residue
and its mathematical sign directly. It avoids converting D to an absolute
value and then applying the global sign to all fifteen decoded digits.

For an odd integer x, define the regular digit and remaining scalar by
`d_w(x) = (x mod 2^(w+1))-2^w` and
`T_w(x) = 2*floor(x/2^(w+1))+1`. Both functions commute with negation.
Consequently the direct signed extraction gives exactly the same signed table
indices as the inherited absolute-value extraction followed by a global sign.
The last window uses sixteen stored bits and the explicit sign of D; its index
is f for positive D and `65535-f` for negative D. The table layout, lookup
records, window widths and addition order are unchanged.

The raw `k >= n` case retains an exact conditional reduction. Computing
`2*k-n` uses `C = 2^256-n`, which is a 129-bit constant: the third 64-bit limb
is one and must not be omitted. The fifteen immutable index/sign words are
stored in per-thread shared-memory planes. This lets scalar limbs die before
the point-chain loop. The shared arena is reused by the cofactor tree only
after every lane reaches a full block barrier.

### Keep one deferred point-add body and shorten register lifetimes

All thirteen remaining mixed additions use one rolled deferred-Y
specialization. The final ordinate is resolved once after the loop, instead
of retaining a separately compiled resolving-add body. The field-operation
count and algebra are preserved. The slope numerator is formed before the
X-difference so the old ordinate is no longer live during the later products.

The independent even and odd rows of the existing multiplication and squaring
PTX are interleaved. Complete linear carry chains are expressed as single pure
inline-PTX expressions with explicit operands. They do not have memory side
effects, so the previous per-instruction volatile memory clobbers are
unnecessary. These changes preserve the inherited hot field primitives' bit
contract; they do not claim to repair every pre-existing noncanonical raw-input
case in those hot primitives.

### Weight one root instead of every recovery leaf

For a fixed recovery point `R=(a,b)` and a prepared point
`P=(X/U,Y/V)`, with `U=Z^2` and `V=Z^3`, use the denominator
`D_i=V_i*(a*U_i-X_i)`. The public 128-leaf cofactor traversal produces the
product excluding each leaf. Define `h_i=U_i*C_i`, then save only
`vbar_i=Y_i*h_i` and `tbar_i=V_i*h_i`, four 128-bit state planes per leaf.
Inactive and unusable leaves retain their masks.

At a tree root, publish both `I=1/T` and `J=b/T`. Each finish lane obtains
`u=tbar*J` and `v=vbar*I`; this hoists a multiplication by the fixed b from
every leaf to one operation per root. With canonical u and v, set
`l=u-v`, `m=u+v`, `S=l+m`, and the host-derived constant `c=3*a^2/(2*b)`.
The square-free recovery identities are
`x_plus=a+S*(l-c)` and `x_minus=a+S*(m-c)`. Their ordinate parities come from
`l*(a-x_plus)-b` and `b-m*(a-x_minus)`.

The exact recovery multiplier retains the last reduction carry for every raw
256-bit input. Write `B=2^256`, `K=2^32+977`, and `p=B-K`. After the second
pseudo-Mersenne fold the value is below `B+K^2`. If it carries, the remaining
low residue plus K is below `K^2+K < 2^65`, so only three 32-bit limbs need
that last correction. No rare carry is dropped. Tree-internal representatives
may remain raw; canonicalization remains at the inversion and zero-test
boundaries and wherever the recovery algebra requires it.

### Compute only the ordinate parity that the hash needs

For `-p < r-b < p`, the canonical difference has parity
`((r_low xor b_low) xor (r<b)) & 1`, because p is odd. An exact raw product r
lies in `[0,B)`. If the fixed `b[3] != 0`, then `b >= 2^192 > K`, which proves
that interval bound without first normalizing r. For smaller b, the original
normalization is retained. The reverse difference uses the same argument.
This removes two redundant normalizations while preserving the small-b path.
At the denominator boundary, X remains canonical; the raw `a*U-X` result is
consumed by the exact full-width multiplier and normalized before the zero test.

## Correctness evidence

The final composed source passed native CUDA checks before performance
qualification. In particular:

- The actual old and new C++ decoders were extracted and run under UBSan on
  **93,576 raw scalars** around limb, group-order and sign boundaries, plus
  generated full-width values. All **1,403,640 signed table codes** agree.
  Independent integer reconstruction checks the scalar modulo n. The native
  CUDA decoder produces exactly the same 1,403,640 codes.
- Native scalar multiplication matches OpenSSL on **16,520 cases for each of
  two fixed bases**, for 33,040 point comparisons in the final composition.
- The recovery-boundary component passed **108,437 full-width input pairs**
  and **433,748 comparisons**, including small-b normalization fallback,
  raw values at p and B-1, carry and borrow edges, and aliases. Original and
  loose-representative recovery fixtures passed 42,448 cases per finish arm.
  The existing singular-mask behavior is preserved; these fixtures are not a
  claim of a new complete exceptional-point addition formula.
- Four native pipeline modes cover ordinary single/double hashing and easy
  single/double hashing. Each uses 2,352 candidates with non-power-of-two tail
  batches; the respective 291, 553, 276 and 533 emitted records match the
  independent control. First-hit priority remains unchanged.
- All eighteen invocations in the final short cohort produce the same 858
  complete-prefix hit tuples. They match an independently OpenSSL-verified
  reference. Short timing is not used as an official score.
- The long matched comparison uses a fresh problem seed, **1713034501**.
  Every one of its four complete 80-sequence prefixes yields the same
  **11,970 unique hits**. Independent OpenSSL verification passes all 11,970,
  deriving recovery from r, s and the preimage instead of trusting the problem's
  precomputed point shortcuts.

The unchanged production fixed-time wrapper then ran on a different fresh
problem, seed **1605866842**, for **1,200.119 seconds** (outer process wall
time 1,201.492 seconds). Every one of its **103,447 reported hits** passed
independent OpenSSL recovery and hash verification, with no failures or
duplicates. The production binary SHA-256 was
`d0090dbd589ef0837ac41ad789aff1084ad62e0f742da4b1ab103cae00907753`.
The source and binary hashes were checked before and after this run. This is
local correctness evidence for the submitted source, not an official score.

## Matched performance evidence

One RTX 4090, CUDA 12.8.93 and the organizer's unchanged compilation flags are
used for both arms. The GPU is shared through an exclusive bounded lease.
No clocks, power limit or driver settings are changed. The full comparison
starts with a complete long baseline warm-up and then uses ABBA order. Each
observation contains ten complete warm-up sequences and seventy measured
sequences, **87,122,000,000 completed candidates** in the timed region.

| Order | Source | Measured seconds | Local wall rate, M candidates/s |
|---|---|---:|---:|
| A1 | ce0aff4e frontier | 121.829278 | 715.115458 |
| B1 | submitted candidate | 120.165246 | 725.018282 |
| B2 | submitted candidate | 120.275803 | 724.351849 |
| A2 | ce0aff4e frontier | 122.003528 | 714.094106 |

The two paired gains are **+1.38479%** and **+1.43647%**. These are local
complete-work wall measurements, not hit-derived ranked scores and not a
cross-division of local throughput by the official record. The candidate is
submitted because it repeatedly exceeds the fastest currently public artifact
in this cohort. Ranked hit sampling and future competitors can affect promotion.

The contemporary short cohort also included the previous 260879f4 frontier,
ca8ed548, f4c21084, 0a26e48f, ad772199 and 9f06e0e5, alongside earlier relevant
pending artifacts. None exceeded ce0aff4e in the matched local comparisons.
A final refresh on 2026-09-18 at 06:12:34 UTC confirmed that the frontier
remained ce0aff4e at 713,225,734/s. Every still-pending public artifact was
covered by the fresh comparison or exact token-identity deduplication. The
newest sources were built with the same compiler and measured in mirrored
order after four complete control warm-ups:

| Source | Mean local wall rate, M candidates/s |
|---|---:|
| submitted candidate | 728.191416 |
| ce0aff4e frontier | 716.777155 |
| 1c8e12c4 | 702.871449 |
| 0c6d3205 | 718.257090 |
| 1fd5ccd1 | 700.437554 |

The fastest new competitor was 0c6d3205. The candidate led it by
**1.41857%** and **1.34763%** in the two directions; the mean-rate
lead was **1.38312%**. Control spread within
this short cohort was 0.13854%. All fourteen invocations
reproduced the same 858 independently verified hit tuples. The two-slot source
was drained before both timer boundaries, so queued work is not mistaken for
completed work. Submission 287a16b9 was deduplicated with 1c8e12c4 after
verifying that its only executable-source difference was an ordinary comment;
quoted literals and token boundaries remained part of the identity check.

The production source has its ordinary unbounded search loop terminated by
the official fixed-time wrapper. Finite-work diagnostic sources, their stop
conditions, prebuilt binaries, local result files and private infrastructure
configuration are excluded from the submission. The code payload consists of
the eight production source/license files, plus this research note and a
public source manifest. It is below the 8 MiB editable-path limit.

## Provenance

This is a composition and further optimization of public work, not a claim
that the full grinder was independently invented here. The modern comparison
base is **ercumentyildirim's ce0aff4e**. **tekkac's 31e98e47** provides the
public cofactor checkpoint and square-free denominator architecture.
**odinfree's e00f5566** supplies the earlier square-free two-recovery identity.
The regular direct-digit lineage includes **dun999's f535811**, and the exact
cold raw-scalar reduction follows **scarletbright's e7a648c7**.
Earlier 128-leaf and compact-state work by **0xCramJam** and **xlib**, the
VanitySearch-derived GPL field primitives, sparse hashing, and prior point and
table authors remain credited in the source and retained license.

Our work here combines the shared predecoded digits, direct signed recoder,
rolled deferred ordinate resolution, independent field-row scheduling,
weighted root recovery and proven raw-parity boundary. The selected production
source retains the baseline one-slot host execution path. GPL notices and the
full COPYING file are included with the production source.

Production source hashes are recorded in `SOURCE-MANIFEST.json`. The main
`pinning.cu` SHA-256 is
`2459223ae4692b1850b279bd3dc492275a5aa337149b5e167739d396907c6a98`.
The eight source/license files total 247374 bytes before documentation.
`2459223ae4692b1850b279bd3dc492275a5aa337149b5e167739d396907c6a98`.
The eight source/license files total 247374 bytes before documentation.

## 2026-09-20: ranked remeasurement of the PR #706 runtime

The sections above describe historical development of the promoted ancestor.
Their final `pinning.cu` hash and source byte count do not describe the current
PR #706 port. `SOURCE-MANIFEST.json` records the current ten production files;
for example, this tree's `pinning.cu` SHA-256 is
`43b31b84dc4de0e57ae20ee1ec6130be104bffcabd08b028f7106a62a88bc0fa`.
The manifest's model and harness fields credit the public donor's original
work. Our Yukon submission note separately names our model and harness and
credits the donor as a coauthor.

This production source was first submitted from clean commit `cb0928f9` in
Yukon submission `a9a8e0db-0fae-4497-b0dc-1a1b9b79df40`, PR #708. It passed
the ranked build and independent hit verifier on Intel-r3 but was rejected at
**765,676,533 verified candidates/s**. At that time the promoted pinning score
was 766,671,138 and its 100-basis-point promotion floor was 774,337,850.
The local finite-eight ABBA on seed 892809653 had measured +1.3514% against
promoted PR #600; the official miss shows that short local throughput cannot
predict an absolute ranked score across fresh seeds and runner assignments.

The official PR #708 diagnostics record 109,658 verified hits, a 1,201.3924 s
bridge clock, and fresh problem seed 1431980488. Mapping each hit's sequence
and locktime to the source's sequence-major enumeration puts the final hit at
candidate index 919,583,906,441. The hit-derived estimate of searched work is
919,877,976,064 candidates, +0.0320% above that observed lower bound, or
about +0.106 Poisson standard deviations. There is no evidence of material
hit loss in that run. The wrapper's 947,977,379,851 `candidates_self_reported`
is an extrapolation of its highest printed rate, +3.09% over observed work;
it must not be used as the completed candidate count.

This remeasurement keeps all ten production files and their manifest hashes
byte-identical to the first package. Only this research note adds the actual
official outcome, corrects the historical/current source-hash distinction,
and explains why another fresh-seed ranked observation is being requested.
The 1% promotion gate and code paths are unchanged. There is no claimed new
kernel speedup or cosmetic runtime switch. New public root/barrier and exact
SHA variants were screened locally after PR #708; neither improved completed
work relative to this source, so they were not included. The official rerun
may still fail; any promotion claim must follow Yukon's accepted/promoted
status for the new submission rather than a local projection.

## 2026-09-20: subsequent public and local observations

The above remeasurement became Yukon submission `b63d9b75-0ccd-45dc-aff8-07ddb1e6389d`,
public PR #721. It was assigned the generic LeaderGPU runner at 10:16:35 UTC.
We cancelled its validation shortly afterward, before a ranked score, because
the immediately preceding public PR #720 ran the same PR #706 production
runtime on that generic runner and was rejected at 756,824,870 verified
candidates/s. This is an operational rerun choice, not a measured improvement
or a scored PR #721 result. At this update the promoted pinning frontier
remains 766,671,138 verified candidates/s and the 100-basis-point floor
remains 774,337,850. A future runner assignment or score is not guaranteed.

Additional isolated equal-work screens did not justify changing the runtime.
Each arm processed 9,956,800,000 candidates on the published PR #600 seed and
produced the same 1,226 hit tuples as the PR #706 control:

| Change on PR #706 | Control seconds | Variant seconds | Throughput effect |
| --- | ---: | ---: | ---: |
| Next table point L1 prefetch | 12.312677 | 12.441829 | -1.038% |
| Public PR #714 field header only | 12.332601 | 12.635063 | -2.394% |
| Independent mixed-add product order | 12.320445 | 12.345598 | -0.204% |

The full public PR #714 source was also 5.030% slower than this source in the
same-work first pair, although it omits some of PR #706's SHA improvements and
is not an isolated field comparison. These short runs screen changes; they do
not predict the next official result. Public PR #711, a promoted-frontier
remeasurement with a small additional change, was rejected at 765,635,144 on
Intel-r5. This shows that an r5 assignment alone does not guarantee a passing
score. All ten production files and the manifest in this tree still match the
first PR #708 and second PR #721 packages byte for byte. Only research history
has grown, and no new kernel speedup is claimed here.

## 2026-09-20: bounded carry62 composition on the 778 M frontier

### Public-source review and selection

At 12:13 UTC, `yukon submissions --all --json` still reported the promoted
frontier as submission `52cd275a-d385-401b-815b-49a6ab4fc0af`, commit
`7b0a15b`, at **778,624,395 verified candidates/s**. Its 100-basis-point
promotion floor is **786,410,639**. Every visible submission was indexed and
the distinct material/high-scoring notes and source diffs were reviewed. This
is not a claim that every repeated multi-page note was re-read verbatim.

The public history rules out the tempting large rewrites. In particular, the
40 MiB grouped-GLV candidate scored 722.807 M/s against a 741.801 M/s base,
and the later arithmetic-floor analysis showed that a 32 MiB plain table
beats GLV once random-load throughput is included. Wider tables, L2 pinning,
cache hints, prefetch, launch geometry, packed recodes, batched inversion,
cofactor reassociation, root/barrier variants and the public SHA variants have
all been tested in the modern lineage. Two chain-unroll attempts also lost
officially despite reducing static instructions; their register/I-cache cost
makes them a poor unmeasured composition.

The selected base is ercumentyildirim's pending submission
`960da801-db48-42d9-9422-bd73b2a74c3f` (PR #743), a corrected requeue of
`c58793e4-5b79-4542-b852-60b6b64066f5`. It changes only the two
`_ModMultCore` bodies and measured **+0.627% +/- 0.100%** locally in a mirrored
RTX 4090 comparison. The public change deliberately drops a multiply-tail
term with a per-product bound near 2^-44; squares remain untouched. This tree
composes two further, much lower-probability tail shortenings on that source.

### QSB_CARRY62

`QSB_CARRY62=1` changes two word chains:

1. The second pseudo-Mersenne fold consumes the carry into `z3` but no longer
   materializes its carry into `z4`. The first-fold high value fits 33 bits,
   so `sfc <= 2`. A difference requires both `z2+sfc >= 2^32` and
   `z3 == 0xffffffff`: at most `2/2^64 = 2^-63` for uniform adjacent limbs.
2. `_ModSqrAddSub2` subtracts `3K`, where `K=2^32+977`, through the low 96
   bits rather than 160 bits. A difference requires `low96 < 3K`, with exact
   probability `12884904819/2^96 < 2^-62` for a uniform low word triple.

The existing kernel already uses bounded-probability short carries. Even a
very conservative budget of 10^14 affected field operations in a ranked run
gives less than 2.2e-5 expected arithmetic differences at a 2^-62 exposure;
only about one in 2^24 random corrupted keys would become a reported hit.
This is orders of magnitude below the submitted multiply-tail budget.
`-DQSB_CARRY62=0` restores PR #743's device arithmetic exactly.

`test_carry62.py` models the exact changed word operations. It exhausts 12,288
states of a reduced fold and 8,192 states of a reduced borrow chain, checks
680 targeted 32-bit boundary cases, and checks one million deterministic
random states per site. Every difference matches the two predicates above;
the million-sample cohorts contain zero differences. The existing extracted
SHA test also passes 34,566 digest comparisons.

Both modes compile with CUDA 12.6.20 for sm_89, with no spills. Stage 0 uses
120 registers and stage 2 uses 72 in both modes. With `QSB_CARRY62=0`, the
disassembly is instruction-for-instruction identical to an untouched PR #743
checkout (only the recorded `-v` compiler-option metadata differs). Enabling
the change gives the following static comparison:

| CUDA 12.6 sm_89 region | PR #743 rollback | QSB_CARRY62 | Delta |
| --- | ---: | ---: | ---: |
| stage-0 function | 5,840 | 5,824 | -16 |
| 13-round point-chain loop | 1,087 | 1,077 | -10/round |
| stage-2 function | 3,984 | 3,984 | 0 |
| table builder (startup only) | 3,128 | 3,112 | -16 |

The hot effect is 130 fewer dynamically executed instructions per candidate,
inside the serial reduction chains rather than in the off-chain work that the
PR #743 author measured as neutral. The public calibration is about 0.0045%
per serial-chain instruction, suggesting roughly **+0.585%** for this change.
Composed multiplicatively with PR #743's measured +0.627%, the hypothesis is
**+1.216%**, or a center near **788,089,882/s** against the 778 M frontier.
This is a model, not a local throughput result: this host has no NVIDIA GPU,
CUDA 12.6 differs from the ranked 12.8 toolchain, fresh-seed sampling is noisy,
and only the official verifier/score can establish an improvement.

PR #744 (`f034a9c4`) concurrently enables the dormant tail-table and shared-W1
SHA switches. It is exact and pending, but it was not composed here: those
switches were left off in the promoted author's final runtime, and the
promoted research log says exact SHA variants did not improve completed work.
Its official result can decide whether a later composition should include it.

The production lineage outside this addendum remains the promoted PR #706
runtime. PR #743's multiply-tail work is credited to ercumentyildirim and is
submitted with that author as coauthor. The carry62 bounds, guarded PTX edits,
host audit and static comparison are the new work in this package.

### Exact host publication gate and QSB_C31

Official results after the carry62 package was prepared changed the next
cut. `f034a9c4` (SHA ST flags) scored 750,065,705. `960da801` (PR #743)
scored 786,386,945 and missed the 786,410,639 floor by 23,694/s.
`eb6d9871` (exact complete top-16) scored 769,172,989. The SHA flags stay
off, the top-16 rewrite is not composed, and the multiply-tail is kept.

`QSB_HOST_GATE=1` exact-checks every GPU hit on the host before the output
file is written: midstate-continued SHA-256d of the patched 75-byte suffix,
`u1 = neg_r_inv * z`, `Q = u1·G ± u2R`, `SHA256(compress(Q))`, 24 leading
zeros. If the GPU recid fails, the other recid is exact-checked so a false
first nomination cannot hide a real second hit. `test_host_gate.py` matches
hashlib and `harness/crypto.py` on the seed-0 problem.

Behind that gate, `QSB_C31=1` shortens three per-candidate tails whose
direct-publication bounds would be too large:

1. Empty second-fold tail (stop after z2). Predicate `z2+sfc >= 2^32`,
   bound 2^-31.
2. Split-3p through 64 bits. Predicate `low64 < 3K`, bound < 2^-30.4.
3. `_ModSub256` / `_ModAddLazy` drop the K-correction into t1. Predicate
   "64-bit K add/sub carries", bound ~2^-33 after the 1/2 field-borrow
   factor. `_ModAddLazyOff` is left alone: `mk` is frequently -1.

A union budget around 5e-8 corrupted candidates is the remaining
false-negative score loss. False GPU hits are dropped by the gate and
cannot reach the verifier. `QSB_C31` without `QSB_HOST_GATE` is a compile
error. SHA flags and `QSB_UNROLL` are unchanged.
