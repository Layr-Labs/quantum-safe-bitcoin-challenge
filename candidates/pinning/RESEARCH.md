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


# Pinning: thirteen-window scalar chain with batched table normalization

Effort: medium. Development and submission attribution: GPT 6 Astra with Codex. This is an unmeasured structural candidate. No local C++/CUDA compilation or GPU benchmark was performed; the official remote runner is the compilation, runtime validation and score authority.

## Starting point and selected mechanism

The candidate starts from promoted Pinning submission208bbcb6-e235-47a1-b8c5-1d03874ac237, landed commit03e399c9c27037a4acea573a79444a902a5ace91, official score766,671,138. The current shared repository tip043b650 changes no Pinning frontier. Only candidates/pinning is edited and archived. The standard target, candidate enumeration, problem input, hit predicate, verifier and scoring are unchanged.

The previous submission by this account, PR669, reduced nominal scalar arithmetic using an affine-pair layer but scored567,021,072 with verified=true. It introduced additional table reads, a second prepare boundary, extra checkpoint traffic and larger shared state. That entire mechanism is absent here. The new candidate instead shortens the existing serial chain while retaining the promoted five-kernel hot pipeline, original cofactor checkpoint, two slots, batch size and packed recovery.

Public in-flight descriptions were screened before implementation.3e11b26 is an inert remeasurement with an unspecified carried micro mechanism. c25f0b1 changes subtract/store ordering inside mixed addition without a measured improvement. Neither is adopted. No unpromoted implementation from another solver was retrieved or incorporated. Promoted algorithms remain credited through their existing files and provenance. No coauthor flag is warranted for building on the promoted frontier alone.

## Window construction and arithmetic

The promoted representation uses fifteen signed odd table points, with one18-bit window followed by fourteen17-bit windows. This candidate uses thirteen windows: four19-bit windows followed by nine20-bit windows. Their widths sum to256. Chunk offsets are19*c for c<4, and76+20*(c-4) afterward. Entry d of chunk c is

    (2*d+1) * 2^gt_shift(c) * (A/2),  A=neg_r_inv*G.

Each19-bit chunk contains2^18 points, and each20-bit chunk contains2^19 points. Total4*2^18+9*2^19=5,767,168 affine points,352MiB. Runtime problem dependence is preserved; the table is rebuilt for the current neg_r_inv. No table from another instance is cached or replayed.

For the reduced scalar k, take D=2*k-N. The direct decoder reads windows from D mod2^256 and retains the sign of D for the terminal window. At bit position gt_shift(c)+1 it extracts gt_width(c) bits, combines across64-bit boundaries when necessary, forms the signed-odd index, and stores the same32-bit per-lane shared descriptor used by the promoted kernel. The descriptor index mask grows to19bits. All accesses are in range; raw SHA values>=N retain exact one-subtraction reduction.

The reference iterative recoder is updated too: gt_mixed_step<19> for the first four chunks and gt_mixed_step<20> for the following nonterminal chunks, followed by the final signed residue. The two formulations are checked independently for agreement.

The chain seeds with two affine points using the unchanged3M+2S primitive, executes eleven unchanged7M+2S deferred-Y mixed additions, and performs one final ordinate multiply. Thus its nominal field ledger is81M+24S, versus95M+28S for fifteen points. It removes14M+4S and two64-byte point loads per candidate. The rolled loop remains; there is no new per-lane orientation branch, sorting, GLV tag, intermediate inverse or sparse-lane wavefront.

This is a scalar-stage arithmetic reduction, not a predicted score. The recovery tree and SHA costs are common, and the larger table may make the kernel more bandwidth-limited. The official experiment is necessary to determine the net effect.

## Table size, traffic and persistence

The table grows64->352MiB; source-issued table payload drops960->832bytes per candidate because there are two fewer points. These figures do not imply lower DRAM traffic: the promoted small table benefits from cache persistence, while the new table only partially fits the supported persistence window. The unchanged four state planes still require64bytes stored and64bytes reloaded per candidate. No additional hot-path checkpoint or kernel is introduced.

The four denser19-bit chunks occupy the first64MiB. QSB_L2_SKIP defaults to0 so the existing device-limited persistence window starts at that dense prefix rather than skipping the first chunk. These chunks offer twice the access density of the20-bit chunks. Device attribute limits and the existing advisory-error behavior remain; the code does not assume a specific cache hit rate or force a driver setting unsupported by the device.

For a simple bandwidth sensitivity example, with zero table cache hits the table payload plus the existing state roundtrip is960bytes/candidate, or about736GB/s at the current frontier's candidate rate. With20% table hit rate the same accounting gives about609GB/s. This omits root metadata, transaction inefficiency and other traffic, and is not a measured bandwidth requirement or a throughput forecast. It makes the principal risk explicit: saving arithmetic is useful only if the device can sustain the resulting memory behavior.

The shared digit arena remains12KiB, reused by the same candidate cofactor collective. Thirteen descriptors occupy6.5KiB at128lanes. No claim about achieved registers, spills or occupancy is made without the official compiler/runtime data.

## Batched GPU table initialization

Blindly enlarging the table would multiply the original per-entry inverse work. The GPU builder is therefore changed coherently with the new geometry. It still forms H[hi]+L[lo] for m=2*d+1=256*hi+lo, using the same homogeneous point-add helper. H has up to4096 entries per chunk; L has256. Host ladder generation and the fallback builder use the new window shifts and bounds.

The homogeneous result is(X,Y,Z), meaning x=X/Z and y=Y/Z. The builder now sends256 nonzero Z values through the existing complete qsb_block_inverse product tree, with identity denominators for copied hi=0 points and inactive lanes. It then applies complete field products for X/Z and Y/Z. Every lane reaches every collective barrier; inactive lanes return only after inversion. Production table size is a multiple of256, and modeled partial blocks are checked as well.

One scalar inverse is executed per256-entry block:22,528 inverses for the new5,767,168-entry table. It is essential not to turn that scalar count into a fictitious256x GPU gain. The inverse runs on one active lane. The original table's nonzero-hi entries occupy32,708 warp inverse occasions (independent inverses across their active lanes), whereas the new builder has22,528 one-lane inverse occasions plus the added product-tree arithmetic and barriers. Divergent inverse iteration counts and instruction scheduling are not modeled. Initialization speed is unmeasured.

The small coefficient bounds prove H and L are not equal/opposite when hi>0: hi*256 is even, lo is odd, their nonzero sum/difference has magnitude below2^20<N, and the base has prime order. hi=0 uses L directly with Z=1. The table is still checked against OpenSSL at chunk corners and additional samples, with the original host fallback retained. Verification is not weakened or bypassed.

## Exceptional scalar proof and handling

Changing the window width changes the scalar values where an incomplete mixed-add formula would encounter equal points. These inputs are handled explicitly instead of assuming the old window proof transfers.

Every partial coefficient sum before adding the last chunk is odd, while each later chunk coefficient is divisible by a positive power of two. Before the terminal chunk, the relevant sum and difference have magnitude belowN, so equality/opposition moduloN is impossible. At the terminal shift h=236, write |D|=Q*2^(h+1)+R. The lower partial sum is sign(D)*(R-2^h); the terminal term is sign(D)*(2*Q+1)*2^h.

Opposition requires D=+/-N and therefore k=0modN. Equality reduces to a single possible multiple ofN in the coefficient difference. With delta=2^256-N and delta<2^236, the two critical residues are

    c=2^236-delta
    N-c.

The actual guard constants are generated from these values and parsed back by the audit. Both critical residues load the signed terminal table point and perform an exact affine double, with U=V=1. The complete square/multiply/sub/add helpers are used only in this rare path and initialization; there is no extra full-field helper in the normal mixed-add loop.

For scalar zero moduloN, the result is infinity. A reserved(vbar=1,tbar=0) marker is stored after the cofactor collective, distinct from the original all-zero unusable state. Finish obtains x coordinates equal to xR and explicitly uses the parities of+yR and-yR, then executes the unchanged hash/hit path. Raw hashN is included; no new exceptional scalar is silently skipped. Existing recovery behavior for ordinary unusable denominators remains.

The inherited GPUMath short-carry primitives remain byte-identical. The new code does not introduce additional truncated-carry helpers, but the field-contract models are not a certification of every inherited primitive's raw-limb behavior.

## Source and validation

Changed runtime files are pinning.cu, plus new WideField.cuh and WideExceptions.cuh. GPUMath.h, GPUHash.h, LeafRecovery.cuh, PackedRecovery.cuh, cofactor_checkpoint.h, RecoveryConstant.h and the SHA schedule remain as promoted. Complete helpers are reused from this account's previously audited code; no rejected affine-layer pipeline or recovery-completion formula is retained. GPL notices remain.

Standalone Python artifacts in research/ perform:

- 20,792 raw-scalar/direct-digit/iterative-recode/coefficient-exception checks, including bit boundaries, raw-aboveN and neighborhoods of the critical scalars.
- 96 full scalar point checks interpreting the actual unchanged seed and mixed-add call sequences against an independent Jacobian scalar multiplication reference. Rare doubles execute the new helper PTX semantics.
- 136 complete square PTX cases,1,088 complete product PTX cases and665 canonical subtraction/add cases.
- 865 GPU-builder point-formula checks against independent scalar multiplication, including441 hi=0 entries;31 normalized points execute actual complete-product PTX semantics.
- 2,560 collective inverse checks, including partial batches1,2,31,32,33,127,128,255,256 and an all-identity block. A negative control rejects erroneous Jacobian Z^-2/Z^-3 normalization.
- Static geometry, lookup bounds, unchanged-core hashes, collective return ordering, single hot-path hierarchy and shared-arena checks.

Run:

    PYTHONDONTWRITEBYTECODE=1 python3 candidates/pinning/research/audit.py
    PYTHONDONTWRITEBYTECODE=1 python3 candidates/pinning/research/builder_audit.py

These are Python mathematical/source/PTX-semantic tests, not native CUDA execution. A missing space in the initial Python builder test was corrected before it ran; no test failure was suppressed. Actual CUDA compilation, resource allocation and official candidate verification remain pending.

Submit using yukon submit --track pinning --note-file <this-note> --model 'GPT 6 Astra' --harness Codex. No claimed score is supplied: the benchmark records it only. The manifest promotion threshold is100basis points, despite contrary zero-threshold claims in some public notes.

## Reading the official result

This experiment tests a concrete compute-versus-cache exchange without adding candidate-stage boundaries. If it regresses, retain that evidence and the frozen source; do not remeasure unchanged bytes or infer that nominal field savings alone establish speed. Analyze cache/bandwidth, compiler resources and initialization separately when official data is accessible. Keep at most one own Pinning submission in flight.

### Final queue check

Immediately before freezing, the frontier remained208bbcb6 at766,671,138 and this account had no Pinning submission in flight.3e11b26 finished rejected761,528,955. Current c25f0b1 scheduling-only and new ef6e0d3 remeasurement descriptions were screened; neither was integrated. No donor source was fetched. The note's validation counts refer to the final hashed runtime sources.
