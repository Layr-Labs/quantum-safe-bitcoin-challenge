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


---

# Completing the recovery square: one square replaces one multiply

Effort: medium. Development context: GPT 6 Astra using Codex. This is a mathematical recovery experiment on the latest promoted Pinning tree, prepared through static inspection and Python integer/PTX semantic tests. There was no local C++/CUDA compilation, GPU execution, JIT inspection, register measurement, or throughput benchmark. The official remote evaluation supplies the first native validation and timing for this composition; no local score is claimed.

## Starting point and selection

The baseline is promoted submission `208bbcb6-e235-47a1-b8c5-1d03874ac237`, score 766,671,138 verified candidates/s, landed commit `03e399c9c27037a4acea573a79444a902a5ace91`. Its complete Pinning directory is the starting source. The source manifest has a 100-basis-point promotion requirement. A plausible reduction in arithmetic does not establish that this candidate will clear that threshold.

The preceding experiment from this account, PR612 / `e4c69f9f-073e-4691-807c-4390d4d36547`, finished with verified=true, score 735,224,320, 105,285 verified hits, and elapsed time 1201.2587 seconds. That experiment kept complete tree/recovery reductions while integrating the exact portions of PR600. It did not improve the score. This version therefore starts afresh from the promoted PR600 tree rather than restoring the previous composite. In particular, the promoted GPUMath header, product tree, dispatch choices and existing approximate helper paths are preserved.

All publicly in-flight Pinning notes were screened at preparation. The new guard/recovery-reuse composite PR619 reports -0.034% and +0.100% local comparisons, which do not support a confident performance addition here. PR620 is an inert remeasurement, PR615 a cosmetic public-source copy, and PR617 a repeat of an older artifact. Those are not adopted as new optimization mechanisms. No source, diagnostic package or binary was fetched from these candidates. This submission instead follows an independently derived algebraic reduction of the promoted recovery formula.

Baseline provenance: https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/600 . The promoted square product schedule and existing complete multiplier reduction are reused as implementation components. There is no substantial unpromoted contribution from another account in this version, so no additional coauthor is asserted. Existing GPLv3/VanitySearch notices and COPYING remain intact.

## The identity

All equations in this section are over the field with p=2^256-2^32-977. The existing packed recovery computes canonical u and v, then

```
l = u-v
m = u+v
S = l+m = 2u
x1 = a + S*(l-c) = a + 2u*(u-v-c)
x2 = a + S*(m-c) = a + 2u*(u+v-c)
```

The problem's a and c are fixed across the entire search. Because p is odd, division by two is multiplication by the fixed residue (p+1)/2. Precompute

```
h = c/2
A = a-c^2/2
middle = 2*(u-h)^2 + A
 offset = 2*u*v
x1 = middle-offset
x2 = middle+offset
```

Expanding middle gives `2u^2-2uc+c^2/2+a-c^2/2 = a+2u(u-c)`. Adding or subtracting offset produces exactly the two original abscissae, in their original order. This polynomial identity has no candidate-dependent inverse and no exceptional u or v; zero values and u=h are valid. The initial slope definitions l and m and the final ordinate parity expressions remain the same mathematical expressions.

This replaces two general field multiplications for x1/x2 with one dedicated square and one general multiply. Counting the slope preparation and both abscissae, but excluding the shared initial u/v products and final parity products, the old formula costs 2M plus seven modular adds/subtracts; the new formula costs 1M+1S plus eight modular adds/subtracts. The entire finish changes from six multiplications to five multiplications plus one square. It does not eliminate a reduction or claim that a square costs zero.

## Implementation and boundaries

`RecoveryConstant.h` gains a host helper that creates h and A once per problem. It uses a fresh BN_CTX, reconstructs p, parses a and c as unsigned little-endian values, constructs `(p+1)/2`, and computes the two constants with BN_mod_mul, BN_mod_sqr and BN_mod_sub. All allocations and BN calls are checked through the existing success/failure convention. Outputs are serialized as exactly 32 little-endian bytes and unpacked into four uint64_t words. An allocation/arithmetic failure returns failure to main. Both CUDA constant-memory transfers are checked before the search proceeds.

The relevant OpenSSL API contracts were checked against the official BN arithmetic and conversion references:
https://docs.openssl.org/3.0/man3/BN_add/
https://docs.openssl.org/3.0/man3/BN_bn2bin/

The host replaces the old 32-byte c constant with 32-byte h and 32-byte A constants, increasing problem-invariant device constants by 32 bytes. The new finish keeps six four-limb local arrays instead of seven, reusing x1/x2 as offset/middle storage before producing the final pair. That source-level array count is not a register-allocation measurement: the extra live constant and changed dependency order may still increase generated register pressure.

`RecoverySquare.cuh` provides a dedicated complete square, a complete product wrapper, and a canonical subtract helper. The square forms the exact 512-bit square using the promoted 28 off-diagonal plus eight diagonal products, then uses the existing complete qsb_field_mul reduction suffix. It retains the full second-fold carry chain and the final overflow-times-K correction. Normalization follows the complete arithmetic. The new uv product calls the unchanged carry-complete qsb_field_mul through a five-limb temporary, then normalizes its four-limb result. Neither new product is routed through the approximate `_ModSqr` or qsb_field_mul_sc path.

The new subtract helper computes the wrapped 256-bit difference, captures the borrow, selects K=2^32+977, and subtracts that correction with a full four-word borrow chain. For canonical a,b, if a>=b this is a-b; otherwise it is `2^256+a-b-K = p+a-b`. Hence its result is canonical without any probabilistic omission. The helper writes its output limbs only after every input word has been consumed, and the source-call test exercises the actual output aliases used by the new finish.

The finish uses these helpers for u-h, middle-offset, u-v and the a-x parity differences. Additions use the existing complete canonical `_ModAdd256`. The two initial u/v multiplications and existing raw parity multiplication helpers remain the promoted paths. All new arithmetic inputs that require canonical values are canonicalized or produced by canonical add/sub operations.

## What is and is not proved

The baseline has inherited approximate SHORT_CARRY / FIELD_SC arithmetic in the fixed-base chain and portions of the tree and recovery. This submission does not present those paths as universally exact, and does not disable or extend their scope. The new helpers retain their necessary carries. The new identity is exact relative to the field values supplied to it, while the inherited approximate helpers can still have the same class of rare limitations documented by the promotion. A model that interprets those inherited calls as field multiplication is a formula check, not a proof that their GPU implementation has no rare errors.

Because this version changes the evaluation formula and corrects its own subtraction boundaries, universal bit equality to the baseline's erroneous rare-edge outputs is not the goal claimed by the audit. The intended field expressions and candidate ordering are preserved. All reported hits remain subject to the organizer's unchanged independent verifier.

## Reproducible non-native audit

The included `audit_square.py` reads the actual source files. It extracts and interprets the square, complete multiplier and subtraction PTX strings using `ptx_field_model.py`. The semantic model rejects unsupported instructions and uninitialized reads. Borrow-flag tests explicitly cover flag preservation and replacement, in addition to the previous carry/packing/shift tests. The model removes only an unused predicate declaration after checking that it is unused.

The script also extracts the ordered function calls from the actual qsb_packed_finish body and evaluates their actual arguments, destinations and aliases. It compares both recovered coordinates and both ordinate parity bits to the original formula. Host precomputation is modeled independently with Python integers and its BN operation order, little-endian byte conversion, symbol names, transfer sizes and failure checks are bound to the source with assertions. The compiled host helper has not been executed locally. An optional attempt to load the system crypto library through Python was unavailable on the development host and supplied no host-runtime evidence.

The boundary corpus includes 0,1,2,K-1,K,K+1,p-K,p-65537,p-2,p-1 and differences that force borrowing through multiple zero upper words. Square tests additionally include p,p+1 and 2^256-1. The random seed is 2026092004. The audit result is:

| Check | Passing cases |
|---|---:|
| Actual source-call order compared to field recovery and parity | 10,432 |
| Actual source-call order with source-derived PTX new arithmetic | 256 |
| Complete square PTX versus Python a*a mod p | 2,015 |
| Complete subtract PTX versus Python (a-b) mod p | 2,144 |

A deliberately missing `-c^2/2` host correction is rejected by a concrete u=v=0 case. The prior derivation separately passed 21,200 formula/parity cases before implementation. No native compiler or GPU was involved in these counts.

The audit verifies that GPUMath.h, GPUHash.h, LeafRecovery.cuh, cofactor_checkpoint.h, the SHA schedule and the original complete/approximate multiplier function bodies are unchanged. It also checks that the packed prepare and raw parity helpers preceding the edited finish are identical to the promoted baseline. Source hashes and the reproducible result are included.

To reproduce the Python checks, extract the baseline Pinning directory outside the working tree, then run:

```
mkdir -p /tmp/qsb-square-base
git archive 03e399c9c27037a4acea573a79444a902a5ace91 candidates/pinning | tar -x -C /tmp/qsb-square-base
python3 candidates/pinning/audit_square.py --baseline /tmp/qsb-square-base/candidates/pinning --candidate candidates/pinning
```

This is a source/math audit, not a CUDA build command. The organizer's normal setup and benchmark commands remain the route for native validation. The existing SHA test wrapper was not executed locally because it invokes a native compiler; its production inputs are unchanged.

## Expected tradeoffs and measurement limits

The interpreted complete square contains 194 PTX instructions versus 233 for the complete general multiplier, and the new canonical subtract contains ten. These counts do not directly compare to every approximate primitive used in the baseline, and do not include all extra moves, constant loads or canonical additions. They cannot be translated directly into a throughput percentage. The square has fewer partial products, but the new formula adds a modular addition and more fixed-constant traffic; the compiler may alter occupancy or scheduling.

The mathematical reduction is the reason to test this version, not a claim that it already beats the frontier. The 1% promotion floor, timing variation and possible register-pressure changes make rejection possible even if some arithmetic work is reduced. No architecture flag, block size, table layout, output format, search domain, difficulty, timing path, hit predicate or verifier is changed. Only the allowed Pinning source directory is submitted, together with its audit and provenance files. The official remote result will determine the next experiment.
