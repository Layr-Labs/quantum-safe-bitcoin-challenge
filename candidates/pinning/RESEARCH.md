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

# Exact product schedules with carry-complete tree and recovery

Effort: medium. Development context: GPT 6 Astra using Codex. This candidate was prepared by source inspection and Python integer/PTX semantic models on a host without a CUDA GPU. No native C++ or CUDA compilation, device execution, register measurement, or local throughput benchmark was performed for this candidate. The official remote evaluator is the first native validation and timing of this composition.

## Baseline and measured motivation

The baseline is promoted Pinning submission `547a64cf-71fb-45f2-babb-fe6c85cad2bf`, score 741,852,708 verified candidates/s, landed commit `57b4c69c0beed7946c6645ae4149c3a19da7d57d`. All unchanged files are taken directly from that promotion. The prior submission from this account, PR592 (`08ab9ad9-ab75-4767-9c96-a2d1662bc559`), completed with verified=true and score 713,112,659. It combined a delta recovery rewrite and guarded normalization removal. That measured composite did not beat the frontier, so neither of those changes is carried here. Window/GLV, batch-size, digest32 interleave, and launch-bound experiments from older rejected candidates are also absent.

The selected public donor is ercumentyildirim's PR600, submission `208bbcb6-e235-47a1-b8c5-1d03874ac237`, commit `a668c4e5fd80db398c13222453f9c9a645612649`:
https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/600

Its public note separates three exact arithmetic/scheduling mechanisms from a fourth mechanism that extends approximate SHORT_CARRY reduction into tree and recovery multiplications. This candidate adopts the first three, and independently applies the exact product-schedule mechanism to the existing carry-complete `qsb_field_mul`. The fourth mechanism is not included: no `qsb_field_mul_sc` is introduced, no tree caller is rerouted to `_ModMultCore`, and the final correction of the exact tree multiplier remains byte-for-byte intact.

The donor reports local 4090 improvements of 3.40% and 3.92% for its full composition, and a static reduction for square cleanup alone. These are donor-reported observations, not measurements of this candidate, and the full-composition gains include the approximate mechanism omitted here. PR604 also reports an isolated schedule-port improvement on a different composite; that is supporting evidence for trying the arithmetic schedule, not a predicted gain for this submission. The only claimed local result here is the source/math audit below.

## Exact paired carries in 8x32 multiplication

Let each input limb be an unsigned 32-bit integer, with maximum M = 2^32-1. The multiplier forms a 512-bit product as even and odd chains of 64-bit partial sums. The odd chain is shifted by one 32-bit word when the two chains merge.

At each of the three affected row boundaries, the old schedule captures an odd carry and an even carry with `addc.u32 carry,0,0`. These values are single bits. It converts each to a separate 64-bit addend. The odd addend feeds a newly created odd limb at bit position 288, 352, or 416; the even addend feeds the adjacent even limb 32 bits higher.

The replacement constructs `odd_lc = odd_cy + (cy << 32)` using `mov.b64 odd_lc,{odd_cy,cy}`. It adds both carries into that newly created odd limb and creates the following even limb with a zero addend. Thus the numerical weight of each carry is unchanged. The fresh odd limb cannot overflow:

```
M*M + odd_cy + (cy << 32) + incoming_carry
<= (2^32-1)^2 + 2 + 2^32
< 2^64.
```

The margin below 2^64 is 4,294,967,293. Subsequent additions still propagate their ordinary carry chains. This is a relocation of a bit at its exact weight, not dropping a rare carry. It works for all 256-bit representatives, including noncanonical inputs at or above p. It removes three conversion instructions in each expanded multiplier, without changing the 64 partial products or allocating additional data planes.

Both SHORT_CARRY configurations of the header receive the same exact product transformation. Their reduction suffixes remain unchanged. The driver-level `qsb_field_mul` receives precisely the same three transformations, but keeps its complete reduction suffix, including propagation through z7 and its final overflow-times-K fold. The latter extension is the difference between this candidate and simply taking the donor header alone, and it exercises the same proved schedule on product-tree and recovery operands without weakening their arithmetic contract.

## Exact square carry cleanup and independent operation order

The dedicated square uses 28 off-diagonal products, doubles their merged sum, and adds the eight diagonal products. At five sites the old schedule creates a fresh high word from one 32x32 product plus incoming carry. The first two such words are `o9=a4*a5+c` and `e10=a4*a6+c`; their outgoing carries are zero because M*M+1 is below 2^64. Therefore the following `o11` and `e12` initial carry values are zero. Those words in turn become a fresh product plus carry; the same bound proves their outgoing carries zero. The final `o13` case follows identically. Even the conservative bound M*M+2 is below 2^64 by 8,589,934,589.

The implementation preserves the incoming carries and omits only these proved-zero outgoing carries and their consumers. It applies this identical prefix rewrite to `_ModSqr` under both configurations and to `_ModSqrAddSub2`. No fused-square reduction instruction is changed. The three ordinary square prefixes are tested before reduction against Python's full, unbounded a*a, so this check does not hide a product error behind a modular comparison.

The mixed-addition change moves `ZZ1 *= PP` after `ZZZ1 *= PPP`. The intervening calculation writes T, uses R, PPP and Q, and neither reads nor writes ZZ1. It does not modify PP. The source audit removes that one call from both bodies and checks that the remaining bodies are identical. The active caller supplies distinct coordinate arrays for X,Y,U,V. This change is intended to expose a more favorable compiler schedule; it does not reduce the point-formula operation count. Its compiler and occupancy effects are unmeasured here.

## Independent audit and negative controls

The included `audit_carries.py` extracts the actual assembly strings from the baseline and candidate. `ptx_field_model.py` evaluates the straight-line integer instructions with explicit 32/64-bit wrapping, carry flags, word packing and funnel shifts. Unsupported instructions and uninitialized registers fail the model instead of being ignored. The unused `take` predicate declaration in the tree multiplier is removed only after asserting it occurs exactly once; no executable statement is removed. The model's own small semantic checks cover flag preservation, flag overwrite, packing and funnel shifts.

The corpus has 4,292 pairs: a Cartesian product of 14 boundary values, 2,048 structured limb pairs, and 2,048 seeded uniformly distributed 256-bit pairs. Boundaries include zero, K, p-K, p-65537, p-1, p, p+1 and 2^256-1. Structured limbs include all-ones, sign-bit boundaries, zero, one and the prime's low word. The deterministic seed is 2026092002.

Results:

| Check | Result |
|---|---:|
| Exact 512-bit multiplication/square prefix executions, baseline and candidate | 42,920, all match Python integer product |
| Full header primitive comparisons, both configurations | 17,168, bit-identical |
| Full carry-complete tree multiplier pairs | 4,292, bit-identical and congruent to Python modulo p |
| Mutant dropping the first even carry | 369 detected mismatches |
| Mutant dropping the first odd carry | 351 detected mismatches |
| Mutant dropping a real square-chain incoming carry | 418 detected mismatches |

The audit also checks that each assembly delta is exactly the enumerated algebraic rewrite, that the tree reduction suffix is unchanged, that the mixed-add call movement is the only non-assembly header delta, and that the rest of the driver is identical to the promoted driver. Unchanged baseline files are compared byte-for-byte. The fused-square check covers its changed product prefix plus source identity of the unchanged reduction suffix; it is not represented as a full fused-function native execution test.

To reproduce the non-native audit in a normal source checkout, first extract the baseline's Pinning directory outside the working tree, then run the included script:

```
mkdir -p /tmp/qsb-carry-baseline
 git archive 57b4c69c0beed7946c6645ae4149c3a19da7d57d candidates/pinning | tar -x -C /tmp/qsb-carry-baseline
python3 candidates/pinning/audit_carries.py --baseline /tmp/qsb-carry-baseline/candidates/pinning --candidate candidates/pinning
```

This uses only Python and Git/tar. The stored `carry-audit-result.json` reports the tested runtime file hashes. The existing SHA test is not run locally because its wrapper invokes a native compiler; SHA production files are unchanged.

## Cost, limitations, and public screening

The source-derived PTX instruction counts change from 228 to 225 and 231 to 228 for the two header multiplication variants, from 194 to 186 and 197 to 189 for the square variants, and from 236 to 233 for the complete tree multiplier. These are model-visible PTX instruction counts, not generated SASS counts, measured register allocations, latency, or throughput. The same algorithm, table size, candidate enumeration, recovery identities, SHA predicates, host overlap, batch size and output interface are retained. Performance may still regress because the compiler can change scheduling or live ranges. The official run is needed to settle that question.

The promoted header already contains approximate SHORT_CARRY reductions and inherited rare field edges. This submission does not claim universal mathematical correctness of that inherited header. The claim is that the new product and square transformations are exact and preserve its output bits, and that the separately complete tree/recovery multiplier remains complete. The finite corpus supports the implementation checks; the carry bounds supply the argument for the new omissions. No new probabilistic carry omission is introduced.

All current public in-flight Pinning notes were screened before preparation. Cosmetic source-copy submissions PR602/603 and the inert remeasurement PR609 supply no new optimization to adopt. PR605 extends approximate reduction and is excluded. PR599 proposed finish launch bounds; the final preflight now reports its official rejection at 711,724,651, so that mechanism is excluded. PR604 includes the same selected schedule alongside signed-table guards, conditional carries and recovery reuse. Only its public schedule evidence is used; no implementation is imported from that composite, and this account's negative recovery result argues against reintroducing the recovery change without a new isolating reason.

The archive changes only candidates/pinning. Public research documentation, the reproducible Python audit and manifest accompany the two runtime changes. Existing licenses and notices are retained. Coauthor credit goes to @ercumentyildirim for the substantial unpromoted schedule work in PR600; the promoted chain remains the baseline attribution. No measured score is claimed for this composition before remote evaluation.
