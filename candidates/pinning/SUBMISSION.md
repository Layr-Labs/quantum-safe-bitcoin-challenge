# Serial16 32 MiB table with certified parity and direct point destinations

Effort: medium. Developed with GPT 6 Astra in Codex. This is a cumulative performance hypothesis for official remote evaluation; no local CUDA or native host compilation was performed and no local score is claimed.

## Promoted base and selection

The base is Saviour1001's promoted submission dcd0147c-8cb3-47f0-8b71-007c87fa7748, source 66fede0cc10d36ad15041861d3eddaad97481ac6. The current shared source e876032f79e6f4f3af2732bbba39403e29f0e227 has the same Pinning runtime. The official frontier is 789,011,576 verified candidates per second, confirmed again before packaging. Its Pinning files were copied into a fresh independent repository and committed as the baseline before applying this candidate. Only candidates/pinning is submitted.

The main hypothesis is that halving the fixed-base table improves effective cache residency enough to offset one extra point addition. The promoted source uses 15 windows and a 64 MiB table; this candidate uses sixteen 16-bit signed windows and a 32 MiB table. It retains the original serial point chain and cofactor traversal. Two independently audited changes reduce endpoint parity work and source-level coordinate copying. These are combined because each targets a different resource, but their gains must not be added arithmetically or assumed positive.

My earlier PR860 scored 779,574,533, 1.19606% below this frontier. Its paired digit transport, three-register seed and rotated chain composite are not used. My subsequent PR890/c7cc4855 implemented only certified parity and direct destinations, and failed with score_missing_or_invalid after the platform reported ENOSPC while extracting an artifact. No valid score or official verified verdict was supplied. That is infrastructure evidence, not a measured regression or success. The present archive is a substantive geometry/exception/lifecycle change, not an unchanged retry. A later public submission PR893 received a numerical score; this only shows that at least one later scoring request completed, not that every platform problem is resolved.

## Window geometry and mathematical exceptions

Let n be the secp256k1 group order and A=neg_r_inv*G for the current runtime problem. Normalize a raw 256-bit hash k modulo n and define D=2*(k mod n)-n. Recoding uses sixteen odd signed digits with width 16. Each of the first fifteen digits is (D mod 2^17)-2^16, followed by the exact signed quotient; the final remaining digit closes the sum. Every digit is odd and nonzero. Chunk c stores (2*i+1)*2^(16*c)*(A/2), i from zero to 32767. Thus 16*32768 affine records of 64 bytes occupy exactly 32 MiB.

The shared direct decoder extracts the same digits from the signed setup representation. Its bit position is 16*c+1; cross-limb extraction is retained, and the final sign comes from the complete signed setup, not a truncated high word. Both CPU builders, GPU ladder indexing, table offsets, table count, serial stride and spot checks follow the same geometry. The low ladder has 256 entries and the high ladder shrinks to 256 entries per chunk. All 524,288 addresses are modeled. The existing table load format and original serial recurrence are retained.

Changing the window count introduces real exceptional point additions. It is not valid to assume the new point chain is complete just because every digit is nonzero. Before the last addition, the signed prefix is odd with magnitude below 2^240, and earlier prefix/summand combinations cannot equal zero or n. The final equality cases occur at k=K0 or n-K0, K0=65535*2^240. The zero scalar gives the identity. These are handled explicitly before the chain: the device classifier compares all four normalized setup limbs and its sign. The host computes [K0]A with checked OpenSSL BN/EC calls for each runtime problem, serializes canonical x,+y,-y into 96 bytes, and uploads them to device constants. The two final-doubling scalars return the corresponding affine point with ZZ=ZZZ=1. No instance-specific coordinates are embedded.

Scalar zero, including raw hash n, returns an identity marker and is tracked by an explicit bitmap. It is not identified by a guessed raw-field sentinel. Finish handles that positively identified scalar as R and -R with the correct recid parity mapping. Other inherited singular recovery cases P=+R or P=-R retain the promoted behavior. This is not a claim of a universally complete ECDSA recovery implementation; it adds explicit handling for the new fixed-base exceptions and does not introduce another candidate skip.

## Bitmap and stream lifetime

The original unused pipeline tree argument transports the identity bitmap. Each slot allocates ceil(BATCH/32) words: 1 MiB for the default 8,388,608 candidates, hence 2 MiB for two slots. One warp ballot and one predicated store publish a word. All lanes reach the ballot before the special-point return; whole inactive blocks return uniformly and tail bits are zero. Each active word is overwritten every generation, so no clearing pass is needed. Prepare and finish use the same slot pointer and CUDA stream. Existing draining prevents a slot from being reused while its prior work remains active. Root kernels do not touch this bitmap. Alternative TREE_OFFLOAD and TREE_OFFLOAD2 configurations are explicitly rejected at compile time because they would use that pointer as a larger legacy checkpoint buffer.

The flag is inspected only in the existing zero saved-denominator branch. An unmarked singular lane retains the original return; a marked identity uses explicit recovery. The ordinary finish path still uses the packed cofactor implementation. Allocation and constant upload errors are checked. L2 skip is zero because all sixteen table planes now have equal density; the original device capacity cap remains.

## Direct destinations and raw arithmetic contract

The changes to _PointAddXYZZT and _PointAddXYZZ_mm write dead intermediate coordinates directly to final destinations. A symbolic audit compares the ordered inputs of every raw primitive for 32 feature combinations, treating each primitive as an arbitrary deterministic function rather than assuming field algebra. Both complete helper bodies can be reversed to the promoted header byte for byte. With fourteen mixed additions, the new chain removes 29 source-level 256-bit copies relative to that chain without this rewrite; this is not a physical register or SASS count.

The inherited truncated multiplier is not representative-invariant and cannot be reassociated based only on modular congruence. Its primitives, approximate carry policy, and cofactor tree are unchanged. The new table geometry intentionally changes the point-operation sequence; exact group reconstruction is checked, but the GPU nomination behavior of this changed sequence remains a matter for official validation. No additional dropped carry, relaxed host verification, or speculative zero-factor omission is introduced.

## Certified endpoint parity calculation


For base beta=2^32, let B=2^256, S=2^224, K=beta+977, p=B-K and T=a*b. Inputs a,b range over [0,B); the fixed ordinate offset is canonical and nonzero, as required by the existing `qsb_sum_parity` caller.

The active promoted multiplier's contract can be expressed as follows. Compute F=low256(T)+K*high256(T), R=low256(F), and h=low32(high256(F)). Its raw output keeps R above bit95 and replaces the low96 bits by low96(R+K*h). The omitted high fold and low96-only final propagation are inherited behavior. Computing canonical multiplication modulo p instead would not in general reproduce this raw result, so that substitution is not made.

Two disjoint product windows suffice to certify the observed parity:

### Low window

Retain eight complete products on diagonal i+j=7 and seven high halves on diagonal i+j=6. Only parity from diagonal8 is needed, so its seven terms are AND/XOR bit products. Higher diagonals cannot affect the low33 bits of floor(T/S).

The omitted nonnegative tail in the scaled product is strictly less than13. Thus a lower head H satisfies H <= T/S < H+13. If low32(H)+12 crosses beta, the certificate declines. Otherwise the relevant top32 bits and parity of high256(T) are fixed. Accumulating the low window modulo2^64 is sufficient because only its low33 bits are observed; overflow beyond them is mathematically irrelevant.

### High window

Retain six complete products on diagonals12..14 and four high halves on diagonal11. Their sum J satisfies J <= T/2^384 < J+9. Consequently:

`J*2^128 <= high256(T) < (J+9)*2^128`.

Compute K*J with four constant-977 products plus shifts and additions. If adding9*K to its low96 bits carries through bit96, decline. Otherwise floor(K*high256(T)/S) is certified. This quotient can require65 bits: the implementation intentionally retains its low64, because only its low32 and the next parity bit are subsequently observed. This is a proven bit projection, not an assumed absent carry.

The low window uses eight wide plus seven high products; the high window uses six wide plus four high products: **14 wide + 11 high = 25 operand products**, plus seven bitwise terms and four constant products. These categories have different instruction costs and cannot simply be equated to 25 GPU instructions.

### Fold, reduction quotient and zero certificate

Add the two certified windows to form f. Decline if low32(f)+13 crosses beta. Otherwise, for r=low32(f), the raw product lies in `[r*S, (r+14)*S-1]`. The inherited truncated low96 second fold cannot change this upper window.

The top32 bits of the canonical offset bound the top quotient of raw+offset. Let lower=r+top32(offset), upper=lower+14. Require the same beta quotient for both, a nonzero lowword of lower, and a lowword of upper other than all ones. These deliberately conservative conditions keep the sum away from both zero and the p=B-K transition. Its reduction quotient and nonzero status are then fixed. The output bit is the XOR of the operand low-bit product, certified high-product bit, fold bit, offset bit, reduction quotient and requested negation.

If **any** certificate fails, the wrapper executes the original `qsb_packed_raw_mul` followed by the original `qsb_sum_parity`. It does not guess the parity or drop a candidate. Other arithmetic configurations also take the original path: the helper is selected only on the device with C31, SHORT_CARRY and FIELD_SC enabled.


## Final source audit and reproduction

The archive includes portable Python checks under research. Run from the repository root:

```
python3 -B candidates/pinning/research/audit_source.py
python3 -B candidates/pinning/research/serial16_model.py
python3 -B candidates/pinning/research/audit_special.py
python3 -B candidates/pinning/research/audit_serial16.py
python3 -B candidates/pinning/research/audit_lifecycle.py
python3 -B candidates/pinning/research/audit_header.py
python3 -B candidates/pinning/research/finish_model.py
python3 -B candidates/pinning/test_host_gate.py
```

The source reversal restores pinning.cu using a recorded edit ledger, restores the two point helper bodies, and removes the parity wrapper and its two call replacements. Other original runtime files remain byte-identical. The compiler, CUDA ABI, register allocation and memory race behavior are not tested by Python.

The geometry model checks 101,288 raw scalar inputs and all table addresses. The final source classifier is parsed and checked against 100,007 normalized scalar cases plus three wrong-sign controls. The source-bound exact curve audit covers 96 full cases across four bases, 1,520 normal additions, and sixteen expected final exceptional cases. The stream model covers 576 generations, slot counts 1/2/3, widths 64/128/256, varying batch tails and 2,305,264 active candidate flags. These models do not prove that every inherited approximate PTX operation computes an exact group result.

The parity audit checks 128,112 executions of the actual header translated statement by statement with u32/u64 wrap, including 7,056 fallback cases. The split model covers 93,964 cases; a separate model checks 425 actual promoted PTX traces. The finish model covers 17,732 synthetic finish cases and 750 endpoint boundary cases. The host gate audit covers 64 midstates and source/verifier bindings. No result is represented as a GPU throughput measurement.

## Performance tradeoffs and next decision

The arithmetic ledger grows from 95M+28S to 102M+30S and table reads grow from 15*64 to 16*64 bytes per ordinary candidate. Halving table size does not halve bandwidth traffic. Under an optimistic model where lookup time halves, the original lookup share must exceed roughly 12.84% just to offset the extra arithmetic. Actual cache misses, bandwidth, instruction count, occupancy and lookup share are unknown. The bitmap ballot, constant comparisons, sparse identity branch and added code may also affect resources. The parity path uses 25 operand products but also needs constant products and certification guards. Ambiguity always falls back to the original operation.

The official remote build is the first native compilation. A negative numerical result should guide rollback or isolated attribution of geometry versus endpoint changes; do not repeat an unchanged archive or insert an inert marker. A platform failure with no valid score provides no ranking evidence. This candidate's exact host publication gate remains mandatory and unchanged; it cannot restore nominations that inherited device approximations missed.

## Public sources and credit

Saviour1001 and the promoted predecessor chain provide the baseline. DrCleverHans's public cbce5501 description suggested direct coordinate destinations. EvanYan1024's [PR863](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/863) and [PR885](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/885) describe partial parity and separate ablations. terrapinelf's [PR887](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/887) reports independent parity-only comparisons, finite hit-set agreement and resource observations on another stack. Those are the authors' reported measurements, not measurements of this candidate. The parity implementation here is independently derived, with full boundary fallback. Their additional carry omissions and reassociated tree are not included.

Recent public descriptions were screened before packaging; PR893 describes frontier remeasurement with a small carried mechanism and does not provide a substantive additional route to integrate. No private task, other author's binary, diagnostics archive or unpromoted source package was fetched. Original VanitySearch/GPL notices and COPYING remain. Attribution is written in this public note, with no additional coauthor metadata, as requested by the submitting account owner.
