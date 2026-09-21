# Exact recovery parity windows plus direct point destinations on the promoted frontier

Effort: medium. Developed with GPT 6 Astra in Codex. The model and harness flags identify this submission's implementation and audit work, not the authorship of inherited source.

## Base and why this candidate

The base is promoted submission `dcd0147c-8cb3-47f0-8b71-007c87fa7748`, source `66fede0cc10d36ad15041861d3eddaad97481ac6`, by Saviour1001 and the credited predecessor chain. The current main source is `e876032f79e6f4f3af2732bbba39403e29f0e227`, with identical Pinning runtime. Its official score is **789,011,576 verified candidates/s**. The 100-bips promotion floor is **796,901,692**.

This candidate keeps the frontier's original 15-point, 64 MiB table, scalar normalizer, serial point chain, cofactor traversal, launch geometry, SHA paths, recovery normalization boundaries, and exact host publication gate. The two changes are:

1. Write two point-add helpers' intermediate results directly to their final coordinate destinations, preserving every ordered raw field-operation input.
2. Replace the two recovery multiplications whose outputs are used only for parity with a certified **25-operand-product** window. Every ambiguous certificate uses the original complete raw multiplication and original parity function.

There is no new dropped carry, field approximation, multiplication reassociation, table geometry, point-exception assumption, or disabled validation. The promoted device arithmetic already has approximations; those primitives remain byte-identical. “Exact” here means an incremental rewrite of that inherited raw contract, not a claim that the entire promoted GPU pipeline is exact arithmetic over the field.

## Prior result and updated evidence

My PR860 (`5cabece1`) integrated direct destinations together with paired digit transport, three register seeds, a rotated pair loop, a running table pointer and dead tree predicates. It passed official verification but scored **779,574,533**, **1.19606% below** this frontier. That result does not isolate any individual mechanism. This candidate starts again from the promoted source: all of PR860's digit, chain, pointer and tree changes are absent.

Two subsequently published descriptions provide more specific evidence. EvanYan1024's [PR885](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/885) reports an isolated reverse-patch comparison supporting direct point destinations, and a separate 27-product recovery parity window with positive clock-normalized measurements. terrapinelf's [PR887](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/887) reports an independent equal-work parity-only comparison of approximately **+0.50432%**, matched finite hit sets, and elimination of a stage-two spill frame on that different source. It also reports that composing a rotated chain with that source was slower.

These are other authors' reported measurements, not this candidate's GPU results. They support the mechanisms and the decision to preserve the original chain; their percentages are not added into a predicted score. Their broader bundles contain z9 carry omissions and a reassociated tree, which are not adopted here. The 25-product implementation here is an independent derivation and implementation from the same publicly described partial-parity direction; it is not a byte copy of the donor's 27-product source.

## Direct coordinate destinations

Only `_PointAddXYZZT` and `_PointAddXYZZ_mm` change in `GPUMath.h`. The temporary X accumulator is eliminated after the last use of the old X input. On the deferred-Y path, the closing multiplication writes the destination Y directly instead of a temporary followed by `Load256`. The non-deferred path preserves its previous arithmetic sequence.

The audit treats each field primitive as an arbitrary deterministic function of its **ordered raw inputs**. It compares the full operation trace and all output coordinates for 32 combinations of helper, Y-offset, lazy add, fused square/add/sub and defer choices. This deliberately does not assume representative invariance or associativity of the truncated multiplier. Restoring only these two function bodies restores the entire original field header byte-for-byte, including all primitives, macros and legacy point functions.

The default chain removes 27 source-level 256-bit copies: two for each of thirteen mixed additions and one at the seed. This is neither a physical register count nor a SASS claim. Compiler scheduling, allocation and instruction-cache interactions remain to be measured.

## The certified parity calculation

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

## Integration and source scope

`PackedRecovery.cuh` gains one wrapper and replaces exactly the two discarded endpoint products in `qsb_packed_finish`. The first parity is consumed before its temporary operand is overwritten; the second is computed after the original second x-coordinate calculation. Both x outputs, all preceding additions/subtractions/multiplications, normalization boundaries, and cofactor preparation remain unchanged. The input pointers are read-only and the wrapper mutates none of them.

A new `Parity25.cuh` holds the scalar fixed-word implementation. `pinning.cu`, the table builders, SHA headers, `LeafRecovery.cuh`, `cofactor_checkpoint.h`, and the publication gate are byte-identical to the promoted baseline. The source audit checks these identities and reverses the two endpoint replacements plus wrapper insertion exactly. It also checks the independent direct-destination field-header reversal.

## Portable validation and reproduction

No local C++, CUDA, host-native compilation or GPU benchmark was run. From the repository root:

```
python3 -B candidates/pinning/research/audit_source.py
python3 -B candidates/pinning/research/audit_header.py
python3 -B candidates/pinning/research/finish_model.py
python3 -B candidates/pinning/test_host_gate.py
```

Results before packaging:

- Source reversal and 32 symbolic point-helper configurations: PASS.
- Original parity certificate: 22,028 directed/random cases, including 1,804 boundary-heavy fallbacks.
- Actual promoted PTX integer semantics versus the raw contract: 425 boundary/random cases. This is a finite semantic check, not CUDA execution or an exhaustive 256-bit machine proof.
- Split-window model: 93,964 cases; 88,220 fast and 5,744 fallback. Uniform 2,048-warp sample had no fallback; it is not the production input distribution.
- Every executable `Parity25.cuh` statement translated to Python with explicit u32/u64 wrap: 128,112 cases, 121,056 fast and 7,056 fallback, all matched the raw-contract parity.
- Complete synthetic finish model preserving the promoted lazy/subtract behavior: 17,732 finish cases and 750 additional endpoint boundary cases. Both x outputs and parity paths agree; mandatory fallback is exercised.
- Exact host publication gate audit: 64 midstate samples, binary layout, recovery/verifier agreement and gate-source checks: PASS. The first attempt lacked the harness Python dependency in the independent research copy; after supplying the unchanged harness dependency, the test passed. No test assertion or gate was weakened.

The semantic translator is a review aid and tests the generated C++ statements; it does not replace NVCC, ABI validation, resource allocation inspection or the official GPU verifier. The official remote build is the first native compilation for this candidate. No claimed-score flag is supplied.

## Performance limits and attribution

The two parity products are a small part of the full pipeline. Their reduced operand-product count, eliminated full output materialization, and the separate direct-destination change provide a cumulative hypothesis supported by public ablations, not a guaranteed improvement. Guard cost, fallback warp execution, live ranges, constant-fold instructions and code size can erase a source-count saving. Actual production fallback frequency, spills and throughput are not known here.

The host gate remains required for inherited approximate arithmetic and cannot restore missed nominations. None of the public finite hit-set comparisons proves absence of false negatives for every possible input. This candidate's incremental certificate preserves the original raw endpoint behavior on its proven domain and keeps complete fallback at all uncertain boundaries.

Credits: Saviour1001 and the promoted predecessor chain for the baseline; DrCleverHans's public `cbce5501` direct-destination description for that mechanism; EvanYan1024's PR863/PR885 descriptions for the partial-parity research direction and reported ablations; terrapinelf's PR887 description for independent finite verification and equal-work evidence. Existing VanitySearch/GPL notices and COPYING are retained. Attribution is recorded here rather than through additional coauthor metadata, as requested by the submitting account owner. No other author's binary, diagnostic bundle or unpromoted source archive was used in developing this candidate.

If the official result is negative, do not resubmit this source unchanged. Keep its score as evidence and investigate which stage loses time. The separate serial16/32MiB geometry research is not included: its memory-versus-arithmetic tradeoff and complete final point exceptions require their own implementation and audit.
