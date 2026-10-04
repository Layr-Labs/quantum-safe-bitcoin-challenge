# Signed-Q5 mathematical review

**Recoding, seed/sign and slot/phi model: PASS. Native all-input field equivalence: not proved.** The production39 hashes and a reversed three-hunk patch bind the candidate to literal b59. This review derives the digit identity independently; it does not rely on the previous Q5 route having shipped.

## Digit identity from the actual table

The active host table builder at A `pinning.cu:5343` and `gt_table_scalar:5386` define segment-zero record `f0` as `(K+f0)A`, with

`C=170559769`, `K=(C+1)*2^99−2^17`.

An ordinary segment at shift `s`, width `b`, stores positive odd multiplier `(2*i+1)*2^(s−1)`. Its magnitude decoder chooses negative when field `f<2^(b−1)`, with `i=2^(b−1)−1−f`; otherwise `i=f−2^(b−1)`. Thus its signed contribution is exactly

`(2*f−2^b+1)*2^(s−1) = f*2^s−(2^(s+b−1)−2^(s−1))`.

Q6 uses intermediate boundaries `18→37→55→73→100`; Q5 uses `18→45→73→100`. In either contiguous partition, the parenthesized biases telescope to `2^99−2^17`. The common top segment contributes `(2*F−C)*2^99`. Adding the common segment-zero bias cancels all biases, giving the exact integer magnitude `M`. XORing every code sign with component sign `s` gives `(-1)^s*M`. This is integer equality, before any reduction modulo the curve order.

The stated exact GLV residual bound has top field **C−1=170559768**, so declared segment-five indices remain valid. The synthetic top field C is still valid table recoding, but `M=C*2^100` exceeds that exact residual bound. For arbitrary 128-bit decoded W, both routes also have equal sums of the actual physical table records: their differing middle fields telescope, and all common records—including any out-of-bound top record alias—are identical. That broader endpoint equality does not establish that an inherited malformed/out-of-bound residual represents the original input scalar.

## Signed windows, seeds and slot-two guard

For `M=W xor (−s)`, let `t'` be the field-top bit of W; the magnitude field top is `t=t' xor s`. Its record index is unchanged by the component XOR, and code sign `(1−t) xor s` simplifies to `1−t'`. In the literal top-aligned PTX window U, `xs=−t'`; `(U xor ~xs)>>(32−b)` equals `2^(b−1)+i`. Adding the literal offset minus `2^(b−1)` gives the correct record, and `v xor (~U & 2^31)` supplies that code sign.

Seed zero returns the magnitude low18 record and mask `−s`. Seed one returns the record and **inverted** sign mask `xs`; `qsb_load_glv_rec<true>` complements it, so the loaded point has code sign `1−t'`. Slot three uses the same end-bit72 window for Q6 `(55,18)` and Q5 `(45,28)`, with different right shifts; the same algebra applies. The Python reference independently checks all four literal LOP3 truth tables.

For Q5, seeds zero/one stay in registers, slot three is segment7, slots four/five are the common segments4/5, and first=1. The chain preloads slot3 and reads slots3..5 before P. Slot2 is never read. The false helper still writes and reads slot2 exactly as Q6 requires. After the chain barrier, TOP5's initial paired leaf stores cover bytes0..2047 before any tree read, including every omitted slot2 byte1024..1535; the ASIC schedule begins at6144 and is separate. There is no later stale slot2 consumer.

## P, phi and exceptional paths

P starts at slot6 in both arms. Normal P5 uses slots6..10/last11; PMIX P6 uses slots6..11/last12. The selector remains `(blockIdx.x &65535)==0`, so CTA0 of every normal 1024-CTA launch keeps P6. A already chooses Q5 there; its Q recoding and all P codes are unchanged. Other blocks switch only Q6 to Q5.

The Q accumulator reaches the same point before the common final Q record and again at the slot6 boundary. Hoisted phi runs exactly there, then the unchanged P terms are added. Both chains therefore represent `phi(Q*A)+P*A` in exact group arithmetic. Different projective coordinates/scales are expected.

With the conditional `NZ_CUT=0` source semantics, Q=0 starts at6, reloads P seeds6/7 and never applies phi; P=Q=0 uses duplicate P seeds and becomes unusable. **The actual A/C default has `QSB_GLV_NZ_CUT=1`, returning3 even for Q=0.** On that executed route, exact Q=0 causes a common final-Q opposite-point cancellation and zero ZZ/ZZZ; the inherited incomplete chain cannot restart from infinity and marks it unusable. A/C have the same ideal exceptional geometry. A synthetic `M=C*2^100` produces common final-Q doubling; it is above the exact residual bound. These conditional statements are not native execution results.

## Bounded check and required runtime qualification

The original development check recorded 15105 bounded magnitudes, both signs/routes: 60420 route rows, 332310 code/seed-mask checks, 60420 common-prefix checks, 4108 full128-bit synthetic endpoint checks, five actual borrow-skip constructions, and 16396 slot/phi/selector schedule cases. All passed in that development check; rerun the accompanying public script to produce its own source-bound receipt. No compile or GPU run is performed by this script.

Actual inherited `DECODE_CUT`, `HIGH15_NOFB`, no-pre-reduction, field multiply/square carry cuts and Y-offset borrow cut remain enabled. In particular, the field carry cuts can behave differently for the changed projective intermediates. The exact digit/group proof therefore does **not** prove all-input native bitwise equality or universal canonical correctness.

Before service timing, mandatory native qualification must bind native2ae/eb9f and the same table/constants, verify the known target and both recovery IDs through the canonical host gate, and retain the full finite canonical hit set with identical candidate intervals/credits. Direct field/path qualification should include CTA0 P6 and ordinary P5 blocks that were Q6, both component signs, zero components on the **actual default** route, transitions at18/37/45/55/73/100, field midpoints/extremes, upper residual bounds, and negative low32-borrow cases. Compare usable status and normalized affine/recovered outputs; raw projective buffers are not an equality oracle. Label inherited exceptional/cut misses separately instead of allowing a recoding proof to excuse a newly missing/extra known or canonical hit.
