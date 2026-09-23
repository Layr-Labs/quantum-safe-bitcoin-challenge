# Exact wide-product accumulation

## Complete product construction

Let W=2^32, with eight input digits a_i and b_j. Compute each of the 64
ordinary products with `mul.wide.u32` and split its two returned words:

```
a_i*b_j = L_ij + W*H_ij
0 <= L_ij < W
0 <= H_ij <= W-2
```

A group of consecutive rows is accumulated in two passes per row. For its
first row, the eight low words initialize the low-word accumulator. A
complete carry chain then adds the eight high words one position higher.
For every following row, each wide multiply produces a low word and a saved
high word. The complete low-word addition pass adds the low words at the
row's offset and saves its outgoing carry in the new top word. The complete
high-word pass then adds the saved high words one position higher, including
that saved carry. Carries are kept between all word additions that require
them; plain multiplication and packing do not overwrite the PTX carry flag.

This is ordinary unsigned integer multiplication. After processing i rows
from a group's start, its accumulator is the product of the corresponding
i-digit integer and b, bounded strictly below W^(i+8). Every intermediate
partial sum is nonnegative and no greater than the completed prefix. The
new top word therefore contains the entire outgoing carry from the low
pass, and the high pass cannot require an unrepresented word beyond the
proven prefix width. This bound is not a claim that an internal carry is
usually zero: all internal carries are propagated.

The selected partition is **seven rows plus one row**. The first seven rows
produce a 15-word prefix, the final row produces nine words, and a complete
nine-word chain merges the final row shifted by seven words. The result is
the full sixteen-word, 512-bit product. The merge consumes eight existing
overlap words and a new top word, and retains every incoming carry. Its
upper bound is a*b < W^16. Zero operands, maximal inputs and long low/high
carry chains are all within this construction.

The full product is followed by the literal promoted C31 raw-fold text.
The four returned limbs match the promoted raw output, including its
inherited representation choices; this is not merely a congruent-residue
replacement. No additional omitted carry, exceptional-point exclusion,
normalization assumption or verifier shortcut is introduced.

## Why this differs from the failed MAC route

The product prefix has **64 plain wide multiplies and 127 32-bit additions**.
There is no `mad.wide`, no `mad.hi` or `mad.lo`, and no zero-extended
64-bit MAC addend. Splitting a wide result into its two real words is still
subject to native allocation; it is not claimed to be physically free. It
avoids the specific repeated construction of a new `{carry,0}` wide seed
and the fused wide-addend lowering that motivated the preceding experiment.

The promoted prefix has the same 64 wide multiplies and 134 width-normalized
word arithmetic operations. We explored all 128 ordered row partitions of
this plain-wide formulation. Seven-plus-one was selected because the source
model simultaneously lowers arithmetic cost and the peak live-word estimate,
without the long dependency chain of the earlier split-MAD formulations.

| Prefix | Wide products | Other word arithmetic | SSA live words | Depth with wide weight 1 / 2 |
|---|---:|---:|---:|---:|
| Promoted |64|134|40|31 / 32|
| Selected seven-plus-one |64|127|32|29 / 30|

With a wide-product weight of one, the abstract arithmetic totals are 198
and 191; with a weight of two they are 262 and 255. Moves and pairs are
represented as aliases for these counts. These figures are source-level
sensitivities, **not SASS counts, hardware cycles, allocated registers,
occupancy or a throughput estimate**. The whole hot multiplication is
replaced, so repeated point-chain multiplications can amplify an arithmetic
saving, but the exact native result and end-to-end effect remain unmeasured.
The new source can still lose because of allocation, move insertion,
compiler scheduling, code expansion or interaction with launch bounds.

The preceding split-MAD search covered 4,374 complete row/group choices and
13 modeled variants, but exposed a tradeoff between high-MAD lowering cost
and long carry chains. Those unsubmitted prototypes were not copied into
this release. This new formulation keeps ordinary wide products throughout.
It is a concrete repair to the failed mechanism, not an unchanged-code redraw.

