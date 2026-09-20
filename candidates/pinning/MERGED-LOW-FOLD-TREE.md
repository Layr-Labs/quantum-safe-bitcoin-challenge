The runtime in this package combines complete packed64 low-fold arithmetic
and the top-two cofactor traversal. It derives from the verified submitted e5
source. The scheduling ideas were reviewed in public commit
f2977f08057f31af3150cc46cf461d2d26f92447; every upper carry, final correction,
canonical boundary and exceptional final point path remains complete here.

Let `B=2^256`, `K=2^32+977`, and `p=B-K`. For a full-width product `T<B^2`, the
first fold computes

```
S = (T mod B) + floor(T/B)*K = r + B*h.
```

Since `S <= (B-1)*(K+1)`, its high part satisfies `h<=K`. Write
`h=z8+2^32*z9`. Then `z9` is zero or one; when it is one, `z8<=977`.
Consequently `q=z8+977*z9` fits in 32 bits for every input in the domain.
The new `mad.lo.u32` does not discard a significant bit.

The second fold adds

```
h*K = 977*z8 + 2^32*(z8+977*z9) + 2^64*z9.
```

If `z0` is the low word of `r`, the packed 64-bit operation computes
`L=z0+977*z8+2^32*q`, retaining its carry. Its low word becomes the new `z0`;
its next word is added to `z1`; and `z9+floor(L/2^64)`, plus that addition's
carry, is added to `z2`. This is the same integer addition as the old low
three-word fold, including the carry leaving bit 95.

Both implementations then capture that actual carry. A nonzero carry still
propagates through all five upper words. An overflow from the complete upper
propagation still invokes the parent's final K correction. No carry is omitted
based on its estimated probability. The transformation applies to point
multiplication, point square, and the tree multiplier.

`prepare_merged_lowfold_cpu.py` extracts the actual generated PTX and checks
5,207 input pairs per function. It verifies exact raw equality with the parent,
the independent integer residue, equality of the carry-path decision, and the
actual first-fold high part against the expression above. All three carry paths
are exercised. See `paired-tree-merged-lowfold-0021/cpu-source-bound-proof.json`.

For the cofactor tree, the original upsweep places its top two subproducts at
`2*N-4` and `2*N-3`. In the original downsweep, the count-two stage copies the
opposite subproduct into `excluded[N-4+tid]`. Thus the count-four stage, for
`tid=0..3`, has the exact ordered operands

```
parent  = products[2*N-4 + ((tid & 1) ^ 1)]
sibling = products[2*N-8 + (tid ^ 2)].
```

The merged stage issues these four multiplications in lanes zero through three,
and the unchanged ordered root multiplication in lane four. It writes the four
results to `excluded[N-8+tid]` and resumes the original count-eight downsweep.
Every later operand expression is unchanged. This preserves raw outputs for any
deterministic multiplication primitive, without relying on reassociation modulo p.

The upsweep's existing warp barrier publishes the top subproducts. Lanes zero
through four are in the same warp, and the barrier after their writes publishes
all exclusions needed by the next stage. Later cross-warp barriers are retained.
The compile-time small-tree path for N=2, 4, and 8 uses the original traversal.
Exact-source native checks passed for the selected merged source, including
the full point and pipeline suites, direct tree multiplication and both recovery
formulations. Their original outputs were independently checked locally. Local
CUDA12.8.93 SM89 compilation uses96prepare and70finish registers with no stack
or spill bytes in the three primary specializations. Performance qualification
and official submission status are recorded separately in RESEARCH.md.

The shipped audit_paired_upper_fold.py interprets the actual point multiply,
point square and tree multiply PTX on5207 input pairs each, including all three
upper-carry paths. The original derivation additionally checked exact raw parent
equality and the real first-fold h against the integer expression above.
The tree index proof compared exact ordered multiplication expression DAGs for
ten powers of two from2through1024, with2046excluded leaves and every root.
These CPU proofs complement the native results; they are not throughput scores.
