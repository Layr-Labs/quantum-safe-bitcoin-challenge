## Exact product construction

Let R = 2^128, and write a = a0 + R*a1 and b = b0 + R*b1, where every half
is in [0,R). Define wrapping differences da = (a0-a1) mod R and db =
(b0-b1) mod R, with borrow bits sa = [a0<a1] and sb = [b0<b1]. Then

```
L  = a0*b0
H  = a1*b1
D  = da*db
Ds = D - R*(sa*db + sb*da) + R^2*sa*sb
C  = L + H - Ds = a0*b1 + a1*b0
a*b = L + R*C + R^2*H
```

Each of the three 128-by-128 products uses sixteen `mul.wide.u32`
operations, reducing the full product prefix from 64 to 48 such operations.
Every partial product, difference borrow, sign correction and final merge
carry is retained. The middle C is at most 257 bits; its 257th bit is
explicitly captured and included in the final 512-bit assembly. Dropping
that bit is not an optimization in this implementation.

Ds lies strictly between -R^2 and R^2. Its four 64-bit low limbs and one
32-bit sign-extension word are sufficient. The sign-extension operations use
modulo-2^32 arithmetic while retaining the two's-complement signed value.
After Ds is formed, the wrapping differences and borrow masks can die before
the diagonal products. C is formed once and merged into L and H in one
six-limb carry chain rather than repeatedly traversing a large accumulator.

The previous unsubmitted masked version still had too much overhead. It used
four 64-bit operand masks plus an explicit sign-product correction. This
release predicates each *whole* high-half subtraction chain on its borrow
predicate. A disabled chain changes neither its destination nor CC; when a
chain executes, its initial `sub.cc` starts a fresh borrow chain. There is no
conditional branch or lane exchange, and all lanes still participate in the
same enclosing collective kernels.

The final independent repair removes the separate `sa*sb` calculation.
`ma32` is zero when sa is zero and UINT32_MAX when sa is one. In the second
chain, enabled exactly when sb is one, the top-word instruction is

```
@pb subc.u32 d4, d4, ma32;
```

Subtracting ma32 adds sa modulo 2^32, so this instruction simultaneously
propagates the borrow and injects the required +sa*sb into the sign word.
This is an exact identity, including both-negative, zero-difference and
all-ones boundary cases. It adds no probability-based carry omission.
Predicate and extended carry semantics follow NVIDIA's
[PTX ISA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

## Preserve the actual raw arithmetic contract

The new complete 512-bit product is followed by the existing promoted C31
reduction text, with scratch names isolated inside the helper. This preserves
raw output bits, including inherited short-carry behavior. The comparison is
not merely an equality modulo the curve prime, and no reassociation of
approximate field operations is used. No new exceptional values are skipped.

`PredicatedMul48.cuh` contains the generated inline PTX and returns four
64-bit limbs. All eight input limbs are captured before output operands are
written, preserving input/output alias cases used by the in-place wrappers.
`GPUMath.h` dispatches its device `_ModMultCore` to this helper when
`QSB_PRED_MUL48 && QSB_C31 && QSB_SHORT_CARRY`. The switch defaults to one.
Setting `QSB_PRED_MUL48=0` retains the original promoted implementation.
Non-C31 and non-short-carry paths, and the host transcription, retain their
original implementations. The existing negative-Y seeded MAC and fused
square routines are not rewritten by this patch.

The promoted launch geometry, scalar enumeration, fixed-base tables,
cofactor topology, parity handling, hashing and exact publication gate are
preserved. Root-group arithmetic that calls the separate carry-complete
`qsb_field_mul` remains unchanged. Only the Pinning editable directory is
part of this submission; the harness and scoring rules are unchanged.

## Cost analysis and its limits

The source ledger counts each 64-bit add/subtract/bitwise operation as two
word operations, each 32-bit operation as one, and lists predicate operations
separately. Pack/unpack register aliases are treated as free in the liveness
model. The table describes the *product prefix*, before the identical fold.

| Prefix | Wide products | Other word arithmetic | Predicate ops | Peak live words | Peak live predicates |
|---|---:|---:|---:|---:|---:|
| Promoted schoolbook | 64 | 134 | 0 | 40 | 0 |
| Prior masked signed middle | 48 | 166 | 0 | 44 | 0 |
| Predicated, separate sign repair | 48 | 153 | 3 | 40 | 2 |
| Submitted fused sign repair | 48 | 152 | 2 | 41 | 2 |

The new repair removes twelve normalized operations from the prior masked
version when predicate operations are included, and reduces its peak live
word estimate by three. Relative to the promoted prefix it trades sixteen
wide products for twenty other operations and two live predicates, with
one additional peak live word. In a simple additive sensitivity model,
a wide product must cost more than 1.25 unit operations for that ledger to
favor the new prefix. That number is not a native instruction-cost claim.
With a multiply weight of two, the conservative dependency model reports
35 units for this prefix versus 32 for the original. The additional sign
repair dependency is a real concern, not omitted from the comparison.

Including the unchanged reduction, the ordinary raw multiply has 57 wide
products rather than 73. This is an instruction-count change, not a promised
percentage improvement in throughput. Predicated instructions may issue even
when some lanes are inactive; no half-warp issue saving is assumed. Native
fusion, predicate handling, register allocation, spills, occupancy, instruction
cache and surrounding point state can outweigh the reduction in products.
Only the official device result can establish whether this trade is useful.
We selected it for remote testing because it changes a broad arithmetic hot
path and concretely repairs the overhead that prevented the earlier
Karatsuba candidate from being submitted.

## Verification actually performed

The independent product models passed 6,816 full-width input pairs for the
promoted product and each new variant. The cases include a Cartesian set of
extreme 128-bit halves, 4,096 deterministic random pairs, prime-adjacent
values, all-ones chains and varying bit boundaries. All four borrow-sign
quadrants are represented. Every full 512-bit result equals Python integer
multiplication. An additional 1,145 tests execute the actual complete new
PTX plus raw fold and compare its output bits with the promoted PTX.

After copying into the release, `test_mul48.py` extracts the actual header
literals again and passes 1,193 source-bound full-product and raw-output
pairs. It checks dispatch selection and identical fold text. Four directed
predicate/CC tests check inactive-chain carry preservation and active-chain
carry replacement. A negative control that omits the fused sign injection
fails 278 of the packaged input pairs, confirming that the check exercises
this necessary correction. A temporary assertion in the Python test compared
a list with an integer; indexing the one-output result corrected that test
harness mistake before the passing run. No production arithmetic fix was
needed for it.

The existing host-gate Python test also passes: 64 independent SHA256d
midstate samples, problem layout, recovery agreement with the verifier and
source gate presence. Source comparison confines production changes to
`GPUMath.h` and the added helper. The carried production headers and main
search source are byte-identical to the new promoted baseline. Include closure,
source hashes, archive scope and `git diff --check` are checked before upload.
These are static and Python semantic checks, not CUDA assembly, device race
checks, hardware register reports or throughput measurements.

Reproduce the packaged model checks from the repository root:

```
python3 -B candidates/pinning/test_mul48.py
python3 -B candidates/pinning/test_host_gate.py
```

`mul48_research/` contains the narrow integer PTX interpreter, original raw
reference, original product reference, submitted product prefix and source
cost result. The reference is copied from the immutable promoted source,
not calculated with the same Karatsuba formulas as the implementation.
The emitted header SHA-256 is
`7d2bb9d259cd5a950f5ba1d9f6ffe2e730f81c5360a6105ab90a1fa5c1fa797d`.

