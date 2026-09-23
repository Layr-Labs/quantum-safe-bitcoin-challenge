# High-half parity experiment

This is a **rejected static performance experiment**, not a production change.
Base: `9f239c386c7e99f8815103d9c6cc4465d7c5a9ba`.

The production narrow parity window uses 18 wide products. This variant uses
9 wide products and 9 high-half products: it does not halve the number of
multiplications. It discards the low halves of the products in columns 6 and
13, then widens the existing exact-fallback guards.

## Proof

Let B=2^32 and Dk be the sum of the products ai*bj with i+j=k. The original
27-product window computes

```
M = D7 + floor((D6 + floor(D5/B))/B)
H = D14 + floor((D13 + floor(D12/B))/B).
```

Replace each carry expression by the sum of the individual high halves.
There are seven terms in D6 and two in D13. Thus the new values underestimate
M by at most 12 and H by at most 4. Both bounds are attained when all input
limbs equal 0xffff0001. The accumulation must retain at least 35 bits.

The inherited quotient estimate is `q=H+977*(H>>32)+uint32(M)+beta7`.
Provided M's low word does not wrap, its error is at most 12+4+977=993.
The guards

```
uint32(M) < 0xfffffff3 && uint32(q) < 0xfffff478
```

imply the original window accepts and preserve its two parity bits. The
existing full-product fallback handles all other cases. This proves agreement
with the inherited window, not a new exactness guarantee for all of the
production field primitives, which already use documented truncated carries.

## Validation and decision

`python3 candidates/pinning/research_parity/audit.py` extracts the actual PTX,
translates its unsigned arithmetic into native C++, and compares it against
independent column arithmetic and the inherited field fallback. The saved
result covers 2,161,566 rows and 8,646,264 parity comparisons. Directed boundary
cases are deliberately overrepresented; their fallback fraction is not a
prediction for random GPU candidates.

CUDA 12.8.93, native sm_89, full ranked finish kernel:

| Metric | Promoted control | High-half variant |
|---|---:|---:|
| SASS instructions excluding NOP | 4,038 | 4,043 |
| Registers/thread | 64 | 64 |
| Spill bytes | 0 | 0 |

The compiled screening arm used a slightly more conservative q guard
0xfffff0a7; the audited bound above changes only that immediate. The full
instruction census, including opcodes, is in `compile_result.json`.
No GPU execution or throughput measurement occurred. There is no static
instruction-count benefit justifying a submission.

The actual experimental header is retained as `ParityHighParts.cuh`. To
reproduce compilation, copy the production include closure into a temporary
directory, replace only `ParityWindow.cuh` with this header, and run
`nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24 -Xptxas=-v -o pinning pinning.cu -lcrypto -lm`
followed by `cuobjdump -sass pinning`. Use a CUDA-compatible host toolchain.
