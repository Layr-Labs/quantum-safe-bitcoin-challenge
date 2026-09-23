# Deferred X anchor: rejected static prototype

This isolated research does not change the production include closure. It adds
an affine abscissa anchor to the inherited deferred ordinate. GPL attribution
to the existing VanitySearch/public point and field code remains applicable;
the generated control is the exact snapshot recorded in `source_manifest.json`.

With ordinary XYZZ state `(X,Y,U,V)` and affine anchor `(a,b)`, store
`D=X-aU` and `N=-Y-bV`. For the next affine point `(x,y)`:

```
P = (x-a)U-D
R = (y+b)V+N
U' = UP²; V' = VP³; Q = xU'
D' = R²+P³-3Q
N' = RD'
```

These equations give the ordinary mixed-add result with the next anchor
`(x,y)`. The first two-affine addition already computes `D` as an intermediate,
so it needs no extra multiply. The last addition returns `X=D'+Q`, reusing its
live `Q`, also without an extra multiply. The mathematical point cost remains
7M+2S per mixed addition. However an extra affine-x subtraction, a longer fused
square tail, the live x anchor, and the final-step branch cost more in compiled
code than the removed modular correction saves.

CUDA 12.8.93, sm_89, `-O3 -DQSB_ZEROS_N=24 -Xptxas=-v`:

| Diagnostic | Control | X-anchor |
|---|---:|---:|
| Stage-0 registers | 124 | 126 |
| Stage-0 static SASS instructions | 6,056 | 6,096 |
| 13-iteration point-loop instructions | 1,030 | 1,075 |
| Stage-2 static SASS instructions | 4,048 | 4,048 |
| Stage-0 spills | 0 | 0 |

The +45 instructions per round give +585 per complete point chain. This
prototype is therefore not selected for production. There is no device timing
or claimed throughput result; native compilation cannot measure performance.

`audit.py` executes the actual C++ seed and mixed-add bodies with exact modular
primitive shims against independent affine Python arithmetic. It checks 2,037
complete 15-point chains and 28,577 intermediate point states without mismatch.
Eleven singular chains are counted and excluded: these formulas retain the
inherited incomplete exceptional-point behavior. This proves neither the
approximate device reducer nor whole-kernel hit recall. The generated fused
square inherits approximate carry cuts and was only compile-screened.

Reproduce from `candidates/pinning/research_anchor`:

```
rtk proxy python3 build.py
rtk proxy python3 audit.py
rtk proxy /tmp/qsb-cuda/nvcc-local -O3 -DQSB_ZEROS_N=24 -arch=sm_89 -Xptxas=-v -o control.bin control/pinning.cu -lcrypto -lm
rtk proxy /tmp/qsb-cuda/nvcc-local -O3 -DQSB_ZEROS_N=24 -arch=sm_89 -Xptxas=-v -o anchor.bin anchor/pinning.cu -lcrypto -lm
```

`audit_seeded_sub.py` and `audit_direct_point.py` are independent checks of the
separate `research_arithmetic` direct subtract-MAC experiment. They accept the
actual helper/header paths as arguments. Their JSON records bind the source
hashes and distinguish exact integer-product checks, approximate field
boundary differences, exact-field point equations, and lack of GPU execution.
The signed zero-bias variant has a directed integer-product counterexample;
the positive-B variant has a correct integer seed. Subtracting K only through
the low 64 bits adds a borrow boundary. Decrementing the first-fold high limb
instead adds a different boundary when that limb is zero. Neither approximate
tail is described as universally exact.

The direct-point audit checks 2,032 complete chains and 28,521 intermediate
states without mismatch over exact field shims (16 singular exclusions).
The actual PTX multiplication reassociation check has 2,210 differences among
12,167 directed boundary triples: every difference contains an identified
approximate primitive error. All 6,000 random triples agree. Consequently the
reordered product is algebraically correct but must not be described as
bit-identical under the inherited approximate arithmetic.

To reproduce the arithmetic audits after generated copies have been removed:

```
rtk proxy python3 ../research_arithmetic/generate.py subseed_bias
rtk proxy python3 audit_direct_point.py
rtk proxy python3 audit_seeded_sub.py ../research_arithmetic/_build/subseed_bias/negative_y_mac.cuh
rtk proxy python3 audit_seeded_sub.py
rtk proxy python3 audit_reassociation.py
```

Generated binaries, copied full source trees, and large SASS dumps were removed
after screening. Scripts and compact JSON evidence remain.
