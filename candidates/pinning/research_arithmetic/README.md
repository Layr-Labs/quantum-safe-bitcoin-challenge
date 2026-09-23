# Subtraction inside the field product: arithmetic research, 2026-09-23

No screened arm is promoted, and no GPU timing was available. The experiments
preserve production source and use the pinned upstream commit in
`source_manifest.json`. The result is a concrete negative finding: eliminating
a source-level modular subtraction did not produce a compelling compiled-loop
improvement with CUDA 12.8.93.

The promising algebraic change forms `P=x2*U-X` inside the multiplier, updates
`Unew=U*P²`, and obtains the unchanged quantity `Q=x2*Unew`. This removes the
need to materialize the intermediate `x2*U` while preserving the mixed XYZZ
addition identity and the multiplication count. The integer multiplier can
seed its existing even row with `-X`, then inject `1-borrow` at bit 256 to
compute **`a*b+2^256-c` exactly**, with no signed partial-product debt.

Since `2^256 = K (mod p)`, `K=2^32+977`, the bias must be removed. The three
screened placements are after reduction (`subseed_bias`), in the first low
64-bit accumulator (`subseed_lowbias`), or by decrementing the first-fold high
word before the second fold (`subseed_foldbias`). Each shortened correction
introduces an explicit exceptional carry/borrow case and requires the inherited
exact host publication gate. These are not exact field primitives. The fold
variant's additional failure is `z8==0`; its probability cannot be inferred from
a uniform high product word, because products have a nonuniform distribution.

The zero-bias signed version has a directed counterexample: `a=b=0,c=1`
produces integer `2^320-1` instead of `2^512-1`. Its initial borrow stops at
`e4`. The one-instruction static saving does not validate that variant.

## Whole-loop native sm_89 screen

All listed arms have 602 wide integer multiply-add instructions per rolled
point-add iteration, no spills, and unchanged shared-memory use.

| Arm | Loop instructions | Registers |
|---|---:|---:|
| Pinned baseline | 1030 | 124 |
| Double Q before one subtraction | 1033 | 124 |
| Remove fused square's 3p bias | 1034 | 126 |
| 64-bit square add/sub chains | 1030 | 124 |
| 64-bit double-Q chains | 1036 | 120 |
| Reordered Q only | 1035 | 128 |
| Signed subtract seed, known borrow defect | 1029 | 126 |
| Positive seed, subtract K after reduction | 1034 | 128 |
| Positive seed, subtract K in e0 | 1034 | 126 |
| Positive seed, remove bias in second fold | 1031 | 126 |
| Fold variant, update ZZZ before Q | 1031 | 118 |

The remaining dependency orders worsened the loop to 1037–1040 instructions.
The production-style `compute_52` PTX assembled for `sm_89` confirms 1030,
1029, and 1034 for baseline, signed seed, and final-K positive seed respectively.
Register reductions do not by themselves imply an occupancy or throughput gain.
The approach could still change dependency latency, but this requires matched
GPU measurement. No percentage speedup is claimed.

## Reproduce

`generate.py` recreates a selected arm from Git and applies its compact patch.
All generated files remain inside `candidates/pinning`. Example, from the
benchmark repository:

```sh
rtk proxy python3 candidates/pinning/research_arithmetic/generate.py baseline
rtk proxy python3 candidates/pinning/research_arithmetic/generate.py fold_pvzq
```

Inside each printed arm directory, the native compile used:

```sh
rtk proxy /tmp/qsb-cuda/nvcc-local -O3 -arch=sm_89 -cubin -Xptxas=-v pinning.cu -o native.cubin
rtk proxy /tmp/qsb-cuda/toolkit/bin/cuobjdump --dump-sass native.cubin
```

Save compiler output as `native.log` and disassembly as `native.sass`, then use
`analyze.py --directory <generated-parent>`. The second path used
`nvcc-local -O3 -arch=compute_52 -ptx`, followed by
`ptxas -O3 -arch=sm_89 -v` and the same disassembly. Pass `--mode jit` after
saving outputs as `jit.log` and `jit.sass`.

The cached `nvcc-local` wrapper selects GCC 13 and compatible glibc headers;
the system GCC 15 headers cannot compile this CUDA 12.8 source directly.
The independent actual-PTX semantic audit lives in
`../research_anchor/audit_seeded_sub.py`; its results distinguish exact biased
512-bit products from the explicitly approximate reduction tails.
