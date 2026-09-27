# Integer Tensor Core field-product screen

This transient-fragment mapping is **not selected** after a native CUDA screen.
The submitted PR74 production include closure remains unchanged. The initial
mathematical screen is retained alongside the new packing/carry prototype.

For 32 little-endian bytes of each operand, define zero-extended matrices

```text
A[r,k] = a[k-r]             r=0..15, k=0..63
B[k,c] = b[7+8c-k]          c=0..7
D = A[:,0:32] B[0:32,:] + A[:,32:64] B[32:64,:]
```

Then `D[r,c]` is convolution coefficient `7+8c-r`. Substituting `i=k-r`
proves the identity. Every contributing term has `k=i+r` between 0 and46;
the two tiles contain the entire sum. Rows0..7 select coefficients0..63 once
each; coefficient63 is zero. Unsigned byte operands give a coefficient bound
of2,080,800. The byte carry bound8160 is inductive and fits signed32 together
with the coefficient.

The mathematical shapes correspond to two dense integer `m16n8k32` operations.
NVIDIA documents unsigned byte inputs, signed32 accumulators and support from
sm80 onward, including sm89. `fragment.cuh` now implements the per-lane layout,
register shuffles, byte packing, output extraction and a ballot-based carry
normalizer. `micro.cu` includes separate coefficient/product kernels and an
exactly extracted current scalar field-multiply control. These are isolated
research kernels with no production integration or GPU runtime test.
[PTX ISA, integer MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)

Run from the benchmark directory:

```sh
python3 -B candidates/pinning/research/tensor_field/screen.py
```

The source-bound report records2,192 successful full products and canonical
residues, including every pair of byte positions, maximal bytes, modulus
boundaries, random operands and squares. All128 matrix outputs are checked,
including duplicate/unused coefficients. Mutations dropping the second tile
and treating unsigned bytes as signed are rejected. Integer folds use Python
arbitrary precision; they supply no device reduction implementation.

This concrete mapping issues8192 logical byte MAC slots for1024 distinct
useful products (12.5%). It consumes1536 packed operand-fragment bytes across
both calls from64 original bytes and holds128 accumulator words across the
warp. These are logical volumes, **not** measured register allocation, global
traffic or emitted instructions. The prototype supplies these fragments with
register shuffles and funnel shifts; the native costs are recorded below.

The main unresolved cost is ownership: one warp handles one independent field
product, compared with32 independent candidates in the submitted schedule.
Two MMA instructions cannot be compared directly with one thread's scalar
multiply count. Carry propagation, pseudo-Mersenne reduction, persistent point
state, table gathers, checkpoint layouts and external inversion must all fit a
cooperative schedule. The current matrix is not claimed to be optimal.

`check_fragments.py` separately checks2,192 complete products through the
source-extracted packing indices and documented lane layout, plus8,688 carry
cases. It models collectives on the CPU; it does not execute the CUDA header.
A wrong-row mutation initially survived an all-0xff witness because that
pattern hides some byte shifts. Distinct byte values expose it; this corrected
mutation test now passes. No production source bug was found.

The CUDA12.8.93 sm89 compile yields:

| Kernel | Products per warp | Static non-NOP instructions | Registers | Spills |
| --- | ---: | ---: | ---: | ---: |
| Packed tensor coefficients | 1 | 127 | 40 | 0 |
| Tensor512-bit integer product | 1 | 188 | 40 | 0 |
| Current canonical field product | 32 | 170 | 40 | 0 |

The tensor product contains twoIMMA instructions,28 indexed shuffles, one
up-shuffle and two ballots. Field reduction is still absent. Counts include
kernel guards/I/O and refer to different output semantics; they are not dynamic
cycle counts or throughput measurements. Nevertheless, the current transient
packing implementation gives no credible substantial gain case. Adding a full
cooperative point pipeline is not justified by this result.

`native-results.json`, `comparison.json` and `fragment-results.json` bind the
tested sources. Revisit only with a materially better packing/ownership or
persistent representation, and cost the full curve/reduction pipeline. This
does not prove an optimality bound or reject every possible Tensor Core design.
The compiler VM was stopped after the build.

Gemini completed a review but supplied unsupported cycle estimates and a
wrong carry bound; those were corrected in `external-review.json`. Grok timed
out without a visible review. No external model code was imported. The initial
paper applicability review is in the subset research inbox's `sixth-review.json`.
