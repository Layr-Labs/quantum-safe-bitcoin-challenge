# Corrected PR24 arithmetic base

The preferred public research base is PR24, commit
`6e76a74fed8e6e5b8439e64ec20f586085f37d52`. Its four source/license files are
preserved in `pr24_reference/`, with hashes and attribution in
`PROVENANCE.json`. This is a research copy; the production entry point remains
unchanged. Imported compiled binaries and build stamps were excluded.

`pr24_corrected/` is the same source with a repair to both `_ModMultCore` and
`_ModSqr`. Its header SHA256 is
`6bcbe655aaf6a1ffe2b90d036e07fe621e7c2e6c5d47518b35a263fe083cb493`.
It is not submitted, GPU executed, or performance measured. On September 16 it
passed real CUDA compilation and linking in a local ARM Linux VM on this Mac,
with both explicit `sm_89` and the official default architecture flags.

## Source-level evidence

Run from the benchmark root:

```sh
python3 candidates/pinning/research/check_pr24_field.py
# Optionally materialize another fresh isolated candidate:
python3 candidates/pinning/research/check_pr24_field.py --output /tmp/fresh-pr24-fixed
```

The test compiles the actual host multiplication and dedicated square bodies.
Unlike the subset square host wrapper, pinning's square contains an independent
limb implementation, so this tests two different host product schedules. Inputs
cover full-width values, near-modulus carry cases, and output/input aliases.

It also extracts and interprets the actual inline PTX strings. The small model
supports only their straight-line integer instruction subset. Unsupported
instructions, uninitialized registers and malformed statements fail. Explicit
semantic probes check carry preservation/overwrite, packing and funnel shifts.
An initial brace-operand tokenizer error failed immediately and was fixed
before collecting results. The model is not a CUDA compiler or GPU emulator.

| Path | Calls | Original wrong residues | Repaired wrong residues |
|---|---:|---:|---:|
| Actual host multiply | 60,540 | 69 | 0 |
| Actual host square | 40,360 | 84 | 0 |
| PTX multiply semantic model | 2,180 | 23 | 0 |
| PTX square semantic model | 2,180 | 42 | 0 |

Host repaired outputs are canonical in all cases. PTX-model counts refer to the
inline assembly result **before** the common C++ canonical subtraction; some
are noncanonical but congruent at that intermediate boundary. The common
normalization code runs after both host and device branches. These targeted
counts are not a failure-rate estimate for benchmark coordinates.

## Bounded third fold

Let `B=2^256` and `C=2^32+977`. For a product below `B^2`, the first fold
`low + C*high` has high part at most C. The second fold is at most
`B-1+C^2`. If its final carry is one, the low remainder is below `C^2`.
The third fold is therefore below `C^2+C < 2^96`. If the carry is zero,
the third fold adds zero. It is enough to propagate through words z0, z1 and
z2; propagating into z3..z7 is unnecessary. The repair adds five PTX arithmetic
instructions to each assembly block, plus the common canonical subtraction.
PTX counts are not SASS counts or a prediction of latency, registers or speed.

The highest second-fold addition must use `addc.cc.u32 z7` before capturing
the carry. NVIDIA specifies that `addc` writes carry-out only with `.cc`.
The model uses the documented packing and funnel-shift conventions too.
[NVIDIA PTX instruction semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-addc).

A mutation that removes that `.cc` is caught with canonical inputs
`a=p-1`, `b=p-2^224`. The right answer is `2^224`; the mutant incorrectly adds
`2^32+977`. This exercises stale carry-in versus actual carry-out, supplementing
the original `a=b=p-65537` dropped-carry witness. Gemini proposed this invalid
flag pattern again in its latest review; it was rejected rather than imported.

## Review boundaries and next work

The external reviews also asserted that classical arithmetic was exhausted,
Montgomery reduction necessarily cost a full dense multiply, and affine
batching could not win. Those universal conclusions were not proved and are
not accepted. A particular failed Karatsuba implementation is not a lower
bound for all alternative schedules. Likewise, an operation-count reduction
does not establish a faster implementation.

The repaired base is ready for source-derived point/pipeline checks when a new
performance architecture is selected. Any next use must retain both repairs,
refresh pending submissions, and distinguish inherited performance evidence
from new expectations. GPU execution and timing remain open evidence gaps,
not an automatic prohibition on a well-supported submission.

## Native compilation evidence

`native-base-comparison.json` binds both reference and repaired builds to their
source hashes. CUDA 12.8.93 compiled the exact include closures at `-O3`,
`-arch=sm_89`, and `-DQSB_ZEROS_N=24`. Both ranked prepare kernels use 126
registers and 16,384 shared bytes; both ranked finish kernels use 80 registers
and 24,576 shared bytes. All four report zero stack and zero spills.

The repair increases ranked prepare from 7,200 to 7,816 SASS instruction slots
and finish from 3,096 to 3,288. These are static code sizes, not dynamic work or
throughput. They show a real native cost despite unchanged resource occupancy
limits; do not describe the correctness repair as free. CPU arithmetic and PTX
semantic checks remain separate evidence from actual GPU execution.

See `LOCAL_CUDA.md` for repeatable local compilation and its limits.
