# Original divstep table, read-only global cache: review before measurement

The live warp-limb inverse calls zi_divstep30_column, whose default ZI_LUT
reads the original 832 uint64 words from constant memory. Kernel comments
record a 2 KiB constant L1 and 6.5 KiB table with dependent lookup latency.
The new knob changes ONLY this lookup to __ldg of an aligned immutable
__device__ mirror initialized by the exact same ZI_BY_LUT_INIT macro.
No compact32/40 encoding, decoder, flags, clamping or arithmetic changes.
Scalar/quad oracle paths and host code still use the original constant table.

No per-CTA copy, shared allocation, synchronization or barrier changes: this
probe keeps the 24 KiB occupancy geometry and removes the shared-table setup
overhead of previous experiments. The read-only cache can hold all 6656 bytes;
whether latency improves must be measured. Table has no host upload or mutable
data race; CUDA module initializes it before kernels. Every clipped delta and
six-bit ratio indexes the same 0..831 row as the unchanged path. Roots zero,
partial blocks and Fermat fallback retain their original behavior.

Knob off by default; compact/shared experiments excluded at compile time;
native carrier signature appends the value only when enabled. Native compiler
must actually emit LDG (not LDC) for this probe to be real. First exact check
is the pre-existing tree_audit.cu, followed by unchanged benchmark.sh for
throughput if resources and exactness are viable. Original packing failures
do not settle this no-encoding/no-shared-occupancy memory-space alternative.
