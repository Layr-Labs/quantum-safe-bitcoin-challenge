# First-state producer specialization: code review before the measurement

The live producer is kernel_build_first_flat in window_schedule_shared.cuh,
not the older disabled kernel_build_first referenced by first_stage_audit.cu.
That old test does not compile against today's source, so it is not used as
proof. The existing benchmark.sh subset entry and its unchanged verifier are
the applicable pre-existing integration check.

For eight classes, t / 8 and t % 8 are exactly t >> 3 and t & 7. For other
positive class counts, the original runtime quotient/remainder is retained.
The host only launches after preparing a nonempty class table; zero epochs
produce no launch. The existing epoch bound check precedes descriptor reads
and output writes, covering the partial last block. No new allocation, error,
synchronization or cleanup path is introduced.

Each output state starts at (epoch * QSB_FIRST_SLOTS + class) * 8 uint32 words.
The device allocation is CUDA-aligned and every offset is 32-byte aligned;
two uint4 writes therefore have the required 16-byte alignment. Writes cover
exactly the original eight words in the same order and no padding slot. The
digest's state stride, class selection, SHA rounds and inputs are untouched.
Producer and digest stay ordered in their existing CUDA stream.

The knob is off by default. An enabled probe adds its value to the native
carrier identity to prevent a stale carrier from hiding the experiment.
The zero value adds no bytes, preserving the default carrier signature.
Bits allow mapping and vector-store effects to be separated if the screen
has a positive or surprising signal. No production image is changed yet.

Hypothesis: eight scalar 32-bit stores with 32-byte lane stride waste store
instructions compared with two 128-bit stores; runtime unsigned division is
also avoidable in the actual eight-class configuration. This is a producer
optimization, not another first-buffer packing or digest-arena experiment.
