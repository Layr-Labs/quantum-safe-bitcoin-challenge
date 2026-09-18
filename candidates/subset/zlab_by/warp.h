/* 4-lane lockstep warp emulation for the host cross-check of zi_inverse_quad.
 *
 * WHAT IT MODELS EXACTLY
 *   - the arithmetic of every lane, bit for bit: the shipped device source text
 *     (tests/gpu_epochs/zinv32.cuh) is #included verbatim and compiled by g++
 *     through its own __CUDA_ARCH__-undefined branch, so ZI_DEV becomes
 *     "static inline", ZI_CONST becomes "static const", and zi_ctz32/zi_clz32
 *     use the header's own __builtin_ctz/__builtin_clz bodies.  Nothing in the
 *     algorithm, the table, the packing or the limb arithmetic is retyped.
 *   - lane indexing and cross-lane data flow: lanes 0..3 are four real OS
 *     threads and zi_x() (the only warp primitive in the file, __shfl_sync with
 *     mask 0xF) is a real cross-lane read through a shared slot with a barrier
 *     before and after, so a lane-indexing slip shows up as a wrong answer.
 *   - shuffle-count uniformity: a lane that reaches a different number of zi_x
 *     calls than the others deadlocks at the barrier instead of proceeding, so
 *     a non-uniform loop exit cannot pass silently.
 *   - the call site's lane set: the device call is under `if(tid<4)` with mask
 *     0xF and srcLane always in 0..3, so lanes 4..31 take no part; four
 *     emulated lanes are the whole participating warp slice.
 *
 * WHAT IT DOES NOT MODEL
 *   - __shfl_sync mask/reconvergence semantics beyond a converged 4-lane group
 *     (the call site guarantees convergence: all 256 threads enter the kernel
 *     and the only earlier return is block-uniform).
 *   - scheduling, register allocation and latency -- none of which change the
 *     values computed.
 */
#pragma once
#include <cstdint>
uint32_t qsb_warp_shfl(uint32_t v, int src);   /* defined in xcheck.cpp */
