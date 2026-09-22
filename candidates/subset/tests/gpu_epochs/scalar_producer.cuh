// SPDX-License-Identifier: GPL-3.0-only
// Research: run the unchanged paired SHA work independently of the EC chain.
#pragma once
// Reload cheap lane identity after the long field calls, so it need not stay
// live across them. Volatile special-register reads inhibit early hoisting.
__device__ __forceinline__ unsigned qsb_reload_tid() {
#ifdef __CUDA_ARCH__
    unsigned out; asm volatile("mov.u32 %0, %%tid.x;" : "=r"(out)); return out;
#else
    return threadIdx.x;
#endif
}
__device__ __forceinline__ unsigned qsb_reload_bid() {
#ifdef __CUDA_ARCH__
    unsigned out; asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(out)); return out;
#else
    return blockIdx.x;
#endif
}
template<bool SECOND>
__device__ __forceinline__ bool qsb_scalar_active(unsigned batch, unsigned epochs) {
    const unsigned tid=qsb_reload_tid(), bid=qsb_reload_bid();
    const unsigned ep=QSB_PAIR_MUL*bid+2u*(tid/QSB_SE_WINDOWS)+(SECOND?1u:0u);
    return bid*QSB_SE_BLOCK+tid<batch && ep<epochs;
}
#if QSB_SUBSET_SHA_PRODUCER
#if !ZLAB_DUAL_EPOCH_SHA || !ZLAB_K2S3M || !QSB_PAIR_SHARED
#error "The scalar producer requires paired 3M recovery and dual-epoch SHA"
#endif

// Each consumer block owns eight contiguous planes of QSB_SE_BLOCK words.
// This layout gives adjacent lanes adjacent 64-bit addresses for every word.
__host__ __device__ __forceinline__ size_t qsb_scalar_plane_index(
    unsigned block, unsigned word, unsigned lane) {
    return ((size_t)block * 8u + word) * QSB_SE_BLOCK + lane;
}

__device__ __forceinline__ void qsb_store_scalar(uint64_t *p,uint64_t value) {
#if defined(__CUDA_ARCH__) && QSB_SUBSET_SCALAR_STREAM
    asm volatile("st.global.cs.u64 [%0], %1;" :: "l"(p), "l"(value) : "memory");
#else
    *p=value;
#endif
}
__device__ __forceinline__ uint64_t qsb_read_scalar(const uint64_t *p) {
#if defined(__CUDA_ARCH__) && QSB_SUBSET_SCALAR_STREAM
    uint64_t value;asm("ld.global.cs.u64 %0, [%1];" : "=l"(value) : "l"(p));return value;
#else
    return *p;
#endif
}

__global__ void __launch_bounds__(QSB_SE_BLOCK, QSB_SUBSET_SHA_BLOCKS)
kernel_subset_scalars(const uint32_t *__restrict__ first,
                      uint64_t *__restrict__ scalars, unsigned epochs) {
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & (QSB_SE_WINDOWS - 1);
    const unsigned half = tid / QSB_SE_WINDOWS;
    const unsigned eA = QSB_PAIR_MUL * blockIdx.x + 2u * half;
    // Match the original consumer for inactive A and odd B tails. They are
    // hashed safely but never counted as additional searched candidates.
    const unsigned eA0 = eA < epochs ? eA : 0u;
    const uint32_t *f0 = first + (size_t)eA0 * QSB_FIRST_SLOTS * 8;
    const uint32_t *f1 = eA + 1u < epochs ? f0 + QSB_FIRST_SLOTS * 8 : f0;
    QsbPairEpochZ pair = qsb_pair_epoch_z_value(f0, f1, lane);
    #pragma unroll
    for (unsigned k = 0; k < 4; ++k) {
        qsb_store_scalar(scalars+qsb_scalar_plane_index(blockIdx.x,k,tid),pair.a[k]);
        qsb_store_scalar(scalars+qsb_scalar_plane_index(blockIdx.x,k+4u,tid),pair.b[k]);
    }
}

// One epoch/window hash per thread: the lower register count can support more
// resident warps than the paired producer. Output ownership is unchanged.
__global__ void __launch_bounds__(128, QSB_SUBSET_SINGLE_BLOCKS)
kernel_subset_scalars_single(const uint32_t *__restrict__ first,
                             uint64_t *__restrict__ scalars,unsigned epochs) {
    const unsigned flat=blockIdx.x*128u+threadIdx.x;
    const unsigned ep=flat/QSB_SE_WINDOWS, lane=flat&(QSB_SE_WINDOWS-1);
    const unsigned block=ep/QSB_PAIR_MUL;
    const unsigned tid=((ep%QSB_PAIR_MUL)/2u)*QSB_SE_WINDOWS+lane;
    const unsigned which=ep&1u, ea=ep&~1u;
    const unsigned chosen=ep<epochs?ep:(ea<epochs?ea:0u);
    const uint32_t *f=first+(size_t)chosen*QSB_FIRST_SLOTS*8;
    uint32_t state[8];uint64_t z[4];
    qsb_scheduled_window_hash(state,nullptr,lane,f);
    qsb_pair_second_sha_z(state,z);
    #pragma unroll
    for(unsigned k=0;k<4;k++)
        qsb_store_scalar(scalars+qsb_scalar_plane_index(block,k+4u*which,tid),z[k]);
}

template<bool SECOND>
__device__ __forceinline__ void qsb_load_scalar(
    const uint64_t *__restrict__ scalars, uint64_t *z) {
    #pragma unroll
    for (unsigned k = 0; k < 4; ++k) {
        z[k] = qsb_read_scalar(scalars+qsb_scalar_plane_index(blockIdx.x,k+(SECOND?4u:0u),threadIdx.x));
    }
}

__device__ __forceinline__ QsbPairEpochZ qsb_load_scalar_pair(
    const uint64_t *__restrict__ scalars) {
    QsbPairEpochZ pair;
    qsb_load_scalar<false>(scalars, pair.a);
    qsb_load_scalar<true>(scalars, pair.b);
    return pair;
}
#endif
