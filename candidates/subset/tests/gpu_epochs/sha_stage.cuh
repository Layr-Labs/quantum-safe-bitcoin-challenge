// SPDX-License-Identifier: GPL-3.0-only
// Separate the unchanged paired sighash from curve recovery.
#pragma once
#ifndef QSB_SHA_STAGE
#define QSB_SHA_STAGE 1
#endif
#ifndef QSB_SHA_STAGE_CTAS
#define QSB_SHA_STAGE_CTAS 4
#endif
#ifndef QSB_SHA_STAGE_SINGLE
#define QSB_SHA_STAGE_SINGLE 1
#endif
#ifndef QSB_SHA_STAGE_SINGLE_CTAS
#define QSB_SHA_STAGE_SINGLE_CTAS 12
#endif
#ifndef QSB_STAGE_RELOAD_ID
#define QSB_STAGE_RELOAD_ID 1
#endif
#if QSB_SHA_STAGE
static_assert(QSB_PAIR_SHARED && ZLAB_DUAL_EPOCH_SHA && ZLAB_K2S3M,
              "SHA staging requires the promoted paired path");
#define QSB_STAGE_ARG(p) ,p

// Source lead: Saviour1001 PR1072. Re-read inexpensive CUDA identities after
// long field calls instead of keeping nomination/address identities live.
__device__ __forceinline__ unsigned qsb_stage_tid() {
#if defined(__CUDA_ARCH__) && QSB_STAGE_RELOAD_ID
    unsigned v; asm volatile("mov.u32 %0, %%tid.x;" : "=r"(v)); return v;
#else
    return threadIdx.x;
#endif
}
__device__ __forceinline__ unsigned qsb_stage_bid() {
#if defined(__CUDA_ARCH__) && QSB_STAGE_RELOAD_ID
    unsigned v; asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(v)); return v;
#else
    return blockIdx.x;
#endif
}
template<bool SECOND>
__device__ __forceinline__ unsigned qsb_stage_epoch() {
    return QSB_PAIR_MUL*qsb_stage_bid()
        +2u*(qsb_stage_tid()/(unsigned)QSB_SE_WINDOWS)+(SECOND?1u:0u);
}
template<bool SECOND>
__device__ __forceinline__ bool qsb_stage_active(unsigned batch,unsigned epochs) {
    return qsb_stage_bid()*QSB_SE_BLOCK+qsb_stage_tid()<batch
        && qsb_stage_epoch<SECOND>()<epochs;
}

#if !QSB_SHA_STAGE_SINGLE
__global__ void __launch_bounds__(256,QSB_SHA_STAGE_CTAS) kernel_stage_sighash(
    const uint32_t * __restrict__ first,uint64_t * __restrict__ staged_z,unsigned epochs) {
    const unsigned idx=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned epoch=2u*(idx/(unsigned)QSB_SE_WINDOWS);
    const unsigned lane=idx&((unsigned)QSB_SE_WINDOWS-1u);
    if(epoch>=epochs)return;
    const uint32_t *a=first+(size_t)epoch*QSB_FIRST_SLOTS*8;
    const bool hasB=epoch+1u<epochs;
    const uint32_t *b=hasB?a+QSB_FIRST_SLOTS*8:a;
    QsbPairEpochZ z=qsb_pair_epoch_z_value(a,b,(int)lane);
    #pragma unroll
    for(int k=0;k<4;k++) {
        staged_z[((size_t)epoch*4+k)*QSB_SE_WINDOWS+lane]=z.a[k];
        if(hasB)staged_z[((size_t)(epoch+1u)*4+k)*QSB_SE_WINDOWS+lane]=z.b[k];
    }
}
#else
// One exact epoch/window hash per thread. Keep the existing per-epoch plane
// layout; do not copy the donor's per-consumer-block layout or MAC arithmetic.
__global__ void __launch_bounds__(128,QSB_SHA_STAGE_SINGLE_CTAS)
kernel_stage_sighash_single(const uint32_t * __restrict__ first,
    uint64_t * __restrict__ staged_z,unsigned epochs) {
    const unsigned idx=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned epoch=idx/(unsigned)QSB_SE_WINDOWS;
    const unsigned lane=idx&((unsigned)QSB_SE_WINDOWS-1u);
    if(epoch>=epochs)return;
    const uint32_t *f=first+(size_t)epoch*QSB_FIRST_SLOTS*8;
    uint32_t state[8];uint64_t z[4];
    qsb_scheduled_window_hash(state,nullptr,(int)lane,f);
    qsb_pair_second_sha_z(state,z);
    #pragma unroll
    for(int k=0;k<4;k++)
        staged_z[((size_t)epoch*4+k)*QSB_SE_WINDOWS+lane]=z[k];
}
#endif
__device__ __forceinline__ void qsb_load_staged_z(uint64_t z[4],
    const uint64_t * __restrict__ staged_z,unsigned epoch,unsigned lane) {
    #pragma unroll
    for(int k=0;k<4;k++) {
        z[k]=staged_z[((size_t)epoch*4+k)*QSB_SE_WINDOWS+lane];
    }
}
template<bool SECOND>
__device__ __forceinline__ void qsb_load_staged_consumer_z(uint64_t z[4],
    const uint64_t * __restrict__ staged_z,unsigned epochs) {
    const unsigned ea=qsb_stage_epoch<false>();
    const unsigned a=ea<epochs?ea:0u;
    const unsigned e=SECOND && ea+1u<epochs?a+1u:a;
    qsb_load_staged_z(z,staged_z,e,qsb_stage_tid()&(QSB_SE_WINDOWS-1u));
}
#else
#define QSB_STAGE_ARG(p)
#endif
