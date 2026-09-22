// SPDX-License-Identifier: GPL-3.0-only
// Separate the unchanged paired sighash from curve recovery.
#pragma once
#ifndef QSB_SHA_STAGE
#define QSB_SHA_STAGE 1
#endif
#ifndef QSB_SHA_STAGE_CTAS
#define QSB_SHA_STAGE_CTAS 4
#endif
#if QSB_SHA_STAGE
static_assert(QSB_PAIR_SHARED && ZLAB_DUAL_EPOCH_SHA && ZLAB_K2S3M,
              "SHA staging requires the promoted paired path");
#define QSB_STAGE_ARG(p) ,p

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
__device__ __forceinline__ void qsb_load_staged_z(uint64_t z[4],
    const uint64_t * __restrict__ staged_z,unsigned epoch,unsigned lane) {
    #pragma unroll
    for(int k=0;k<4;k++) {
        z[k]=staged_z[((size_t)epoch*4+k)*QSB_SE_WINDOWS+lane];
    }
}
#else
#define QSB_STAGE_ARG(p)
#endif
