// SPDX-License-Identifier: GPL-3.0-only
#include "inverse_service.cuh"
#define QSB_C31 1
#define QSB_HOST_GATE 1
#include "../GPUMath.h"
#ifndef QSB_SERVICE_BLOCKS
#define QSB_SERVICE_BLOCKS 4
#endif
__device__ __forceinline__ void service_normalize(uint64_t *r) {
    if((r[1]&r[2]&r[3])==UINT64_MAX && r[0]>=UINT64_C(0xfffffffefffffc2f)) {
        r[0]-=UINT64_C(0xfffffffefffffc2f);r[1]=r[2]=r[3]=0;
    }
}
struct BinaryInverse {
    __device__ void operator()(uint64_t out[4],const uint64_t in[4]) const {
        uint64_t work[5]={in[0],in[1],in[2],in[3],0};
        service_normalize(work);_ModInv(work);service_normalize(work);
        for(int i=0;i<4;++i)out[i]=work[i];
    }
};
__global__ __launch_bounds__(128,QSB_SERVICE_BLOCKS) void qsb_inverse_service_math_compile_probe(
    qsb_inverse_service::Slot *slots,unsigned workers,unsigned services) {
    const unsigned worker=blockIdx.x-services;
    if(blockIdx.x<services) {
        qsb_inverse_service::service(slots,workers,blockIdx.x*blockDim.x+threadIdx.x,BinaryInverse{});
        return;
    }
    if(worker<workers && threadIdx.x==0) {
        const uint64_t root[4]={2,3,4,5};uint64_t inverse[4];
        qsb_inverse_service::worker_publish(slots[worker],1,root);
        qsb_inverse_service::worker_wait(slots[worker],1,inverse);
        qsb_inverse_service::worker_stop(slots[worker]);
    }
}
