// SPDX-License-Identifier: GPL-3.0-only
#include "inverse_service.cuh"
struct IdentityInverse {
  __device__ void operator()(uint64_t out[4],const uint64_t in[4]) const {
    for(int j=0;j<4;++j)out[j]=in[j];
  }
};
__global__ __launch_bounds__(128,4) void qsb_inverse_service_compile_probe(
    qsb_inverse_service::Slot *slots,unsigned workers,unsigned services) {
    const unsigned worker=blockIdx.x-services;
    if(blockIdx.x<services) {
        qsb_inverse_service::service(slots,workers,blockIdx.x*blockDim.x+threadIdx.x,IdentityInverse{});
        return;
    }
    if(worker<workers && threadIdx.x==0) {
        const uint64_t root[4]={1,2,3,4};uint64_t inverse[4];
        qsb_inverse_service::worker_publish(slots[worker],1,root);
        qsb_inverse_service::worker_wait(slots[worker],1,inverse);
        qsb_inverse_service::worker_stop(slots[worker]);
    }
}
