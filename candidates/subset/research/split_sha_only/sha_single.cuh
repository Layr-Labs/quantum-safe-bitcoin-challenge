#pragma once
#if !ZLAB_TRIM || !QSB_PAIR_SHARED || !ZLAB_K2S3M || !ZLAB_DUAL_EPOCH_SHA
#error "Single-SHA research requires the default trimmed paired consumer"
#endif
// One independent candidate per thread. Existing public SHA functions retain
// the same schedule, compression order, feed-forward, and scalar byte order.
__global__ void __launch_bounds__(256,2) kernel_sha_only(const uint32_t *first,
        uint64_t *work, unsigned epochs, unsigned stride) {
    const unsigned id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=stride)return;
    const unsigned epoch=id/QSB_SE_WINDOWS,lane=id%QSB_SE_WINDOWS;
    if(epoch>=epochs)return;
    const uint32_t *f=first+(size_t)epoch*QSB_FIRST_SLOTS*8;
    uint32_t state[8];
    qsb_scheduled_window_hash(state,nullptr,(int)lane,f);
    uint64_t z[4];
    qsb_pair_second_sha_z(state,z);
    #pragma unroll
    for(int j=0;j<4;j++)work[(size_t)j*stride+id]=z[j];
}
