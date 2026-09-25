#pragma once
#if !ZLAB_TRIM || !QSB_PAIR_SHARED || !ZLAB_K2S3M || !ZLAB_DUAL_EPOCH_SHA
#error "SHA-only research requires the default trimmed paired configuration"
#endif
__global__ void __launch_bounds__(256,2) kernel_sha_only(const uint32_t *first,
        uint64_t *work, unsigned epochs, unsigned stride) {
    const unsigned tid=threadIdx.x,lane=tid&(QSB_SE_WINDOWS-1);
    const unsigned ea=QSB_PAIR_MUL*blockIdx.x+2*(tid/QSB_SE_WINDOWS);
    if(ea>=epochs)return;
    const bool hasB=ea+1<epochs;
    const uint32_t *fa=first+(size_t)ea*QSB_FIRST_SLOTS*8;
    const uint32_t *fb=hasB?fa+QSB_FIRST_SLOTS*8:fa;
    const QsbPairEpochZ z=qsb_pair_epoch_z_value(fa,fb,lane);
    const unsigned ia=ea*QSB_SE_WINDOWS+lane;
    #pragma unroll
    for(int j=0;j<4;j++){
        work[(size_t)j*stride+ia]=z.a[j];
        if(hasB)work[(size_t)j*stride+ia+QSB_SE_WINDOWS]=z.b[j];
    }
}
