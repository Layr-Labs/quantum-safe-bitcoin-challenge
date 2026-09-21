// Host must fill canonical x, +y, -y for [65535*2^240]A.
// A is this problem's neg_r_inv*G. No fixed problem data is embedded.
// Return 1 requires an explicit P=infinity recovery path in the caller.
#pragma once
__device__ __constant__ uint64_t qsb_serial16_special_xy[12];
__device__ __forceinline__ unsigned qsb_serial16_classify(const uint64_t *M,unsigned negative) {
    if (M[0]==0x402da1732fc9bebfULL && M[1]==0x4551231950b75fc4ULL && M[2]==0x0000000000000001ULL && M[3]==0x0000000000000000ULL && negative==1u) return 1u;
    if (M[0]==0x402da1732fc9bebfULL && M[1]==0x4551231950b75fc4ULL && M[2]==0x0000000000000001ULL && M[3]==0xfffe000000000000ULL && negative==0u) return 2u;
    if (M[0]==0xbfd25e8cd0364141ULL && M[1]==0xbaaedce6af48a03bULL && M[2]==0xfffffffffffffffeULL && M[3]==0x0001ffffffffffffULL && negative==1u) return 3u;
    return 0u;
}
__device__ __forceinline__ void qsb_serial16_special_point(unsigned kind,
    uint64_t *X,uint64_t *Y,uint64_t *U,uint64_t *V) {
    #pragma unroll
    for (int i=0;i<4;i++) {
        X[i]=kind==1u?0u:qsb_serial16_special_xy[i];
        Y[i]=kind==1u?uint64_t(i==0):qsb_serial16_special_xy[(kind==2u?4:8)+i];
        U[i]=V[i]=uint64_t(kind!=1u && i==0);
    }
}
