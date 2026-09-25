#pragma once
#include <stdint.h>
/* Exact direct11 codebook. For normalized k choose signed m<=floor(n/2).
 * Widths sum to255; top f<=2^27-1<T=2^27+1 excludes final doubling.
 * K=(T+1)*2^227-2^16. Biased first and ten odd terms telescope to m.
 * Four hot banks occupy32MiB. Record index<2^29; bit31 is only the sign.
 * All portable helpers are independently CPU checked against big integers. */
__host__ __device__ __forceinline__ unsigned qsb_d11_entries(int c) {
    return c==0 ? 131072u : c==1 ? 131072u : c==2 ? 131072u : c==3 ? 131072u : c==4 ? 33554432u : c==5 ? 33554432u : c==6 ? 33554432u : c==7 ? 33554432u : c==8 ? 33554432u : c==9 ? 67108864u : 67108865u;
}
__host__ __device__ __forceinline__ unsigned qsb_d11_offset(int c) {
    return c==0 ? 0u : c==1 ? 131072u : c==2 ? 262144u : c==3 ? 393216u : c==4 ? 524288u : c==5 ? 34078720u : c==6 ? 67633152u : c==7 ? 101187584u : c==8 ? 134742016u : c==9 ? 168296448u : 235405312u;
}
__host__ __device__ __forceinline__ unsigned qsb_d11_shift(int c) {
    return c==0 ? 0u : c==1 ? 17u : c==2 ? 35u : c==3 ? 53u : c==4 ? 71u : c==5 ? 97u : c==6 ? 123u : c==7 ? 149u : c==8 ? 175u : c==9 ? 201u : 228u;
}
__host__ __device__ __forceinline__ unsigned qsb_d11_ge(const uint64_t a[4],const uint64_t b[4]) {
    for(int i=3;i>=0;i--) {if(a[i]!=b[i])return a[i]>b[i];}
    return 1u;
}
__host__ __device__ __forceinline__ void qsb_d11_sub(uint64_t out[4],const uint64_t a[4],const uint64_t b[4]) {
    uint64_t borrow=0;
    #pragma unroll
    for(int i=0;i<4;i++) {
        const uint64_t ai=a[i],bi=b[i],t=ai-bi;
        out[i]=t-borrow;borrow=(ai<bi)|(t<borrow);
    }
}
__host__ __device__ __forceinline__ void qsb_d11_center(const uint64_t input[4],uint64_t mag[4],unsigned *sign) {
    const uint64_t n[4]={0xbfd25e8cd0364141ULL,0xbaaedce6af48a03bULL,0xfffffffffffffffeULL,0xffffffffffffffffULL};
    const uint64_t halfplus[4]={0xdfe92f46681b20a1ULL,0x5d576e7357a4501dULL,0xffffffffffffffffULL,0x7fffffffffffffffULL};
    uint64_t k[4];
    #pragma unroll
    for(int i=0;i<4;i++)k[i]=input[i];
    if(qsb_d11_ge(k,n))qsb_d11_sub(k,k,n);
    *sign=qsb_d11_ge(k,halfplus);
    if(*sign)qsb_d11_sub(mag,n,k);
    else {
        #pragma unroll
        for(int i=0;i<4;i++)mag[i]=k[i];
    }
}
__host__ __device__ __forceinline__ uint32_t qsb_d11_code(const uint64_t mag[4],unsigned sign,int c) {
    const unsigned shift=qsb_d11_shift(c),limb=shift>>6,bit=shift&63;
    uint64_t field=mag[limb]>>bit;
    if(bit && limb<3)field|=mag[limb+1]<<(64-bit);
    const unsigned width=c<10?qsb_d11_shift(c+1)-shift:27u;
    const uint32_t f=(uint32_t)field&((1u<<width)-1u);
    const uint32_t d=2u*f+1u-(c==0?0u:c==10?134217730u:1u<<width);
    const uint32_t nm=0u-(d>>31),idx=((d^nm)-nm-1u)>>1;
    return (qsb_d11_offset(c)+idx)|(((nm&1u)^sign)<<31);
}
#define QSB_D11_DESC_INIT { \
    {0x1ffffu,0u,0u,17u}, \
    {0x3ffffu,262144u,131072u,18u}, \
    {0x3ffffu,262144u,262144u,18u}, \
    {0x3ffffu,262144u,393216u,18u}, \
    {0x3ffffffu,67108864u,524288u,26u}, \
    {0x3ffffffu,67108864u,34078720u,26u}, \
    {0x3ffffffu,67108864u,67633152u,26u}, \
    {0x3ffffffu,67108864u,101187584u,26u}, \
    {0x3ffffffu,67108864u,134742016u,26u}, \
    {0x7ffffffu,134217728u,168296448u,27u}, \
    {0x7ffffffu,134217730u,235405312u,27u} \
}
