#pragma once
#include "compact_geometry.cuh"
#define COMPACT_CHUNKS MIXED_CHUNKS
#define COMPACT_TOTAL_ENTRIES ((uint32_t)MIXED_TOTAL_ENTRIES)
#define COMPACT_LO 8192
#define COMPACT_HI 8192
__host__ __device__ __forceinline__ unsigned compact_entries(int c){return mixed_entries(c);}
__host__ __device__ __forceinline__ unsigned compact_offset(int c){return mixed_offset(c);}
__host__ __device__ __forceinline__ int compact_shift(int c){return mixed_shift(c);}


__device__ __forceinline__ int32_t compact_recode_step(uint64_t M[4],int sign,int c){
    return c<COMPACT_CHUNKS-1?mixed_step(M,sign,mixed_bits(c)):sign*(int32_t)M[0];
}
__device__ __forceinline__ void compact_recode_signed(const uint64_t k[4],int32_t e[COMPACT_CHUNKS]){
    uint64_t M[4];int sign;gt_recode_setup(k,M,&sign);
    for(int c=0;c<COMPACT_CHUNKS;++c)e[c]=compact_recode_step(M,sign,c);
}
__device__ __forceinline__ void compact_load_signed(const uint8_t *gTX, const uint8_t *gTY,
                                                int c, uint32_t idx, uint64_t neg,
                                                uint64_t gx[4], uint64_t gy[4]) {
    size_t off = ((size_t)compact_offset(c) + idx) * 64;
    (void)gTY;
    const ulonglong2 *tx=(const ulonglong2 *)(gTX+off), *ty=(const ulonglong2 *)(gTX+off+32);
    ulonglong2 x0=tx[0],x1=tx[1],y0=ty[0],y1=ty[1];
    gx[0]=x0.x;gx[1]=x0.y;gx[2]=x1.x;gx[3]=x1.y;
    uint64_t y[4]={y0.x,y0.y,y1.x,y1.y}, yn[4];
    _ModNeg256(yn, y);                                 /* p - y */
    uint64_t m = 0 - neg;                              /* all-ones if negative digit */
    gy[0]=(y[0]&~m)|(yn[0]&m); gy[1]=(y[1]&~m)|(yn[1]&m);
    gy[2]=(y[2]&~m)|(yn[2]&m); gy[3]=(y[3]&~m)|(yn[3]&m);
}
__device__ __forceinline__ void compact_prefetch_point(const uint8_t *table,int c,uint32_t idx){
    size_t off=((size_t)compact_offset(c)+idx)*64;
    asm volatile("prefetch.global.L2 [%0];" :: "l"(table+off));
    asm volatile("prefetch.global.L2 [%0];" :: "l"(table+off+32));
}
__device__ void compact_fixed_xyzz(uint64_t *X, uint64_t *Y, uint64_t *ZZ, uint64_t *ZZZ,
                                      const uint64_t scalar[4], const uint8_t *gTX, const uint8_t *gTY) {
    uint64_t M[4]; int sign; gt_recode_setup(scalar,M,&sign);
    uint32_t idx; uint64_t neg;
    uint64_t x0[4],y0[4],x1[4],y1[4];
    gt_digit_idx(compact_recode_step(M,sign,0), &idx, &neg);
    compact_load_signed(gTX,gTY,0,idx,neg,x0,y0);
    gt_digit_idx(compact_recode_step(M,sign,1), &idx, &neg);
    compact_load_signed(gTX,gTY,1,idx,neg,x1,y1);

    gt_digit_idx(compact_recode_step(M,sign,2), &idx, &neg);
    compact_prefetch_point(gTX,2,idx);
    _PointAddXYZZ_mm(X,Y,ZZ,ZZZ, x0,y0, x1,y1);
    uint64_t cx[4],cy[4];
    #pragma unroll 1
    for(int c=2;c<COMPACT_CHUNKS-1;++c){
        compact_load_signed(gTX,gTY,c,idx,neg,cx,cy);
        gt_digit_idx(compact_recode_step(M,sign,c+1), &idx, &neg);
        compact_prefetch_point(gTX,c+1,idx);
        _PointAddXYZZ<true>(X,Y,ZZ,ZZZ,cx,cy,y0);
        Load256(y0,cy);
    }
    compact_load_signed(gTX,gTY,COMPACT_CHUNKS-1,idx,neg,cx,cy);
    _PointAddXYZZ<false>(X,Y,ZZ,ZZZ,cx,cy,y0);
}
