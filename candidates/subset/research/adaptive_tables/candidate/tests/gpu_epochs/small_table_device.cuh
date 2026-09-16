#pragma once
#define SMALL_CHUNKS 16
#define SMALL_ENTRIES (1u<<15)
#define SMALL_LO 256
#define SMALL_HI 256
__device__ __forceinline__ int32_t small_recode_step(uint64_t M[4], int sign, int c) {
    if (c < 15) {
        int32_t digit = (int32_t)(uint32_t)(M[0] & 0x1FFFFu) - 65536;   /* odd */
        uint64_t r0=(M[0]>>17)|(M[1]<<47);
        uint64_t r1=(M[1]>>17)|(M[2]<<47);
        uint64_t r2=(M[2]>>17)|(M[3]<<47);
        uint64_t r3=(M[3]>>17);
        M[0]=(r0<<1)|1ULL; M[1]=(r1<<1)|(r0>>63); M[2]=(r2<<1)|(r1>>63); M[3]=(r3<<1)|(r2>>63);
        return sign*digit;
    }
    return sign*(int32_t)M[0];   /* c==15: remainder < 2^16, odd */
}
__device__ __forceinline__ void small_recode_signed(const uint64_t k[4], int32_t e[16]) {
    uint64_t M[4]; int sign; gt_recode_setup(k, M, &sign);
    #pragma unroll
    for (int c=0;c<16;c++) e[c]=small_recode_step(M, sign, c);
}
__device__ __forceinline__ void small_load_signed(const uint8_t *gTX, const uint8_t *gTY,
                                                int c, uint32_t idx, uint64_t neg,
                                                uint64_t gx[4], uint64_t gy[4]) {
    size_t off = ((size_t)c * SMALL_ENTRIES + idx) * 32;
    const ulonglong2 *tx=(const ulonglong2 *)(gTX+off), *ty=(const ulonglong2 *)(gTY+off);
    ulonglong2 x0=tx[0],x1=tx[1],y0=ty[0],y1=ty[1];
    gx[0]=x0.x;gx[1]=x0.y;gx[2]=x1.x;gx[3]=x1.y;
    uint64_t y[4]={y0.x,y0.y,y1.x,y1.y}, yn[4];
    _ModNeg256(yn, y);                                 /* p - y */
    uint64_t m = 0 - neg;                              /* all-ones if negative digit */
    gy[0]=(y[0]&~m)|(yn[0]&m); gy[1]=(y[1]&~m)|(yn[1]&m);
    gy[2]=(y[2]&~m)|(yn[2]&m); gy[3]=(y[3]&~m)|(yn[3]&m);
}
__device__ void small_fixed_xyzz(uint64_t *X, uint64_t *Y, uint64_t *ZZ, uint64_t *ZZZ,
                                      const uint64_t scalar[4], const uint8_t *gTX, const uint8_t *gTY) {
    uint64_t M[4]; int sign; gt_recode_setup(scalar,M,&sign);
    uint32_t idx; uint64_t neg;
    uint64_t x0[4],y0[4],x1[4],y1[4];
    gt_digit_idx(small_recode_step(M,sign,0), &idx, &neg);
    small_load_signed(gTX,gTY,0,idx,neg,x0,y0);
    gt_digit_idx(small_recode_step(M,sign,1), &idx, &neg);
    small_load_signed(gTX,gTY,1,idx,neg,x1,y1);
    _PointAddXYZZ_mm(X,Y,ZZ,ZZZ, x0,y0, x1,y1);
    uint64_t cx[4],cy[4];
    #pragma unroll 1
    for (int c=2;c<SMALL_CHUNKS-1;c++){
        gt_digit_idx(small_recode_step(M,sign,c), &idx, &neg);
        small_load_signed(gTX,gTY,c,idx,neg,cx,cy);
        _PointAddXYZZ<true>(X,Y,ZZ,ZZZ, cx,cy, y0);
        Load256(y0, cy);
    }
    gt_digit_idx(small_recode_step(M,sign,SMALL_CHUNKS-1), &idx, &neg);
    small_load_signed(gTX,gTY,SMALL_CHUNKS-1,idx,neg,cx,cy);
    _PointAddXYZZ<false>(X,Y,ZZ,ZZZ, cx,cy, y0);
}
