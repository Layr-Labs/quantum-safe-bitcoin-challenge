/* qsb_real_search.cu — Real pinning search with sequence + locktime variation
 *
 * Reads pinning2.bin (midstate with sequence in suffix)
 * Loops: outer=sequence (0x80000000+), inner=locktime (500000000-1744600000)
 * 4 DER checks per candidate (2 recovery flags × 2 hashes)
 *
 * Build:  nvcc -O3 -o qsb_real qsb_real_search.cu -lcrypto -lm
 * Usage:  ./qsb_real pinning2.bin [easy]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <vector>

#include "GPUMath.h"

static_assert(sizeof(ulonglong2) == 16, "pipeline vector must be 128 bits");
static_assert(alignof(ulonglong2) == 16, "pipeline vector must be 16-byte aligned");

#define MAX_LEN_WORD_PRIME 20
#define MAX_LEN_WORD_AFFIX 4
#define AFFIX_IS_SUFFIX true
#define SIZE_COMBO_MULTI 4
#define COUNT_COMBO_SYMBOLS 100
#define IDX_CUDA_THREAD ((blockIdx.x * blockDim.x) + threadIdx.x)

__device__ __constant__ int MULTI_EIGHT[65] = { 0,
    0+8,0+16,0+24,0+32,0+40,0+48,0+56,0+64,
    64+8,64+16,64+24,64+32,64+40,64+48,64+56,64+64,
    128+8,128+16,128+24,128+32,128+40,128+48,128+56,128+64,
    192+8,192+16,192+24,192+32,192+40,192+48,192+56,192+64,
    256+8,256+16,256+24,256+32,256+40,256+48,256+56,256+64,
    320+8,320+16,320+24,320+32,320+40,320+48,320+56,320+64,
    384+8,384+16,384+24,384+32,384+40,384+48,384+56,384+64,
    448+8,448+16,448+24,448+32,448+40,448+48,448+56,448+64,
};
__device__ __constant__ uint8_t COMBO_SYMBOLS[100] = {
    0x30,0x31,0x32,0x33,0x34,0x35,0x36,0x37,0x38,0x39,
    0x20,0x21,0x22,0x23,0x24,0x25,0x26,0x27,0x28,0x29,0x2A,0x2B,0x2C,0x2D,0x2E,0x2F,
    0x3A,0x3B,0x3C,0x3D,0x3E,0x3F,0x40,0x5B,0x5C,0x5D,0x5E,0x5F,0x60,0x7B,0x7C,0x7D,0x7E,
    0x41,0x42,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4A,0x4B,0x4C,0x4D,0x4E,0x4F,0x50,0x51,0x52,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5A,
    0x61,0x62,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x6B,0x6C,0x6D,0x6E,0x6F,0x70,0x71,0x72,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7A,
    0x00,0x7F,0xFF,0x09,0x0D
};

#include "GPUHash.h"

/* Ten signed odd windows [26x6,25x4], 16 GiB runtime-specific table.
 * Keeps PR64 streamed recoding, deferred XYZZ and no-alias contracts. */
#include "wide_geometry.cuh"
#define GT_CHUNKS WIDE_CHUNKS
#define GT_TOTAL_ENTRIES ((uint32_t)WIDE_TOTAL_ENTRIES)
#define GT_LO 8192
#define GT_HI 8192
__host__ __device__ __forceinline__ unsigned gt_entries(int c){return wide_entries(c);}
__host__ __device__ __forceinline__ unsigned gt_offset(int c){return wide_offset(c);}
__host__ __device__ __forceinline__ int gt_shift(int c){return wide_shift(c);}

/* n = secp256k1 group order, little-endian limbs */
__device__ __constant__ uint64_t GT_ORDER_N[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};

/* k -> 15 signed odd digits. Branchless (no data-dependent BRA) so warps stay
 * convergent; correctness mirrored on CPU by the same source. */
/* Recode state: the odd 2k-representative M (4 limbs) plus a global sign.
 * gt_recode_setup computes it once; gt_mixed_step peels one signed odd digit
 * per chunk and advances M. The window multiply carries this 32-byte state and
 * peels digits on the fly, so the 15-entry digit array never materialises
 * (that array was the largest single spill source). gt_recode_signed keeps the
 * array form for the CPU cross-check; both share the same step logic. */
__device__ __forceinline__ void gt_recode_setup(const uint64_t k[4], uint64_t M[4], int *sign) {
    const uint64_t n0=GT_ORDER_N[0], n1=GT_ORDER_N[1], n2=GT_ORDER_N[2], n3=GT_ORDER_N[3];
    __uint128_t s;
    /* A raw SHA scalar is at least n with probability (2^256-n)/2^256. Keep
     * that exact case, but let the overwhelmingly common path avoid a
     * four-limb subtract and four selects. */
    uint64_t k0=k[0], k1=k[1], k2=k[2], k3=k[3];
    if (k3 == n3 &&
        (k2 > n2 ||
         (k2 == n2 && (k1 > n1 || (k1 == n1 && k0 >= n0))))) {
        s=(__uint128_t)k0-n0; k0=(uint64_t)s; uint64_t kb=(uint64_t)(s>>64)&1;
        s=(__uint128_t)k1-n1-kb; k1=(uint64_t)s; kb=(uint64_t)(s>>64)&1;
        s=(__uint128_t)k2-n2-kb; k2=(uint64_t)s; kb=(uint64_t)(s>>64)&1;
        s=(__uint128_t)k3-n3-kb; k3=(uint64_t)s;
    }
    uint64_t t0=k0<<1;
    uint64_t t1=(k1<<1)|(k0>>63);
    uint64_t t2=(k2<<1)|(k1>>63);
    uint64_t t3=(k3<<1)|(k2>>63);
    uint64_t tc=(k3>>63);
    s=(__uint128_t)t0-n0;    uint64_t d0=(uint64_t)s; uint64_t br=(s>>64)&1;
    s=(__uint128_t)t1-n1-br; uint64_t d1=(uint64_t)s; br=(s>>64)&1;
    s=(__uint128_t)t2-n2-br; uint64_t d2=(uint64_t)s; br=(s>>64)&1;
    s=(__uint128_t)t3-n3-br; uint64_t d3=(uint64_t)s; br=(s>>64)&1;
    uint64_t ge = tc | (1u - (uint64_t)br);
    uint64_t gm = 0 - ge;
    uint64_t m0=(t0&~gm)|(d0&gm), m1=(t1&~gm)|(d1&gm), m2=(t2&~gm)|(d2&gm), m3=(t3&~gm)|(d3&gm);
    uint64_t odd = m0 & 1ULL;
    s=(__uint128_t)n0-m0;    uint64_t p0=(uint64_t)s; br=(s>>64)&1;
    s=(__uint128_t)n1-m1-br; uint64_t p1=(uint64_t)s; br=(s>>64)&1;
    s=(__uint128_t)n2-m2-br; uint64_t p2=(uint64_t)s; br=(s>>64)&1;
    s=(__uint128_t)n3-m3-br; uint64_t p3=(uint64_t)s;
    uint64_t om = 0 - odd;
    M[0]=(m0&om)|(p0&~om); M[1]=(m1&om)|(p1&~om); M[2]=(m2&om)|(p2&~om); M[3]=(m3&om)|(p3&~om);
    *sign = (int)odd*2 - 1;
}

template<int BITS>
__device__ __forceinline__ int32_t gt_mixed_step(uint64_t M[4], int sign) {
    int32_t digit=(int32_t)(M[0]&((1u<<(BITS+1))-1))-(1<<BITS);
    uint64_t r0=(M[0]>>(BITS+1))|(M[1]<<(63-BITS));
    uint64_t r1=(M[1]>>(BITS+1))|(M[2]<<(63-BITS));
    uint64_t r2=(M[2]>>(BITS+1))|(M[3]<<(63-BITS));
    uint64_t r3=M[3]>>(BITS+1);
    M[0]=(r0<<1)|1ULL; M[1]=(r1<<1)|(r0>>63);
    M[2]=(r2<<1)|(r1>>63); M[3]=(r3<<1)|(r2>>63);
    return sign*digit;
}
__device__ __forceinline__ void gt_recode_signed(const uint64_t k[4],int32_t e[GT_CHUNKS]){
    uint64_t M[4];int sign;gt_recode_setup(k,M,&sign);
    #pragma unroll
    for(int c=0;c<GT_CHUNKS-1;c++)e[c]=wide_step(M,sign,wide_bits(c));
    e[GT_CHUNKS-1]=sign*(int32_t)M[0];
}

/* Load table point (c, idx) into (gx,gy); negate y (p - y) when neg_mask != 0.
 * Branchless: y is selected between y and p-y by a mask. */
__device__ __forceinline__ void gt_load_signed_flat(const uint8_t *__restrict__ gTable,
                                                     uint32_t base, uint32_t idx,
                                                     uint64_t neg_mask,
                                                     uint64_t *__restrict__ gx,
                                                     uint64_t *__restrict__ gy) {
    size_t off = ((size_t)base + idx) * 64;
    const ulonglong2 *tx=(const ulonglong2 *)(gTable+off);
    const ulonglong2 *ty=(const ulonglong2 *)(gTable+off+32);
    ulonglong2 x0=__ldg(tx),x1=__ldg(tx+1),y0=__ldg(ty),y1=__ldg(ty+1);
    gx[0]=x0.x;gx[1]=x0.y;gx[2]=x1.x;gx[3]=x1.y;
    uint64_t m=neg_mask;
    uint64_t r0=y0.x^m, r1=y0.y^m, r2=y1.x^m, r3=y1.y^m;
    uint64_t c0=0xFFFFFFFEFFFFFC30ULL&m;
    UADDO1(r0,c0); UADDC1(r1,m); UADDC1(r2,m); UADD1(r3,m);
    gy[0]=r0; gy[1]=r1; gy[2]=r2; gy[3]=r3;
}

__device__ __forceinline__ void gt_load_signed(const uint8_t *gTable,
                                                int c, uint32_t idx, uint64_t neg,
                                                uint64_t gx[4], uint64_t gy[4]) {
    gt_load_signed_flat(gTable, gt_offset(c), idx, neg, gx, gy);
}

/* Decode one signed table digit. */
__device__ __forceinline__ void gt_digit_idx(int32_t ec, uint32_t *idx, uint64_t *neg) {
    int32_t mask = ec >> 31;
    uint32_t ae = ((uint32_t)ec ^ (uint32_t)mask) - (uint32_t)mask;
    *idx = (ae - 1) >> 1;
    *neg = (uint64_t)(int64_t)mask;
}

/* Signed-digit fixed-base multiply, accumulating INTERNALLY in XYZZ (x=X/ZZ,
 * y=Y/ZZZ). Seed the first two chunks with a deferred-Y mmadd (3M+2S), adjust
 * each next point's y by the preceding affine anchor, and defer the new anchor
 * term through every intermediate addition. Only the final addition resolves
 * Y exactly. This costs 3M+2S + 7*(7M+2S) + 8M+2S = 60M+18S across all
 * 10 points; the former15-point chain cost95M+28S. Fully unrolling
 * inlines the asm multiply ~150x past ptxas' budget); the back-edge is a
 * uniform loop-counter
 * branch, and every signed odd digit is non-zero so there is NO data-dependent
 * branch and no chunk is skipped. Next chunk's table point loaded one step ahead.
 *
 * Production consumes the raw XYZZ output directly. This avoids the three
 * field multiplications formerly used to convert it to homogeneous projective
 * form before recovery. */
__device__ void _FixedBaseSignedXYZZ(
    uint64_t *__restrict__ X, uint64_t *__restrict__ Y,
    uint64_t *__restrict__ ZZ, uint64_t *__restrict__ ZZZ,
    const int32_t *__restrict__ e, const uint8_t *__restrict__ gTable) {
    uint32_t idx; uint64_t neg;
    uint64_t x0[4],y0[4],x1[4],y1[4];
    gt_digit_idx(e[0], &idx, &neg); gt_load_signed(gTable,0,idx,neg,x0,y0);
    gt_digit_idx(e[1], &idx, &neg); gt_load_signed(gTable,1,idx,neg,x1,y1);
    _PointAddXYZZ_mm(X,Y,ZZ,ZZZ, x0,y0, x1,y1);
    /* No software prefetch: keeping the next table point live alongside the
     * 128-byte XYZZ accumulator raised spills (92/64 -> measured worse). Load
     * each chunk just-in-time; the loads are still independent (indices known
     * from the recoded digits) so the hardware overlaps them. */
    uint64_t cx[4],cy[4];
    #pragma unroll 1
    for (int c=2;c<GT_CHUNKS-1;c++){
        gt_digit_idx(e[c], &idx, &neg); gt_load_signed(gTable,c,idx,neg,cx,cy);
        _PointAddXYZZ<true>(X,Y,ZZ,ZZZ, cx,cy, y0);
        Load256(y0, cy);                /* current affine y anchors next madd */
    }
    gt_digit_idx(e[GT_CHUNKS-1], &idx, &neg);
    gt_load_signed(gTable,GT_CHUNKS-1,idx,neg,cx,cy);
    _PointAddXYZZ<false>(X,Y,ZZ,ZZZ, cx,cy, y0);
}

/* Production scalar-entry form: consume the mixed signed digits as they are
 * generated instead of materializing an address-taken digit array. */
__device__ void _FixedBaseSignedXYZZScalar(uint64_t *__restrict__ X,uint64_t *__restrict__ Y,uint64_t *__restrict__ ZZ,uint64_t *__restrict__ ZZZ,const uint64_t *__restrict__ k,const uint8_t *__restrict__ table){
    uint64_t M[4];int sign;gt_recode_setup(k,M,&sign);
    uint32_t idx;uint64_t neg;uint64_t x0[4],y0[4],x1[4],y1[4];
    int32_t e=wide_step(M,sign,26);gt_digit_idx(e,&idx,&neg);gt_load_signed_flat(table,wide_offset(0),idx,neg,x0,y0);
    e=wide_step(M,sign,26);gt_digit_idx(e,&idx,&neg);gt_load_signed_flat(table,wide_offset(1),idx,neg,x1,y1);
    _PointAddXYZZ_mm(X,Y,ZZ,ZZZ,x0,y0,x1,y1);
    uint64_t x[4],y[4];unsigned offset=wide_offset(2);
    #pragma unroll 1
    for(int c=2;c<WIDE_CHUNKS-1;++c){
        e=wide_step(M,sign,wide_bits(c));gt_digit_idx(e,&idx,&neg);gt_load_signed_flat(table,offset,idx,neg,x,y);
        _PointAddXYZZ<true>(X,Y,ZZ,ZZZ,x,y,y0);Load256(y0,y);offset+=wide_entries(c);
    }
    e=sign*(int32_t)M[0];gt_digit_idx(e,&idx,&neg);gt_load_signed_flat(table,offset,idx,neg,x,y);
    _PointAddXYZZ<false>(X,Y,ZZ,ZZZ,x,y,y0);
}

/* _FixedBaseSignedAffine: removed -- dead with the diagnostic kernel. */

/* DER checks */
__device__ int gpu_is_valid_der(const uint8_t *d, int l) {
    if(l<9||d[0]!=0x30) return 0;
    int tl=d[1]; if(tl+3!=l) return 0;
    int idx=2;
    for(int p=0;p<2;p++){
        if(idx>=l-1||d[idx]!=0x02) return 0; idx++;
        int il=d[idx]; idx++;
        if(il==0||idx+il>l-1) return 0;
        if(il>1&&d[idx]==0&&!(d[idx+1]&0x80)) return 0;
        if(d[idx]&0x80) return 0; idx+=il;}
    return idx==l-1;
}
__device__ int gpu_is_der_easy(const uint8_t *d, int l) { return l>=9&&(d[0]>>4)==3; }

/* gpu_is_on_curve and gpu_der_r_on_curve: removed -- dead with the diagnostic kernel. */

#ifndef QSB_ZEROS_N
#define QSB_ZEROS_N 24
#endif
__device__ int gpu_leading_zero_bits(const uint8_t *h) {
    int z = 0;
    for (int i = 0; i < 32; i++) {
        if (h[i] == 0) { z += 8; continue; }
        unsigned v = h[i]; int c = 0;
        while ((v & 0x80u) == 0) { c++; v <<= 1; }
        return z + c;
    }
    return z;
}
/* gpu_bench_oncurve: removed -- it has no caller. */
__device__ int gpu_bench_valid(const uint8_t *h) {
    return gpu_leading_zero_bits(h) >= QSB_ZEROS_N;  /* leading-zeros gate only; no on-curve(h) check */
}
/* The digest bytes are the SHA state words in big-endian order, so the ranked
 * leading-zero gate can inspect those words directly. */
__device__ __forceinline__ int gpu_bench_valid_words(const uint32_t *hs) {
    int ok = 1;
    #pragma unroll
    for (int i = 0; i < QSB_ZEROS_N / 32; i++) ok &= (hs[i] == 0u);
#if (QSB_ZEROS_N % 32) != 0
    ok &= ((hs[QSB_ZEROS_N / 32] >> (32 - (QSB_ZEROS_N % 32))) == 0u);
#endif
    return ok;
}


/* ============================================================
 * Kernel: searches locktime range for a fixed sequence value
 * ============================================================ */

/* kernel_debug_pin_one_point: removed from benchmark builds. */

__device__ __forceinline__ void qsb_field_mul(uint64_t *out,uint64_t *a,uint64_t *b){
    uint64_t r0,r1,r2,r3;
    asm(
        "{\n"
        "\t.reg .u32 a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7;\n"
        "\t.reg .u64 e0,e1,e2,e3,e4,e5,e6,e7,o0,o1,o2,o3,o4,o5,o6,t,lc;\n"
        "\t.reg .u32 cy,o15;\n"
        "\t.reg .u32 x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;\n"
        "\t.reg .u32 y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14;\n"
        "\tmov.b64 {a0,a1}, %4;\n"
        "\tmov.b64 {a2,a3}, %5;\n"
        "\tmov.b64 {a4,a5}, %6;\n"
        "\tmov.b64 {a6,a7}, %7;\n"
        "\tmov.b64 {b0,b1}, %8;\n"
        "\tmov.b64 {b2,b3}, %9;\n"
        "\tmov.b64 {b4,b5}, %10;\n"
        "\tmov.b64 {b6,b7}, %11;\n"
        "\tmul.wide.u32 e0, a0, b0; mul.wide.u32 e1, a0, b2; mul.wide.u32 e2, a0, b4; mul.wide.u32 e3, a0, b6;\n"
        "\tmul.wide.u32 t, a1, b1; add.cc.u64 e1, e1, t;\n"
        "\tmul.wide.u32 t, a1, b3; addc.cc.u64 e2, e2, t;\n"
        "\tmul.wide.u32 t, a1, b5; addc.cc.u64 e3, e3, t;\n"
        "\tmul.wide.u32 t, a1, b7; addc.u64 e4, t, 0;\n"
        "\tmul.wide.u32 t, a2, b0; add.cc.u64 e1, e1, t;\n"
        "\tmul.wide.u32 t, a2, b2; addc.cc.u64 e2, e2, t;\n"
        "\tmul.wide.u32 t, a2, b4; addc.cc.u64 e3, e3, t;\n"
        "\tmul.wide.u32 t, a2, b6; addc.cc.u64 e4, e4, t;\n"
        "\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n"
        "\tmul.wide.u32 t, a3, b1; add.cc.u64 e2, e2, t;\n"
        "\tmul.wide.u32 t, a3, b3; addc.cc.u64 e3, e3, t;\n"
        "\tmul.wide.u32 t, a3, b5; addc.cc.u64 e4, e4, t;\n"
        "\tmul.wide.u32 t, a3, b7; addc.u64 e5, t, lc;\n"
        "\tmul.wide.u32 t, a4, b0; add.cc.u64 e2, e2, t;\n"
        "\tmul.wide.u32 t, a4, b2; addc.cc.u64 e3, e3, t;\n"
        "\tmul.wide.u32 t, a4, b4; addc.cc.u64 e4, e4, t;\n"
        "\tmul.wide.u32 t, a4, b6; addc.cc.u64 e5, e5, t;\n"
        "\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n"
        "\tmul.wide.u32 t, a5, b1; add.cc.u64 e3, e3, t;\n"
        "\tmul.wide.u32 t, a5, b3; addc.cc.u64 e4, e4, t;\n"
        "\tmul.wide.u32 t, a5, b5; addc.cc.u64 e5, e5, t;\n"
        "\tmul.wide.u32 t, a5, b7; addc.u64 e6, t, lc;\n"
        "\tmul.wide.u32 t, a6, b0; add.cc.u64 e3, e3, t;\n"
        "\tmul.wide.u32 t, a6, b2; addc.cc.u64 e4, e4, t;\n"
        "\tmul.wide.u32 t, a6, b4; addc.cc.u64 e5, e5, t;\n"
        "\tmul.wide.u32 t, a6, b6; addc.cc.u64 e6, e6, t;\n"
        "\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n"
        "\tmul.wide.u32 t, a7, b1; add.cc.u64 e4, e4, t;\n"
        "\tmul.wide.u32 t, a7, b3; addc.cc.u64 e5, e5, t;\n"
        "\tmul.wide.u32 t, a7, b5; addc.cc.u64 e6, e6, t;\n"
        "\tmul.wide.u32 t, a7, b7; addc.u64 e7, t, lc;\n"
        "\tmul.wide.u32 o0, a0, b1; mul.wide.u32 o1, a0, b3; mul.wide.u32 o2, a0, b5; mul.wide.u32 o3, a0, b7;\n"
        "\tmul.wide.u32 t, a1, b0; add.cc.u64 o0, o0, t;\n"
        "\tmul.wide.u32 t, a1, b2; addc.cc.u64 o1, o1, t;\n"
        "\tmul.wide.u32 t, a1, b4; addc.cc.u64 o2, o2, t;\n"
        "\tmul.wide.u32 t, a1, b6; addc.cc.u64 o3, o3, t;\n"
        "\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n"
        "\tmul.wide.u32 t, a2, b1; add.cc.u64 o1, o1, t;\n"
        "\tmul.wide.u32 t, a2, b3; addc.cc.u64 o2, o2, t;\n"
        "\tmul.wide.u32 t, a2, b5; addc.cc.u64 o3, o3, t;\n"
        "\tmul.wide.u32 t, a2, b7; addc.u64 o4, t, lc;\n"
        "\tmul.wide.u32 t, a3, b0; add.cc.u64 o1, o1, t;\n"
        "\tmul.wide.u32 t, a3, b2; addc.cc.u64 o2, o2, t;\n"
        "\tmul.wide.u32 t, a3, b4; addc.cc.u64 o3, o3, t;\n"
        "\tmul.wide.u32 t, a3, b6; addc.cc.u64 o4, o4, t;\n"
        "\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n"
        "\tmul.wide.u32 t, a4, b1; add.cc.u64 o2, o2, t;\n"
        "\tmul.wide.u32 t, a4, b3; addc.cc.u64 o3, o3, t;\n"
        "\tmul.wide.u32 t, a4, b5; addc.cc.u64 o4, o4, t;\n"
        "\tmul.wide.u32 t, a4, b7; addc.u64 o5, t, lc;\n"
        "\tmul.wide.u32 t, a5, b0; add.cc.u64 o2, o2, t;\n"
        "\tmul.wide.u32 t, a5, b2; addc.cc.u64 o3, o3, t;\n"
        "\tmul.wide.u32 t, a5, b4; addc.cc.u64 o4, o4, t;\n"
        "\tmul.wide.u32 t, a5, b6; addc.cc.u64 o5, o5, t;\n"
        "\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n"
        "\tmul.wide.u32 t, a6, b1; add.cc.u64 o3, o3, t;\n"
        "\tmul.wide.u32 t, a6, b3; addc.cc.u64 o4, o4, t;\n"
        "\tmul.wide.u32 t, a6, b5; addc.cc.u64 o5, o5, t;\n"
        "\tmul.wide.u32 t, a6, b7; addc.u64 o6, t, lc;\n"
        "\tmul.wide.u32 t, a7, b0; add.cc.u64 o3, o3, t;\n"
        "\tmul.wide.u32 t, a7, b2; addc.cc.u64 o4, o4, t;\n"
        "\tmul.wide.u32 t, a7, b4; addc.cc.u64 o5, o5, t;\n"
        "\tmul.wide.u32 t, a7, b6; addc.cc.u64 o6, o6, t;\n"
        "\taddc.u32 o15, 0, 0;\n"
        "\tmov.b64 {x0,x1}, e0;\n"
        "\tmov.b64 {x2,x3}, e1;\n"
        "\tmov.b64 {x4,x5}, e2;\n"
        "\tmov.b64 {x6,x7}, e3;\n"
        "\tmov.b64 {x8,x9}, e4;\n"
        "\tmov.b64 {x10,x11}, e5;\n"
        "\tmov.b64 {x12,x13}, e6;\n"
        "\tmov.b64 {x14,x15}, e7;\n"
        "\tmov.b64 {y1,y2}, o0;\n"
        "\tmov.b64 {y3,y4}, o1;\n"
        "\tmov.b64 {y5,y6}, o2;\n"
        "\tmov.b64 {y7,y8}, o3;\n"
        "\tmov.b64 {y9,y10}, o4;\n"
        "\tmov.b64 {y11,y12}, o5;\n"
        "\tmov.b64 {y13,y14}, o6;\n"
        "\tadd.cc.u32 x1, x1, y1;\n"
        "\taddc.cc.u32 x2, x2, y2;\n"
        "\taddc.cc.u32 x3, x3, y3;\n"
        "\taddc.cc.u32 x4, x4, y4;\n"
        "\taddc.cc.u32 x5, x5, y5;\n"
        "\taddc.cc.u32 x6, x6, y6;\n"
        "\taddc.cc.u32 x7, x7, y7;\n"
        "\taddc.cc.u32 x8, x8, y8;\n"
        "\taddc.cc.u32 x9, x9, y9;\n"
        "\taddc.cc.u32 x10, x10, y10;\n"
        "\taddc.cc.u32 x11, x11, y11;\n"
        "\taddc.cc.u32 x12, x12, y12;\n"
        "\taddc.cc.u32 x13, x13, y13;\n"
        "\taddc.cc.u32 x14, x14, y14;\n"
        "\taddc.u32 x15, x15, o15;\n"
        "\t.reg .u64 r0,r1,r2,r3,h0,h1,h2,h3,f0,f1,f2,f3,g0,g1,g2,g3;\n"
        "\t.reg .u32 f8,g8,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,w0,w1,w2,w3,w4,w5,w6,w7,m0,m1,m2;\n"
        "\tmov.b64 r0, {x0,x1}; mov.b64 r1, {x2,x3}; mov.b64 r2, {x4,x5}; mov.b64 r3, {x6,x7};\n"
        "\tmov.b64 h0, {x8,x9}; mov.b64 h1, {x10,x11}; mov.b64 h2, {x12,x13}; mov.b64 h3, {x14,x15};\n"
        "\tmul.wide.u32 t, x8, 977;  add.cc.u64  f0, r0, t;\n"
        "\tmul.wide.u32 t, x10, 977; addc.cc.u64 f1, r1, t;\n"
        "\tmul.wide.u32 t, x12, 977; addc.cc.u64 f2, r2, t;\n"
        "\tmul.wide.u32 t, x14, 977; addc.cc.u64 f3, r3, t;\n"
        "\taddc.u32 f8, 0, 0;\n"
        "\tmul.wide.u32 t, x9, 977;  add.cc.u64  g0, h0, t;\n"
        "\tmul.wide.u32 t, x11, 977; addc.cc.u64 g1, h1, t;\n"
        "\tmul.wide.u32 t, x13, 977; addc.cc.u64 g2, h2, t;\n"
        "\tmul.wide.u32 t, x15, 977; addc.cc.u64 g3, h3, t;\n"
        "\taddc.u32 g8, 0, 0;\n"
        "\tmov.b64 {z0,z1}, f0;\n"
        "\tmov.b64 {z2,z3}, f1;\n"
        "\tmov.b64 {z4,z5}, f2;\n"
        "\tmov.b64 {z6,z7}, f3;\n"
        "\tmov.b64 {w0,w1}, g0;\n"
        "\tmov.b64 {w2,w3}, g1;\n"
        "\tmov.b64 {w4,w5}, g2;\n"
        "\tmov.b64 {w6,w7}, g3;\n"
        "\tadd.cc.u32  z1, z1, w0;\n"
        "\taddc.cc.u32 z2, z2, w1;\n"
        "\taddc.cc.u32 z3, z3, w2;\n"
        "\taddc.cc.u32 z4, z4, w3;\n"
        "\taddc.cc.u32 z5, z5, w4;\n"
        "\taddc.cc.u32 z6, z6, w5;\n"
        "\taddc.cc.u32 z7, z7, w6;\n"
        "\taddc.cc.u32 z8, f8, w7;\n"
        "\taddc.u32    z9, g8, 0;\n"
        "\tmul.wide.u32 t, z8, 977; mov.b64 {m0,m1}, t;\n"
        "\tmad.lo.u32 m1, z9, 977, m1;\n"
        "\tadd.cc.u32 m1, m1, z8;\n"
        "\taddc.u32 m2, z9, 0;\n"
        "\tadd.cc.u32 z0, z0, m0; addc.cc.u32 z1, z1, m1; addc.cc.u32 z2, z2, m2;\n"
        "\taddc.cc.u32 z3, z3, 0;\n"
        "\taddc.cc.u32 z4, z4, 0;\n"
        "\taddc.cc.u32 z5, z5, 0;\n"
        "\taddc.cc.u32 z6, z6, 0;\n"
        "\taddc.cc.u32 z7, z7, 0;\n"
        "    .reg .u32 cf, k0, k1, v0, v1, v2, v3, v4, v5, v6, v7, borrow;\n"
        "    .reg .pred take;\n"
        "    addc.u32 cf, 0, 0;\n"
        "    mul.lo.u32 k0, cf, 977;\n"
        "    add.cc.u32 z0, z0, k0;\n"
        "    addc.cc.u32 z1, z1, cf;\n"
        "    addc.cc.u32 z2, z2, 0;\n"
        "    addc.cc.u32 z3, z3, 0;\n"
        "    addc.cc.u32 z4, z4, 0;\n"
        "    addc.cc.u32 z5, z5, 0;\n"
        "    addc.cc.u32 z6, z6, 0;\n"
        "    addc.u32 z7, z7, 0;\n"
        "mov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.b64 %2, {z4,z5}; mov.b64 %3, {z6,z7};\n"
        "\t}\n"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
          "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    out[0]=r0;out[1]=r1;out[2]=r2;out[3]=r3;out[4]=0;
}

/* qsb_field_mul is an exact residue in [0,2^256), while _ModInv expects its
 * input below p and callers expect canonical leaf inverses.  Tree-internal
 * products need only congruent representatives, so normalize the root and 256
 * returned leaves instead of all 765 products. */
__device__ __forceinline__ void qsb_field_normalize(uint64_t *r) {
    if ((r[1] & r[2] & r[3]) == UINT64_MAX &&
        r[0] >= 0xFFFFFFFEFFFFFC2FULL) {
        r[0] -= 0xFFFFFFFEFFFFFC2FULL;
        r[1] = r[2] = r[3] = 0;
    }
}

/* qsb_warp_inverse: removed -- it has no caller. */

// Share one inverse across all 256 lanes with a work-efficient binary product
// tree. Each level is packed after the preceding level, and every level stores
// its left half before its right half. Thus both operands of a multiply are
// contiguous across a warp. Products are immutable during the downward pass;
// the 255 internal inverses use a second, smaller packed array. Limb-major
// storage gives adjacent lanes adjacent 64-bit words instead of a 32-byte AoS
// stride. Whole-block participation is required: the caller maps inactive and
// unusable tail lanes to the multiplicative identity before entering here.
__device__ __forceinline__ void qsb_block_inverse(uint64_t *value) {
    __shared__ uint64_t products[4][512];
    __shared__ uint64_t inverses[4][256];
    int tid=threadIdx.x;

    #pragma unroll
    for(int k=0;k<4;k++)products[k][tid]=value[k];
    __syncthreads();

    // Level (offset,count) pairs are (0,256), (256,128), (384,64), ...,
    // (508,2), (510,1). The final root iteration is executed only by lane zero,
    // so it can invert the result immediately without another synchronization.
    int offset=0;
    #pragma unroll 1
    for(int count=256;count>1;count>>=1){
        int half=count>>1;
        if(tid<half){
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++){
                a[k]=products[k][offset+tid];
                b[k]=products[k][offset+half+tid];
            }
            a[4]=b[4]=0;
            qsb_field_mul(out,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)products[k][offset+count+tid]=out[k];
        }
        offset+=count;
        if(count>2)__syncthreads();
    }

    if(tid==0){
        uint64_t root[5];
        #pragma unroll
        for(int k=0;k<4;k++)root[k]=products[k][510];
        root[4]=0;
        qsb_field_normalize(root);
        _ModInv(root);
        #pragma unroll
        for(int k=0;k<4;k++)inverses[k][254]=root[k];
    }
    __syncthreads();

    // If I=1/(L*R), then I*R=1/L and I*L=1/R. One thread per child
    // therefore expands all internal inverse levels with 254 multiplies.
    // Internal inverse index = product index - 256.
    offset=508;
    #pragma unroll 1
    for(int count=2;count<256;count<<=1){
        int half=count>>1;
        if(tid<count){
            int local_parent=tid&(half-1);
            uint64_t parent_inv[5],sibling[5],child_inv[5];
            #pragma unroll
            for(int k=0;k<4;k++){
                parent_inv[k]=inverses[k][offset+count-256+local_parent];
                sibling[k]=products[k][offset+(tid^half)];
            }
            parent_inv[4]=sibling[4]=0;
            qsb_field_mul(child_inv,parent_inv,sibling);
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-256+tid]=child_inv[k];
        }
        offset-=count<<1;
        __syncthreads();
    }

    // The leaf level has no shared inverse destination or following barrier.
    // Its 256 child inverses can be returned directly to the callers.
    uint64_t parent_inv[5],sibling[5];
    #pragma unroll
    for(int k=0;k<4;k++){
        parent_inv[k]=inverses[k][tid&127];
        sibling[k]=products[k][tid^128];
    }
    parent_inv[4]=sibling[4]=0;
    qsb_field_mul(value,parent_inv,sibling);
    qsb_field_normalize(value);
}

#include "warp_checkpoint.cuh"

/* Batch the per-search-CTA roots one level further. Groups of 256 roots use
 * the same checkpointed tree helpers, then one 256-lane CTA batch-inverts all
 * group roots. A full 16M candidate batch therefore executes one _ModInv
 * instead of 65,536 independent inversions. */
__global__ void __launch_bounds__(256,2) qsb_root_group_prepare(
    const uint64_t *roots, int count, uint64_t *super_roots,
    uint64_t *root_checkpoint
) {
    int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    bool active=i<count;
    uint64_t r[5]={active?roots[(size_t)i*4u]:1ULL,
                   active?roots[(size_t)i*4u+1]:0ULL,
                   active?roots[(size_t)i*4u+2]:0ULL,
                   active?roots[(size_t)i*4u+3]:0ULL,0};
    qsb_block_product_checkpoint(r,super_roots,root_checkpoint);
}

__global__ void __launch_bounds__(256,1) qsb_invert_super_roots(
    uint64_t *super_roots, int count
) {
    int tid=(int)threadIdx.x;
    bool active=tid<count;
    uint64_t r[5]={active?super_roots[(size_t)tid*4u]:1ULL,
                   active?super_roots[(size_t)tid*4u+1]:0ULL,
                   active?super_roots[(size_t)tid*4u+2]:0ULL,
                   active?super_roots[(size_t)tid*4u+3]:0ULL,0};
    qsb_block_inverse(r);
    if(active){
        #pragma unroll
        for(int k=0;k<4;k++)super_roots[(size_t)tid*4u+k]=r[k];
    }
}

__global__ void __launch_bounds__(256,2) qsb_root_group_finish(
    uint64_t *roots, int count, const uint64_t *super_roots,
    const uint64_t *root_checkpoint
) {
    int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    bool active=i<count;
    uint64_t r[5]={active?roots[(size_t)i*4u]:1ULL,
                   active?roots[(size_t)i*4u+1]:0ULL,
                   active?roots[(size_t)i*4u+2]:0ULL,
                   active?roots[(size_t)i*4u+3]:0ULL,0};
    qsb_block_inverse_checkpoint(r,super_roots,root_checkpoint);
    if(active){
        #pragma unroll
        for(int k=0;k<4;k++)roots[(size_t)i*4u+k]=r[k];
    }
}


/* Shared-denominator recovery directly from XYZZ coordinates.
 *
 * P has affine coordinates xP=X/ZZ and yP=Y/ZZZ, with ZZZ^2=ZZ^3.
 * For affine R=(xR,yR), let d=xR*ZZ-X=ZZ*(xR-xP). The collective
 * inverts W=ZZ^2*d. X is dead after d is formed, so overwrite it with d and
 * keep only four field elements live across the block-wide inverse. */
__device__ __forceinline__ void qsb_xyzz_finish_prepare(
    uint64_t *X_D, uint64_t *ZZ, uint64_t *xR, uint64_t *W
) {
    uint64_t t[4];
    _ModMult(t, xR, ZZ);
    _ModSub256(t, t, X_D);
    Load256(X_D, t);             /* X_D becomes d */
    _ModSqr(W, ZZ);
    _ModMult(W, X_D);            /* W = ZZ^2*d */
    W[4] = 0;
}

/* Across the kernel boundary, X_D has been replaced by C=ZZ*d^2 and ZZ by
 * W=ZZ^2*d. With inv=1/W, h=inv*ZZZ=A/(B*d) is the common slope scale and
 * delta=inv*C=d/ZZ=xR-xP. Thus xs=2*xR-delta=xP+xR. The y formulas are
 * anchored at R, avoiding reconstruction of affine yP:
 *   y1 = lambda1*(xR-x1)-yR
 *   y2 = -(m2*(xR-x2)-yR).
 * Returns the two y parities in bits 0 and 1. C, W, and ZZZ are deliberately
 * reused as delta, xs, and h. */
__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed(
    uint64_t *C, uint64_t *Y, uint64_t *W, uint64_t *ZZZ,
    uint64_t *inv, uint64_t *xR, uint64_t *yR,
    uint64_t *x1, uint64_t *x2
) {
    uint64_t yb[4], m[4], t[4], s[4];

    _ModMult(yb, yR, ZZZ);       /* yR*B */
    _ModMult(ZZZ, inv);          /* h = B/(A^2*d) = A/(B*d) */

    _ModMult(C, inv);            /* delta = C/W = d/ZZ */
    _ModAdd256(W, xR, xR);
    _ModSub256(W, C);            /* xs = xP+xR = 2*xR-delta */

    _ModSub256(m, yb, Y);
    _ModMult(m, ZZZ);            /* lambda1 = (yR*B-Y)*h */
    _ModSqr(x1, m);
    _ModSub256(x1, W);
    _ModSub256(t, xR, x1);
    _ModMult(s, m, t);
    _ModSub256(s, yR);
    uint32_t parities = (uint32_t)(s[0] & 1ULL);

    _ModAdd256(m, yb, Y);
    _ModMult(m, ZZZ);            /* m2 = (yR*B+Y)*h = -lambda2 */
    _ModSqr(x2, m);
    _ModSub256(x2, W);
    _ModSub256(t, xR, x2);
    _ModMult(s, m, t);
    _ModSub256(s, yR);
    /* y2=-s. Since p is odd, field negation flips its parity. */
    parities |= (uint32_t)(((s[0] & 1ULL) ^ 1ULL) << 1);
    return parities;
}


/* The canonical pinning tail has one sequence-dependent block followed by
 * an 11-byte locktime-dependent tail. Host code hashes the first block once
 * per sequence. These words contain only the fixed bytes of the second block. */
__device__ __constant__ uint32_t pin_tail_words[3];
__device__ __constant__ uint64_t pin_u2rx_words[4];
__device__ __constant__ uint64_t pin_u2ry_words[4];

template<bool FAST_TAIL, int STAGE>
__global__ void __launch_bounds__(256, STAGE == 0 ? 2 : 3) kernel_pinning_pipeline(
    const uint32_t *__restrict__ d_midstate,
    const uint8_t *__restrict__ d_suffix,    /* suffix template */
    int suffix_len,             /* total suffix including lt+sighash */
    int seq_offset,             /* offset of sequence in suffix */
    int lt_offset,              /* offset of locktime in suffix */
    int total_preimage_len,
    uint32_t seq_value,         /* current sequence value */
    uint32_t start_lt,          /* starting locktime for this batch */
    const uint64_t *__restrict__ d_neg_r_inv,
    const uint64_t *__restrict__ d_u2rx, const uint64_t *__restrict__ d_u2ry,
    const uint64_t *__restrict__ d_neg2u2rx, const uint64_t *__restrict__ d_neg2u2ry,
    const uint8_t *__restrict__ d_gt,
    uint32_t *__restrict__ d_hit_cnt, uint32_t *__restrict__ d_hit_idx,
    int batch_size, int easy_mode, int single_hash, uint32_t group_slot,
    ulonglong2 *__restrict__ saved, uint64_t *__restrict__ roots,
    uint64_t *__restrict__ tree
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x * blockDim.x >= batch_size) return;
    int active = idx < batch_size;
    uint32_t lt = start_lt + (uint32_t)(active ? idx : 0);

    uint64_t qx[4], qy[4], qzz[4], qzzz[4], prod[5];
    if (STAGE==0) {
    uint32_t state[8];
    if (FAST_TAIL) {
        // This specialization is selected only for single_hash, normal mode.
        easy_mode = 0;
        single_hash = 1;
        #pragma unroll
        for (int i=0;i<8;i++) state[i]=d_midstate[i];
        uint32_t blk[16] = {
            pin_tail_words[0] | (lt & 0xffu),
            ((lt & 0xff00u) << 16) | (lt & 0xff0000u) |
                ((lt >> 16) & 0xff00u) | pin_tail_words[1],
            pin_tail_words[2],
            0,0,0,0,0,0,0,0,0,0,0,0,9995u*8u
        };
        _SHA256Transform(state,blk);
    } else {
        /* Copy suffix, set sequence + locktime */
        uint8_t buf[192];
        for(int i=0;i<suffix_len;i++) buf[i]=d_suffix[i];
        buf[seq_offset]=(seq_value)&0xFF; buf[seq_offset+1]=(seq_value>>8)&0xFF;
        buf[seq_offset+2]=(seq_value>>16)&0xFF; buf[seq_offset+3]=(seq_value>>24)&0xFF;
        buf[lt_offset]=(lt)&0xFF; buf[lt_offset+1]=(lt>>8)&0xFF;
        buf[lt_offset+2]=(lt>>16)&0xFF; buf[lt_offset+3]=(lt>>24)&0xFF;

        /* SHA-256 padding */
        buf[suffix_len]=0x80;
        for(int i=suffix_len+1;i<192;i++) buf[i]=0;
        int nblk=(suffix_len<56)?1:2;
        uint64_t bit_len=(uint64_t)total_preimage_len*8;
        int last=nblk*64-8;
        buf[last]=(bit_len>>56)&0xFF;buf[last+1]=(bit_len>>48)&0xFF;
        buf[last+2]=(bit_len>>40)&0xFF;buf[last+3]=(bit_len>>32)&0xFF;
        buf[last+4]=(bit_len>>24)&0xFF;buf[last+5]=(bit_len>>16)&0xFF;
        buf[last+6]=(bit_len>>8)&0xFF;buf[last+7]=bit_len&0xFF;

        for(int i=0;i<8;i++) state[i]=d_midstate[i];
        for(int b=0;b<nblk;b++){
            uint32_t blk[16]; for(int i=0;i<16;i++)
                blk[i]=((uint32_t)buf[b*64+i*4]<<24)|((uint32_t)buf[b*64+i*4+1]<<16)|
                       ((uint32_t)buf[b*64+i*4+2]<<8)|(uint32_t)buf[b*64+i*4+3];
            _SHA256Transform(state,blk);
        }

    }

    /* Second SHA-256: the first digest is already in big-endian words. */
    uint32_t b2[16];
    #pragma unroll
    for(int i=0;i<8;i++) b2[i]=state[i];
    b2[8]=0x80000000u;
    #pragma unroll
    for(int i=9;i<15;i++) b2[i]=0;
    b2[15]=256;
    uint32_t s2[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    _SHA256Transform(s2,b2);

    /* Scalar from the SHA-256 state words, in little-endian limbs. */
    uint64_t z[4];
    z[0] = ((uint64_t)s2[6] << 32) | (uint64_t)s2[7];
    z[1] = ((uint64_t)s2[4] << 32) | (uint64_t)s2[5];
    z[2] = ((uint64_t)s2[2] << 32) | (uint64_t)s2[3];
    z[3] = ((uint64_t)s2[0] << 32) | (uint64_t)s2[1];
    /* neg_r_inv is folded into fixed base A = neg_r_inv*G. Recoding z
     * directly yields z*A = (neg_r_inv*z mod n)*G without a per-candidate
     * scalar multiplication. */
    /* u1*G as raw XYZZ via the signed 16 GiB runtime A-table. */
    _FixedBaseSignedXYZZScalar(qx,qy,qzz,qzzz,z,d_gt);

    /* Recover P+R and P-R together with one shared denominator inverse. The
     * prepare-only xR copy dies before the collective; reload R afterward so
     * its eight limbs do not lengthen the inverse's already pressured state. */
    {
        uint64_t prep_xR[4]={pin_u2rx_words[0],pin_u2rx_words[1],
                             pin_u2rx_words[2],pin_u2rx_words[3]};
        qsb_xyzz_finish_prepare(qx,qzz,prep_xR,prod);
    }
    bool usable = active && ((prod[0] | prod[1] | prod[2] | prod[3]) != 0);
    /* Preserve exactly four fields across the kernel boundary.  The finish
     * needs C=ZZ*d^2 and W=ZZ^2*d, but no longer needs d or ZZ separately. */
    if(usable){
        _ModSqr(qx,qx);
        _ModMult(qx,qzz);        /* qx becomes C */
    }
    Load256(qzz,prod);           /* qzz becomes W */
    if (!usable) {
        prod[0]=1; prod[1]=prod[2]=prod[3]=prod[4]=0;
    }
    if(active){
        /* Eight vector planes retain SoA coalescing while pairing adjacent
         * limbs into naturally aligned 128-bit stores. */
        size_t state_plane_stride=(size_t)batch_size;
        size_t state_idx=(size_t)idx;
        saved[0u*state_plane_stride+state_idx]=make_ulonglong2(qx[0],qx[1]);
        saved[1u*state_plane_stride+state_idx]=make_ulonglong2(qx[2],qx[3]);
        saved[2u*state_plane_stride+state_idx]=make_ulonglong2(qy[0],qy[1]);
        saved[3u*state_plane_stride+state_idx]=make_ulonglong2(qy[2],qy[3]);
        saved[4u*state_plane_stride+state_idx]=make_ulonglong2(qzz[0],qzz[1]);
        saved[5u*state_plane_stride+state_idx]=make_ulonglong2(qzz[2],qzz[3]);
        saved[6u*state_plane_stride+state_idx]=make_ulonglong2(qzzz[0],qzzz[1]);
        saved[7u*state_plane_stride+state_idx]=make_ulonglong2(qzzz[2],qzzz[3]);
    }
    qsb_block_product_checkpoint(prod,roots,tree);
    return;
    } else {

    bool usable = false;
    if(active){
        size_t state_plane_stride=(size_t)batch_size;
        size_t state_idx=(size_t)idx;
        ulonglong2 qx01=saved[0u*state_plane_stride+state_idx];
        ulonglong2 qx23=saved[1u*state_plane_stride+state_idx];
        ulonglong2 qy01=saved[2u*state_plane_stride+state_idx];
        ulonglong2 qy23=saved[3u*state_plane_stride+state_idx];
        ulonglong2 qzz01=saved[4u*state_plane_stride+state_idx];
        ulonglong2 qzz23=saved[5u*state_plane_stride+state_idx];
        ulonglong2 qzzz01=saved[6u*state_plane_stride+state_idx];
        ulonglong2 qzzz23=saved[7u*state_plane_stride+state_idx];
        qx[0]=qx01.x; qx[1]=qx01.y; qx[2]=qx23.x; qx[3]=qx23.y;
        qy[0]=qy01.x; qy[1]=qy01.y; qy[2]=qy23.x; qy[3]=qy23.y;
        qzz[0]=qzz01.x; qzz[1]=qzz01.y; qzz[2]=qzz23.x; qzz[3]=qzz23.y;
        qzzz[0]=qzzz01.x; qzzz[1]=qzzz01.y; qzzz[2]=qzzz23.x; qzzz[3]=qzzz23.y;
        usable = ((qzz[0] | qzz[1] | qzz[2] | qzz[3]) != 0);
    }
    /* qzz carries the original, pre-identity-substitution W, so this exactly
     * recreates the promoted kernel's usability decision without a flag. */
    #pragma unroll
    for(int limb=0;limb<4;limb++)prod[limb]=usable?qzz[limb]:(limb==0?1ULL:0ULL);
    prod[4]=0;
    qsb_block_inverse_checkpoint(prod,roots,tree);
    if (!usable) return;
    uint64_t u2rx[4]={pin_u2rx_words[0],pin_u2rx_words[1],
                      pin_u2rx_words[2],pin_u2rx_words[3]};
    uint64_t u2ry[4]={pin_u2ry_words[0],pin_u2ry_words[1],
                      pin_u2ry_words[2],pin_u2ry_words[3]};
    uint64_t q1x[4],q2x[4];
    uint32_t y_parities = qsb_xyzz_finish_precomputed(
        qx,qy,qzz,qzzz,prod,u2rx,u2ry,
        q1x,q2x);

    /* Check both pubkeys × 2 hashes */
    #pragma unroll
    for(int ri=0;ri<2;ri++){
        uint64_t sx0=ri ? q2x[0] : q1x[0];
        uint64_t sx1=ri ? q2x[1] : q1x[1];
        uint64_t sx2=ri ? q2x[2] : q1x[2];
        uint64_t sx3=ri ? q2x[3] : q1x[3];
        uint32_t x0=(uint32_t)sx0, x1=(uint32_t)(sx0>>32);
        uint32_t x2=(uint32_t)sx1, x3=(uint32_t)(sx1>>32);
        uint32_t x4=(uint32_t)sx2, x5=(uint32_t)(sx2>>32);
        uint32_t x6=(uint32_t)sx3, x7=(uint32_t)(sx3>>32);
        uint32_t pb[16];
        pb[0]=__byte_perm(x7,0x2+(uint8_t)((y_parities>>ri)&1u),0x4321);
        pb[1]=__byte_perm(x7,x6,0x0765);pb[2]=__byte_perm(x6,x5,0x0765);
        pb[3]=__byte_perm(x5,x4,0x0765);pb[4]=__byte_perm(x4,x3,0x0765);
        pb[5]=__byte_perm(x3,x2,0x0765);pb[6]=__byte_perm(x2,x1,0x0765);
        pb[7]=__byte_perm(x1,x0,0x0765);pb[8]=__byte_perm(x0,0x80,0x0456);
        pb[9]=0;pb[10]=0;pb[11]=0;pb[12]=0;pb[13]=0;pb[14]=0;pb[15]=0x108;
        uint32_t hs[8];_SHA256Initialize(hs);_SHA256Transform(hs,pb);
        int vv;
        if (!FAST_TAIL && easy_mode) {
            uint8_t h[32];
            for(int i=0;i<8;i++){h[i*4]=(hs[i]>>24)&0xFF;h[i*4+1]=(hs[i]>>16)&0xFF;
                h[i*4+2]=(hs[i]>>8)&0xFF;h[i*4+3]=hs[i]&0xFF;}
            vv=gpu_is_der_easy(h,32);
        } else {
            vv=gpu_bench_valid_words(hs);
        }
        if(vv){
            uint32_t pos=atomicAdd(d_hit_cnt,1);
            if(pos<1024){
                if(FAST_TAIL)
                    d_hit_idx[pos]=(lt & 0x7FFFFFFFu)|(ri<<31);
                else
                    d_hit_idx[pos]=((uint32_t)idx)|(group_slot<<24)|(ri<<30);
            }
            return;
        }
        if (FAST_TAIL || single_hash) continue;  /* Config A: only one hash iteration */
        uint8_t h[32];
        for(int i=0;i<8;i++){h[i*4]=(hs[i]>>24)&0xFF;h[i*4+1]=(hs[i]>>16)&0xFF;
            h[i*4+2]=(hs[i]>>8)&0xFF;h[i*4+3]=hs[i]&0xFF;}
        uint8_t pp[64];memset(pp,0,64);memcpy(pp,h,32);pp[32]=0x80;pp[62]=1;pp[63]=0;
        uint32_t bb2[16];for(int i=0;i<16;i++)bb2[i]=((uint32_t)pp[i*4]<<24)|((uint32_t)pp[i*4+1]<<16)|
            ((uint32_t)pp[i*4+2]<<8)|(uint32_t)pp[i*4+3];
        uint32_t h2s[8];_SHA256Initialize(h2s);_SHA256Transform(h2s,bb2);
        if (!FAST_TAIL && easy_mode) {
            uint8_t h2[32];
            for(int i=0;i<8;i++){h2[i*4]=(h2s[i]>>24)&0xFF;h2[i*4+1]=(h2s[i]>>16)&0xFF;
                h2[i*4+2]=(h2s[i]>>8)&0xFF;h2[i*4+3]=h2s[i]&0xFF;}
            vv=gpu_is_der_easy(h2,32);
        } else {
            vv=gpu_bench_valid_words(h2s);
        }
        if(vv){
            uint32_t pos=atomicAdd(d_hit_cnt,1);
            if(pos<1024)d_hit_idx[pos]=((uint32_t)idx)|(group_slot<<24)|(ri<<30)|(1u<<31);
            return;
        }
    }
    }
}

#include "wide_fused_kernel.cuh"
static bool wide_use_fused=false;

template<bool FAST_TAIL>
static void launch_pinning_pipeline(
    const uint32_t *d_midstate, const uint8_t *d_suffix,
    int suffix_len, int seq_offset, int lt_offset, int total_preimage_len,
    uint32_t seq_value, uint32_t start_lt,
    const uint64_t *d_neg_r_inv,
    const uint64_t *d_u2rx, const uint64_t *d_u2ry,
    const uint64_t *d_neg2u2rx, const uint64_t *d_neg2u2ry,
    uint8_t *d_gt, uint32_t *d_hit_cnt, uint32_t *d_hit_idx,
    int batch_size, int easy_mode, int single_hash, uint32_t group_slot,
    ulonglong2 *saved, uint64_t *roots, uint64_t *tree,
    uint64_t *super_roots, uint64_t *root_checkpoint
) {
    int blocks=(batch_size+255)/256;
    if(FAST_TAIL&&wide_use_fused){
        kernel_pinning_fused<FAST_TAIL><<<blocks,256>>>(
            d_midstate,d_suffix,suffix_len,seq_offset,lt_offset,total_preimage_len,
            seq_value,start_lt,d_neg_r_inv,d_u2rx,d_u2ry,d_neg2u2rx,d_neg2u2ry,
            d_gt,d_hit_cnt,d_hit_idx,batch_size,easy_mode,single_hash,group_slot,
            saved,roots,tree);
        cudaError_t err=cudaGetLastError();
        if(err!=cudaSuccess){fprintf(stderr,"Fused search launch failed: %s\n",cudaGetErrorString(err));exit(2);}
        return;
    }
    kernel_pinning_pipeline<FAST_TAIL,0><<<blocks,256>>>(
        d_midstate,d_suffix,suffix_len,seq_offset,lt_offset,total_preimage_len,
        seq_value,start_lt,d_neg_r_inv,d_u2rx,d_u2ry,d_neg2u2rx,d_neg2u2ry,
        d_gt,d_hit_cnt,d_hit_idx,batch_size,easy_mode,single_hash,group_slot,
        saved,roots,tree);
    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess){
        fprintf(stderr,"Pipeline prepare launch failed: %s\n",cudaGetErrorString(err));
        exit(2);
    }
    int root_groups=(blocks+255)/256;
    if(root_groups>256){
        fprintf(stderr,"Pipeline batch exceeds two-level inverse capacity\n");
        exit(2);
    }
    qsb_root_group_prepare<<<root_groups,256>>>(
        roots,blocks,super_roots,root_checkpoint);
    err=cudaGetLastError();
    if(err!=cudaSuccess){
        fprintf(stderr,"Root-group prepare launch failed: %s\n",cudaGetErrorString(err));
        exit(2);
    }
    qsb_invert_super_roots<<<1,256>>>(super_roots,root_groups);
    err=cudaGetLastError();
    if(err!=cudaSuccess){
        fprintf(stderr,"Super-root inverse launch failed: %s\n",cudaGetErrorString(err));
        exit(2);
    }
    qsb_root_group_finish<<<root_groups,256>>>(
        roots,blocks,super_roots,root_checkpoint);
    err=cudaGetLastError();
    if(err!=cudaSuccess){
        fprintf(stderr,"Root-group finish launch failed: %s\n",cudaGetErrorString(err));
        exit(2);
    }
    kernel_pinning_pipeline<FAST_TAIL,2><<<blocks,256>>>(
        d_midstate,d_suffix,suffix_len,seq_offset,lt_offset,total_preimage_len,
        seq_value,start_lt,d_neg_r_inv,d_u2rx,d_u2ry,d_neg2u2rx,d_neg2u2ry,
        d_gt,d_hit_cnt,d_hit_idx,batch_size,easy_mode,single_hash,group_slot,
        saved,roots,tree);
    err=cudaGetLastError();
    if(err!=cudaSuccess){
        fprintf(stderr,"Pipeline finish launch failed: %s\n",cudaGetErrorString(err));
        exit(2);
    }
}

#include "wide_table_kernels.cuh"

/* ============================================================
 * Host code
 * ============================================================ */

extern "C" {
#include <openssl/sha.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
}

/* Affine (x,y) of a point, as the 4+4 little-endian limbs the table uses. */
static void gt_point_to_limbs(EC_GROUP *grp, EC_POINT *pt, BIGNUM *x, BIGNUM *y,
                              BN_CTX *ctx, uint64_t out[8]) {
    uint8_t xb[32], yb[32];
    memset(xb, 0, 32); memset(yb, 0, 32);
    EC_POINT_get_affine_coordinates_GFp(grp, pt, x, y, ctx);
    BN_bn2bin(x, xb + (32 - BN_num_bytes(x)));
    BN_bn2bin(y, yb + (32 - BN_num_bytes(y)));
    for (int j = 0; j < 16; j++) { uint8_t t = xb[j]; xb[j] = xb[31-j]; xb[31-j] = t; }
    for (int j = 0; j < 16; j++) { uint8_t t = yb[j]; yb[j] = yb[31-j]; yb[31-j] = t; }
    memcpy(out,     xb, 32);
    memcpy(out + 4, yb, 32);
}

static void gt_require(bool ok,const char *where){if(!ok){fprintf(stderr,"OpenSSL table builder failure: %s\n",where);exit(2);}}
// Batched affine ladder normalization, as independently described in public
// MakiRH4 PR46 and jacklightChen PR53; extended to odd13-bit lows and wide highs.
static void gt_build_ladders(uint64_t *hL,uint64_t *hH,const uint8_t neg_r_inv[32]){
    EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);BN_CTX *ctx=BN_CTX_new();
    BIGNUM *x=BN_new(),*y=BN_new(),*shift=BN_new(),*order=BN_new(),*inv2=BN_new(),*nri=BN_new(),*coef=BN_new();
    EC_POINT *base=grp?EC_POINT_new(grp):nullptr,*step=grp?EC_POINT_new(grp):nullptr;
    gt_require(grp&&ctx&&x&&y&&shift&&order&&inv2&&nri&&coef&&base&&step,"allocate");
    gt_require(EC_GROUP_get_order(grp,order,ctx),"group order");gt_require(BN_set_word(shift,2),"set2");
    gt_require(BN_mod_inverse(inv2,shift,order,ctx)!=nullptr,"inverse2");
    gt_require(BN_lebin2bn(neg_r_inv,32,nri)!=nullptr,"runtime coefficient");
    gt_require(BN_mod_mul(coef,inv2,nri,order,ctx),"fold base coefficient");
    gt_require(!BN_is_zero(coef),"nonzero runtime base");
    gt_require(EC_POINT_mul(grp,base,coef,nullptr,nullptr,ctx),"base point");
    memset(hL,0,(size_t)GT_CHUNKS*GT_LO*8*sizeof(uint64_t));
    memset(hH,0,(size_t)GT_CHUNKS*GT_HI*8*sizeof(uint64_t));
    for(int ch=0;ch<GT_CHUNKS;++ch){
        if(ch){gt_require(BN_set_word(shift,1u<<wide_bits(ch-1)),"window shift");gt_require(EC_POINT_mul(grp,base,nullptr,base,shift,ctx),"shifted base");}
        std::vector<EC_POINT*> points;std::vector<uint64_t*> destinations;
        gt_require(EC_POINT_dbl(grp,step,base,ctx),"low step");
        for(int lo=1;lo<GT_LO;lo+=2){
            EC_POINT *pt=EC_POINT_new(grp);gt_require(pt!=nullptr,"low point allocate");
            gt_require(lo==1?EC_POINT_copy(pt,base):EC_POINT_add(grp,pt,points.back(),step,ctx),"low ladder");
            points.push_back(pt);destinations.push_back(hL+((size_t)ch*GT_LO+lo)*8);
        }
        gt_require(BN_set_word(shift,GT_LO),"high step scalar");
        gt_require(EC_POINT_mul(grp,step,nullptr,base,shift,ctx),"high step");
        int high_count=(int)((uint64_t)gt_entries(ch)*2/GT_LO);
        for(int hi=1;hi<high_count;++hi){
            EC_POINT *pt=EC_POINT_new(grp);gt_require(pt!=nullptr,"high point allocate");
            gt_require(hi==1?EC_POINT_copy(pt,step):EC_POINT_add(grp,pt,points.back(),step,ctx),"high ladder");
            points.push_back(pt);destinations.push_back(hH+((size_t)ch*GT_HI+hi)*8);
        }
        gt_require(EC_POINTs_make_affine(grp,points.size(),points.data(),ctx),"batch normalize");
        for(size_t i=0;i<points.size();++i){gt_point_to_limbs(grp,points[i],x,y,ctx,destinations[i]);EC_POINT_free(points[i]);}
    }
    EC_POINT_free(base);EC_POINT_free(step);BN_free(x);BN_free(y);BN_free(shift);BN_free(order);BN_free(inv2);BN_free(nri);BN_free(coef);EC_GROUP_free(grp);BN_CTX_free(ctx);
}


/* Spot-check the built table against OpenSSL. The builder runs on hardware this
 * code has never executed on, so a silent wrong table -- which would simply
 * produce zero verifiable hits and burn the whole run -- must be caught here
 * and fall back, not discovered from the scorecard. */
static int gt_spot_check(const uint8_t *gTable, int samples,
                         const uint8_t neg_r_inv[32]) {
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *x = BN_new(), *y = BN_new(), *k = BN_new(), *inv2 = BN_new(), *order = BN_new(),
           *nri = BN_new(), *half_nri = BN_new();
    EC_POINT *pt = EC_POINT_new(grp);
    uint64_t want[8];
    int ok = 1;
    unsigned seed = 0x9e3779b9u;
    EC_GROUP_get_order(grp, order, ctx);
    BN_set_word(k, 2); BN_mod_inverse(inv2, k, order, ctx);   /* inv2 = 2^-1 mod n */
    BN_lebin2bn(neg_r_inv, 32, nri);
    BN_mod_mul(half_nri, inv2, nri, order, ctx);              /* (2^-1 * neg_r_inv) mod n = A/2 scalar */
    for (int t = 0; t < samples && ok; t++) {
        /* always include the corners of each chunk, then pseudo-random entries */
        int ch, i;
        if (t < GT_CHUNKS * 8) {
            ch = t / 8;
            const int corner[8] = {0,1,2,4095,4096,4097,(int)gt_entries(ch)-2,(int)gt_entries(ch)-1};
            i = corner[t % 8];
        } else {
            seed = seed * 1664525u + 1013904223u;
            ch = (int)(seed >> 28) % GT_CHUNKS;
            i  = (int)((seed >> 4) & (gt_entries(ch) - 1));
        }
        /* want = (2i+1) * 2^gt_shift(ch) * (A/2). */
        BN_one(k);
        BN_lshift(k, k, gt_shift(ch));
        BN_mul_word(k, (BN_ULONG)(2*i + 1));
        BN_mod_mul(k, k, half_nri, order, ctx);
        EC_POINT_mul(grp, pt, k, NULL, NULL, ctx);
        gt_point_to_limbs(grp, pt, x, y, ctx, want);
        size_t off = ((size_t)gt_offset(ch) + i) * 64;
        uint8_t sample[64];
        if(cudaMemcpy(sample,gTable+off,64,cudaMemcpyDeviceToHost)!=cudaSuccess){ok=0;break;}
        if (memcmp(sample,want,32)!=0 || memcmp(sample+32,want+4,32)!=0) {
            fprintf(stderr, "  GTable spot check FAILED at chunk %d entry %d\n", ch, i);
            ok = 0;
        }
    }
    BN_free(x); BN_free(y); BN_free(k); BN_free(inv2); BN_free(order); BN_free(nri); BN_free(half_nri);
    EC_POINT_free(pt); EC_GROUP_free(grp); BN_CTX_free(ctx);
    return ok;
}

#include "wide_table_host.cuh"
#include "wide_search_tuner.cuh"

/* Params loader for pinning2.bin */
typedef struct {
    uint32_t midstate[8];
    uint32_t suffix_len;
    uint8_t *suffix;
    uint32_t total_preimage_len;
    uint32_t seq_offset;
    uint32_t lt_offset;
    uint8_t neg_r_inv[32];
    uint8_t u2r_x[32];
    uint8_t u2r_y[32];
} pinning2_params_t;

static int load_pinning2(const char *fn, pinning2_params_t *p) {
    FILE *f = fopen(fn, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fn); return -1; }
    if (fread(p->midstate, 4, 8, f) != 8) goto err;
    for (int i=0;i<8;i++) {
        uint8_t *b=(uint8_t*)&p->midstate[i];
        p->midstate[i]=((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|b[3];
    }
    if (fread(&p->suffix_len, 4, 1, f) != 1) goto err;
    p->suffix = (uint8_t*)malloc(p->suffix_len + 16); /* extra for lt+sighash */
    if (fread(p->suffix, 1, p->suffix_len, f) != p->suffix_len) goto err;
    if (fread(&p->total_preimage_len, 4, 1, f) != 1) goto err;
    if (fread(&p->seq_offset, 4, 1, f) != 1) goto err;
    if (fread(&p->lt_offset, 4, 1, f) != 1) goto err;
    if (fread(p->neg_r_inv, 1, 32, f) != 32) goto err;
    if (fread(p->u2r_x, 1, 32, f) != 32) goto err;
    if (fread(p->u2r_y, 1, 32, f) != 32) goto err;
    fclose(f);
    /* In NEW pipeline format, the suffix already includes locktime + sighash_type
     * at lt_offset..lt_offset+7 (placed there by cmd_export). No additional
     * placeholder writes needed.
     *
     * Old format used to write placeholders here; that wrote 8 bytes BEYOND
     * suffix_len (into uninitialized malloc memory) which on the GPU got hashed
     * as if they were part of the message — silently corrupting first_sha256
     * by 8 zero bytes. Removed. */
    printf("  Loaded: preimage=%u, suffix=%u, seq@%u, lt@%u\n",
           p->total_preimage_len, p->suffix_len, p->seq_offset, p->lt_offset);
    return 0;
err:
    fprintf(stderr, "Error reading %s\n", fn);
    fclose(f); return -1;
}


int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <pinning2.bin> [gpu_index] [total_gpus] [global_offset] [easy]\n", argv[0]);
        printf("  total_gpus: total GPUs across ALL machines (default: local count)\n");
        printf("  global_offset: this machine's GPU offset (default: 0)\n");
        return 1;
    }
    int gpu_index = (argc >= 3) ? atoi(argv[2]) : 0;
    int total_gpus_override = (argc >= 4) ? atoi(argv[3]) : 0;
    int global_offset = (argc >= 5) ? atoi(argv[4]) : 0;
    int easy = 0;
    for (int i = 3; i < argc; i++) if (strcmp(argv[i], "easy") == 0) easy = 1;
    int single_hash = 0;
    for (int i = 3; i < argc; i++) if (strcmp(argv[i], "single_hash") == 0) single_hash = 1;
    /* Optional seq_start=0xHEX argument: skip ahead in pin space (e.g. to find
     * the SECOND pin after the first one was already used and yielded zero
     * digest hits). Default: 0x80000000. */
    uint32_t seq_start_override = 0;
    for (int i = 3; i < argc; i++) {
        if (strncmp(argv[i], "seq_start=", 10) == 0) {
            seq_start_override = (uint32_t)strtoul(argv[i] + 10, NULL, 0);
        }
    }

    /* Use the specified GPU */
    wide_cuda_require(cudaSetDevice(gpu_index),"select device");
    wide_cuda_require(cudaDeviceSetLimit(cudaLimitStackSize,4096),"set stack limit");

    cudaDeviceProp prop; wide_cuda_require(cudaGetDeviceProperties(&prop,gpu_index),"device properties");
    printf("QSB Real Pinning Search (seq+lt) [GPU %d]\n", gpu_index);
    printf("  GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    pinning2_params_t pp;
    if (load_pinning2(argv[1], &pp) < 0) return 1;

    /* Reserve the large runtime table only on the GPU. Builder scratch is
     * released before allocating the search checkpoints. */
    size_t free_bytes=0,total_bytes=0;
    wide_cuda_require(cudaMemGetInfo(&free_bytes,&total_bytes),"device memory query");
    const size_t gt_sz=(size_t)GT_TOTAL_ENTRIES*64;
    const size_t search_bytes=(5ull<<29)+(4ull<<20)+(8ull<<10);
    if(free_bytes<gt_sz+search_bytes+(64ull<<20)){
        fprintf(stderr,"Insufficient GPU memory for16GiB table and search checkpoints: %.2fGiB free\n",(double)free_bytes/(1ull<<30));return 2;
    }
    uint8_t *d_gt=nullptr;
    wide_cuda_require(cudaMalloc(&d_gt,gt_sz),"allocate wide table");
    wide_build_table(d_gt,pp.neg_r_inv);

    /* Upload midstate */
    uint32_t *d_mid; cudaMalloc(&d_mid, 32);
    cudaMemcpy(d_mid, pp.midstate, 32, cudaMemcpyHostToDevice);

    /* Build suffix template. In the NEW pipeline format (combined_suffix), the
     * suffix loaded from pinning.bin ALREADY includes:
     *   [prefix_remainder] [seq_template] [output_count=0] [lt_template] [sighash_type]
     * So pp.suffix_len already accounts for lt+sighash. The kernel processes
     * exactly pp.suffix_len bytes; no +8 fudge needed.
     *
     * (The previous +8 was a leftover from the OLD format where pinning.bin
     * stored only [remainder + seq + outcount] and load_pinning2 had to append
     * lt+sighash placeholders at runtime. With new format that's already done by
     * the export step.) */
    uint8_t *suffix_template = (uint8_t*)calloc(256, 1);
    memcpy(suffix_template, pp.suffix, pp.suffix_len);
    int gpu_suffix_len = pp.suffix_len;

    uint8_t *d_suffix; cudaMalloc(&d_suffix, 256);
    cudaMemcpy(d_suffix, suffix_template, 256, cudaMemcpyHostToDevice);

    printf("  Full suffix: %d bytes, seq@%d, lt@%d\n",
           gpu_suffix_len, pp.seq_offset, pp.lt_offset);
    printf("  Mode: %s\n", easy ? "EASY" : "REAL");

    /* Upload EC constants */
    uint64_t *d_nri, *d_u2rx, *d_u2ry, *d_neg2u2rx, *d_neg2u2ry;
    cudaMalloc(&d_nri,32); cudaMalloc(&d_u2rx,32); cudaMalloc(&d_u2ry,32);
    cudaMalloc(&d_neg2u2rx,32); cudaMalloc(&d_neg2u2ry,32);
    cudaMemcpy(d_nri, pp.neg_r_inv, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2rx, pp.u2r_x, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2ry, pp.u2r_y, 32, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(pin_u2rx_words, pp.u2r_x, sizeof(pp.u2r_x));
    cudaMemcpyToSymbol(pin_u2ry_words, pp.u2r_y, sizeof(pp.u2r_y));

    /* Compute neg_2u2R */
    {
        EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);
        BN_CTX *ctx=BN_CTX_new();
        BIGNUM *bx=BN_new(),*by=BN_new();
        uint8_t be[32];
        for(int i=0;i<32;i++) be[i]=pp.u2r_x[31-i]; BN_bin2bn(be,32,bx);
        for(int i=0;i<32;i++) be[i]=pp.u2r_y[31-i]; BN_bin2bn(be,32,by);
        EC_POINT *pt=EC_POINT_new(grp);
        EC_POINT_set_affine_coordinates_GFp(grp,pt,bx,by,ctx);
        EC_POINT *dbl=EC_POINT_new(grp);
        EC_POINT_dbl(grp,dbl,pt,ctx);
        EC_POINT_invert(grp,dbl,ctx);
        BIGNUM *dx=BN_new(),*dy=BN_new();
        EC_POINT_get_affine_coordinates_GFp(grp,dbl,dx,dy,ctx);
        uint8_t dxb[32],dyb[32]; memset(dxb,0,32);memset(dyb,0,32);
        BN_bn2bin(dx,dxb+(32-BN_num_bytes(dx)));
        BN_bn2bin(dy,dyb+(32-BN_num_bytes(dy)));
        uint64_t n2x[4],n2y[4];
        for(int i=0;i<4;i++){n2x[i]=0;n2y[i]=0;
            for(int b=0;b<8;b++){n2x[i]|=(uint64_t)dxb[31-i*8-b]<<(b*8);
                n2y[i]|=(uint64_t)dyb[31-i*8-b]<<(b*8);}}
        cudaMemcpy(d_neg2u2rx,n2x,32,cudaMemcpyHostToDevice);
        cudaMemcpy(d_neg2u2ry,n2y,32,cudaMemcpyHostToDevice);
        BN_free(bx);BN_free(by);BN_free(dx);BN_free(dy);
        EC_POINT_free(pt);EC_POINT_free(dbl);
        EC_GROUP_free(grp);BN_CTX_free(ctx);
    }

    const bool fast_tail = single_hash && !easy && pp.suffix_len == 75 &&
        pp.seq_offset == 31 && pp.lt_offset == 67 && pp.total_preimage_len == 9995;
    if (fast_tail) {
        uint32_t words[3] = {
            ((uint32_t)pp.suffix[64]<<24) | ((uint32_t)pp.suffix[65]<<16) |
                ((uint32_t)pp.suffix[66]<<8),
            pp.suffix[71],
            ((uint32_t)pp.suffix[72]<<24) | ((uint32_t)pp.suffix[73]<<16) |
                ((uint32_t)pp.suffix[74]<<8) | 0x80u
        };
        cudaError_t copy_err = cudaMemcpyToSymbol(pin_tail_words, words, sizeof(words));
        if (copy_err != cudaSuccess) {
            fprintf(stderr, "Failed to upload fixed SHA tail: %s\n",
                    cudaGetErrorString(copy_err));
            return 1;
        }
        printf("  SHA path: per-sequence midstate + one static tail block\n");
    }


    uint32_t *d_hit_cnt, *d_hit_idx;
    cudaMalloc(&d_hit_cnt, 4); cudaMalloc(&d_hit_idx, 1024*4);

    int BATCH = 16777216;  /* 16M: amortize launch/sync/copy overhead */
    int BLKSZ = 256;
    int GRDSZ = (BATCH+BLKSZ-1)/BLKSZ;
    int ROOT_GRDSZ=(GRDSZ+255)/256;
    ulonglong2 *d_pipeline_state=NULL;
    uint64_t *d_pipeline_roots=NULL,*d_pipeline_tree=NULL;
    uint64_t *d_super_roots=NULL,*d_root_checkpoint=NULL;
    size_t pipeline_state_bytes=(size_t)BATCH*8u*sizeof(ulonglong2);
    size_t pipeline_root_bytes=(size_t)GRDSZ*4u*sizeof(uint64_t);
    size_t pipeline_tree_bytes=(size_t)GRDSZ*4u*QSB_CHECKPOINT_STRIDE*sizeof(uint64_t);
    size_t super_root_bytes=(size_t)ROOT_GRDSZ*4u*sizeof(uint64_t);
    size_t root_checkpoint_bytes=(size_t)ROOT_GRDSZ*4u*QSB_CHECKPOINT_STRIDE*sizeof(uint64_t);
    cudaError_t pipeline_err=cudaMalloc(&d_pipeline_state,pipeline_state_bytes);
    if(pipeline_err==cudaSuccess)
        pipeline_err=cudaMalloc(&d_pipeline_roots,pipeline_root_bytes);
    if(pipeline_err==cudaSuccess)
        pipeline_err=cudaMalloc(&d_pipeline_tree,pipeline_tree_bytes);
    if(pipeline_err==cudaSuccess)
        pipeline_err=cudaMalloc(&d_super_roots,super_root_bytes);
    if(pipeline_err==cudaSuccess)
        pipeline_err=cudaMalloc(&d_root_checkpoint,root_checkpoint_bytes);
    if(pipeline_err!=cudaSuccess){
        fprintf(stderr,"Pipeline allocation failed: %s\n",cudaGetErrorString(pipeline_err));
        return 1;
    }
    if(((uintptr_t)d_pipeline_state & (alignof(ulonglong2)-1u)) != 0){
        fprintf(stderr,"Pipeline state allocation is not 16-byte aligned\n");
        return 1;
    }
    printf("  Pipeline checkpoints: %.0f MiB state + %.0f MiB tree + %.0f MiB roots + %.2f MiB root tree\n",
           (double)pipeline_state_bytes/(1024*1024),
           (double)pipeline_tree_bytes/(1024*1024),
           (double)pipeline_root_bytes/(1024*1024),
           (double)(super_root_bytes+root_checkpoint_bytes)/(1024*1024));

    /* Safe ranges */
    uint32_t LT_MIN = 500000000;   /* timestamp interpretation */
    uint32_t LT_MAX = 1744600000;  /* current time (approx) */
    uint32_t SEQ_MIN = 0x80000000; /* bit 31 set — avoids BIP68 */
    if (seq_start_override) {
        SEQ_MIN = seq_start_override;
        printf("  seq_start override: 0x%08x\n", SEQ_MIN);
    }
    uint32_t lt_range = LT_MAX - LT_MIN;

    /* How many GPUs total (for interleaving across all machines) */
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus < 1) num_gpus = 1;
    int effective_total = (total_gpus_override > 0) ? total_gpus_override : num_gpus;
    int effective_id = global_offset + gpu_index;

    printf("\n  === Search: lt=[%u,%u] (%u), seq=[0x%08X+], GPU %d (global %d of %d) ===\n",
           LT_MIN, LT_MAX, lt_range, SEQ_MIN, gpu_index, effective_id, effective_total);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    uint64_t total_searched = 0;
    WideSearchTuner schedule_tuner(fast_tail);
    int found = 0;

    /* Each GPU handles sequences: SEQ_MIN + effective_id, SEQ_MIN + effective_id + effective_total, ... */

    /* ── DEBUG MODE ──
     * If argv contains "debug" followed by <seq_hex> <lt>, run the
     * single-point diagnostic kernel and exit. Use this to investigate a
     * specific (seq, lt) that the production kernel claims is a hit but
     * which CPU verification rejects.
     *
     * Example:
     *   ./qsb_real pinning.bin 0 single_hash debug 0x80006137 1317906633
     */
    {
        int debug_idx = -1;
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "debug") == 0) { debug_idx = i; break; }
        }
        /* diagnostic kernel and launch removed from benchmark builds */
    }

    /* Benchmark runs for a fixed window ended by the harness's timeout.
     * The loop no longer stops at the first hit; hits are appended per batch.
     */
    for (uint32_t seq = SEQ_MIN + effective_id; ; seq += effective_total) {
        if (fast_tail) {
            uint8_t block[64];
            memcpy(block, pp.suffix, sizeof(block));
            for(int i=0;i<4;i++) block[pp.seq_offset+i]=(uint8_t)(seq>>(8*i));
            SHA256_CTX ctx;
            SHA256_Init(&ctx);
            for(int i=0;i<8;i++) ctx.h[i]=pp.midstate[i];
            SHA256_Transform(&ctx,block);
            cudaError_t copy_err = cudaMemcpy(d_mid,ctx.h,32,cudaMemcpyHostToDevice);
            if (copy_err != cudaSuccess) {
                fprintf(stderr, "Failed to upload per-sequence SHA state: %s\n",
                        cudaGetErrorString(copy_err));
                return 1;
            }
        }

        /* Queue complete pipelines before the synchronous counter readback.
         * Ranked fast mode packs the absolute 31-bit locktime plus recid, so
         * all pipelines for one sequence can share a readback.  The fallback
         * keeps six group-slot bits above its 24-bit batch index. */
        static const uint32_t MAX_READBACK_GROUP = 128;
        uint32_t readback_group = fast_tail ? MAX_READBACK_GROUP : 64;
        for (uint32_t group_off = 0; group_off < lt_range;
             group_off += readback_group * (uint32_t)BATCH) {
            readback_group=schedule_tuner.group_limit(fast_tail?MAX_READBACK_GROUP:64);
            uint64_t group_start_count=total_searched;
            schedule_tuner.begin();
            uint32_t h_hit = 0;
            cudaError_t err = cudaMemsetAsync(d_hit_cnt, 0, 4);
            if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

            uint32_t group_lt[MAX_READBACK_GROUP];
            uint32_t group_count = 0;
            for (; group_count < readback_group; group_count++) {
                uint32_t lt_off = group_off + group_count * (uint32_t)BATCH;
                if (lt_off >= lt_range) break;
                uint32_t batch_lt = LT_MIN + lt_off;
                int batch_sz = (lt_off + BATCH <= lt_range) ? BATCH : (lt_range - lt_off);
                group_lt[group_count] = batch_lt;

                if (fast_tail) {
                    launch_pinning_pipeline<true>(
                        d_mid, d_suffix, gpu_suffix_len,
                        pp.seq_offset, pp.lt_offset,
                        pp.total_preimage_len,
                        seq, batch_lt,
                        d_nri, d_u2rx, d_u2ry, d_neg2u2rx, d_neg2u2ry,
                        d_gt,
                        d_hit_cnt, d_hit_idx,
                        batch_sz, easy, single_hash, group_count,
                        d_pipeline_state,d_pipeline_roots,d_pipeline_tree,
                        d_super_roots,d_root_checkpoint);
                } else {
                    launch_pinning_pipeline<false>(
                        d_mid, d_suffix, gpu_suffix_len,
                        pp.seq_offset, pp.lt_offset,
                        pp.total_preimage_len,
                        seq, batch_lt,
                        d_nri, d_u2rx, d_u2ry, d_neg2u2rx, d_neg2u2ry,
                        d_gt,
                        d_hit_cnt, d_hit_idx,
                        batch_sz, easy, single_hash, group_count,
                        d_pipeline_state,d_pipeline_roots,d_pipeline_tree,
                        d_super_roots,d_root_checkpoint);
                }
                total_searched += batch_sz;
            }

            schedule_tuner.end();
            err = cudaMemcpy(&h_hit, d_hit_cnt, 4, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
            schedule_tuner.observe(total_searched-group_start_count);
            if (h_hit > 0) {
                uint32_t hits[1024];
                int nh = (h_hit > 1024) ? 1024 : h_hit;
                wide_cuda_require(cudaMemcpy(hits,d_hit_idx,nh*4,cudaMemcpyDeviceToHost),"hit records copy");

                printf("\n  *** HIT! seq=0x%08X ***\n", seq);
                mkdir("results", 0755);
                char fname[256];
                snprintf(fname, sizeof(fname), "results/pinning_hit_%d.txt", gpu_index);
                FILE *f = fopen(fname, "a");
                if (f) {
                    for (int h = 0; h < nh; h++) {
                        uint32_t raw = hits[h];
                        uint32_t lt;
                        int ri, hc;
                        if (fast_tail) {
                            lt = raw & 0x7FFFFFFFu;
                            ri = (raw >> 31) & 1;
                            hc = 0;
                        } else {
                            uint32_t slot = (raw >> 24) & 0x3Fu;
                            if (slot >= group_count) continue;
                            lt = group_lt[slot] + (raw & 0x00FFFFFFu);
                            ri = (raw >> 30) & 1;
                            hc = (raw >> 31) & 1;
                        }
                        fprintf(f, "sequence=%u\nlocktime=%u\nhash_choice=%d\nrecid=%d\n",
                                seq, lt, hc, ri);
                        printf("  seq=0x%08X lt=%u hc=%d recid=%d\n", seq, lt, hc, ri);
                    }
                    fclose(f);
                }
                found = 1;
            }

            /* Check if another GPU found it */
            if ((total_searched % (50*1024*1024)) <
                (uint64_t)readback_group * (uint64_t)BATCH) {
                char check[256];
                for (int g = 0; g < num_gpus; g++) {
                    if (g == gpu_index) continue;
                    snprintf(check, sizeof(check), "results/pinning_hit_%d.txt", g);
                    FILE *cf = fopen(check, "r");
                    if (cf) { fclose(cf); printf("  GPU %d found hit, stopping.\n", g); found = 1; break; }
                }
            }
        }

        /* Progress every 10 sequences */
        uint32_t seqs_done = (seq - SEQ_MIN - effective_id) / effective_total + 1;
        if (seqs_done % 10 == 0 || found) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
            double rate = total_searched / elapsed;
            printf("  [GPU %d] seq #%u (0x%08X), %luM total, %.1fM/s, %.0fs\n",
                   gpu_index, seqs_done, seq, total_searched/1000000, rate/1e6, elapsed);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    printf("\n  Done: %luM in %.0fs (%.1fM/s), found=%d\n",
           total_searched/1000000, elapsed, total_searched/elapsed/1e6, found);

    return 0;
}
