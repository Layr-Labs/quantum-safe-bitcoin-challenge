/* pinning.cu -- QSB pinning grinder (benchmark track).
 *
 * Per candidate (sequence, locktime): SHA256d of the 9995-byte preimage, ECDSA
 * public-key recovery for both recids, SHA256 of each compressed key, and the
 * leading-zero gate.
 *
 * This version carries the subset track's promoted EC pipeline over to
 * pinning (see candidates/subset/TREE_INVERSE.md for provenance):
 *   - neg_r_inv folded into a signed-digit 32 MiB fixed-base table built on the
 *     GPU, so z is recoded directly and no per-candidate scalar mulmod runs;
 *   - XYZZ accumulation for u1*G;
 *   - one shared-denominator finish for P+R and P-R (both recids);
 *   - one field inversion per 256-thread block via a binary product tree.
 * The pinning front end hashes one SHA-256 block per candidate from a
 * per-sequence midstate: suffix block A (which holds `sequence`) is compressed
 * on the host once per sequence; only block B (which holds `locktime`) and the
 * second SHA-256 run on the GPU.
 *
 * Build:  nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm
 * Usage:  ./pinning <pinning.bin> [gpu_index] [total_gpus] [global_offset] [single_hash]
 *
 * GPUMath.h, GPUHash.h and square32.cuh derive from VanitySearch (GPLv3);
 * this file and the executable are GPLv3-governed as one unit (see COPYING).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <openssl/sha.h>

#include "GPUMath.h"

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

#define ASSEMBLY_SIGMA 1  /* funnel-shift sigma macros in GPUHash.h (test) */
#include "GPUHash.h"

/* ============================================================
 * Signed-digit fixed-base geometry (B1).
 *
 * Table entry (c,d) = (2d+1) * 2^(16c) * (G/2), d in [0, 2^15), where
 * G/2 = (2^-1 mod n) * G. Two coordinate arrays (SoA), 16 chunks * 2^15
 * entries * 32 B = 16 MiB each, 32 MiB total -- half the previous 64 MiB, to
 * stay L2-resident on the 4090 (a 772 MiB w=20 table lost 20%).
 *
 * gt_recode_signed turns the 256-bit scalar k into 16 signed ODD digits e_c
 * (|e_c| < 2^16) with  sum_c e_c * 2^(16c) == 2k (mod n). Then
 *   sum_c e_c * 2^(16c) * (G/2) = ((2k) mod-n representative) * (G/2) = k*G,
 * because n*(G/2) = O so any 2k-congruent representative works. A negative
 * digit selects the same table point with y negated (p - y) -- free. Every
 * digit is odd hence non-zero, so the window multiply is branchless (no skip),
 * which is what lets the next step's table loads issue an iteration ahead.
 *
 * Derivation of the odd digits (Joye-Tunstall regular recoding): make a 2k-
 * representative M odd (M = m0 if m0=2k mod n is odd, else M = n-m0 with a
 * global sign flip; n odd so exactly one of m0, n-m0 is odd, both < 2^256).
 * Then repeatedly e = (M mod 2^17) - 2^16 (odd, in (-2^16,2^16)); M becomes
 * 2*(M>>17)+1, which stays odd -- so every extracted digit is odd. 15 windowed
 * digits + the (< 2^16) remainder = 16 digits. Verified in Python over 10^5 k.
 * ============================================================ */
#define GT_CHUNKS   16
#define GT_BITS     16
#define GT_ENTRIES  (1u << 15)       /* odd multiples 1,3,..,2^16-1 -> 2^15 entries */
#define GT_LO       256              /* L[lo] = lo * base_c,  lo in [1,255] (lo odd used) */
#define GT_HI       256              /* H[hi] = hi * 256 * base_c, hi in [1,255]; H[0]=O  */
/* Two coordinate arrays of GT_CHUNKS*GT_ENTRIES*32 B; total must be 32 MiB. */
static_assert((unsigned long long)GT_CHUNKS * GT_ENTRIES * 32ULL * 2ULL == 32ULL*1024*1024,
              "signed-digit G-table must be exactly 32 MiB total (16 MiB per array)");

/* n = secp256k1 group order, little-endian limbs */
__device__ __constant__ uint64_t GT_ORDER_N[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};

/* k -> 16 signed odd digits. Branchless (no data-dependent BRA) so warps stay
 * convergent; correctness mirrored on CPU by the same source. */
/* Recode state: the odd 2k-representative M (4 limbs) plus a global sign.
 * gt_recode_setup computes it once; gt_recode_step peels one signed odd digit
 * per chunk and advances M. The window multiply carries this 32-byte state and
 * peels digits on the fly, so the 16-entry digit array never materialises
 * (that array was the largest single spill source). gt_recode_signed keeps the
 * array form for the CPU cross-check; both share the same step logic. */
__device__ __forceinline__ void gt_recode_setup(const uint64_t k[4], uint64_t M[4], int *sign) {
    const uint64_t n0=GT_ORDER_N[0], n1=GT_ORDER_N[1], n2=GT_ORDER_N[2], n3=GT_ORDER_N[3];
    __uint128_t s;
    /* Reduce the input mod n first: the caller may pass a raw hash z (>= n).
     * k < 2^256 < 2n, so one conditional subtract suffices; then 2*(k mod n) < 2n
     * and the 2k-mod-n step below (one more subtract) is exact. For a k already
     * < n this is a no-op. */
    s=(__uint128_t)k[0]-n0;    uint64_t kd0=(uint64_t)s; uint64_t kb=(uint64_t)(s>>64)&1;
    s=(__uint128_t)k[1]-n1-kb; uint64_t kd1=(uint64_t)s; kb=(uint64_t)(s>>64)&1;
    s=(__uint128_t)k[2]-n2-kb; uint64_t kd2=(uint64_t)s; kb=(uint64_t)(s>>64)&1;
    s=(__uint128_t)k[3]-n3-kb; uint64_t kd3=(uint64_t)s; kb=(uint64_t)(s>>64)&1;
    uint64_t km = (uint64_t)0 - (1ULL - kb);   /* all-ones if k >= n (no borrow) */
    uint64_t k0=(k[0]&~km)|(kd0&km), k1=(k[1]&~km)|(kd1&km),
             k2=(k[2]&~km)|(kd2&km), k3=(k[3]&~km)|(kd3&km);
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

__device__ __forceinline__ int32_t gt_recode_step(uint64_t M[4], int sign, int c) {
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

__device__ __forceinline__ void gt_recode_signed(const uint64_t k[4], int32_t e[16]) {
    uint64_t M[4]; int sign; gt_recode_setup(k, M, &sign);
    #pragma unroll
    for (int c=0;c<16;c++) e[c]=gt_recode_step(M, sign, c);
}

/* Load table point (c, idx) into (gx,gy); negate y (p - y) when neg != 0.
 * Branchless: y is selected between y and p-y by a mask. */
__device__ __forceinline__ void gt_load_signed(const uint8_t *gTX, const uint8_t *gTY,
                                                int c, uint32_t idx, uint64_t neg,
                                                uint64_t gx[4], uint64_t gy[4]) {
    size_t off = ((size_t)c * GT_ENTRIES + idx) * 32;
    const ulonglong2 *tx=(const ulonglong2 *)(gTX+off), *ty=(const ulonglong2 *)(gTY+off);
    ulonglong2 x0=tx[0],x1=tx[1],y0=ty[0],y1=ty[1];
    gx[0]=x0.x;gx[1]=x0.y;gx[2]=x1.x;gx[3]=x1.y;
    uint64_t y[4]={y0.x,y0.y,y1.x,y1.y}, yn[4];
    _ModNeg256(yn, y);                                 /* p - y */
    uint64_t m = 0 - neg;                              /* all-ones if negative digit */
    gy[0]=(y[0]&~m)|(yn[0]&m); gy[1]=(y[1]&~m)|(yn[1]&m);
    gy[2]=(y[2]&~m)|(yn[2]&m); gy[3]=(y[3]&~m)|(yn[3]&m);
}

/* Branchless windowed fixed-base multiply in homogeneous projective coords.
 * 16 signed digits -> 1 seed load + 15 mixed adds; the next chunk's load is
 * issued one iteration ahead. Returns (qx,qy,qz) WITHOUT affine conversion so
 * the caller shares one inverse across the recid finish. */
__device__ __forceinline__ void gt_digit_idx(int32_t ec, uint32_t *idx, uint64_t *neg) {
    uint32_t ae = (uint32_t)(ec < 0 ? -ec : ec);   /* branchless SEL, not BRA */
    *idx = (ae - 1) >> 1;
    *neg = (ec < 0) ? 1ULL : 0ULL;
}

/* Signed-digit fixed-base multiply, accumulating INTERNALLY in XYZZ (x=X/ZZ,
 * y=Y/ZZZ). Seed with an mmadd of the first two chunks' points (4M+2S), then 14
 * madd (8M+2S each) -- vs 15 homogeneous adds at 9M+2S, so ~ -14M/candidate for
 * the +3M end conversion below. Rolled loop (fully unrolling inlines the asm
 * multiply ~150x past ptxas' budget); the back-edge is a uniform loop-counter
 * branch, and every signed odd digit is non-zero so there is NO data-dependent
 * branch and no chunk is skipped. Next chunk's table point loaded one step ahead.
 *
 * The OUTPUT is homogeneous projective (qx,qy,qz) -- identical signature to the
 * previous multiply -- so the downstream conjugate pair + block inverse are
 * unchanged. Convert XYZZ->homogeneous once: X'=X*ZZZ, Y'=Y*ZZ, Z'=ZZ*ZZZ
 * (X'/Z' = X/ZZ = x, Y'/Z' = Y/ZZZ = y). */
__device__ void _FixedBaseSignedProj(uint64_t *qx, uint64_t *qy, uint64_t *qz,
                                      const int32_t e[16], const uint8_t *gTX, const uint8_t *gTY) {
    uint32_t idx; uint64_t neg;
    uint64_t x0[4],y0[4],x1[4],y1[4];
    gt_digit_idx(e[0], &idx, &neg); gt_load_signed(gTX,gTY,0,idx,neg,x0,y0);
    gt_digit_idx(e[1], &idx, &neg); gt_load_signed(gTX,gTY,1,idx,neg,x1,y1);
    uint64_t X[4],Y[4],ZZ[4],ZZZ[4];
    _PointAddXYZZ_mm(X,Y,ZZ,ZZZ, x0,y0, x1,y1);        /* seed = P0 + P1 */
    /* No software prefetch: keeping the next table point live alongside the
     * 128-byte XYZZ accumulator raised spills (92/64 -> measured worse). Load
     * each chunk just-in-time; the loads are still independent (indices known
     * from the recoded digits) so the hardware overlaps them. */
    uint64_t cx[4],cy[4];
    #pragma unroll 1
    for (int c=2;c<16;c++){
        gt_digit_idx(e[c], &idx, &neg); gt_load_signed(gTX,gTY,c,idx,neg,cx,cy);
        _PointAddXYZZ(X,Y,ZZ,ZZZ, cx,cy);
    }
    _ModMult(qx, X, ZZZ);      /* X' = X*ZZZ */
    _ModMult(qy, Y, ZZ);       /* Y' = Y*ZZ  */
    _ModMult(qz, ZZ, ZZZ);     /* Z' = ZZ*ZZZ */
    qz[4]=0;
}

/* ===== BENCHMARK GATE : replaces the DER check (see candidates/README.md) =====
 * Relaxed validity: recovered-key hash h has >= QSB_ZEROS_N leading zero BITS
 * AND, read as a big-endian 256-bit integer, is a valid secp256k1 x-coordinate
 * (retained EC-point check). Set N at compile time: nvcc ... -DQSB_ZEROS_N=24
 * Matches the Python verifier (leading_zero_bits(h) >= N only). */
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
/* gpu_bench_oncurve: removed with gpu_is_on_curve, its only callee. The ranked
 * gate is leading-zero bits only; no on-curve check follows it. */

__device__ int gpu_bench_valid(const uint8_t *h) {
    return gpu_leading_zero_bits(h) >= QSB_ZEROS_N;  /* leading-zeros gate only; no on-curve(h) check */
}
/* Same gate, read straight off the SHA-256 state words. h is those words in
 * big-endian order, so "the first QSB_ZEROS_N bits are zero" is a test on the
 * top bits of hs[0], hs[1], ... The ranked path therefore never materialises
 * the 32-byte digest or walks it a byte at a time. */
__device__ __forceinline__ int gpu_bench_valid_words(const uint32_t *hs) {
    int ok = 1;
    #pragma unroll
    for (int i = 0; i < QSB_ZEROS_N / 32; i++) ok &= (hs[i] == 0u);
#if (QSB_ZEROS_N % 32) != 0
    ok &= ((hs[QSB_ZEROS_N / 32] >> (32 - (QSB_ZEROS_N % 32))) == 0u);
#endif
    return ok;
}

// Use the promoted 8x32 multiply schedule for the inverse product tree.
// Preserve the final reduction carry and canonicalize before _ModInv.
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
    // A 256-bit result can exceed p only when its upper 192 bits are all ones.
    if ((r1 & r2 & r3) == UINT64_MAX && r0 >= 0xFFFFFFFEFFFFFC2FULL) {
        r0 -= 0xFFFFFFFEFFFFFC2FULL;
        r1 = r2 = r3 = 0;
    }
    out[0]=r0;out[1]=r1;out[2]=r2;out[3]=r3;out[4]=0;
}
/* Shared-denominator affine finish for both recovery flags.
 * P = (X:Y:Z) homogeneous projective, R = (xR, yR) affine (u2R).
 * Stage 1 produces the value to invert, W = Z*(xR*Z - X). */
__device__ __forceinline__ void qsb_affine_finish_prepare(uint64_t *X, uint64_t *Z, uint64_t *xR, uint64_t *D, uint64_t *W) {
    uint64_t t[4];
    _ModMult(t, xR, Z);
    _ModSub256(D, t, X);
    W[4] = 0;
    _ModMult(W, Z, D);
}

/* Stage 2: inv = 1/W. Outputs Q1 = P + R and Q2 = P - R in affine form. */
__device__ __forceinline__ void qsb_affine_finish(uint64_t *X, uint64_t *Y, uint64_t *Z, uint64_t *D, uint64_t *inv,
                                                  uint64_t *xR, uint64_t *yR,
                                                  uint64_t *x1, uint64_t *y1, uint64_t *x2, uint64_t *y2) {
    uint64_t iZ[4], xP[4], yP[4], z2[4], id[4], s[4], t[4], m1[4], m2[4], sq[4], xs[4];
    _ModMult(iZ, inv, D);          /* 1/Z */
    _ModMult(xP, X, iZ);
    _ModMult(yP, Y, iZ);
    _ModSqr(z2, Z);
    _ModMult(id, inv, z2);         /* 1/(xR - xP) */
    _ModSub256(s, yR, yP);
    _ModMult(m1, s, id);           /* lambda1 */
    _ModAdd256(t, yR, yP);
    _ModMult(m2, t, id);           /* -lambda2 */
    _ModAdd256(xs, xP, xR);
    _ModSqr(sq, m1);
    _ModSub256(x1, sq, xs);
    _ModSub256(t, xP, x1);
    _ModMult(y1, m1, t);
    _ModSub256(y1, y1, yP);
    _ModSqr(sq, m2);
    _ModSub256(x2, sq, xs);
    _ModSub256(t, xP, x2);
    _ModMult(y2, m2, t);
    _ModAdd256(y2, y2, yP);
    _ModNeg256(y2);                /* y2 = -(m2*(xP - x2) + yP) */
}

#include "tree_inverse.cuh"

/* ============================================================
 * Pinning kernel: one thread per candidate, 256 threads per block.
 *
 * Candidate (seq, lt) with lt = lt_base + blockIdx*256 + threadIdx.
 * d_midA   : SHA-256 state after the fixed prefix AND suffix block A (holds seq)
 * d_wB     : the 16 big-endian message words of suffix block B (with padding
 *            and length) with the four locktime bytes zeroed
 * lt_rel   : byte offset of locktime inside block B (0..60)
 * Hits record the launch-local index and recid; the host rebuilds lt.
 * ============================================================ */
__global__ void __launch_bounds__(256, 2) kernel_pin(
    const uint32_t * __restrict__ d_midA,
    const uint32_t * __restrict__ d_wB,
    int lt_rel,
    uint32_t lt_base,
    const uint64_t * __restrict__ d_u2rx, const uint64_t * __restrict__ d_u2ry,
    uint8_t * __restrict__ d_gtX, uint8_t * __restrict__ d_gtY,
    uint32_t *d_hit_cnt, uint32_t *d_hit_idx
) {
    uint32_t idx = (uint32_t)blockIdx.x * 256u + (uint32_t)threadIdx.x;
    uint32_t lt = lt_base + idx;

    /* Block B: template words plus the four locktime bytes (little-endian
     * value, placed at byte offsets lt_rel..lt_rel+3 of the big-endian words). */
    uint32_t blk[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) blk[i] = d_wB[i];
    #pragma unroll
    for (int b = 0; b < 4; b++) {
        int p = lt_rel + b;
        blk[p >> 2] |= ((lt >> (8 * b)) & 0xFFu) << (24 - 8 * (p & 3));
    }
    uint32_t state[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = d_midA[i];
    _SHA256Transform(state, blk);

    /* Second SHA-256 over the 32-byte first digest (the state words). */
    uint32_t b2[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) b2[i] = state[i];
    b2[8] = 0x80000000u;
    b2[9] = 0; b2[10] = 0; b2[11] = 0; b2[12] = 0; b2[13] = 0; b2[14] = 0;
    b2[15] = 0x00000100u;
    uint32_t s2[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                      0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    _SHA256Transform(s2, b2);

    /* z = big-endian digest; z[0] is the low limb. The table base already
     * carries neg_r_inv, so recoding z yields u1*G directly. */
    uint64_t z[4];
    z[0] = ((uint64_t)s2[6] << 32) | (uint64_t)s2[7];
    z[1] = ((uint64_t)s2[4] << 32) | (uint64_t)s2[5];
    z[2] = ((uint64_t)s2[2] << 32) | (uint64_t)s2[3];
    z[3] = ((uint64_t)s2[0] << 32) | (uint64_t)s2[1];
    int32_t gte[16]; gt_recode_signed(z, gte);
    uint64_t qx[4], qy[4], qz[5]; _FixedBaseSignedProj(qx, qy, qz, gte, d_gtX, d_gtY);

    /* recid 0: Q = u1G + u2R ; recid 1: Q = u1G - u2R (R with odd y). */
    uint64_t u2rx[4] = {d_u2rx[0], d_u2rx[1], d_u2rx[2], d_u2rx[3]};
    uint64_t u2ry[4] = {d_u2ry[0], d_u2ry[1], d_u2ry[2], d_u2ry[3]};
    uint64_t fD[4], prod[5] = {0, 0, 0, 0, 0};
    qsb_affine_finish_prepare(qx, qz, u2rx, fD, prod);
    /* Every launch covers whole blocks, so every lane is active. */
    qsb_block_inverse_tree(prod);
    uint64_t q1x[4], q1y[4], q2x[4], q2y[4];
    qsb_affine_finish(qx, qy, qz, fD, prod, u2rx, u2ry, q1x, q1y, q2x, q2y);

    int v = 0, recid = 0;
    uint64_t *pts_x[2] = {q1x, q2x};
    uint64_t *pts_y[2] = {q1y, q2y};
    for (int ri = 0; ri < 2 && !v; ri++) {
        uint32_t *x32 = (uint32_t *)pts_x[ri];
        uint32_t pb[16];
        uint8_t prefix_byte = 0x2 + (uint8_t)(pts_y[ri][0] & 1);
        pb[0] = __byte_perm(x32[7], prefix_byte, 0x4321);
        pb[1] = __byte_perm(x32[7], x32[6], 0x0765); pb[2] = __byte_perm(x32[6], x32[5], 0x0765);
        pb[3] = __byte_perm(x32[5], x32[4], 0x0765); pb[4] = __byte_perm(x32[4], x32[3], 0x0765);
        pb[5] = __byte_perm(x32[3], x32[2], 0x0765); pb[6] = __byte_perm(x32[2], x32[1], 0x0765);
        pb[7] = __byte_perm(x32[1], x32[0], 0x0765); pb[8] = __byte_perm(x32[0], 0x80, 0x0456);
        pb[9] = 0; pb[10] = 0; pb[11] = 0; pb[12] = 0; pb[13] = 0; pb[14] = 0; pb[15] = 0x108;
        uint32_t hs[8]; _SHA256Initialize(hs); _SHA256Transform(hs, pb);
        if (gpu_bench_valid_words(hs)) { v = 1; recid = ri; }
    }

    if (v) {
        uint32_t p = atomicAdd(d_hit_cnt, 1);
        if (p < 1024) d_hit_idx[p] = idx | ((uint32_t)recid << 31);
    }
}

/* ============================================================
 * Fixed-base table construction on the GPU (signed-digit table)
 *
 * Entry (ch, d) is (2d+1) * 2^(16*ch) * (G/2) in affine form, limbs little-
 * endian -- the layout _FixedBaseSignedProj indexes. base_c = 2^(16c)*(G/2).
 *
 * Building it on the host would cost a modular inversion per entry through
 * OpenSSL. Split the odd index instead: with m = 2d+1 = hi*256 + lo,
 *
 *     m*base_c = hi*(256*base_c) + lo*base_c = H[hi] + L[lo]
 *
 * so the host only produces two short ladders per chunk (L[lo]=lo*base_c,
 * H[hi]=hi*256*base_c) and every table entry is ONE independent mixed addition
 * plus one inversion -- perfectly parallel, one inversion per thread.
 *
 * m is odd so lo is odd (never 0); L[0] is never referenced. H[0] is the
 * identity (m < 256) -> copy L[lo]. H[hi] == +-L[lo] would need m == 0 (mod n),
 * impossible for m in [1, 2^16-1]. Base G/2 = (2^-1 mod n)*G.
 * ============================================================ */

/* Geometry (GT_CHUNKS/GT_ENTRIES/GT_LO/GT_HI) is defined once near the top,
 * beside gt_recode_signed / _FixedBaseSignedProj. Base of chunk c is
 * base_c = 2^(16c) * (G/2). Entry (c,d) = (2d+1)*base_c with m = 2d+1 odd;
 * split m = hi*256 + lo, lo odd in [1,255], hi in [0,255]:
 *     m*base_c = H[hi] + L[lo],  L[lo] = lo*base_c,  H[hi] = hi*256*base_c.
 * H[0] is the identity (m < 256) -> copy L[lo]; lo is always odd so never 0,
 * so L[0] is never referenced. H[hi] == +-L[lo] would need m == 0 (mod n),
 * impossible for m in [1, 2^16-1]. */
__global__ void kernel_build_gtable(
    const uint64_t * __restrict__ d_L,   /* [GT_CHUNKS][GT_LO][8] : x[4] then y[4] */
    const uint64_t * __restrict__ d_H,   /* [GT_CHUNKS][GT_HI][8] */
    uint8_t * __restrict__ gTableX, uint8_t * __restrict__ gTableY)
{
    uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (uint64_t)GT_CHUNKS * GT_ENTRIES) return;
    int ch = (int)(t >> 15);
    int d  = (int)(t & (GT_ENTRIES - 1));
    int m  = 2*d + 1;                        /* odd multiple in [1, 2^16-1] */
    int hi = m >> 8, lo = m & 255;           /* lo odd, hi in [0,255] */

    const uint64_t *Hp = d_H + ((size_t)ch * GT_HI + hi) * 8;
    const uint64_t *Lp = d_L + ((size_t)ch * GT_LO + lo) * 8;

    uint64_t rx[4], ry[4];
    if (hi == 0) {
        for (int k = 0; k < 4; k++) { rx[k] = Lp[k]; ry[k] = Lp[4 + k]; }
    } else {
        uint64_t px[4], py[4], pz[5] = {1, 0, 0, 0, 0}, qx[4], qy[4];
        for (int k = 0; k < 4; k++) {
            px[k] = Hp[k]; py[k] = Hp[4 + k];
            qx[k] = Lp[k]; qy[k] = Lp[4 + k];
        }
        _PointAddSecp256k1(px, py, pz, qx, qy);
        _ModInv(pz);
        _ModMult(px, pz); _ModMult(py, pz);
        for (int k = 0; k < 4; k++) { rx[k] = px[k]; ry[k] = py[k]; }
    }
    /* Limbs are little-endian in memory, which is exactly the table's byte
     * order, so the store is a straight copy. */
    size_t off = ((size_t)ch * GT_ENTRIES + d) * 32;
    memcpy(gTableX + off, rx, 32);
    memcpy(gTableY + off, ry, 32);
}

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

/* The two ladders the GPU builder needs: L[ch][lo] = lo * 2^(16ch) * G and
 * H[ch][hi] = hi * 256 * 2^(16ch) * G. Index 0 of each is the identity and is
 * left zeroed; the kernel treats it as such. 8176 real points, against the
 * 1,048,576 the host would otherwise have to make affine one at a time. */
/* Build the ladders for base A/2 where A = neg_r_inv * G (problem-dependent).
 * With the table on base A, recoding z directly gives z*A = z*neg_r_inv*G =
 * (neg_r_inv*z mod n)*G = u1*G, so the kernel skips gpu_scalar_mulmod. neg_r_inv
 * comes from the runtime problem (little-endian 32 bytes), so the ladders are
 * rebuilt per instance and NOT cached across problems (anti-replay). */
static void gt_build_ladders(uint64_t *hL, uint64_t *hH, const uint8_t neg_r_inv[32]) {
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *x = BN_new(), *y = BN_new(), *shift = BN_new(), *inv2 = BN_new(),
           *order = BN_new(), *nri = BN_new(), *bscal = BN_new();
    EC_POINT *base = EC_POINT_new(grp), *step = EC_POINT_new(grp), *acc = EC_POINT_new(grp);
    /* base = A/2 = (2^-1 * neg_r_inv mod n) * G */
    EC_GROUP_get_order(grp, order, ctx);
    BN_set_word(shift, 2); BN_mod_inverse(inv2, shift, order, ctx);
    BN_lebin2bn(neg_r_inv, 32, nri);                     /* neg_r_inv is LE, like d_nri */
    BN_mod_mul(bscal, inv2, nri, order, ctx);            /* (2^-1 * neg_r_inv) mod n */
    EC_POINT_mul(grp, base, bscal, NULL, NULL, ctx);     /* base = bscal * G = A/2 */
    memset(hL, 0, (size_t)GT_CHUNKS * GT_LO * 8 * sizeof(uint64_t));
    memset(hH, 0, (size_t)GT_CHUNKS * GT_HI * 8 * sizeof(uint64_t));
    for (int ch = 0; ch < GT_CHUNKS; ch++) {
        if (ch > 0) { BN_set_word(shift, 65536); EC_POINT_mul(grp, base, NULL, base, shift, ctx); }
        EC_POINT_copy(acc, base);
        for (int lo = 1; lo < GT_LO; lo++) {                 /* L[lo] = lo * B */
            gt_point_to_limbs(grp, acc, x, y, ctx, hL + ((size_t)ch * GT_LO + lo) * 8);
            EC_POINT_add(grp, acc, acc, base, ctx);
        }
        BN_set_word(shift, 256);                             /* step = 256 * B */
        EC_POINT_mul(grp, step, NULL, base, shift, ctx);
        EC_POINT_copy(acc, step);
        for (int hi = 1; hi < GT_HI; hi++) {                 /* H[hi] = hi * 256 * B */
            gt_point_to_limbs(grp, acc, x, y, ctx, hH + ((size_t)ch * GT_HI + hi) * 8);
            EC_POINT_add(grp, acc, acc, step, ctx);
        }
    }
    BN_free(x); BN_free(y); BN_free(shift); BN_free(inv2); BN_free(order);
    BN_free(nri); BN_free(bscal);
    EC_POINT_free(base); EC_POINT_free(step); EC_POINT_free(acc);
    EC_GROUP_free(grp); BN_CTX_free(ctx);
}

/* Spot-check the built table against OpenSSL. The builder runs on hardware this
 * code has never executed on, so a silent wrong table -- which would simply
 * produce zero verifiable hits and burn the whole run -- must be caught here
 * and fall back, not discovered from the scorecard. */
static int gt_spot_check(const uint8_t *gTableX, const uint8_t *gTableY, int samples,
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
        if (t < GT_CHUNKS * 4) {
            ch = t / 4;
            static const int corner[4] = {0, 1, 2, GT_ENTRIES - 1};
            i = corner[t % 4];
        } else {
            seed = seed * 1664525u + 1013904223u;
            ch = (int)(seed >> 28) % GT_CHUNKS;
            i  = (int)((seed >> 4) & (GT_ENTRIES - 1));
        }
        /* want = (2i+1) * 2^(16*ch) * (A/2) = (2i+1) * 2^(16*ch) * (inv2*neg_r_inv) * G (mod n) */
        BN_one(k);
        BN_lshift(k, k, 16 * ch);
        BN_mul_word(k, (BN_ULONG)(2*i + 1));
        BN_mod_mul(k, k, half_nri, order, ctx);
        EC_POINT_mul(grp, pt, k, NULL, NULL, ctx);
        gt_point_to_limbs(grp, pt, x, y, ctx, want);
        size_t off = ((size_t)ch * GT_ENTRIES + i) * 32;
        if (memcmp(gTableX + off, want,     32) != 0 ||
            memcmp(gTableY + off, want + 4, 32) != 0) {
            fprintf(stderr, "  GTable spot check FAILED at chunk %d entry %d\n", ch, i);
            ok = 0;
        }
    }
    BN_free(x); BN_free(y); BN_free(k); BN_free(inv2); BN_free(order); BN_free(nri); BN_free(half_nri);
    EC_POINT_free(pt); EC_GROUP_free(grp); BN_CTX_free(ctx);
    return ok;
}

/* OpenSSL fallback builder (only if the GPU builder's spot check fails). Emits
 * the signed table: entry (ch,d) = (2d+1) * 2^(16ch) * (G/2). Walks odd
 * multiples by stepping 2*base_c per entry (acc = base_c, 3base_c, ...). */
static void compute_gtable(uint8_t *gTableX, uint8_t *gTableY, const uint8_t neg_r_inv[32]) {
    size_t gt_bytes = (size_t)GT_CHUNKS * GT_ENTRIES * 32;
    /* No cache: base A/2 is problem-dependent (neg_r_inv fresh per instance). */
    printf("  Computing GTable (OpenSSL fallback)...\n");
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *x = BN_new(), *y = BN_new(), *shift = BN_new(), *inv2 = BN_new(), *order = BN_new(),
           *nri = BN_new(), *bscal = BN_new();
    EC_POINT *base = EC_POINT_new(grp), *pt = EC_POINT_new(grp), *two_base = EC_POINT_new(grp);
    /* base = A/2 = (2^-1 * neg_r_inv mod n) * G */
    EC_GROUP_get_order(grp, order, ctx);
    BN_set_word(shift, 2); BN_mod_inverse(inv2, shift, order, ctx);
    BN_lebin2bn(neg_r_inv, 32, nri);
    BN_mod_mul(bscal, inv2, nri, order, ctx);
    EC_POINT_mul(grp, base, bscal, NULL, NULL, ctx);
    for (int ch = 0; ch < GT_CHUNKS; ch++) {
        if (ch > 0) { BN_set_word(shift, 65536); EC_POINT_mul(grp, base, NULL, base, shift, ctx); }
        BN_set_word(shift, 2); EC_POINT_mul(grp, two_base, NULL, base, shift, ctx);  /* 2*base_c */
        EC_POINT_copy(pt, base);                                                     /* (2*0+1)*base_c */
        for (unsigned d = 0; d < GT_ENTRIES; d++) {
            EC_POINT_get_affine_coordinates_GFp(grp, pt, x, y, ctx);
            uint8_t xb[32], yb[32]; memset(xb,0,32); memset(yb,0,32);
            BN_bn2bin(x, xb+(32-BN_num_bytes(x)));
            BN_bn2bin(y, yb+(32-BN_num_bytes(y)));
            for(int j=0;j<16;j++){uint8_t t=xb[j];xb[j]=xb[31-j];xb[31-j]=t;}
            for(int j=0;j<16;j++){uint8_t t=yb[j];yb[j]=yb[31-j];yb[31-j]=t;}
            size_t off = ((size_t)ch * GT_ENTRIES + d) * 32;
            memcpy(gTableX + off, xb, 32);
            memcpy(gTableY + off, yb, 32);
            if (d < GT_ENTRIES - 1) EC_POINT_add(grp, pt, pt, two_base, ctx);
        }
    }
    BN_free(x);BN_free(y);BN_free(shift);BN_free(inv2);BN_free(order);BN_free(nri);BN_free(bscal);
    EC_POINT_free(base);EC_POINT_free(pt);EC_POINT_free(two_base);
    EC_GROUP_free(grp);BN_CTX_free(ctx);
    (void)gt_bytes;
}
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


/* Launch size: 32768 blocks x 256 threads = 8,388,608 candidates. */
#define QSB_PIN_LAUNCH_BLOCKS 32768

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <pinning.bin> [gpu_index] [total_gpus] [global_offset] [single_hash]\n", argv[0]);
        return 1;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    int gpu_index = (argc >= 3) ? atoi(argv[2]) : 0;
    int total_gpus_override = (argc >= 4) ? atoi(argv[3]) : 0;
    int global_offset = (argc >= 5) ? atoi(argv[4]) : 0;
    /* Only the single-hash gate is implemented (the benchmark verifies
     * SHA256(compress(Q)) only); the flag is accepted for interface parity. */

    cudaSetDevice(gpu_index);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, gpu_index);
    printf("QSB Pinning Search [GPU %d]\n", gpu_index);
    printf("  GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    pinning2_params_t pp;
    if (load_pinning2(argv[1], &pp) < 0) return 1;

    /* Shape: fixed prefix block-aligned, sequence entirely in suffix block A,
     * locktime entirely in block B, and block B plus padding fits one block. */
    uint32_t prefix_len = pp.total_preimage_len - pp.suffix_len;
    if (pp.suffix_len <= 64 || pp.suffix_len - 64 > 55 || (prefix_len % 64) != 0
        || pp.seq_offset + 4 > 64 || pp.lt_offset < 64 || pp.lt_offset + 4 > pp.suffix_len) {
        fprintf(stderr, "ERROR: unsupported pinning shape (suffix=%u seq@%u lt@%u total=%u)\n",
                pp.suffix_len, pp.seq_offset, pp.lt_offset, pp.total_preimage_len);
        return 1;
    }
    int lt_rel = (int)pp.lt_offset - 64;
    uint32_t h_wB[16];
    {
        uint8_t bb[64];
        memset(bb, 0, sizeof(bb));
        int rem = (int)pp.suffix_len - 64;
        memcpy(bb, pp.suffix + 64, (size_t)rem);
        bb[rem] = 0x80;
        uint64_t bit_len = (uint64_t)pp.total_preimage_len * 8;
        for (int i = 0; i < 8; i++) bb[56 + i] = (uint8_t)(bit_len >> (56 - 8 * i));
        for (int b = 0; b < 4; b++) bb[lt_rel + b] = 0;
        for (int i = 0; i < 16; i++)
            h_wB[i] = ((uint32_t)bb[i*4] << 24) | ((uint32_t)bb[i*4+1] << 16)
                    | ((uint32_t)bb[i*4+2] << 8) | (uint32_t)bb[i*4+3];
    }

    size_t gt_sz = (size_t)GT_CHUNKS*GT_ENTRIES*32;
    uint8_t *d_gtX, *d_gtY;
    cudaMalloc(&d_gtX,gt_sz);cudaMalloc(&d_gtY,gt_sz);
    {
        /* Build the fixed-base table on the GPU. The host only produces the two
         * small ladders; the million entries are one parallel addition each.
         * The result is then spot-checked against OpenSSL, and anything that
         * does not match falls back to the original host builder -- a wrong
         * table yields zero verifiable hits, so it must never reach the run. */
        struct timespec ta, tb; clock_gettime(CLOCK_MONOTONIC, &ta);
        size_t lb = (size_t)GT_CHUNKS*GT_LO*8*sizeof(uint64_t);
        size_t hb = (size_t)GT_CHUNKS*GT_HI*8*sizeof(uint64_t);
        uint64_t *hL=(uint64_t*)malloc(lb), *hH=(uint64_t*)malloc(hb);
        if(!hL||!hH){ fprintf(stderr,"OOM: gtable ladders\n"); return 1; }
        gt_build_ladders(hL,hH,pp.neg_r_inv);
        uint64_t *dL=NULL,*dH=NULL; cudaMalloc(&dL,lb); cudaMalloc(&dH,hb);
        cudaMemcpy(dL,hL,lb,cudaMemcpyHostToDevice);
        cudaMemcpy(dH,hH,hb,cudaMemcpyHostToDevice);
        free(hL); free(hH);
        int gt_total = GT_CHUNKS*GT_ENTRIES;
        kernel_build_gtable<<<(gt_total+255)/256,256>>>(dL,dH,d_gtX,d_gtY);
        cudaDeviceSynchronize();
        cudaError_t gerr = cudaGetLastError();
        cudaFree(dL); cudaFree(dH);
        uint8_t *chk_x=(uint8_t*)malloc(gt_sz), *chk_y=(uint8_t*)malloc(gt_sz);
        if(!chk_x||!chk_y){ fprintf(stderr,"OOM: gtable check\n"); return 1; }
        int gt_ok = (gerr==cudaSuccess);
        if(gt_ok){
            cudaMemcpy(chk_x,d_gtX,gt_sz,cudaMemcpyDeviceToHost);
            cudaMemcpy(chk_y,d_gtY,gt_sz,cudaMemcpyDeviceToHost);
            gt_ok = gt_spot_check(chk_x,chk_y,GT_CHUNKS*4+192,pp.neg_r_inv);
        }
        clock_gettime(CLOCK_MONOTONIC, &tb);
        double gt_secs=(tb.tv_sec-ta.tv_sec)+(tb.tv_nsec-ta.tv_nsec)/1e9;
        if(gt_ok){
            printf("  GTable built on GPU in %.2fs (%d entries/array, %.0f MiB total, spot check passed)\n",
                   gt_secs, gt_total, (double)(2*gt_sz)/(1024*1024));
        } else {
            printf("  GTable GPU build rejected (%s); using the host builder\n",
                   gerr!=cudaSuccess ? cudaGetErrorString(gerr) : "spot check failed");
            compute_gtable(chk_x,chk_y,pp.neg_r_inv);
            cudaMemcpy(d_gtX,chk_x,gt_sz,cudaMemcpyHostToDevice);
            cudaMemcpy(d_gtY,chk_y,gt_sz,cudaMemcpyHostToDevice);
        }
        fflush(stdout);
        free(chk_x); free(chk_y);
    }

    uint32_t *d_wB; cudaMalloc(&d_wB, sizeof(h_wB));
    cudaMemcpy(d_wB, h_wB, sizeof(h_wB), cudaMemcpyHostToDevice);
    uint32_t *d_midA; cudaMalloc(&d_midA, 32);
    uint64_t *d_u2rx, *d_u2ry;
    cudaMalloc(&d_u2rx, 32); cudaMalloc(&d_u2ry, 32);
    cudaMemcpy(d_u2rx, pp.u2r_x, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2ry, pp.u2r_y, 32, cudaMemcpyHostToDevice);

    cudaDeviceSetLimit(cudaLimitStackSize, 32768);
    uint32_t *d_hit_cnt, *d_hit_idx;
    cudaMalloc(&d_hit_cnt, 4); cudaMalloc(&d_hit_idx, 1024 * 4);

    /* Safe ranges (as in the production grinder). */
    const uint32_t LT_MIN = 500000000u;
    const uint32_t LT_MAX = 1744600000u;
    const uint32_t SEQ_MIN = 0x80000000u;
    const uint32_t lt_range = LT_MAX - LT_MIN;

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus < 1) num_gpus = 1;
    int effective_total = (total_gpus_override > 0) ? total_gpus_override : num_gpus;
    int effective_id = global_offset + gpu_index;
    printf("  Search: lt=[%u,%u), seq=0x%08X+%d step %d, %d blocks x 256 per launch\n",
           LT_MIN, LT_MAX, SEQ_MIN, effective_id, effective_total, QSB_PIN_LAUNCH_BLOCKS);

    struct timespec t0, t_last;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    t_last = t0;
    uint64_t total_searched = 0, hit_total = 0;
    mkdir("results", 0755);
    char fname[64];
    snprintf(fname, sizeof(fname), "results/pinning_hit_%d.txt", gpu_index);

    for (uint32_t seq = SEQ_MIN + (uint32_t)effective_id; ; seq += (uint32_t)effective_total) {
        /* Compress suffix block A (holds seq) once for this sequence. */
        uint8_t blockA[64];
        memcpy(blockA, pp.suffix, 64);
        for (int b = 0; b < 4; b++) blockA[pp.seq_offset + b] = (uint8_t)(seq >> (8 * b));
        SHA256_CTX sctx;
        SHA256_Init(&sctx);
        for (int i = 0; i < 8; i++) sctx.h[i] = pp.midstate[i];
        SHA256_Transform(&sctx, blockA);
        uint32_t midA[8];
        for (int i = 0; i < 8; i++) midA[i] = sctx.h[i];
        cudaMemcpy(d_midA, midA, 32, cudaMemcpyHostToDevice);

        for (uint32_t lt_off = 0; lt_range - lt_off >= 256u; ) {
            uint32_t left_blocks = (lt_range - lt_off) / 256u;
            int nblk = (left_blocks < (uint32_t)QSB_PIN_LAUNCH_BLOCKS)
                       ? (int)left_blocks : QSB_PIN_LAUNCH_BLOCKS;
            uint32_t n = (uint32_t)nblk * 256u;
            uint32_t lt_base = LT_MIN + lt_off;

            uint32_t h_hit = 0;
            cudaMemcpy(d_hit_cnt, &h_hit, 4, cudaMemcpyHostToDevice);
            kernel_pin<<<nblk, 256>>>(d_midA, d_wB, lt_rel, lt_base,
                                      d_u2rx, d_u2ry, d_gtX, d_gtY,
                                      d_hit_cnt, d_hit_idx);
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
            total_searched += n;
            lt_off += n;

            cudaMemcpy(&h_hit, d_hit_cnt, 4, cudaMemcpyDeviceToHost);
            if (h_hit > 0) {
                uint32_t nh = (h_hit > 1024u) ? 1024u : h_hit;
                static uint32_t hits[1024];
                cudaMemcpy(hits, d_hit_idx, nh * 4, cudaMemcpyDeviceToHost);
                FILE *f = fopen(fname, "a");
                if (f) {
                    for (uint32_t h = 0; h < nh; h++) {
                        uint32_t raw = hits[h];
                        uint32_t lt = lt_base + (raw & 0x7FFFFFFFu);
                        int ri = (int)(raw >> 31);
                        fprintf(f, "sequence=%u\nlocktime=%u\nhash_choice=0\nrecid=%d\n", seq, lt, ri);
                    }
                    fclose(f);
                }
                hit_total += nh;
            }

            struct timespec t_now;
            clock_gettime(CLOCK_MONOTONIC, &t_now);
            double since = (t_now.tv_sec - t_last.tv_sec) + (t_now.tv_nsec - t_last.tv_nsec) / 1e9;
            if (since >= 10.0) {
                double el = (t_now.tv_sec - t0.tv_sec) + (t_now.tv_nsec - t0.tv_nsec) / 1e9;
                printf("  [GPU %d] seq 0x%08X (%lluM/0M) %.1fM/s hits=%llu %.0fs\n",
                       gpu_index, seq, (unsigned long long)(total_searched / 1000000),
                       total_searched / el / 1e6, (unsigned long long)hit_total, el);
                t_last = t_now;
            }
        }
    }
    return 0;
}
