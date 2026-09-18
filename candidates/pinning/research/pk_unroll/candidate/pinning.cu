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

#include "GPUMath.h"

static_assert(sizeof(ulonglong2) == 16, "pipeline vector must be 128 bits");
static_assert(alignof(ulonglong2) == 16, "pipeline vector must be 16-byte aligned");

/* ---- Experiment switches (fable round 2). Defaults are set at the end of
 * this block; ab.sh overrides them with -D on the build line. ---- */
#ifndef QSB_TREE_N
#define QSB_TREE_N 128        /* leaves per candidate product tree = prepare/finish block size (256, 128 or 64) */
#endif
#ifndef QSB_S0_SHM
#define QSB_S0_SHM 0          /* 1: keep the recode state and the y anchor in shared memory (register relief) */
#endif
#if QSB_TREE_N != 256 && QSB_TREE_N != 128 && QSB_TREE_N != 64
#error "QSB_TREE_N must be 256, 128 or 64"
#endif
#ifndef QSB_BATCH
#define QSB_BATCH 16777216    /* candidates per pipeline launch */
#endif
#ifndef QSB_PREFETCH
#define QSB_PREFETCH 0        /* 0: none, 1: next chunk one step ahead, 2: all chunks up front */
#endif
#ifndef QSB_STREAM
#define QSB_STREAM 0          /* 1: .cs (evict-first) hints on pipeline state/tree traffic */
#endif
#ifndef QSB_TREE_OFFLOAD
#define QSB_TREE_OFFLOAD 0    /* 1: build the leaf product tree in a dense kernel, not in prepare */
#endif
#ifndef QSB_S0_THREADS
#define QSB_S0_THREADS 256    /* prepare-kernel block size (only free when the tree is offloaded) */
#endif
#ifndef QSB_S0_BLOCKS
#define QSB_S0_BLOCKS 2       /* prepare-kernel resident blocks per SM */
#endif
#ifndef QSB_TREE_BLOCKS
#define QSB_TREE_BLOCKS 4     /* dense tree kernel resident blocks per SM */
#endif
#ifndef QSB_TREE_OFFLOAD2
#define QSB_TREE_OFFLOAD2 0   /* 1: expand the leaf inverses in a dense kernel, not in finish */
#endif
#ifndef QSB_S2_THREADS
#define QSB_S2_THREADS 256    /* finish-kernel block size (only free when the down-tree is offloaded) */
#endif
#ifndef QSB_S2_BLOCKS
#define QSB_S2_BLOCKS 3       /* finish-kernel resident blocks per SM */
#endif
#if QSB_TREE_N != 256 && QSB_S2_THREADS == 256
#undef QSB_S2_THREADS
#define QSB_S2_THREADS QSB_TREE_N
#undef QSB_S2_BLOCKS
#define QSB_S2_BLOCKS (768/QSB_TREE_N)    /* keep 768 threads per SM at 80 registers */
#endif
#if QSB_S2_THREADS != QSB_TREE_N && !QSB_TREE_OFFLOAD2
#error "finish block size must equal the tree width unless the inverse tree is offloaded"
#endif
#ifndef QSB_EARLY_LOAD
#define QSB_EARLY_LOAD 0      /* 1: load the next table record inside the mixed addition, once cx/cy die */
#endif
#ifndef QSB_UNROLL
#define QSB_UNROLL 1          /* unroll factor of the 13-iteration chain loop */
#endif
#ifndef QSB_PK_UNROLL
#define QSB_PK_UNROLL 1       /* 1: unroll the two-recid pubkey SHA loop so both chains interleave */
#endif
#ifndef QSB_L2_SKIP
#define QSB_L2_SKIP 0         /* 1: start the persisting-L2 window after chunk 0 (half the access density) */
#endif
#ifndef QSB_HOST_READBACK
#define QSB_HOST_READBACK 0   /* delta A (jungjipdo a91746ca): one blocking readback of counter+indices per batch */
#endif
#ifndef QSB_SPARSE_TAIL
#define QSB_SPARSE_TAIL 1     /* delta B (scarletbright 7f965b4d): sparse-schedule transform for the 11-byte tail block */
#endif
#ifndef QSB_FINAL_TEMPLATE
#define QSB_FINAL_TEMPLATE 1  /* delta C (jacklightChen e582bda4): compile-time final (resolved) XYZZ addition */
#endif
#ifndef QSB_SPARSE_D
#define QSB_SPARSE_D 1        /* delta D (preludebrace bc77eb42, unmeasured): sparse SHA256d-second and pubkey transforms */
#endif
#ifndef QSB_SYM_FINISH
#define QSB_SYM_FINISH 1      /* delta E (xlib 0c6f4c8): symmetric recovery, 6 state planes, K=3xR^2 constant */
#endif
#define QSB_STATE_PLANES (QSB_SYM_FINISH ? 6u : 8u)
#ifndef QSB_PROBE_MASK
#define QSB_PROBE_MASK 0      /* speed probe only: mask table indices to shrink the working set (wrong math) */
#endif
#if QSB_TREE_N != 256 && QSB_S0_THREADS == 256
#undef QSB_S0_THREADS
#define QSB_S0_THREADS QSB_TREE_N
#undef QSB_S0_BLOCKS
#define QSB_S0_BLOCKS (512/QSB_TREE_N)    /* keep 4 x 128 = 8 x 64 = 512 threads per SM */
#endif
#if QSB_S0_THREADS != QSB_TREE_N && !QSB_TREE_OFFLOAD
#error "prepare block size must equal the tree width unless the tree is offloaded"
#endif

__device__ __forceinline__ void qsb_prefetch_l2(const void *p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
/* Streaming (evict-first) 128-bit and 64-bit global accesses. */
__device__ __forceinline__ void qsb_st_v2(ulonglong2 *p, uint64_t a, uint64_t b) {
#if QSB_STREAM
    asm volatile("{ .reg .u64 g; cvta.to.global.u64 g, %0; st.global.cs.v2.u64 [g], {%1,%2}; }"
                 :: "l"(p), "l"(a), "l"(b) : "memory");
#else
    *p = make_ulonglong2(a, b);
#endif
}
__device__ __forceinline__ ulonglong2 qsb_ld_v2(const ulonglong2 *p) {
#if QSB_STREAM
    uint64_t a, b;
    asm volatile("{ .reg .u64 g; cvta.to.global.u64 g, %2; ld.global.cs.v2.u64 {%0,%1}, [g]; }"
                 : "=l"(a), "=l"(b) : "l"(p) : "memory");
    return make_ulonglong2(a, b);
#else
    return *p;
#endif
}
__device__ __forceinline__ void qsb_st_u64(uint64_t *p, uint64_t a) {
#if QSB_STREAM
    asm volatile("{ .reg .u64 g; cvta.to.global.u64 g, %0; st.global.cs.u64 [g], %1; }"
                 :: "l"(p), "l"(a) : "memory");
#else
    *p = a;
#endif
}
__device__ __forceinline__ uint64_t qsb_ld_u64(const uint64_t *p) {
#if QSB_STREAM
    uint64_t a;
    asm volatile("{ .reg .u64 g; cvta.to.global.u64 g, %1; ld.global.cs.u64 %0, [g]; }"
                 : "=l"(a) : "l"(p) : "memory");
    return a;
#else
    return *p;
#endif
}

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

/* Mixed regular odd digits: widths [18,17,...,17], 15 chunks.
 * Chunk c starts at bit 0 when c=0, otherwise 17*c+1. Entry d is
 * (2*d+1)*2^offset*(A/2). Every digit is odd and nonzero; the reconstruction
 * is 2*k modulo the group order, as in the original regular recoder.
 * First chunk has 2^17 entries, others 2^16: 2^20 points, 64 MiB total. */
#define GT_CHUNKS 15
#define GT_TOTAL_ENTRIES (1u << 20)
#define GT_LO 256
#define GT_HI 1024
__host__ __device__ __forceinline__ unsigned gt_entries(int c) {
    return c == 0 ? (1u << 17) : (1u << 16);
}
__host__ __device__ __forceinline__ unsigned gt_offset(int c) {
    return c == 0 ? 0u : (unsigned)(c+1) << 16;
}
__host__ __device__ __forceinline__ int gt_shift(int c) {
    return c == 0 ? 0 : 17*c+1;
}
static_assert(GT_TOTAL_ENTRIES*64ULL == 64ULL*1024*1024,
              "mixed table must contain exactly 64 MiB");

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
__device__ __forceinline__ void gt_recode_signed(const uint64_t k[4], int32_t e[GT_CHUNKS]) {
    uint64_t M[4]; int sign; gt_recode_setup(k,M,&sign);
    e[0]=gt_mixed_step<18>(M,sign);
    #pragma unroll
    for(int c=1;c<GT_CHUNKS-1;c++)e[c]=gt_mixed_step<17>(M,sign);
    e[GT_CHUNKS-1]=sign*(int32_t)M[0];
}

/* Load table point (c, idx) into (gx,gy); negate y (p - y) when neg != 0.
 * Branchless: y is selected between y and p-y by a mask. */
__device__ __forceinline__ void gt_load_signed_flat(const uint8_t *gTable,
                                                     uint32_t base, uint32_t idx,
                                                     uint64_t neg,
                                                     uint64_t gx[4], uint64_t gy[4]) {
    size_t off = ((size_t)base + idx) * 64;
    const ulonglong2 *tx=(const ulonglong2 *)(gTable+off);
    const ulonglong2 *ty=(const ulonglong2 *)(gTable+off+32);
    ulonglong2 x0=tx[0],x1=tx[1],y0=ty[0],y1=ty[1];
    gx[0]=x0.x;gx[1]=x0.y;gx[2]=x1.x;gx[3]=x1.y;
    uint64_t m=0ULL-neg;
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
    uint32_t ae = (uint32_t)(ec < 0 ? -ec : ec);   /* branchless SEL, not BRA */
    *idx = (ae - 1) >> 1;
#if QSB_PROBE_MASK
    *idx &= (uint32_t)QSB_PROBE_MASK;
#endif
    *neg = (ec < 0) ? 1ULL : 0ULL;
}

/* Signed-digit fixed-base multiply, accumulating INTERNALLY in XYZZ (x=X/ZZ,
 * y=Y/ZZZ). Seed the first two chunks with a deferred-Y mmadd (3M+2S), adjust
 * each next point's y by the preceding affine anchor, and defer the new anchor
 * term through every intermediate addition. Only the final addition resolves
 * Y exactly. This costs 3M+2S + 12*(7M+2S) + 8M+2S = 95M+28S across all
 * 15 points instead of 108M+28S. Fully unrolling
 * inlines the asm multiply ~150x past ptxas' budget); the back-edge is a
 * uniform loop-counter
 * branch, and every signed odd digit is non-zero so there is NO data-dependent
 * branch and no chunk is skipped. Next chunk's table point loaded one step ahead.
 *
 * Production consumes the raw XYZZ output directly. This avoids the three
 * field multiplications formerly used to convert it to homogeneous projective
 * form before recovery. */
__device__ void _FixedBaseSignedXYZZ(uint64_t *X, uint64_t *Y,
                                      uint64_t *ZZ, uint64_t *ZZZ,
                                      const int32_t e[GT_CHUNKS],
                                      const uint8_t *gTable) {
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
    for (int c=2;c<GT_CHUNKS;c++){
        gt_digit_idx(e[c], &idx, &neg); gt_load_signed(gTable,c,idx,neg,cx,cy);
        _PointAddXYZZ(X,Y,ZZ,ZZZ, cx,cy, y0, c != GT_CHUNKS-1);
        Load256(y0, cy);                /* current affine y anchors next madd */
    }
}

#if QSB_EARLY_LOAD
/* Mixed addition with the next record's loads issued as soon as the current
 * record is consumed: X2 dies after U2, Y2 after S2. The next digit is peeled
 * one step ahead (one register); the loaded point lands in nx/ny while the
 * remaining 5M+2S of this addition execute. */
__device__ __forceinline__ void _PointAddXYZZ_early(
    uint64_t *X1, uint64_t *Y1, uint64_t *ZZ1, uint64_t *ZZZ1,
    const uint64_t *X2, const uint64_t *Y2, const uint64_t *Yoff, bool defer_y,
    bool do_load, const uint8_t *gTable, uint32_t nbase, uint32_t nidx, uint64_t nneg,
    uint64_t *nx, uint64_t *ny)
{
  uint64_t U2[4], S2[4], P[4], R[4], PP[4], PPP[4], Q[4], T[4];
  _ModMult(U2, (uint64_t *)X2, ZZ1);   // U2 = X2*ZZ1
#if QSB_LAZY
  _ModAddLazy(S2, Y2, Yoff);
#else
  _ModAdd256(S2, (uint64_t *)Y2, (uint64_t *)Yoff);
#endif
  _ModMult(S2, ZZZ1);                  // S2 = (Y2+Yoff)*ZZZ1
  if (do_load) gt_load_signed_flat(gTable, nbase, nidx, nneg, nx, ny);
  _ModSub256(P, U2, X1);
  _ModSub256(R, S2, Y1);
  _ModSqr(PP, P);
  _ModMult(PPP, PP, P);
  _ModMult(Q, U2, PP);
  _ModMult(ZZ1, PP);
  _ModSqr(T, R);
#if QSB_LAZY
  _ModX3Fused(T, T, PPP, Q);
#else
  _ModAdd256(T, T, PPP);
  _ModSub256(T, T, Q);
  _ModSub256(T, T, Q);
#endif
  _ModMult(ZZZ1, PPP);
  _ModSub256(Q, Q, T);
  _ModMult(Q, R);
  if (defer_y) {
    Load256(Y1, Q);
  } else {
    _ModMult(S2, (uint64_t *)Y2, ZZZ1);
    _ModSub256(Y1, Q, S2);
  }
  Load256(X1, T);
}
#endif

/* Production scalar-entry form: consume the mixed signed digits as they are
 * generated instead of materializing an address-taken digit array. */
__device__ void _FixedBaseSignedXYZZScalar(uint64_t *X, uint64_t *Y,
                                            uint64_t *ZZ, uint64_t *ZZZ,
                                            const uint64_t k[4],
                                            const uint8_t *gTable,
                                            uint64_t (*scratch)[2*QSB_TREE_N]) {
#if QSB_S0_SHM
    /* Cold per-thread state lives in the (currently unused) tree buffer:
     * limb-major, lane-contiguous, so each access is one 8-byte shared op.
     * Slot [k][tid] holds M[k]; slot [k][QSB_TREE_N+tid] holds y0[k]. */
    #define QSB_M_LD(i)   (scratch[i][threadIdx.x])
    #define QSB_M_ST(i,v) (scratch[i][threadIdx.x]=(v))
    #define QSB_Y0_LD(i)  (scratch[i][QSB_TREE_N+threadIdx.x])
    #define QSB_Y0_ST(i,v) (scratch[i][QSB_TREE_N+threadIdx.x]=(v))
#else
    (void)scratch;
#endif
    uint64_t M[4]; int sign;
    gt_recode_setup(k, M, &sign);
    uint32_t idx; uint64_t neg;
    uint64_t x0[4],y0[4],x1[4],y1[4];
#if QSB_PREFETCH == 2
    {   /* Warm L2 with every remaining table record before the addition chain.
         * Runs the recoder on a copy; digits 0 and 1 are loaded immediately
         * below, so only chunks 2..14 are prefetched. */
        uint64_t Mp[4]={M[0],M[1],M[2],M[3]};
        (void)gt_mixed_step<18>(Mp,sign);
        (void)gt_mixed_step<17>(Mp,sign);
        uint32_t pb=gt_offset(2);
        #pragma unroll
        for(int c=2;c<GT_CHUNKS;c++){
            int32_t e=(c<GT_CHUNKS-1)?gt_mixed_step<17>(Mp,sign):sign*(int32_t)Mp[0];
            uint32_t pi; uint64_t pn; gt_digit_idx(e,&pi,&pn);
            qsb_prefetch_l2(gTable+((size_t)pb+pi)*64);
            qsb_prefetch_l2(gTable+((size_t)pb+pi)*64+32);
            pb+=1u<<16;
        }
    }
#endif
    int32_t ec=gt_mixed_step<18>(M,sign);
    gt_digit_idx(ec, &idx, &neg); gt_load_signed(gTable,0,idx,neg,x0,y0);
    ec=gt_mixed_step<17>(M,sign);
    gt_digit_idx(ec, &idx, &neg); gt_load_signed(gTable,1,idx,neg,x1,y1);
    _PointAddXYZZ_mm(X,Y,ZZ,ZZZ, x0,y0, x1,y1);
    uint64_t cx[4],cy[4];
    uint32_t table_base=gt_offset(2);
#if QSB_PREFETCH == 1
    /* Peel the digit one chunk ahead and prefetch its record into L2. Only the
     * next signed digit (one register) stays live, not the 64-byte point. */
    int32_t ec_next=gt_mixed_step<17>(M,sign);
    gt_digit_idx(ec_next, &idx, &neg);
    qsb_prefetch_l2(gTable+((size_t)table_base+idx)*64);
    qsb_prefetch_l2(gTable+((size_t)table_base+idx)*64+32);
    #pragma unroll 1
    for (int c=2;c<GT_CHUNKS;c++){
        ec=ec_next;
        if (c<GT_CHUNKS-1) {
            ec_next=(c+1<GT_CHUNKS-1)?gt_mixed_step<17>(M,sign):sign*(int32_t)M[0];
            uint32_t pi; uint64_t pn; gt_digit_idx(ec_next,&pi,&pn);
            qsb_prefetch_l2(gTable+((size_t)table_base+(1u<<16)+pi)*64);
            qsb_prefetch_l2(gTable+((size_t)table_base+(1u<<16)+pi)*64+32);
        }
        gt_digit_idx(ec, &idx, &neg); gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
        _PointAddXYZZ(X,Y,ZZ,ZZZ, cx,cy, y0, c != GT_CHUNKS-1);
        Load256(y0, cy);                /* current affine y anchors next madd */
        table_base += 1u << 16;
    }
#elif QSB_S0_SHM
    #pragma unroll
    for(int i=0;i<4;i++){ QSB_M_ST(i,M[i]); QSB_Y0_ST(i,y0[i]); }
    #pragma unroll 1
    for (int c=2;c<GT_CHUNKS;c++){
        uint64_t Ms[4]; int32_t ecs;
        #pragma unroll
        for(int i=0;i<4;i++) Ms[i]=QSB_M_LD(i);
        if (c<GT_CHUNKS-1) {
            ecs=gt_mixed_step<17>(Ms,sign);
            #pragma unroll
            for(int i=0;i<4;i++) QSB_M_ST(i,Ms[i]);
        } else {
            ecs=sign*(int32_t)Ms[0];
        }
        gt_digit_idx(ecs, &idx, &neg); gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
        uint64_t ya[4];
        #pragma unroll
        for(int i=0;i<4;i++) ya[i]=QSB_Y0_LD(i);
        _PointAddXYZZ(X,Y,ZZ,ZZZ, cx,cy, ya, c != GT_CHUNKS-1);
        #pragma unroll
        for(int i=0;i<4;i++) QSB_Y0_ST(i,cy[i]);   /* current affine y anchors next madd */
        table_base += 1u << 16;
    }
    #undef QSB_M_LD
    #undef QSB_M_ST
    #undef QSB_Y0_LD
    #undef QSB_Y0_ST
#elif QSB_EARLY_LOAD
    /* Chunk 2 is loaded up front; every later record is loaded inside the
     * preceding addition. Two point buffers alternate roles. */
    uint64_t nx[4],ny[4];
    ec=gt_mixed_step<17>(M,sign);
    gt_digit_idx(ec, &idx, &neg); gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
    #pragma unroll 1
    for (int c=2;c<GT_CHUNKS;c+=2){
        /* iteration c: consume cx/cy, load c+1 into nx/ny */
        bool more=(c+1<GT_CHUNKS);
        uint32_t ni=0; uint64_t nn=0;
        if (more) { ec=(c+1<GT_CHUNKS-1)?gt_mixed_step<17>(M,sign):sign*(int32_t)M[0]; gt_digit_idx(ec,&ni,&nn); }
        _PointAddXYZZ_early(X,Y,ZZ,ZZZ, cx,cy, y0, c != GT_CHUNKS-1,
                            more, gTable, table_base+(1u<<16), ni, nn, nx, ny);
        Load256(y0, cy);
        table_base += 1u << 16;
        if (!more) break;
        /* iteration c+1: consume nx/ny, load c+2 into cx/cy */
        bool more2=(c+2<GT_CHUNKS);
        ni=0; nn=0;
        if (more2) { ec=(c+2<GT_CHUNKS-1)?gt_mixed_step<17>(M,sign):sign*(int32_t)M[0]; gt_digit_idx(ec,&ni,&nn); }
        _PointAddXYZZ_early(X,Y,ZZ,ZZZ, nx,ny, y0, c+1 != GT_CHUNKS-1,
                            more2, gTable, table_base+(1u<<16), ni, nn, cx, cy);
        Load256(y0, ny);
        table_base += 1u << 16;
    }
#elif QSB_FINAL_TEMPLATE
    /* Delta C (jacklightChen e582bda4): twelve deferred additions in the
     * rolled loop, then the resolving final addition as its own compile-time
     * specialisation, so the loop body carries no defer_y branch. */
    #pragma unroll 1
    for (int c=2;c<GT_CHUNKS-1;c++){
        ec=gt_mixed_step<17>(M,sign);
        gt_digit_idx(ec, &idx, &neg); gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
        _PointAddXYZZT<true>(X,Y,ZZ,ZZZ, cx,cy, y0);
        Load256(y0, cy);                /* current affine y anchors next madd */
        table_base += 1u << 16;
    }
    ec=sign*(int32_t)M[0];
    gt_digit_idx(ec, &idx, &neg);
    gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
    _PointAddXYZZT<false>(X,Y,ZZ,ZZZ, cx,cy, y0);
#else
#if QSB_UNROLL == 2
    #pragma unroll 2
#elif QSB_UNROLL == 3
    #pragma unroll 3
#else
    #pragma unroll 1
#endif
    for (int c=2;c<GT_CHUNKS;c++){
        ec=(c<GT_CHUNKS-1)?gt_mixed_step<17>(M,sign):sign*(int32_t)M[0];
        gt_digit_idx(ec, &idx, &neg); gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
        _PointAddXYZZ(X,Y,ZZ,ZZZ, cx,cy, y0, c != GT_CHUNKS-1);
        Load256(y0, cy);                /* current affine y anchors next madd */
        table_base += 1u << 16;
    }
#endif
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


/* Sparse-schedule SHA-256 for the Fast 11-byte locktime tail block
 * (delta B, scarletbright 7f965b4d). Pad shape: W[0..2] live, W[3..14]=0,
 * W[15]=9995*8=79960. Continues from an existing midstate. The first 16
 * rounds and the first in-place WMIX drop zero addends; later rounds use the
 * generic SHA256_RND / WMIX schedule. Bit-identical to _SHA256Transform on
 * that padded block. */
__device__ __forceinline__ void _SHA256TransformFastTail11(
    uint32_t state[8], uint32_t w0, uint32_t w1, uint32_t w2)
{
    const uint32_t L = 9995u * 8u; /* 79960 */
    uint32_t t1;
    uint32_t t2;

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    uint32_t w[16];
    w[0] = w0;
    w[1] = w1;
    w[2] = w2;
#pragma unroll
    for (int i = 3; i < 15; i++) w[i] = 0;
    w[15] = L;

    S2Round(a, b, c, d, e, f, g, h, K[0], w[0]);
    S2Round(h, a, b, c, d, e, f, g, K[1], w[1]);
    S2Round(g, h, a, b, c, d, e, f, K[2], w[2]);
    S2Round(f, g, h, a, b, c, d, e, K[3], 0u);
    S2Round(e, f, g, h, a, b, c, d, K[4], 0u);
    S2Round(d, e, f, g, h, a, b, c, K[5], 0u);
    S2Round(c, d, e, f, g, h, a, b, K[6], 0u);
    S2Round(b, c, d, e, f, g, h, a, K[7], 0u);
    S2Round(a, b, c, d, e, f, g, h, K[8], 0u);
    S2Round(h, a, b, c, d, e, f, g, K[9], 0u);
    S2Round(g, h, a, b, c, d, e, f, K[10], 0u);
    S2Round(f, g, h, a, b, c, d, e, K[11], 0u);
    S2Round(e, f, g, h, a, b, c, d, K[12], 0u);
    S2Round(d, e, f, g, h, a, b, c, K[13], 0u);
    S2Round(c, d, e, f, g, h, a, b, K[14], 0u);
    S2Round(b, c, d, e, f, g, h, a, K[15], L);

    {
        w[0] += s0(w[1]);
        w[1] += s1(L) + s0(w[2]);
        w[2] += s1(w[0]);
        w[3]  = s1(w[1]);
        w[4]  = s1(w[2]);
        w[5]  = s1(w[3]);
        w[6]  = s1(w[4]) + L;
        w[7]  = s1(w[5]) + w[0];
        w[8]  = s1(w[6]) + w[1];
        w[9]  = s1(w[7]) + w[2];
        w[10] = s1(w[8]) + w[3];
        w[11] = s1(w[9]) + w[4];
        w[12] = s1(w[10]) + w[5];
        w[13] = s1(w[11]) + w[6];
        w[14] = s1(w[12]) + w[7] + s0(L);
        w[15] += s1(w[13]) + w[8] + s0(w[0]);
    }

    SHA256_RND(16);
    WMIX();
    SHA256_RND(32);
    WMIX();
    SHA256_RND(48);

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

/* Sparse-schedule SHA-256 for the SHA256d second compression (delta D,
 * preludebrace bc77eb42): 32-byte message = first digest as eight words,
 * fixed pad W[8]=0x80000000, W[9..14]=0, W[15]=256, from the SHA-256 IV.
 * Bit-identical to _SHA256Initialize + _SHA256Transform on that block. */
__device__ __forceinline__ void _SHA256TransformDigest32(
    uint32_t out[8], const uint32_t m[8])
{
    uint32_t t1;
    uint32_t t2;

    uint32_t a = 0x6a09e667u;
    uint32_t b = 0xbb67ae85u;
    uint32_t c = 0x3c6ef372u;
    uint32_t d = 0xa54ff53au;
    uint32_t e = 0x510e527fu;
    uint32_t f = 0x9b05688cu;
    uint32_t g = 0x1f83d9abu;
    uint32_t h = 0x5be0cd19u;

    uint32_t w[16];
#pragma unroll
    for (int i = 0; i < 8; i++) w[i] = m[i];

    S2Round(a, b, c, d, e, f, g, h, K[0], w[0]);
    S2Round(h, a, b, c, d, e, f, g, K[1], w[1]);
    S2Round(g, h, a, b, c, d, e, f, K[2], w[2]);
    S2Round(f, g, h, a, b, c, d, e, K[3], w[3]);
    S2Round(e, f, g, h, a, b, c, d, K[4], w[4]);
    S2Round(d, e, f, g, h, a, b, c, K[5], w[5]);
    S2Round(c, d, e, f, g, h, a, b, K[6], w[6]);
    S2Round(b, c, d, e, f, g, h, a, K[7], w[7]);
    S2Round(a, b, c, d, e, f, g, h, K[8], 0x80000000u);
    S2Round(h, a, b, c, d, e, f, g, K[9], 0u);
    S2Round(g, h, a, b, c, d, e, f, K[10], 0u);
    S2Round(f, g, h, a, b, c, d, e, K[11], 0u);
    S2Round(e, f, g, h, a, b, c, d, K[12], 0u);
    S2Round(d, e, f, g, h, a, b, c, K[13], 0u);
    S2Round(c, d, e, f, g, h, a, b, K[14], 0u);
    S2Round(b, c, d, e, f, g, h, a, K[15], 256u);

    {
        /* First schedule expansion; w[9..14]=0 and w[8]/w[15] are the fixed
         * pad words. s0(0)=s1(0)=0, so zero terms vanish. */
        w[0] += s0(w[1]);
        w[1] += s1(256u) + s0(w[2]);
        w[2] += s1(w[0]) + s0(w[3]);
        w[3] += s1(w[1]) + s0(w[4]);
        w[4] += s1(w[2]) + s0(w[5]);
        w[5] += s1(w[3]) + s0(w[6]);
        w[6] += s1(w[4]) + 256u + s0(w[7]);
        w[7] += s1(w[5]) + w[0] + s0(0x80000000u);
        w[8]  = 0x80000000u + s1(w[6]) + w[1];
        w[9]  = s1(w[7]) + w[2];
        w[10] = s1(w[8]) + w[3];
        w[11] = s1(w[9]) + w[4];
        w[12] = s1(w[10]) + w[5];
        w[13] = s1(w[11]) + w[6];
        w[14] = s1(w[12]) + w[7] + s0(256u);
        w[15] = 256u + s1(w[13]) + w[8] + s0(w[0]);
    }

    SHA256_RND(16);
    WMIX();
    SHA256_RND(32);
    WMIX();
    SHA256_RND(48);

    out[0] = 0x6a09e667u + a;
    out[1] = 0xbb67ae85u + b;
    out[2] = 0x3c6ef372u + c;
    out[3] = 0xa54ff53au + d;
    out[4] = 0x510e527fu + e;
    out[5] = 0x9b05688cu + f;
    out[6] = 0x1f83d9abu + g;
    out[7] = 0x5be0cd19u + h;
}

/* Sparse-schedule SHA-256 for the 33-byte compressed public key (delta D):
 * live words pb[0..8], W[9..14]=0, W[15]=0x108, from the SHA-256 IV.
 * Bit-identical to _SHA256Initialize + _SHA256Transform on that block. */
__device__ __forceinline__ void _SHA256TransformPubkey33(
    uint32_t out[8], const uint32_t m[9])
{
    uint32_t t1;
    uint32_t t2;

    uint32_t a = 0x6a09e667u;
    uint32_t b = 0xbb67ae85u;
    uint32_t c = 0x3c6ef372u;
    uint32_t d = 0xa54ff53au;
    uint32_t e = 0x510e527fu;
    uint32_t f = 0x9b05688cu;
    uint32_t g = 0x1f83d9abu;
    uint32_t h = 0x5be0cd19u;

    uint32_t w[16];
#pragma unroll
    for (int i = 0; i < 9; i++) w[i] = m[i];

    S2Round(a, b, c, d, e, f, g, h, K[0], w[0]);
    S2Round(h, a, b, c, d, e, f, g, K[1], w[1]);
    S2Round(g, h, a, b, c, d, e, f, K[2], w[2]);
    S2Round(f, g, h, a, b, c, d, e, K[3], w[3]);
    S2Round(e, f, g, h, a, b, c, d, K[4], w[4]);
    S2Round(d, e, f, g, h, a, b, c, K[5], w[5]);
    S2Round(c, d, e, f, g, h, a, b, K[6], w[6]);
    S2Round(b, c, d, e, f, g, h, a, K[7], w[7]);
    S2Round(a, b, c, d, e, f, g, h, K[8], w[8]);
    S2Round(h, a, b, c, d, e, f, g, K[9], 0u);
    S2Round(g, h, a, b, c, d, e, f, K[10], 0u);
    S2Round(f, g, h, a, b, c, d, e, K[11], 0u);
    S2Round(e, f, g, h, a, b, c, d, K[12], 0u);
    S2Round(d, e, f, g, h, a, b, c, K[13], 0u);
    S2Round(c, d, e, f, g, h, a, b, K[14], 0u);
    S2Round(b, c, d, e, f, g, h, a, K[15], 0x108u);

    {
        /* First schedule expansion; w[9..14]=0 and w[15]=0x108 is fixed. */
        w[0] += s0(w[1]);
        w[1] += s1(0x108u) + s0(w[2]);
        w[2] += s1(w[0]) + s0(w[3]);
        w[3] += s1(w[1]) + s0(w[4]);
        w[4] += s1(w[2]) + s0(w[5]);
        w[5] += s1(w[3]) + s0(w[6]);
        w[6] += s1(w[4]) + 0x108u + s0(w[7]);
        w[7] += s1(w[5]) + w[0] + s0(w[8]);
        w[8] += s1(w[6]) + w[1];
        w[9]  = s1(w[7]) + w[2];
        w[10] = s1(w[8]) + w[3];
        w[11] = s1(w[9]) + w[4];
        w[12] = s1(w[10]) + w[5];
        w[13] = s1(w[11]) + w[6];
        w[14] = s1(w[12]) + w[7] + s0(0x108u);
        w[15] = 0x108u + s1(w[13]) + w[8] + s0(w[0]);
    }

    SHA256_RND(16);
    WMIX();
    SHA256_RND(32);
    WMIX();
    SHA256_RND(48);

    out[0] = 0x6a09e667u + a;
    out[1] = 0xbb67ae85u + b;
    out[2] = 0x3c6ef372u + c;
    out[3] = 0xa54ff53au + d;
    out[4] = 0x510e527fu + e;
    out[5] = 0x9b05688cu + f;
    out[6] = 0x1f83d9abu + g;
    out[7] = 0x5be0cd19u + h;
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

#define QSB_CHECKPOINT_NODES 254
#define QSB_CHECKPOINT_STRIDE 256
/* Candidate trees may be narrower than the 256-wide root-group trees. */
#define QSB_CAND_STRIDE (QSB_TREE_N)
/* Shared scratch for the prepare kernel: the product tree (2N leaves x 32 B)
 * is dead during the fixed-base chain, so the chain may park cold per-thread
 * state there when QSB_S0_SHM is set. */
__device__ __forceinline__ uint64_t (*qsb_prepare_scratch())[2*QSB_TREE_N] {
    __shared__ uint64_t products[4][2*QSB_TREE_N];
    return products;
}

/* Split form of qsb_block_inverse.  The prepare kernel checkpoints the 254
 * internal non-root product-tree nodes to global memory and publishes the raw root.
 * A small intervening kernel normalizes and inverts each root.  The finish
 * kernel restores the immutable product tree, expands the supplied root
 * inverse, and returns canonical leaf inverses.  The packed node numbering is
 * identical to qsb_block_inverse; leaves come from the saved W field and node
 * 510 is omitted because roots owns it. */
template<int N>
__device__ __forceinline__ void qsb_block_product_checkpoint(
    uint64_t *value, uint64_t *roots, uint64_t *checkpoint, uint64_t (*products)[2*N]
) {
    int tid=threadIdx.x;
    size_t block_base=(size_t)blockIdx.x*4u*N;

    #pragma unroll
    for(int k=0;k<4;k++)products[k][tid]=value[k];
    __syncthreads();

    int offset=0;
    #pragma unroll 1
    for(int count=N;count>1;count>>=1){
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
            int node=offset+count+tid;
            #pragma unroll
            for(int k=0;k<4;k++){
                products[k][node]=out[k];
                if(node<2*N-2)
                    qsb_st_u64(&checkpoint[block_base+(size_t)k*N+node-N],out[k]);
            }
        }
        offset+=count;
        if(count>2)__syncthreads();
    }

    if(tid==0){
        #pragma unroll
        for(int k=0;k<4;k++)roots[(size_t)blockIdx.x*4u+k]=products[k][2*N-2];
    }
}
template<int N>
__device__ __forceinline__ void qsb_block_product_checkpoint(
    uint64_t *value, uint64_t *roots, uint64_t *checkpoint
) {
    __shared__ uint64_t products[4][2*N];
    qsb_block_product_checkpoint<N>(value,roots,checkpoint,products);
}

template<int N>
__device__ __forceinline__ void qsb_block_inverse_checkpoint(
    uint64_t *value, const uint64_t *roots, const uint64_t *checkpoint
) {
    __shared__ uint64_t products[4][2*N];
    __shared__ uint64_t inverses[4][N];
    int tid=threadIdx.x;
    size_t block_base=(size_t)blockIdx.x*4u*N;

    /* Each lane supplies its saved W leaf and all but the last two lanes
     * restore one internal node. Lane zero also publishes the external root inverse.
     * One barrier makes both immutable inputs visible to the downward pass. */
    #pragma unroll
    for(int k=0;k<4;k++){
        products[k][tid]=value[k];
        if(tid<N-2)
            products[k][N+tid]=qsb_ld_u64(&checkpoint[block_base+(size_t)k*N+tid]);
        if(tid==0)inverses[k][N-2]=roots[(size_t)blockIdx.x*4u+k];
    }
    __syncthreads();

    int offset=2*N-4;
    #pragma unroll 1
    for(int count=2;count<N;count<<=1){
        int half=count>>1;
        if(tid<count){
            int local_parent=tid&(half-1);
            uint64_t parent_inv[5],sibling[5],child_inv[5];
            #pragma unroll
            for(int k=0;k<4;k++){
                parent_inv[k]=inverses[k][offset+count-N+local_parent];
                sibling[k]=products[k][offset+(tid^half)];
            }
            parent_inv[4]=sibling[4]=0;
            qsb_field_mul(child_inv,parent_inv,sibling);
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-N+tid]=child_inv[k];
        }
        offset-=count<<1;
        __syncthreads();
    }

    uint64_t parent_inv[5],sibling[5];
    #pragma unroll
    for(int k=0;k<4;k++){
        parent_inv[k]=inverses[k][tid&(N/2-1)];
        sibling[k]=products[k][tid^(N/2)];
    }
    parent_inv[4]=sibling[4]=0;
    qsb_field_mul(value,parent_inv,sibling);
    qsb_field_normalize(value);
}

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
    qsb_block_product_checkpoint<256>(r,super_roots,root_checkpoint);
}

/* One 256-lane CTA per 256 group roots; each CTA runs its own _ModInv, so
 * batches with more than 65,536 candidate trees need no third tree level. */
__global__ void __launch_bounds__(256,1) qsb_invert_super_roots(
    uint64_t *super_roots, int count
) {
    int tid=(int)(blockIdx.x*256u+threadIdx.x);
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
    qsb_block_inverse_checkpoint<256>(r,super_roots,root_checkpoint);
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
__device__ __constant__ uint64_t pin_u2rk_words[4];   /* K = 3*xR^2 (delta E) */

/* Delta E (xlib 0c6f4c8). With I=1/W and V=ZZZ, t=V^2*I=1/(xR-xP). Let
 * u=yR*t and v=Y*V*I, so u-v and -(u+v) are the slopes for P+R and P-R.
 * K=3*xR^2 is fixed for the entire problem. The shared x base is
 * F=2*u^2-K*t+xR; H=2*u*v gives x_plus=F-H, x_minus=F+H. Both y
 * coordinates are anchored at R. Returns their parities in bits 0,1.
 * Only Y, ZZZ and W cross the kernel boundary (six planes). */
__device__ __forceinline__ uint32_t qsb_xyzz_finish_symmetric(
    uint64_t *Y, uint64_t *V, uint64_t *inv,
    uint64_t *xR, uint64_t *yR, uint64_t *K,
    uint64_t *x_plus, uint64_t *x_minus
) {
    uint64_t h[4], u[4], v[4], f[4];
    _ModMult(h, V, inv);         /* h = V*I */
    _ModMult(V, h);              /* V becomes t = V*h */
    _ModMult(u, yR, V);          /* u = yR*t */
    _ModMult(v, Y, h);           /* v = Y*h */

    /* GPUMath's square drops a final carry for some near-p operands. For
     * upper-half u, square the equivalent negative representative; keep u
     * unchanged for H and y. */
    qsb_field_normalize(u);
    if(u[3] >> 63) _ModNeg256(h, u);
    else Load256(h, u);
    _ModSqr(f, h);
    _ModAdd256(f, f, f);
    _ModMult(h, K, V);           /* h becomes K*t */
    _ModSub256(f, h);
    _ModAdd256(f, f, xR);        /* F = 2*u^2-K*t+xR */
    _ModMult(h, u, v);
    _ModAdd256(h, h, h);         /* H = 2*u*v */
    _ModSub256(x_plus, f, h);
    _ModAdd256(x_minus, f, h);
    qsb_field_normalize(x_plus);
    qsb_field_normalize(x_minus);

    _ModSub256(h, u, v);
    _ModSub256(V, xR, x_plus);
    _ModMult(h, V);
    _ModSub256(h, yR);
    qsb_field_normalize(h);
    uint32_t parities = (uint32_t)(h[0] & 1ULL);

    _ModAdd256(h, u, v);
    _ModSub256(V, xR, x_minus);
    _ModMult(h, V);
    _ModSub256(V, yR, h);
    qsb_field_normalize(V);
    parities |= (uint32_t)((V[0] & 1ULL) << 1);
    return parities;
}

template<bool FAST_TAIL, int STAGE>
__global__ void __launch_bounds__(STAGE == 0 ? QSB_S0_THREADS : QSB_S2_THREADS,
                                  STAGE == 0 ? QSB_S0_BLOCKS : QSB_S2_BLOCKS) kernel_pinning_pipeline(
    const uint32_t *d_midstate,
    const uint8_t *d_suffix,    /* suffix template */
    int suffix_len,             /* total suffix including lt+sighash */
    int seq_offset,             /* offset of sequence in suffix */
    int lt_offset,              /* offset of locktime in suffix */
    int total_preimage_len,
    uint32_t seq_value,         /* current sequence value */
    uint32_t start_lt,          /* starting locktime for this batch */
    const uint64_t *d_neg_r_inv,
    const uint64_t *d_u2rx, const uint64_t *d_u2ry,
    const uint64_t *d_neg2u2rx, const uint64_t *d_neg2u2ry,
    uint8_t *d_gt,
    uint32_t *d_hit_cnt, uint32_t *d_hit_idx,
    int batch_size, int easy_mode, int single_hash,
    ulonglong2 *saved, uint64_t *roots, uint64_t *tree
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
#if QSB_SPARSE_TAIL
        /* W[0..2] live locktime-patched words; W[3..14]=0; W[15]=79960. */
        uint32_t w0 = pin_tail_words[0] | (lt & 0xffu);
        uint32_t w1 = ((lt & 0xff00u) << 16) | (lt & 0xff0000u) |
                ((lt >> 16) & 0xff00u) | pin_tail_words[1];
        uint32_t w2 = pin_tail_words[2];
        _SHA256TransformFastTail11(state, w0, w1, w2);
#else
        uint32_t blk[16] = {
            pin_tail_words[0] | (lt & 0xffu),
            ((lt & 0xff00u) << 16) | (lt & 0xff0000u) |
                ((lt >> 16) & 0xff00u) | pin_tail_words[1],
            pin_tail_words[2],
            0,0,0,0,0,0,0,0,0,0,0,0,9995u*8u
        };
        _SHA256Transform(state,blk);
#endif
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
    uint32_t s2[8];
#if QSB_SPARSE_D
    _SHA256TransformDigest32(s2, state);
#else
    {
    uint32_t b2[16];
    #pragma unroll
    for(int i=0;i<8;i++) b2[i]=state[i];
    b2[8]=0x80000000u;
    #pragma unroll
    for(int i=9;i<15;i++) b2[i]=0;
    b2[15]=256;
    const uint32_t iv[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                          0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    #pragma unroll
    for(int i=0;i<8;i++) s2[i]=iv[i];
    _SHA256Transform(s2,b2);
    }
#endif

    /* Scalar from the SHA-256 state words, in little-endian limbs. */
    uint64_t z[4];
    z[0] = ((uint64_t)s2[6] << 32) | (uint64_t)s2[7];
    z[1] = ((uint64_t)s2[4] << 32) | (uint64_t)s2[5];
    z[2] = ((uint64_t)s2[2] << 32) | (uint64_t)s2[3];
    z[3] = ((uint64_t)s2[0] << 32) | (uint64_t)s2[1];
    /* neg_r_inv is folded into fixed base A = neg_r_inv*G. Recoding z
     * directly yields z*A = (neg_r_inv*z mod n)*G without a per-candidate
     * scalar multiplication. */
    /* u1*G as raw XYZZ via the signed 64 MiB A-table. */
    _FixedBaseSignedXYZZScalar(qx,qy,qzz,qzzz,z,d_gt,qsb_prepare_scratch());

    /* Recover P+R and P-R together with one shared denominator inverse. The
     * prepare-only xR copy dies before the collective; reload R afterward so
     * its eight limbs do not lengthen the inverse's already pressured state. */
    {
        uint64_t prep_xR[4]={pin_u2rx_words[0],pin_u2rx_words[1],
                             pin_u2rx_words[2],pin_u2rx_words[3]};
        qsb_xyzz_finish_prepare(qx,qzz,prep_xR,prod);
    }
    bool usable = active && ((prod[0] | prod[1] | prod[2] | prod[3]) != 0);
#if QSB_SYM_FINISH
    /* Delta E: only Y, ZZZ and W cross the kernel boundary (W stays in
     * planes 4-5 so the tree kernels are unchanged). */
    Load256(qzz,prod);           /* qzz becomes W */
    if (!usable) {
        prod[0]=1; prod[1]=prod[2]=prod[3]=prod[4]=0;
    }
    if(active){
        size_t state_plane_stride=(size_t)batch_size;
        size_t state_idx=(size_t)idx;
        qsb_st_v2(&saved[0u*state_plane_stride+state_idx],qy[0],qy[1]);
        qsb_st_v2(&saved[1u*state_plane_stride+state_idx],qy[2],qy[3]);
        qsb_st_v2(&saved[2u*state_plane_stride+state_idx],qzzz[0],qzzz[1]);
        qsb_st_v2(&saved[3u*state_plane_stride+state_idx],qzzz[2],qzzz[3]);
        qsb_st_v2(&saved[4u*state_plane_stride+state_idx],qzz[0],qzz[1]);
        qsb_st_v2(&saved[5u*state_plane_stride+state_idx],qzz[2],qzz[3]);
    }
#else
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
        qsb_st_v2(&saved[0u*state_plane_stride+state_idx],qx[0],qx[1]);
        qsb_st_v2(&saved[1u*state_plane_stride+state_idx],qx[2],qx[3]);
        qsb_st_v2(&saved[2u*state_plane_stride+state_idx],qy[0],qy[1]);
        qsb_st_v2(&saved[3u*state_plane_stride+state_idx],qy[2],qy[3]);
        qsb_st_v2(&saved[4u*state_plane_stride+state_idx],qzz[0],qzz[1]);
        qsb_st_v2(&saved[5u*state_plane_stride+state_idx],qzz[2],qzz[3]);
        qsb_st_v2(&saved[6u*state_plane_stride+state_idx],qzzz[0],qzzz[1]);
        qsb_st_v2(&saved[7u*state_plane_stride+state_idx],qzzz[2],qzzz[3]);
    }
#endif
#if QSB_TREE_OFFLOAD
    /* The leaf product tree is built by qsb_leaf_tree_prepare from the saved
     * W plane, so this kernel has no shared memory and no barriers. */
    (void)prod; (void)roots; (void)tree;
#else
    qsb_block_product_checkpoint<QSB_TREE_N>(prod,roots,tree,qsb_prepare_scratch());
#endif
    return;
    } else {

    bool usable = false;
    if(active){
        size_t state_plane_stride=(size_t)batch_size;
        size_t state_idx=(size_t)idx;
#if QSB_SYM_FINISH
        ulonglong2 qy01=qsb_ld_v2(&saved[0u*state_plane_stride+state_idx]);
        ulonglong2 qy23=qsb_ld_v2(&saved[1u*state_plane_stride+state_idx]);
        ulonglong2 qzzz01=qsb_ld_v2(&saved[2u*state_plane_stride+state_idx]);
        ulonglong2 qzzz23=qsb_ld_v2(&saved[3u*state_plane_stride+state_idx]);
        ulonglong2 qzz01=qsb_ld_v2(&saved[4u*state_plane_stride+state_idx]);
        ulonglong2 qzz23=qsb_ld_v2(&saved[5u*state_plane_stride+state_idx]);
        (void)qx;
#else
        ulonglong2 qx01=qsb_ld_v2(&saved[0u*state_plane_stride+state_idx]);
        ulonglong2 qx23=qsb_ld_v2(&saved[1u*state_plane_stride+state_idx]);
        ulonglong2 qy01=qsb_ld_v2(&saved[2u*state_plane_stride+state_idx]);
        ulonglong2 qy23=qsb_ld_v2(&saved[3u*state_plane_stride+state_idx]);
        ulonglong2 qzz01=qsb_ld_v2(&saved[4u*state_plane_stride+state_idx]);
        ulonglong2 qzz23=qsb_ld_v2(&saved[5u*state_plane_stride+state_idx]);
        ulonglong2 qzzz01=qsb_ld_v2(&saved[6u*state_plane_stride+state_idx]);
        ulonglong2 qzzz23=qsb_ld_v2(&saved[7u*state_plane_stride+state_idx]);
        qx[0]=qx01.x; qx[1]=qx01.y; qx[2]=qx23.x; qx[3]=qx23.y;
#endif
        qy[0]=qy01.x; qy[1]=qy01.y; qy[2]=qy23.x; qy[3]=qy23.y;
        qzz[0]=qzz01.x; qzz[1]=qzz01.y; qzz[2]=qzz23.x; qzz[3]=qzz23.y;
        qzzz[0]=qzzz01.x; qzzz[1]=qzzz01.y; qzzz[2]=qzzz23.x; qzzz[3]=qzzz23.y;
        usable = ((qzz[0] | qzz[1] | qzz[2] | qzz[3]) != 0);
    }
#if QSB_TREE_OFFLOAD2
    /* qsb_leaf_tree_finish replaced the W plane by the canonical leaf inverse
     * (zero for unusable lanes), so qzz already holds 1/W. */
    if (!usable) return;
    Load256(prod,qzz); prod[4]=0;
    (void)roots; (void)tree;
#else
    /* qzz carries the original, pre-identity-substitution W, so this exactly
     * recreates the promoted kernel's usability decision without a flag. */
    #pragma unroll
    for(int limb=0;limb<4;limb++)prod[limb]=usable?qzz[limb]:(limb==0?1ULL:0ULL);
    prod[4]=0;
    qsb_block_inverse_checkpoint<QSB_TREE_N>(prod,roots,tree);
    if (!usable) return;
#endif
    uint64_t u2rx[4]={pin_u2rx_words[0],pin_u2rx_words[1],
                      pin_u2rx_words[2],pin_u2rx_words[3]};
    uint64_t u2ry[4]={pin_u2ry_words[0],pin_u2ry_words[1],
                      pin_u2ry_words[2],pin_u2ry_words[3]};
    uint64_t q1x[4],q2x[4];
#if QSB_SYM_FINISH
    uint64_t u2rk[4]={pin_u2rk_words[0],pin_u2rk_words[1],
                      pin_u2rk_words[2],pin_u2rk_words[3]};
    uint32_t y_parities = qsb_xyzz_finish_symmetric(
        qy,qzzz,prod,u2rx,u2ry,u2rk,
        q1x,q2x);
#else
    uint32_t y_parities = qsb_xyzz_finish_precomputed(
        qx,qy,qzz,qzzz,prod,u2rx,u2ry,
        q1x,q2x);
#endif

    /* Check both pubkeys × 2 hashes */
#if QSB_PK_UNROLL
    #pragma unroll
#else
    #pragma unroll 1
#endif
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
        uint32_t hs[8];
#if QSB_SPARSE_D
        _SHA256TransformPubkey33(hs,pb);   /* pb[9..14]=0, pb[15]=0x108 folded in */
#else
        pb[9]=0;pb[10]=0;pb[11]=0;pb[12]=0;pb[13]=0;pb[14]=0;pb[15]=0x108;
        _SHA256Initialize(hs);_SHA256Transform(hs,pb);
#endif
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
            if(pos<1024)d_hit_idx[pos]=((uint32_t)idx)|(ri<<30);
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
            if(pos<1024)d_hit_idx[pos]=((uint32_t)idx)|(ri<<30)|(1u<<31);
            return;
        }
    }
    }
}

#if QSB_TREE_OFFLOAD
/* Dense product-tree kernel: one 256-leaf tree per CTA over the saved W plane.
 * Candidate i is leaf (i mod 256) of tree (i / 256), exactly the numbering the
 * finish kernel restores. Inactive lanes and zero denominators enter as the
 * identity, as before. It runs at several CTAs per SM, so its barriers and
 * the tree's shrinking active set no longer idle the 124-register prepare
 * kernel. */
__global__ void __launch_bounds__(256,QSB_TREE_BLOCKS) qsb_leaf_tree_prepare(
    const ulonglong2 *saved, int batch_size, uint64_t *roots, uint64_t *tree
) {
    int idx=(int)(blockIdx.x*256u+threadIdx.x);
    uint64_t prod[5]={1ULL,0ULL,0ULL,0ULL,0ULL};
    if(idx<batch_size){
        size_t s=(size_t)batch_size;
        ulonglong2 w01=qsb_ld_v2(&saved[4u*s+(size_t)idx]);
        ulonglong2 w23=qsb_ld_v2(&saved[5u*s+(size_t)idx]);
        if((w01.x|w01.y|w23.x|w23.y)!=0ULL){
            prod[0]=w01.x; prod[1]=w01.y; prod[2]=w23.x; prod[3]=w23.y;
        }
    }
    qsb_block_product_checkpoint<256>(prod,roots,tree);
}
#endif

#if QSB_TREE_OFFLOAD2
/* Dense inverse-expansion kernel: restores the checkpointed tree, expands the
 * group-supplied root inverse down to the 256 leaves and overwrites the W
 * plane in place with each lane's canonical inverse (zero when the lane was
 * unusable, which the finish kernel tests exactly as it tested W). */
__global__ void __launch_bounds__(256,QSB_TREE_BLOCKS) qsb_leaf_tree_finish(
    ulonglong2 *saved, int batch_size, const uint64_t *roots, const uint64_t *tree
) {
    int idx=(int)(blockIdx.x*256u+threadIdx.x);
    bool active=idx<batch_size, usable=false;
    uint64_t prod[5]={1ULL,0ULL,0ULL,0ULL,0ULL};
    size_t s=(size_t)batch_size;
    if(active){
        ulonglong2 w01=qsb_ld_v2(&saved[4u*s+(size_t)idx]);
        ulonglong2 w23=qsb_ld_v2(&saved[5u*s+(size_t)idx]);
        usable=(w01.x|w01.y|w23.x|w23.y)!=0ULL;
        if(usable){ prod[0]=w01.x; prod[1]=w01.y; prod[2]=w23.x; prod[3]=w23.y; }
    }
    qsb_block_inverse_checkpoint<256>(prod,roots,tree);
    if(active){
        if(!usable){ prod[0]=prod[1]=prod[2]=prod[3]=0ULL; }
        qsb_st_v2(&saved[4u*s+(size_t)idx],prod[0],prod[1]);
        qsb_st_v2(&saved[5u*s+(size_t)idx],prod[2],prod[3]);
    }
}
#endif

template<bool FAST_TAIL>
static void launch_pinning_pipeline(
    const uint32_t *d_midstate, const uint8_t *d_suffix,
    int suffix_len, int seq_offset, int lt_offset, int total_preimage_len,
    uint32_t seq_value, uint32_t start_lt,
    const uint64_t *d_neg_r_inv,
    const uint64_t *d_u2rx, const uint64_t *d_u2ry,
    const uint64_t *d_neg2u2rx, const uint64_t *d_neg2u2ry,
    uint8_t *d_gt, uint32_t *d_hit_cnt, uint32_t *d_hit_idx,
    int batch_size, int easy_mode, int single_hash,
    ulonglong2 *saved, uint64_t *roots, uint64_t *tree,
    uint64_t *super_roots, uint64_t *root_checkpoint
) {
    int blocks=(batch_size+QSB_TREE_N-1)/QSB_TREE_N;
    int blocks0=(batch_size+QSB_S0_THREADS-1)/QSB_S0_THREADS;
    kernel_pinning_pipeline<FAST_TAIL,0><<<blocks0,QSB_S0_THREADS>>>(
        d_midstate,d_suffix,suffix_len,seq_offset,lt_offset,total_preimage_len,
        seq_value,start_lt,d_neg_r_inv,d_u2rx,d_u2ry,d_neg2u2rx,d_neg2u2ry,
        d_gt,d_hit_cnt,d_hit_idx,batch_size,easy_mode,single_hash,
        saved,roots,tree);
    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess){
        fprintf(stderr,"Pipeline prepare launch failed: %s\n",cudaGetErrorString(err));
        exit(2);
    }
#if QSB_TREE_OFFLOAD
    qsb_leaf_tree_prepare<<<blocks,256>>>(saved,batch_size,roots,tree);
    err=cudaGetLastError();
    if(err!=cudaSuccess){
        fprintf(stderr,"Leaf-tree launch failed: %s\n",cudaGetErrorString(err));
        exit(2);
    }
#endif
    int root_groups=(blocks+255)/256;
    qsb_root_group_prepare<<<root_groups,256>>>(
        roots,blocks,super_roots,root_checkpoint);
    err=cudaGetLastError();
    if(err!=cudaSuccess){
        fprintf(stderr,"Root-group prepare launch failed: %s\n",cudaGetErrorString(err));
        exit(2);
    }
    qsb_invert_super_roots<<<(root_groups+255)/256,256>>>(super_roots,root_groups);
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
#if QSB_TREE_OFFLOAD2
    qsb_leaf_tree_finish<<<blocks,256>>>(saved,batch_size,roots,tree);
    err=cudaGetLastError();
    if(err!=cudaSuccess){
        fprintf(stderr,"Leaf-inverse launch failed: %s\n",cudaGetErrorString(err));
        exit(2);
    }
#endif
    int blocks2=(batch_size+QSB_S2_THREADS-1)/QSB_S2_THREADS;
    kernel_pinning_pipeline<FAST_TAIL,2><<<blocks2,QSB_S2_THREADS>>>(
        d_midstate,d_suffix,suffix_len,seq_offset,lt_offset,total_preimage_len,
        seq_value,start_lt,d_neg_r_inv,d_u2rx,d_u2ry,d_neg2u2rx,d_neg2u2ry,
        d_gt,d_hit_cnt,d_hit_idx,batch_size,easy_mode,single_hash,
        saved,roots,tree);
    err=cudaGetLastError();
    if(err!=cudaSuccess){
        fprintf(stderr,"Pipeline finish launch failed: %s\n",cudaGetErrorString(err));
        exit(2);
    }
}

/* ============================================================
 * Fixed-base table construction on the GPU (signed-digit table)
 *
 * Entry (ch, d) is (2d+1) * 2^gt_shift(ch) * (A/2) in affine form, limbs
 * little-endian -- the layout _FixedBaseSignedXYZZScalar indexes it.
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
 * impossible for m below 2^18. Base A/2 uses A=neg_r_inv*G.
 * ============================================================ */

/* Mixed geometry (GT_CHUNKS/GT_TOTAL_ENTRIES/GT_LO/GT_HI) is defined once near the top,
 * beside gt_recode_signed / _FixedBaseSignedXYZZScalar. Base of chunk c is
 * base_c = 2^gt_shift(c) * (A/2). Entry (c,d) = (2d+1)*base_c with m odd;
 * split m = hi*256 + lo, lo odd in [1,255], hi below 1024:
 *     m*base_c = H[hi] + L[lo],  L[lo] = lo*base_c,  H[hi] = hi*256*base_c.
 * H[0] is the identity (m < 256) -> copy L[lo]; lo is always odd so never 0,
 * so L[0] is never referenced. H[hi] == +-L[lo] would need m == 0 (mod n),
 * impossible for m below 2^18. */
__global__ void kernel_build_gtable(
    const uint64_t * __restrict__ d_L,   /* [GT_CHUNKS][GT_LO][8] : x[4] then y[4] */
    const uint64_t * __restrict__ d_H,   /* [GT_CHUNKS][GT_HI][8] */
    uint8_t * __restrict__ gTable)
{
    uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= GT_TOTAL_ENTRIES) return;
    int ch=t<(1u<<17)?0:1+(int)((t-(1u<<17))>>16);
    int d=(int)(t-gt_offset(ch));
    int m  = 2*d + 1;                        /* odd multiple below 2^18 */
    int hi = m >> 8, lo = m & 255;           /* lo odd; hi < 1024 */

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
    size_t off = ((size_t)gt_offset(ch) + d) * 64;
    memcpy(gTable + off,      rx, 32);
    memcpy(gTable + off + 32, ry, 32);
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

/* Batch-affine ladder build (delta C, jacklightChen e582bda4, after PR46):
 * the point sequence is unchanged; EC_POINTs_make_affine replaces one
 * inversion per point by one batched inversion per ladder. */
static void gt_batch_ladder(EC_GROUP *grp, const EC_POINT *step, int count,
                            uint64_t *out, BIGNUM *x, BIGNUM *y, BN_CTX *ctx) {
    EC_POINT *points[GT_HI];
    if(count<1 || count>=GT_HI) { fprintf(stderr,"Invalid ladder size\n");exit(2); }
    for(int i=0;i<count;i++) {
        points[i]=EC_POINT_new(grp);
        if(!points[i]) { fprintf(stderr,"Ladder allocation failed\n");exit(2); }
        int ok=i==0 ? EC_POINT_copy(points[i],step)
                    : EC_POINT_add(grp,points[i],points[i-1],step,ctx);
        if(!ok) { fprintf(stderr,"Ladder addition failed\n");exit(2); }
    }
    if(!EC_POINTs_make_affine(grp,(size_t)count,points,ctx)) {
        fprintf(stderr,"Ladder batch normalization failed\n");exit(2);
    }
    for(int i=0;i<count;i++) {
        gt_point_to_limbs(grp,points[i],x,y,ctx,out+(size_t)(i+1)*8);
        EC_POINT_free(points[i]);
    }
}

/* The two ladders the GPU builder needs: L[ch][lo] = lo * base_ch and
 * H[ch][hi] = hi * 256 * base_ch. Index 0 of each is the identity and is
 * left zeroed; the kernel treats it as such. 12,002 real points, against the
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
    EC_POINT *base = EC_POINT_new(grp), *step = EC_POINT_new(grp);
    /* base = A/2 = (2^-1 * neg_r_inv mod n) * G */
    EC_GROUP_get_order(grp, order, ctx);
    BN_set_word(shift, 2); BN_mod_inverse(inv2, shift, order, ctx);
    BN_lebin2bn(neg_r_inv, 32, nri);                     /* neg_r_inv is LE, like d_nri */
    BN_mod_mul(bscal, inv2, nri, order, ctx);            /* (2^-1 * neg_r_inv) mod n */
    EC_POINT_mul(grp, base, bscal, NULL, NULL, ctx);     /* base = bscal * G = A/2 */
    memset(hL, 0, (size_t)GT_CHUNKS * GT_LO * 8 * sizeof(uint64_t));
    memset(hH, 0, (size_t)GT_CHUNKS * GT_HI * 8 * sizeof(uint64_t));
    for (int ch = 0; ch < GT_CHUNKS; ch++) {
        if (ch > 0) { BN_set_word(shift, ch==1 ? (1u<<18) : (1u<<17)); EC_POINT_mul(grp, base, NULL, base, shift, ctx); }
        gt_batch_ladder(grp,base,GT_LO-1,                    /* L[lo] = lo * B */
            hL+(size_t)ch*GT_LO*8,x,y,ctx);
        BN_set_word(shift, 256);                             /* step = 256 * B */
        EC_POINT_mul(grp, step, NULL, base, shift, ctx);
        gt_batch_ladder(grp,step,(ch==0?1024:512)-1,         /* H[hi] = hi * 256 * B */
            hH+(size_t)ch*GT_HI*8,x,y,ctx);
    }
    BN_free(x); BN_free(y); BN_free(shift); BN_free(inv2); BN_free(order);
    BN_free(nri); BN_free(bscal);
    EC_POINT_free(base); EC_POINT_free(step);
    EC_GROUP_free(grp); BN_CTX_free(ctx);
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
        if (t < GT_CHUNKS * 4) {
            ch = t / 4;
            const int corner[4] = {0, 1, 2, (int)gt_entries(ch) - 1};
            i = corner[t % 4];
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
        if (memcmp(gTable + off,      want,     32) != 0 ||
            memcmp(gTable + off + 32, want + 4, 32) != 0) {
            fprintf(stderr, "  GTable spot check FAILED at chunk %d entry %d\n", ch, i);
            ok = 0;
        }
    }
    BN_free(x); BN_free(y); BN_free(k); BN_free(inv2); BN_free(order); BN_free(nri); BN_free(half_nri);
    EC_POINT_free(pt); EC_GROUP_free(grp); BN_CTX_free(ctx);
    return ok;
}

/* OpenSSL fallback builder (only if the GPU builder's spot check fails). Emits
 * the signed table: entry (ch,d) = (2d+1) * 2^gt_shift(ch) * (A/2). Walks odd
 * multiples by stepping 2*base_c per entry (acc = base_c, 3base_c, ...). */
static void compute_gtable(uint8_t *gTable, const uint8_t neg_r_inv[32]) {
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
        if (ch > 0) { BN_set_word(shift, ch==1 ? (1u<<18) : (1u<<17)); EC_POINT_mul(grp, base, NULL, base, shift, ctx); }
        BN_set_word(shift, 2); EC_POINT_mul(grp, two_base, NULL, base, shift, ctx);  /* 2*base_c */
        EC_POINT_copy(pt, base);                                                     /* (2*0+1)*base_c */
        for (unsigned d = 0; d < gt_entries(ch); d++) {
            EC_POINT_get_affine_coordinates_GFp(grp, pt, x, y, ctx);
            uint8_t xb[32], yb[32]; memset(xb,0,32); memset(yb,0,32);
            BN_bn2bin(x, xb+(32-BN_num_bytes(x)));
            BN_bn2bin(y, yb+(32-BN_num_bytes(y)));
            for(int j=0;j<16;j++){uint8_t t=xb[j];xb[j]=xb[31-j];xb[31-j]=t;}
            for(int j=0;j<16;j++){uint8_t t=yb[j];yb[j]=yb[31-j];yb[31-j]=t;}
            size_t off = ((size_t)gt_offset(ch) + d) * 64;
            memcpy(gTable + off,      xb, 32);
            memcpy(gTable + off + 32, yb, 32);
            if (d < gt_entries(ch) - 1) EC_POINT_add(grp, pt, pt, two_base, ctx);
        }
    }
    BN_free(x);BN_free(y);BN_free(shift);BN_free(inv2);BN_free(order);BN_free(nri);BN_free(bscal);
    EC_POINT_free(base);EC_POINT_free(pt);EC_POINT_free(two_base);
    EC_GROUP_free(grp);BN_CTX_free(ctx);
}

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
    cudaSetDevice(gpu_index);

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, gpu_index);
    printf("QSB Real Pinning Search (seq+lt) [GPU %d]\n", gpu_index);
    printf("  GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    pinning2_params_t pp;
    if (load_pinning2(argv[1], &pp) < 0) return 1;

    /* GTable */
    size_t gt_sz = (size_t)GT_TOTAL_ENTRIES*64;
    uint8_t *d_gt;
    cudaMalloc(&d_gt,gt_sz);
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
        int gt_total = GT_TOTAL_ENTRIES;
        kernel_build_gtable<<<(gt_total+255)/256,256>>>(dL,dH,d_gt);
        cudaDeviceSynchronize();
        cudaError_t gerr = cudaGetLastError();
        cudaFree(dL); cudaFree(dH);
        uint8_t *chk_table=(uint8_t*)malloc(gt_sz);
        if(!chk_table){ fprintf(stderr,"OOM: gtable check\n"); return 1; }
        int gt_ok = (gerr==cudaSuccess);
        if(gt_ok){
            cudaMemcpy(chk_table,d_gt,gt_sz,cudaMemcpyDeviceToHost);
            gt_ok = gt_spot_check(chk_table,GT_CHUNKS*4+192,pp.neg_r_inv);
        }
        clock_gettime(CLOCK_MONOTONIC, &tb);
        double gt_secs=(tb.tv_sec-ta.tv_sec)+(tb.tv_nsec-ta.tv_nsec)/1e9;
        if(gt_ok){
            printf("  GTable built on GPU in %.2fs (%d points, %.0f MiB total, spot check passed)\n",
                   gt_secs, gt_total, (double)gt_sz/(1024*1024));
        } else {
            printf("  GTable GPU build rejected (%s); using the host builder\n",
                   gerr!=cudaSuccess ? cudaGetErrorString(gerr) : "spot check failed");
            compute_gtable(chk_table,pp.neg_r_inv);
            cudaMemcpy(d_gt,chk_table,gt_sz,cudaMemcpyHostToDevice);
        }
        fflush(stdout);
        free(chk_table);
    }

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
        {   /* K=3*xR^2 is invariant across all candidates in this problem (delta E). */
            BIGNUM *field=BN_new(),*bk=BN_new();
            if(!field || !bk || !EC_GROUP_get_curve_GFp(grp,field,NULL,NULL,ctx) ||
               !BN_mod_sqr(bk,bx,field,ctx) || !BN_mul_word(bk,3) ||
               !BN_nnmod(bk,bk,field,ctx)) {
                fprintf(stderr,"Failed to precompute recovery K\n");
                return 1;
            }
            uint8_t kb[32]={0};
            BN_bn2bin(bk,kb+(32-BN_num_bytes(bk)));
            uint64_t kw[4]={0,0,0,0};
            for(int i=0;i<4;i++)for(int b=0;b<8;b++)
                kw[i]|=(uint64_t)kb[31-i*8-b]<<(b*8);
            cudaError_t kerr=cudaMemcpyToSymbol(pin_u2rk_words,kw,sizeof(kw));
            if(kerr!=cudaSuccess){
                fprintf(stderr,"Failed to upload recovery K: %s\n",cudaGetErrorString(kerr));
                return 1;
            }
            BN_free(field); BN_free(bk);
        }
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

    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    /* Pin the fixed-base table in L2. The 64 MiB table is sized to be
     * L2-resident on AD102's 72 MB L2, but the pipeline streams ~2.1 GiB of
     * per-candidate state through the same cache every 16M batch, which evicts
     * it. Advisory: if the device or driver refuses, the run is unaffected. */
    {
        int max_persist = 0, max_window = 0;
        cudaDeviceGetAttribute(&max_persist, cudaDevAttrMaxPersistingL2CacheSize, gpu_index);
        cudaDeviceGetAttribute(&max_window, cudaDevAttrMaxAccessPolicyWindowSize, gpu_index);
        size_t want = gt_sz < (size_t)max_persist ? gt_sz : (size_t)max_persist;
        /* Chunk 0 holds 2^17 entries for one access per candidate, the other
         * chunks 2^16 each: pinning the dense chunks first captures more of the
         * 15 random reads. The window stays inside the table. */
        size_t skip = QSB_L2_SKIP ? (size_t)gt_entries(0) * 64u : 0u;
        if (want > gt_sz - skip) want = gt_sz - skip;
        if (want > 0 && max_window > 0) {
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, want);
            cudaStreamAttrValue av = {};
            av.accessPolicyWindow.base_ptr  = (void *)(d_gt + skip);
            av.accessPolicyWindow.num_bytes = want < (size_t)max_window ? want : (size_t)max_window;
            av.accessPolicyWindow.hitRatio  = 1.0f;
            av.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
            av.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
            cudaError_t pe = cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &av);
            printf("  L2 persistence: %.0f MiB pinned (max %.0f MiB, window %.0f MiB) %s\n",
                   (double)av.accessPolicyWindow.num_bytes/(1024*1024),
                   (double)max_persist/(1024*1024), (double)max_window/(1024*1024),
                   pe==cudaSuccess?"ok":cudaGetErrorString(pe));
            fflush(stdout);
        }
    }
    uint32_t *d_hit_cnt, *d_hit_idx;
#if QSB_HOST_READBACK
    /* Delta A (jungjipdo a91746ca): counter and indices contiguous, so one
     * blocking copy per batch replaces synchronize + two copies. */
    {
        cudaError_t hit_err = cudaMalloc(&d_hit_cnt, (1 + 1024)*sizeof(uint32_t));
        if (hit_err != cudaSuccess) {
            fprintf(stderr, "Hit buffer allocation failed: %s\n", cudaGetErrorString(hit_err));
            return 1;
        }
        d_hit_idx = d_hit_cnt + 1;
        hit_err = cudaMemset(d_hit_cnt, 0, (1 + 1024)*sizeof(uint32_t));
        if (hit_err != cudaSuccess) {
            fprintf(stderr, "Hit buffer initialization failed: %s\n", cudaGetErrorString(hit_err));
            return 1;
        }
    }
#else
    cudaMalloc(&d_hit_cnt, 4); cudaMalloc(&d_hit_idx, 1024*4);
#endif

    int BATCH = QSB_BATCH; /* 16M: amortize launch/sync/copy overhead */
    int BLKSZ = 256;
    (void)BLKSZ;
    int GRDSZ = (BATCH+QSB_TREE_N-1)/QSB_TREE_N;
    int ROOT_GRDSZ=(GRDSZ+255)/256;
    ulonglong2 *d_pipeline_state=NULL;
    uint64_t *d_pipeline_roots=NULL,*d_pipeline_tree=NULL;
    uint64_t *d_super_roots=NULL,*d_root_checkpoint=NULL;
    size_t pipeline_state_bytes=(size_t)BATCH*QSB_STATE_PLANES*sizeof(ulonglong2);
    size_t pipeline_root_bytes=(size_t)GRDSZ*4u*sizeof(uint64_t);
    size_t pipeline_tree_bytes=(size_t)GRDSZ*4u*QSB_CAND_STRIDE*sizeof(uint64_t);
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

        /* Search all safe locktimes for this sequence */
        for (uint32_t lt_off = 0; lt_off < lt_range; lt_off += BATCH) {
            uint32_t batch_lt = LT_MIN + lt_off;
            int batch_sz = (lt_off + BATCH <= lt_range) ? BATCH : (lt_range - lt_off);

            uint32_t h_hit = 0;
            cudaMemcpy(d_hit_cnt, &h_hit, 4, cudaMemcpyHostToDevice);

            if (fast_tail) {
                launch_pinning_pipeline<true>(
                    d_mid, d_suffix, gpu_suffix_len,
                    pp.seq_offset, pp.lt_offset,
                    pp.total_preimage_len,
                    seq, batch_lt,
                    d_nri, d_u2rx, d_u2ry, d_neg2u2rx, d_neg2u2ry,
                    d_gt,
                    d_hit_cnt, d_hit_idx,
                    batch_sz, easy, single_hash,
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
                    batch_sz, easy, single_hash,
                    d_pipeline_state,d_pipeline_roots,d_pipeline_tree,
                    d_super_roots,d_root_checkpoint);
            }
#if QSB_HOST_READBACK
            /* The blocking default-stream copy waits for all kernels and
             * returns the counter plus the same first 64 indices reported below. */
            uint32_t hit_report[1 + 64];
            cudaError_t err = cudaMemcpy(hit_report, d_hit_cnt, sizeof(hit_report),
                                         cudaMemcpyDeviceToHost);
            if (err == cudaSuccess) err = cudaGetLastError();
            if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

            total_searched += batch_sz;
            h_hit = hit_report[0];
            if (h_hit > 0) {
                const uint32_t *hits = hit_report + 1;
                int nh = (h_hit > 64) ? 64 : h_hit;
#else
            cudaDeviceSynchronize();

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

            total_searched += batch_sz;

            cudaMemcpy(&h_hit, d_hit_cnt, 4, cudaMemcpyDeviceToHost);
            if (h_hit > 0) {
                uint32_t hits[64];
                int nh = (h_hit > 64) ? 64 : h_hit;
                cudaMemcpy(hits, d_hit_idx, nh*4, cudaMemcpyDeviceToHost);
#endif

                printf("\n  *** HIT! seq=0x%08X ***\n", seq);
                mkdir("results", 0755);
                char fname[256];
                snprintf(fname, sizeof(fname), "results/pinning_hit_%d.txt", gpu_index);
                FILE *f = fopen(fname, "a");
                if (f) {
                    for (int h = 0; h < nh; h++) {
                        uint32_t raw = hits[h];
                        uint32_t lt = batch_lt + (raw & 0x3FFFFFFF);
                        int ri = (raw >> 30) & 1;
                        int hc = (raw >> 31) & 1;
                        fprintf(f, "sequence=%u\nlocktime=%u\nhash_choice=%d\nrecid=%d\n",
                                seq, lt, hc, ri);
                        printf("  seq=0x%08X lt=%u hc=%d recid=%d\n", seq, lt, hc, ri);
                    }
                    fclose(f);
                }
                found = 1;
            }

            /* Check if another GPU found it */
            if ((total_searched % (50*1024*1024)) < (uint64_t)BATCH) {
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
