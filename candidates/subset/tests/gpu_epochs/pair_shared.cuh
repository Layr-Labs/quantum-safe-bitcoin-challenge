// First-state producer from dun999 PR212; paired finish from PR258.
// Shared pre-inverse finish derived from dun999 PR258, ac9a6164.
// Window SHA cache is inherited from odinfree; retain all parent notices.
#pragma once
#include "parity_window.cuh"
#ifndef QSB_PAIR_SHARED
#define QSB_PAIR_SHARED 1
#endif
#define QSB_PAIR_MUL (QSB_PAIR_SHARED ? 2 : 1)
#if QSB_PAIR_SHARED
__device__ __forceinline__ void qsb_k2s_pre(
    uint64_t *Y, uint64_t *ZZ, uint64_t *ZZZ, uint64_t *yR, uint64_t *m1, uint64_t *m2
) {
    uint64_t yb[4];
    _ModMult(yb, yR, ZZZ);
    _ModSub256(m1, yb, Y);
    _ModMult(m1, ZZ);
    _ModAdd256(m2, yb, Y);
    _ModMult(m2, ZZ);
}

__device__ __forceinline__ uint32_t qsb_k2s_post(
    uint64_t *m1, uint64_t *m2, uint64_t *inv, uint64_t *xR, uint64_t *yR,
    uint64_t *x1, uint64_t *x2
) {
    uint64_t t[4], sum[4];
    uint64_t cc[4]={QSB_U2R_C[0],QSB_U2R_C[1],QSB_U2R_C[2],QSB_U2R_C[3]};
    _ModMult(m1, inv);
    _ModMult(m2, inv);
    _ModAdd256(sum, m1, m2);
    _ModSub256(t, m1, cc);
    _ModMult(x1, sum, t);
    _ModAdd256(x1, x1, xR);
    _ModSub256(t, xR, x1);
    _ModMult(t, m1);
    _ModSub256(t, yR);
    uint32_t parities = (uint32_t)(t[0] & 1ULL);
    _ModSub256(t, m2, cc);
    _ModMult(x2, sum, t);
    _ModAdd256(x2, x2, xR);
    _ModSub256(t, xR, x2);
    _ModMult(t, m2);
    _ModSub256(t, yR);
    parities |= (uint32_t)(((t[0] & 1ULL) ^ 1ULL) << 1);
    return parities;
}

/* ---- ZLAB_K2S3M: the same paired finish in 3M instead of 4M --------------
 * qsb_k2s_pre multiplies BOTH slope numerators by ZZ and qsb_k2s_post
 * multiplies BOTH by inv: four multiplies to apply the single scale
 * h = ZZ/W.  Park (yR*ZZZ - Y), (yR*ZZZ + Y) and ZZ instead -- 12 words rather
 * than 8 -- and post forms h = ZZ*inv once (1M) and applies it twice (2M).
 * Net -1M per candidate.
 *   0 = the 4M pair above (kill switch; byte-identical to the frontier).
 *   1 = 3M for the PARKED candidate A only (ship default).  Candidate B keeps
 *       the 4M pair, so it still carries 8 words in registers across the block
 *       inverse and the kernel's register/spill profile is untouched there. */
#ifndef ZLAB_K2S3M
#define ZLAB_K2S3M 1
#endif
#if ZLAB_K2S3M
// Research: owizdom 4f367236, carried in DPZZxlz cbcb7bb and fkiene 2cf35a3.
// Only speculative paired preparation changes; exact replay uses original helpers.
#ifndef QSB_SPEC_PREPARE_PAIR
#define QSB_SPEC_PREPARE_PAIR 1
#endif
#if QSB_SPEC_PREPARE_PAIR
#define QSB_PRE_MUL QSB_FMUL
#define QSB_PRE_SUB QSB_FSUB
#define QSB_PRE_ADD QSB_FADD
#else
#define QSB_PRE_MUL X_FMUL
#define QSB_PRE_SUB X_FSUB
#define QSB_PRE_ADD X_FADD
#endif
__device__ __forceinline__ void qsb_k2s_pre3(
    uint64_t *Y, uint64_t *ZZ, uint64_t *ZZZ, uint64_t *yR, uint64_t *n
) {
    uint64_t yb[4];
    QSB_PRE_MUL(yb, yR, ZZZ);
    QSB_PRE_SUB(n, yb, Y);
    QSB_PRE_ADD(n + 4, yb, Y);
    Load256(n + 8, ZZ);
}
/* Filter-only copy of qsb_xyzz_finish_prepare (the exact front keeps the original). */
__device__ __forceinline__ void qsb_xyzz_finish_prepare_f(
    uint64_t *X_D, uint64_t *ZZ, uint64_t *ZZZ, uint64_t *xR, uint64_t *W
) {
    uint64_t t[4];
    QSB_PRE_MUL(t, xR, ZZ);
    QSB_PRE_SUB(t, t, X_D);
    Load256(X_D, t);             /* X_D becomes d */
    QSB_PRE_MUL(W, ZZZ, X_D);       /* W = ZZZ*d */
    W[4] = 0;
}
#undef QSB_PRE_MUL
#undef QSB_PRE_SUB
#undef QSB_PRE_ADD
/* h = ZZ*inv is the common slope scale: m1 = n[0..3]*h, m2 = n[4..7]*h.  The
 * tail from _ModAdd256(sum,...) on is the tail of qsb_k2s_post unchanged. */
__device__ __forceinline__ uint32_t qsb_k2s_post3(
    uint64_t *n, uint64_t *inv, uint64_t *xR, uint64_t *yR,
    uint64_t *x1, uint64_t *x2
) {
    uint64_t t[4], sum[4], m1[4], m2[4];
    uint64_t cc[4]={QSB_U2R_C[0],QSB_U2R_C[1],QSB_U2R_C[2],QSB_U2R_C[3]};
    QSB_FMUL(n + 8, n + 8, inv);   /* h = ZZ/W, formed once */
    QSB_FMUL(m1, n, n + 8);
    QSB_FMUL(m2, n + 4, n + 8);
    QSB_FADD(sum, m1, m2);
    QSB_FSUB(t, m1, cc);
    QSB_FMUL(x1, sum, t);
    QSB_FADD(x1, x1, xR);
    QSB_FSUB(t, xR, x1);
    uint32_t parity1;
#if QSB_PARITY_WINDOW
    if(!qsb_pw_sub(&parity1, t, m1, yR))
#endif
    {
        QSB_FMUL(t, t, m1);
        QSB_FSUB(t, t, yR);
        parity1=(uint32_t)(t[0]&1ULL);
    }
    uint32_t parities = parity1;
    QSB_FSUB(t, m2, cc);
    QSB_FMUL(x2, sum, t);
    QSB_FADD(x2, x2, xR);
    QSB_FSUB(t, xR, x2);
    uint32_t parity2;
#if QSB_PARITY_WINDOW
    if(!qsb_pw_sub(&parity2, t, m2, yR))
#endif
    {
        QSB_FMUL(t, t, m2);
        QSB_FSUB(t, t, yR);
        parity2=(uint32_t)(t[0]&1ULL);
    }
    parities |= (parity2 ^ 1U) << 1;
    return parities;
}
#endif
__device__ __forceinline__ int qsb_k2s_front(
    const epoch_desc_t *ep, const uint32_t *first, int lane, const uint8_t *d_gt,
    uint64_t *u2rx, uint64_t *u2ry, uint64_t *prod, uint64_t *m1, uint64_t *m2
) {
    uint32_t state[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = ep->mid[i];
    qsb_scheduled_window_hash(state, ep, lane, first);
    uint32_t b2[16];
    #pragma unroll
    for (int i=0;i<8;i++) b2[i]=state[i];
    b2[8]=0x80000000;
    #pragma unroll
    for (int i=9;i<15;i++) b2[i]=0;
    b2[15]=0x00000100;
    uint32_t s2[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    _SHA256Transform(s2, b2);
    uint64_t z[4];
    z[0] = ((uint64_t)s2[6] << 32) | (uint64_t)s2[7];
    z[1] = ((uint64_t)s2[4] << 32) | (uint64_t)s2[5];
    z[2] = ((uint64_t)s2[2] << 32) | (uint64_t)s2[3];
    z[3] = ((uint64_t)s2[0] << 32) | (uint64_t)s2[1];
    uint64_t qx[4],qy[4],qzz[4],qzzz[4];
    uint32_t unused_flag=0;
    qsb_filter_chain_trial(qx,qy,qzz,qzzz,z,d_gt,unused_flag);
    qsb_xyzz_finish_prepare(qx,qzz,qzzz,u2rx,prod);
    qsb_k2s_pre(qy,qzz,qzzz,u2ry,m1,m2);
    return (prod[0]|prod[1]|prod[2]|prod[3]) != 0;
}
#if ZLAB_K2S3M
__device__ __forceinline__ int qsb_k2s_front3(
    const epoch_desc_t *ep, const uint32_t *first, int lane, const uint8_t *d_gt,
    uint64_t *u2rx, uint64_t *u2ry, uint64_t *prod, uint64_t *n
) {
    uint32_t state[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = ep->mid[i];
    qsb_scheduled_window_hash(state, ep, lane, first);
    uint32_t b2[16];
    #pragma unroll
    for (int i=0;i<8;i++) b2[i]=state[i];
    b2[8]=0x80000000;
    #pragma unroll
    for (int i=9;i<15;i++) b2[i]=0;
    b2[15]=0x00000100;
    uint32_t s2[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    _SHA256Transform(s2, b2);
    uint64_t z[4];
    z[0] = ((uint64_t)s2[6] << 32) | (uint64_t)s2[7];
    z[1] = ((uint64_t)s2[4] << 32) | (uint64_t)s2[5];
    z[2] = ((uint64_t)s2[2] << 32) | (uint64_t)s2[3];
    z[3] = ((uint64_t)s2[0] << 32) | (uint64_t)s2[1];
    uint64_t qx[4],qy[4],qzz[4],qzzz[4];
    uint32_t unused_flag=0;
    qsb_filter_chain_trial(qx,qy,qzz,qzzz,z,d_gt,unused_flag);
    qsb_xyzz_finish_prepare_f(qx,qzz,qzzz,u2rx,prod);
    qsb_k2s_pre3(qy,qzz,qzzz,u2ry,n);
    return (prod[0]|prod[1]|prod[2]|prod[3]) != 0;
}
#endif
#if ZLAB_DUAL_EPOCH_SHA && ZLAB_K2S3M
struct QsbPairEpochZ {uint64_t a[4],b[4];};
__device__ __forceinline__ void qsb_pair_second_sha_z(uint32_t *state,uint64_t *z){
    uint32_t b2[16];
    #pragma unroll
    for(int i=0;i<8;i++)b2[i]=state[i];
    b2[8]=0x80000000;
    #pragma unroll
    for(int i=9;i<15;i++)b2[i]=0;
    b2[15]=0x00000100;
    uint32_t s2[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    _SHA256Transform(s2,b2);
    z[0]=((uint64_t)s2[6]<<32)|(uint64_t)s2[7];
    z[1]=((uint64_t)s2[4]<<32)|(uint64_t)s2[5];
    z[2]=((uint64_t)s2[2]<<32)|(uint64_t)s2[3];
    z[3]=((uint64_t)s2[0]<<32)|(uint64_t)s2[1];
}
__device__ __forceinline__ QsbPairEpochZ qsb_pair_epoch_z_value(
    const uint32_t*firstA,const uint32_t*firstB,int lane){
    uint32_t stateA[8],stateB[8];
    qsb_scheduled_window_hash_pair(stateA,stateB,lane,firstA,firstB);
    QsbPairEpochZ out;
    qsb_pair_second_sha_z(stateA,out.a);
    qsb_pair_second_sha_z(stateB,out.b);
    return out;
}
__device__ __forceinline__ int qsb_k2s_front3_z(
    const uint64_t*z,const uint8_t*d_gt,uint64_t*u2rx,uint64_t*u2ry,
    uint64_t*prod,uint64_t*n){
    uint64_t qx[4],qy[4],qzz[4],qzzz[4];
    uint32_t unused_flag=0;
    qsb_filter_chain_trial(qx,qy,qzz,qzzz,z,d_gt,unused_flag);
    qsb_xyzz_finish_prepare_f(qx,qzz,qzzz,u2rx,prod);   /* same finish as qsb_k2s_front3 */
    qsb_k2s_pre3(qy,qzz,qzzz,u2ry,n);
    return (prod[0]|prod[1]|prod[2]|prod[3])!=0;
}
#endif
__device__ __forceinline__ int qsb_k2s_front_exact(
    const epoch_desc_t *ep, const uint32_t *first, int lane, const uint8_t *d_gt,
    uint64_t *u2rx, uint64_t *u2ry, uint64_t *prod, uint64_t *m1, uint64_t *m2
) {
    uint32_t state[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = ep->mid[i];
    qsb_scheduled_window_hash(state, ep, lane, first);
    uint32_t b2[16];
    #pragma unroll
    for (int i=0;i<8;i++) b2[i]=state[i];
    b2[8]=0x80000000;
    #pragma unroll
    for (int i=9;i<15;i++) b2[i]=0;
    b2[15]=0x00000100;
    uint32_t s2[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    _SHA256Transform(s2, b2);
    uint64_t z[4];
    z[0] = ((uint64_t)s2[6] << 32) | (uint64_t)s2[7];
    z[1] = ((uint64_t)s2[4] << 32) | (uint64_t)s2[5];
    z[2] = ((uint64_t)s2[2] << 32) | (uint64_t)s2[3];
    z[3] = ((uint64_t)s2[0] << 32) | (uint64_t)s2[1];
    uint64_t qx[4],qy[4],qzz[4],qzzz[4];
    _FixedBaseSignedXYZZStream(qx,qy,qzz,qzzz,z,d_gt);
    qsb_xyzz_finish_prepare(qx,qzz,qzzz,u2rx,prod);
    qsb_k2s_pre(qy,qzz,qzzz,u2ry,m1,m2);
    return (prod[0]|prod[1]|prod[2]|prod[3]) != 0;
}

/* QSB_GATE_PAIR (kill switch): 1 = hash both recovery-id pubkeys in one interleaved SHA-256 block
 * (two independent dependency chains -> ILP), then test ri=0 before ri=1 exactly as the loop did.
 * Same arithmetic per stream; the only difference is that ri=1 is also hashed when ri=0 passes (rare). */
#ifndef QSB_GATE_PAIR
#define QSB_GATE_PAIR 1
#endif
#if QSB_GATE_PAIR
__device__ __forceinline__ void qsb_gate_block(uint32_t *pb, const uint64_t *qx, uint32_t parity) {
    uint64_t sx0=qx[0], sx1=qx[1], sx2=qx[2], sx3=qx[3];
    uint32_t x32[8]={(uint32_t)sx0,(uint32_t)(sx0>>32),(uint32_t)sx1,(uint32_t)(sx1>>32),
                     (uint32_t)sx2,(uint32_t)(sx2>>32),(uint32_t)sx3,(uint32_t)(sx3>>32)};
    uint8_t prefix_byte = 0x2+(uint8_t)(parity&1u);
    pb[0]=__byte_perm(x32[7],prefix_byte,0x4321);
    pb[1]=__byte_perm(x32[7],x32[6],0x0765);pb[2]=__byte_perm(x32[6],x32[5],0x0765);
    pb[3]=__byte_perm(x32[5],x32[4],0x0765);pb[4]=__byte_perm(x32[4],x32[3],0x0765);
    pb[5]=__byte_perm(x32[3],x32[2],0x0765);pb[6]=__byte_perm(x32[2],x32[1],0x0765);
    pb[7]=__byte_perm(x32[1],x32[0],0x0765);pb[8]=__byte_perm(x32[0],0x80,0x0456);
    pb[9]=0;pb[10]=0;pb[11]=0;pb[12]=0;pb[13]=0;pb[14]=0;pb[15]=0x108;
}
#define QSB_GP_WMIX(w) { \
w[0] += s1(w[14]) + w[9] + s0(w[1]);   w[1] += s1(w[15]) + w[10] + s0(w[2]); \
w[2] += s1(w[0]) + w[11] + s0(w[3]);   w[3] += s1(w[1]) + w[12] + s0(w[4]); \
w[4] += s1(w[2]) + w[13] + s0(w[5]);   w[5] += s1(w[3]) + w[14] + s0(w[6]); \
w[6] += s1(w[4]) + w[15] + s0(w[7]);   w[7] += s1(w[5]) + w[0] + s0(w[8]); \
w[8] += s1(w[6]) + w[1] + s0(w[9]);    w[9] += s1(w[7]) + w[2] + s0(w[10]); \
w[10] += s1(w[8]) + w[3] + s0(w[11]);  w[11] += s1(w[9]) + w[4] + s0(w[12]); \
w[12] += s1(w[10]) + w[5] + s0(w[13]); w[13] += s1(w[11]) + w[6] + s0(w[14]); \
w[14] += s1(w[12]) + w[7] + s0(w[15]); w[15] += s1(w[13]) + w[8] + s0(w[0]); }
#define QSB_GP_R2(A,B,C,D,E,F,G,H,k,i) \
    S2Round(A##0,B##0,C##0,D##0,E##0,F##0,G##0,H##0,K[(k)+(i)],w0[i]); \
    S2Round(A##1,B##1,C##1,D##1,E##1,F##1,G##1,H##1,K[(k)+(i)],w1[i]);
#define QSB_GP_RND(k) { \
    QSB_GP_R2(a,b,c,d,e,f,g,h,k,0)  QSB_GP_R2(h,a,b,c,d,e,f,g,k,1) \
    QSB_GP_R2(g,h,a,b,c,d,e,f,k,2)  QSB_GP_R2(f,g,h,a,b,c,d,e,k,3) \
    QSB_GP_R2(e,f,g,h,a,b,c,d,k,4)  QSB_GP_R2(d,e,f,g,h,a,b,c,k,5) \
    QSB_GP_R2(c,d,e,f,g,h,a,b,k,6)  QSB_GP_R2(b,c,d,e,f,g,h,a,k,7) \
    QSB_GP_R2(a,b,c,d,e,f,g,h,k,8)  QSB_GP_R2(h,a,b,c,d,e,f,g,k,9) \
    QSB_GP_R2(g,h,a,b,c,d,e,f,k,10) QSB_GP_R2(f,g,h,a,b,c,d,e,k,11) \
    QSB_GP_R2(e,f,g,h,a,b,c,d,k,12) QSB_GP_R2(d,e,f,g,h,a,b,c,k,13) \
    QSB_GP_R2(c,d,e,f,g,h,a,b,k,14) QSB_GP_R2(b,c,d,e,f,g,h,a,k,15) }
/* Two independent single-block SHA-256 compressions from the initial state, interleaved round by round. */
__device__ __forceinline__ void qsb_sha256_init_transform_pair(uint32_t *o0, uint32_t *w0, uint32_t *o1, uint32_t *w1) {
    uint32_t t1, t2;
    uint32_t a0=I[0],b0=I[1],c0=I[2],d0=I[3],e0=I[4],f0=I[5],g0=I[6],h0=I[7];
    uint32_t a1=I[0],b1=I[1],c1=I[2],d1=I[3],e1=I[4],f1=I[5],g1=I[6],h1=I[7];
    QSB_GP_RND(0);  QSB_GP_WMIX(w0); QSB_GP_WMIX(w1);
    QSB_GP_RND(16); QSB_GP_WMIX(w0); QSB_GP_WMIX(w1);
    QSB_GP_RND(32); QSB_GP_WMIX(w0); QSB_GP_WMIX(w1);
    QSB_GP_RND(48);
    o0[0]=I[0]+a0;o0[1]=I[1]+b0;o0[2]=I[2]+c0;o0[3]=I[3]+d0;o0[4]=I[4]+e0;o0[5]=I[5]+f0;o0[6]=I[6]+g0;o0[7]=I[7]+h0;
    o1[0]=I[0]+a1;o1[1]=I[1]+b1;o1[2]=I[2]+c1;o1[3]=I[3]+d1;o1[4]=I[4]+e1;o1[5]=I[5]+f1;o1[6]=I[6]+g1;o1[7]=I[7]+h1;
}
// H0-only gate derived independently from Saviour1001 38eb0bc public research.
// Original full-digest helper remains available for replay and larger targets.
#ifndef QSB_GATE_H0
#define QSB_GATE_H0 1
#endif
#if QSB_GATE_H0 && defined(QSB_ZEROS_N) && QSB_ZEROS_N >= 1 && QSB_ZEROS_N <= 32
__device__ __forceinline__ void qsb_sha256_gate_h0_pair(uint32_t *o0, uint32_t *w0, uint32_t *o1, uint32_t *w1) {
    uint32_t t1, t2;
    uint32_t a0=I[0],b0=I[1],c0=I[2],d0=I[3],e0=I[4],f0=I[5],g0=I[6],h0=I[7];
    uint32_t a1=I[0],b1=I[1],c1=I[2],d1=I[3],e1=I[4],f1=I[5],g1=I[6],h1=I[7];
    QSB_GP_RND(0);  QSB_GP_WMIX(w0); QSB_GP_WMIX(w1);
    QSB_GP_RND(16); QSB_GP_WMIX(w0); QSB_GP_WMIX(w1);
    QSB_GP_RND(32); QSB_GP_WMIX(w0); QSB_GP_WMIX(w1);
    QSB_GP_R2(a,b,c,d,e,f,g,h,48,0)
    QSB_GP_R2(h,a,b,c,d,e,f,g,48,1)
    QSB_GP_R2(g,h,a,b,c,d,e,f,48,2)
    QSB_GP_R2(f,g,h,a,b,c,d,e,48,3)
    QSB_GP_R2(e,f,g,h,a,b,c,d,48,4)
    QSB_GP_R2(d,e,f,g,h,a,b,c,48,5)
    QSB_GP_R2(c,d,e,f,g,h,a,b,48,6)
    QSB_GP_R2(b,c,d,e,f,g,h,a,48,7)
    QSB_GP_R2(a,b,c,d,e,f,g,h,48,8)
    QSB_GP_R2(h,a,b,c,d,e,f,g,48,9)
    QSB_GP_R2(g,h,a,b,c,d,e,f,48,10)
    QSB_GP_R2(f,g,h,a,b,c,d,e,48,11)
    QSB_GP_R2(e,f,g,h,a,b,c,d,48,12)
    QSB_GP_R2(d,e,f,g,h,a,b,c,48,13)
    QSB_GP_R2(c,d,e,f,g,h,a,b,48,14)
    *o0=I[0]+a0+S1(f0)+Ch(f0,g0,h0)+K[63]+w0[15]+S0(b0)+Maj(b0,c0,d0);
    *o1=I[0]+a1+S1(f1)+Ch(f1,g1,h1)+K[63]+w1[15]+S0(b1)+Maj(b1,c1,d1);
}
#endif

#undef QSB_GP_RND
#undef QSB_GP_R2
#undef QSB_GP_WMIX
#endif
__device__ __forceinline__ int qsb_k2s_gate(uint64_t *q1x, uint64_t *q2x, uint32_t y_parities, int *recid_out) {
#if QSB_GATE_PAIR
    uint32_t pb0[16], pb1[16], hs0[8], hs1[8];
    qsb_gate_block(pb0, q1x, y_parities);
    qsb_gate_block(pb1, q2x, y_parities>>1);
    qsb_sha256_init_transform_pair(hs0, pb0, hs1, pb1);
    if(gpu_bench_valid_words(hs0)){*recid_out=0;return 1;}
    if(gpu_bench_valid_words(hs1)){*recid_out=1;return 1;}
    return 0;
#else
    for(int ri=0;ri<2;ri++){
        uint64_t sx0=ri ? q2x[0] : q1x[0];
        uint64_t sx1=ri ? q2x[1] : q1x[1];
        uint64_t sx2=ri ? q2x[2] : q1x[2];
        uint64_t sx3=ri ? q2x[3] : q1x[3];
        uint32_t x32[8]={(uint32_t)sx0,(uint32_t)(sx0>>32),(uint32_t)sx1,(uint32_t)(sx1>>32),
                         (uint32_t)sx2,(uint32_t)(sx2>>32),(uint32_t)sx3,(uint32_t)(sx3>>32)};
        uint32_t pb[16];
        uint8_t prefix_byte = 0x2+(uint8_t)((y_parities>>ri)&1u);
        pb[0]=__byte_perm(x32[7],prefix_byte,0x4321);
        pb[1]=__byte_perm(x32[7],x32[6],0x0765);pb[2]=__byte_perm(x32[6],x32[5],0x0765);
        pb[3]=__byte_perm(x32[5],x32[4],0x0765);pb[4]=__byte_perm(x32[4],x32[3],0x0765);
        pb[5]=__byte_perm(x32[3],x32[2],0x0765);pb[6]=__byte_perm(x32[2],x32[1],0x0765);
        pb[7]=__byte_perm(x32[1],x32[0],0x0765);pb[8]=__byte_perm(x32[0],0x80,0x0456);
        pb[9]=0;pb[10]=0;pb[11]=0;pb[12]=0;pb[13]=0;pb[14]=0;pb[15]=0x108;
        uint32_t hs[8];_SHA256Initialize(hs);_SHA256Transform(hs,pb);
        if(gpu_bench_valid_words(hs)){*recid_out=ri;return 1;}
    }
    return 0;
#endif
}
// Full SHA fallback preserves every supported difficulty and original replay.
__device__ __forceinline__ int qsb_k2s_gate_h0(
    uint64_t *q1x,uint64_t *q2x,uint32_t y_parities,int *recid_out) {
#if QSB_GATE_PAIR && QSB_GATE_H0 && defined(QSB_ZEROS_N) && QSB_ZEROS_N >= 1 && QSB_ZEROS_N <= 32
    uint32_t pb0[16],pb1[16],h0,h1;
    qsb_gate_block(pb0,q1x,y_parities);
    qsb_gate_block(pb1,q2x,y_parities>>1);
    qsb_sha256_gate_h0_pair(&h0,pb0,&h1,pb1);
    if((h0>>(32-QSB_ZEROS_N))==0){*recid_out=0;return 1;}
    if((h1>>(32-QSB_ZEROS_N))==0){*recid_out=1;return 1;}
    return 0;
#else
    return qsb_k2s_gate(q1x,q2x,y_parities,recid_out);
#endif
}


struct QsbPairFront {uint64_t words[12];int ok;};
__device__ __noinline__ QsbPairFront qsb_pair_front_value(
    const epoch_desc_t*ep,const uint32_t*first,int lane,const uint8_t*d_gt,
    uint64_t rx0,uint64_t rx1,uint64_t rx2,uint64_t rx3,
    uint64_t ry0,uint64_t ry1,uint64_t ry2,uint64_t ry3){
    uint64_t rx[4]={rx0,rx1,rx2,rx3},ry[4]={ry0,ry1,ry2,ry3};
    uint64_t prod[5],m1[4],m2[4];QsbPairFront out;
    out.ok=qsb_k2s_front(ep,first,lane,d_gt,rx,ry,prod,m1,m2);
    Load256(out.words,prod);Load256(out.words+4,m1);Load256(out.words+8,m2);
    return out;
}

__device__ __noinline__ int qsb_pair_tail_value(
    uint64_t a0,uint64_t a1,uint64_t a2,uint64_t a3,
    uint64_t b0,uint64_t b1,uint64_t b2,uint64_t b3,
    uint64_t v0,uint64_t v1,uint64_t v2,uint64_t v3,
    uint64_t rx0,uint64_t rx1,uint64_t rx2,uint64_t rx3,
    uint64_t ry0,uint64_t ry1,uint64_t ry2,uint64_t ry3){
    uint64_t m1[4]={a0,a1,a2,a3},m2[4]={b0,b1,b2,b3};
    uint64_t inv[4]={v0,v1,v2,v3};
    uint64_t rx[4]={rx0,rx1,rx2,rx3},ry[4]={ry0,ry1,ry2,ry3};
    uint64_t q1x[4],q2x[4];int recid=0;
    uint32_t par=qsb_k2s_post(m1,m2,inv,rx,ry,q1x,q2x);
    return qsb_k2s_gate(q1x,q2x,par,&recid) ? recid+1 : 0;
}

// Only this exact check authorizes a hit record. The speculative calculation
// cannot bypass it, and neither the external verifier nor its inputs changes.
__device__ __noinline__ int qsb_pair_verify_candidate(
    const epoch_desc_t*ep,const uint32_t*first,int lane,const uint8_t*d_gt){
    uint64_t rx[4]={QSB_U2R[0],QSB_U2R[1],QSB_U2R[2],QSB_U2R[3]};
    uint64_t ry[4]={QSB_U2R[4],QSB_U2R[5],QSB_U2R[6],QSB_U2R[7]};
    uint64_t inv[5],m1[4],m2[4],x1[4],x2[4];
    if(!qsb_k2s_front_exact(ep,first,lane,d_gt,rx,ry,inv,m1,m2))return 0;
    _ModInv(inv); // Nonzero canonical denominator; independent scalar inverse.
    uint32_t par=qsb_k2s_post(m1,m2,inv,rx,ry,x1,x2);
    int recid=0;
    return qsb_k2s_gate(x1,x2,par,&recid)?recid+1:0;
}
#if ZLAB_K2S3M

struct QsbPairFront3 {uint64_t words[16];int ok;};
#if ZLAB_DUAL_EPOCH_SHA
__device__ __noinline__ QsbPairFront3 qsb_pair_front3_z_value(
    uint64_t z0,uint64_t z1,uint64_t z2,uint64_t z3,const uint8_t*d_gt,
    uint64_t rx0,uint64_t rx1,uint64_t rx2,uint64_t rx3,
    uint64_t ry0,uint64_t ry1,uint64_t ry2,uint64_t ry3){
    uint64_t z[4]={z0,z1,z2,z3};
    uint64_t rx[4]={rx0,rx1,rx2,rx3},ry[4]={ry0,ry1,ry2,ry3};
    uint64_t prod[5],n[12];QsbPairFront3 out;
    out.ok=qsb_k2s_front3_z(z,d_gt,rx,ry,prod,n);
    Load256(out.words,prod);
    #pragma unroll
    for(int k=0;k<12;k++)out.words[4+k]=n[k];
    return out;
}
#endif
__device__ __noinline__ QsbPairFront3 qsb_pair_front3_value(
    const epoch_desc_t*ep,const uint32_t*first,int lane,const uint8_t*d_gt,
    uint64_t rx0,uint64_t rx1,uint64_t rx2,uint64_t rx3,
    uint64_t ry0,uint64_t ry1,uint64_t ry2,uint64_t ry3){
    uint64_t rx[4]={rx0,rx1,rx2,rx3},ry[4]={ry0,ry1,ry2,ry3};
    uint64_t prod[5],n[12];QsbPairFront3 out;
    out.ok=qsb_k2s_front3(ep,first,lane,d_gt,rx,ry,prod,n);
    Load256(out.words,prod);
    #pragma unroll
    for(int k=0;k<12;k++)out.words[4+k]=n[k];
    return out;
}

__device__ __noinline__ int qsb_pair_tail3_value(
    uint64_t a0,uint64_t a1,uint64_t a2,uint64_t a3,
    uint64_t b0,uint64_t b1,uint64_t b2,uint64_t b3,
    uint64_t c0,uint64_t c1,uint64_t c2,uint64_t c3,
    uint64_t v0,uint64_t v1,uint64_t v2,uint64_t v3,
    uint64_t rx0,uint64_t rx1,uint64_t rx2,uint64_t rx3,
    uint64_t ry0,uint64_t ry1,uint64_t ry2,uint64_t ry3){
    uint64_t n[12]={a0,a1,a2,a3,b0,b1,b2,b3,c0,c1,c2,c3};
    uint64_t inv[4]={v0,v1,v2,v3};
    uint64_t rx[4]={rx0,rx1,rx2,rx3},ry[4]={ry0,ry1,ry2,ry3};
    uint64_t q1x[4],q2x[4];int recid=0;
    uint32_t par=qsb_k2s_post3(n,inv,rx,ry,q1x,q2x);
    return qsb_k2s_gate_h0(q1x,q2x,par,&recid) ? recid+1 : 0;
}
#endif
#endif
