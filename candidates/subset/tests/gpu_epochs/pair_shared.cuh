// First-state producer from dun999 PR212; paired finish from PR258.
// Shared pre-inverse finish derived from dun999 PR258, ac9a6164.
// Window SHA cache is inherited from odinfree; retain all parent notices.
#pragma once
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

__device__ __forceinline__ int qsb_k2s_gate(uint64_t *q1x, uint64_t *q2x, uint32_t y_parities, int *recid_out) {
#if QSB_PAIR_SHA
    /* Both recovery pubkeys hashed by one interleaved pair of one-block
     * SHA-256 compressions: the two states share the K fetch and the message
     * schedule loop, so the second hash rides in the issue slots the first
     * one leaves idle on its dependent-round critical path. Two separate
     * arrays, not pw[2][16]: an indexed 2-D temp would land in local memory. */
    uint32_t pb0[16], pb1[16];
    qsb_pair_sha_preimage(q1x[0],q1x[1],q1x[2],q1x[3],y_parities,pb0);
    qsb_pair_sha_preimage(q2x[0],q2x[1],q2x[2],q2x[3],y_parities>>1,pb1);
    uint32_t hs0[8],hs1[8];
    zlab_sha256_pair_h0(pb0,pb1,hs0,hs1);
    /* recid 0 is still preferred on a double hit, as in the sequential loop. */
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
#endif
