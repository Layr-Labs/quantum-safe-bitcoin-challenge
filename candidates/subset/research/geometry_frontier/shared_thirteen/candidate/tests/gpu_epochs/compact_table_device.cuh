#pragma once
#include "compact_geometry.cuh"
#define COMPACT_CHUNKS MIXED_CHUNKS
#define COMPACT_TOTAL_ENTRIES ((uint32_t)MIXED_TOTAL_ENTRIES)
#define COMPACT_LO 8192
#define COMPACT_HI 8192
__host__ __device__ __forceinline__ unsigned compact_entries(int c){return mixed_entries(c);}
__host__ __device__ __forceinline__ unsigned compact_offset(int c){return mixed_offset(c);}
__host__ __device__ __forceinline__ int compact_shift(int c){return mixed_shift(c);}


// Direct regular-digit mechanism: PR120, re-derived there from dun999.
__device__ __forceinline__ uint32_t gt_field_bits_v(const uint64_t m[4], unsigned pos) {
    unsigned li = pos >> 6, sh = pos & 63u;
    uint64_t lo = li == 0 ? m[0] : li == 1 ? m[1] : li == 2 ? m[2] : m[3];
    uint64_t hi = li == 0 ? m[1] : li == 1 ? m[2] : li == 2 ? m[3] : 0ULL;
    return (uint32_t)((lo >> sh) | ((hi << 1) << (63u - sh)));
}

__device__ __forceinline__ void gt_direct_digit(const uint64_t M[4], uint64_t sflag,
                                                unsigned pos, unsigned w, bool last,
                                                uint32_t *idx, uint64_t *neg) {
    uint32_t f = gt_field_bits_v(M, pos) & ((1u << w) - 1u);
    uint32_t t = f >> (w - 1);
    *idx = last ? (f & ((1u << (w - 1)) - 1u)) : ((f ^ (t - 1u)) & ((1u << (w - 1)) - 1u));
    *neg = (last ? 0ULL : (uint64_t)(t ^ 1u)) ^ sflag;
}

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


__device__ __forceinline__ void qsb_asym_last_add(
    uint64_t *X1, uint64_t *Y1, uint64_t *ZZ1, uint64_t *ZZZ1,
    const uint64_t *X2, const uint64_t *Y2, const uint64_t *Yoff)
{
  uint64_t U2[4];
  uint64_t S2[4];
  uint64_t P[4];
  uint64_t R[4];
  uint64_t PP[4];
  uint64_t PPP[4];
  uint64_t Q[4];
  uint64_t T[4];

  _ModMult(U2, (uint64_t *)X2, ZZ1);   // U2 = X2*ZZ1
  _ModAdd256(S2, (uint64_t *)Y2, (uint64_t *)Yoff);
  _ModMult(S2, ZZZ1);                  // S2 = (Y2+Yoff)*ZZZ1
  _ModSub256(P, U2, X1);               // P  = U2 - X1
  _ModSub256(R, S2, Y1);               // R  = S2 - Y1

  // The cold-first ordering has no exceptional prefix before this final add.
  // A zero x difference is either doubling (R=0) or opposite points (R!=0).
  // Construct 2*(X2,Y2) directly from the affine table point in the rare case.
  if (!(P[0]|P[1]|P[2]|P[3])) {
    if (R[0]|R[1]|R[2]|R[3]) {
      for(int i=0;i<4;++i) X1[i]=Y1[i]=ZZ1[i]=ZZZ1[i]=0;
      return;
    }
    uint64_t xx[4],yy[4],yyyy[4],s[4],m[4],t[4],tmp[4];
    _ModSqr(xx,(uint64_t *)X2);
    _ModSqr(yy,(uint64_t *)Y2);
    _ModSqr(yyyy,yy);
    _ModMult(s,(uint64_t *)X2,yy);
    _ModAdd256(s,s,s);_ModAdd256(s,s,s); // s=4*x*y^2
    _ModAdd256(m,xx,xx);_ModAdd256(m,m,xx); // m=3*x^2
    _ModSqr(t,m);_ModSub256(t,t,s);_ModSub256(t,t,s);
    _ModSub256(tmp,s,t);_ModMult(Y1,m,tmp);
    _ModAdd256(yyyy,yyyy,yyyy);_ModAdd256(yyyy,yyyy,yyyy);_ModAdd256(yyyy,yyyy,yyyy);
    _ModSub256(Y1,Y1,yyyy);Load256(X1,t);
    _ModAdd256(ZZ1,yy,yy);_ModAdd256(ZZ1,ZZ1,ZZ1);
    _ModMult(ZZZ1,(uint64_t *)Y2,yy);
    _ModAdd256(ZZZ1,ZZZ1,ZZZ1);_ModAdd256(ZZZ1,ZZZ1,ZZZ1);_ModAdd256(ZZZ1,ZZZ1,ZZZ1);
    return;
  }
  _ModSqr(PP, P);                      // PP = P^2
  _ModMult(PPP, PP, P);                // PPP = P*PP
  _ModMult(Q, U2, PP);                 // V  = U2*PP
  _ModMult(ZZ1, PP);                   // ZZ3; PP dies before the R^2/Y3 tail

  _ModSqr(T, R);                       // R^2
  _ModAdd256(T, T, PPP);
  _ModSub256(T, T, Q);
  _ModSub256(T, T, Q);                 // X3 = R^2 + PPP - 2V

  _ModMult(ZZZ1, PPP);                 // ZZZ3
  _ModSub256(Q, Q, T);                 // V - X3
  _ModMult(Q, R);                      // R*(V - X3)
  if (false) {
    Load256(Y1, Q);                    // actual Y3 = Y1 - Y2*ZZZ3
  } else {
    _ModMult(S2, (uint64_t *)Y2, ZZZ1);// affine Y2*ZZZ3
    _ModSub256(Y1, Q, S2);             // exact Y3
  }

  Load256(X1, T);                      // X3
}

__device__ void compact_fixed_xyzz(uint64_t *X, uint64_t *Y, uint64_t *ZZ, uint64_t *ZZZ,
                                      const uint64_t scalar[4], const uint8_t *gTX, const uint8_t *gTY) {
    uint64_t M[4]; int sign; gt_recode_setup(scalar,M,&sign);
    uint32_t idx;uint64_t neg;
    uint64_t x0[4],y0[4],x1[4],y1[4];
    gt_direct_digit(M,(uint64_t)(sign<0),157u,25u,false,&idx,&neg);
    compact_load_signed(gTX,gTY,9,idx,neg,x0,y0);
    gt_direct_digit(M,(uint64_t)(sign<0),182u,25u,false,&idx,&neg);
    compact_load_signed(gTX,gTY,10,idx,neg,x1,y1);
    gt_direct_digit(M,(uint64_t)(sign<0),207u,25u,false,&idx,&neg);
    _PointAddXYZZ_mm_def(X,Y,ZZ,ZZZ, x0,y0, x1,y1);
    uint64_t cx[4],cy[4];
    #pragma unroll 1
    for(int i=0;i<10;++i){
        int c=i<2?i+11:i-2;
        int next=i==1?0:c+1;
        compact_load_signed(gTX,gTY,c,idx,neg,cx,cy);
        gt_direct_digit(M,(uint64_t)(sign<0),(unsigned)mixed_shift(next)+1u,
                        (unsigned)mixed_bits(next),next==12,&idx,&neg);
        if(next<9)compact_prefetch_point(gTX,next,idx);
        _PointAddXYZZ_def(X,Y,ZZ,ZZZ,cx,cy,y0,true);
        Load256(y0,cy);
    }
    compact_load_signed(gTX,gTY,8,idx,neg,cx,cy);
    qsb_asym_last_add(X,Y,ZZ,ZZZ,cx,cy,y0);
}

// Experimental reuse of the inverse arena before collective inversion.
// Per-thread SoA slots: M at tid, anchor at256+tid, ZZ at512+tid, ZZZ at768+tid.
// Volatile accesses intentionally prevent carrying those values across adds.
__device__ __forceinline__ uint32_t qsb_shared_field_bits(
    volatile uint64_t arena[4][1024], unsigned tid, unsigned pos) {
    unsigned li=pos>>6, sh=pos&63u;
    uint64_t lo=arena[li][tid];
    uint64_t hi=li<3 ? arena[li+1][tid] : 0ULL;
    return (uint32_t)((lo>>sh)|((hi<<1)<<(63u-sh)));
}
__device__ __forceinline__ void qsb_shared_direct_digit(
    volatile uint64_t arena[4][1024], unsigned tid, uint64_t sflag,
    unsigned pos, unsigned w, bool last, uint32_t *idx, uint64_t *neg) {
    uint32_t f=qsb_shared_field_bits(arena,tid,pos)&((1u<<w)-1u);
    uint32_t t=f>>(w-1);
    *idx=last?(f&((1u<<(w-1))-1u)):((f^(t-1u))&((1u<<(w-1))-1u));
    *neg=(last?0ULL:(uint64_t)(t^1u))^sflag;
}

__device__ void qsb_PointAddXYZZ_shared_z_def(uint64_t *X1, uint64_t *Y1, volatile uint64_t arena[4][1024], unsigned tid,
                                  const uint64_t *X2, const uint64_t *Y2,
                                  const uint64_t *Yoff, bool defer_y)
{
  uint64_t U2[4];
  uint64_t S2[4];
  uint64_t P[4];
  uint64_t R[4];
  uint64_t PP[4];
  uint64_t PPP[4];
  uint64_t Q[4];
  uint64_t T[4];

    {
    uint64_t ZZ1[4];
    #pragma unroll
    for(int k=0;k<4;k++)ZZ1[k]=arena[k][512+tid];
    _ModMult(U2, (uint64_t *)X2, ZZ1);
  }   // U2 = X2*ZZ1
  _ModAdd256(S2, (uint64_t *)Y2, (uint64_t *)Yoff);
    {
    uint64_t ZZZ1[4];
    #pragma unroll
    for(int k=0;k<4;k++)ZZZ1[k]=arena[k][768+tid];
    _ModMult(S2, ZZZ1);
  }                  // S2 = (Y2+Yoff)*ZZZ1
  _ModSub256(P, U2, X1);               // P  = U2 - X1
  _ModSub256(R, S2, Y1);               // R  = S2 - Y1
  _ModSqr(PP, P);                      // PP = P^2
  _ModMult(PPP, PP, P);                // PPP = P*PP
  _ModMult(Q, U2, PP);                 // V  = U2*PP
    {
    uint64_t ZZ1[4];
    #pragma unroll
    for(int k=0;k<4;k++)ZZ1[k]=arena[k][512+tid];
    _ModMult(ZZ1, PP);
    #pragma unroll
    for(int k=0;k<4;k++)arena[k][512+tid]=ZZ1[k];
  }                   // ZZ3; PP dies before the R^2/Y3 tail

  _ModSqr(T, R);                       // R^2
  _ModAdd256(T, T, PPP);
  _ModSub256(T, T, Q);
  _ModSub256(T, T, Q);                 // X3 = R^2 + PPP - 2V

    {
    uint64_t ZZZ1[4];
    #pragma unroll
    for(int k=0;k<4;k++)ZZZ1[k]=arena[k][768+tid];
    _ModMult(ZZZ1, PPP);
    #pragma unroll
    for(int k=0;k<4;k++)arena[k][768+tid]=ZZZ1[k];
  }                 // ZZZ3
  _ModSub256(Q, Q, T);                 // V - X3
  _ModMult(Q, R);                      // R*(V - X3)
  if (defer_y) {
    Load256(Y1, Q);                    // actual Y3 = Y1 - Y2*ZZZ3
  } else {
      {
    uint64_t ZZZ1[4];
    #pragma unroll
    for(int k=0;k<4;k++)ZZZ1[k]=arena[k][768+tid];
    _ModMult(S2, (uint64_t *)Y2, ZZZ1);
  }// affine Y2*ZZZ3
    _ModSub256(Y1, Q, S2);             // exact Y3
  }

  Load256(X1, T);                      // X3
}
__device__ void compact_fixed_xyzz_shared(uint64_t *X, uint64_t *Y, uint64_t *ZZ, uint64_t *ZZZ,
                                      const uint64_t scalar[4], const uint8_t *gTX, const uint8_t *gTY, volatile uint64_t arena[4][1024], unsigned tid) {
    int sign;
    {
        uint64_t M[4];gt_recode_setup(scalar,M,&sign);
        #pragma unroll
        for(int k=0;k<4;k++)arena[k][tid]=M[k];
    }
    uint32_t idx;uint64_t neg;
    uint64_t x0[4],y0[4],x1[4],y1[4];
    qsb_shared_direct_digit(arena,tid,(uint64_t)(sign<0),157u,25u,false,&idx,&neg);
    compact_load_signed(gTX,gTY,9,idx,neg,x0,y0);
    qsb_shared_direct_digit(arena,tid,(uint64_t)(sign<0),182u,25u,false,&idx,&neg);
    compact_load_signed(gTX,gTY,10,idx,neg,x1,y1);
    qsb_shared_direct_digit(arena,tid,(uint64_t)(sign<0),207u,25u,false,&idx,&neg);
    #pragma unroll
    for(int k=0;k<4;k++)arena[k][256+tid]=y0[k];
    _PointAddXYZZ_mm_def(X,Y,ZZ,ZZZ, x0,y0, x1,y1);
    #pragma unroll
    for(int k=0;k<4;k++){arena[k][512+tid]=ZZ[k];arena[k][768+tid]=ZZZ[k];}
    uint64_t cx[4],cy[4];
    #pragma unroll 1
    for(int i=0;i<10;++i){
        int c=i<2?i+11:i-2;
        int next=i==1?0:c+1;
        compact_load_signed(gTX,gTY,c,idx,neg,cx,cy);
        qsb_shared_direct_digit(arena,tid,(uint64_t)(sign<0),(unsigned)mixed_shift(next)+1u,
                                (unsigned)mixed_bits(next),next==12,&idx,&neg);
        if(next<9)compact_prefetch_point(gTX,next,idx);
        uint64_t anchor[4];
        #pragma unroll
        for(int k=0;k<4;k++)anchor[k]=arena[k][256+tid];
        #pragma unroll
        for(int k=0;k<4;k++)arena[k][256+tid]=cy[k];
        qsb_PointAddXYZZ_shared_z_def(X,Y,arena,tid,cx,cy,anchor,true);
    }
    compact_load_signed(gTX,gTY,8,idx,neg,cx,cy);
    uint64_t anchor[4];
    #pragma unroll
    for(int k=0;k<4;k++)anchor[k]=arena[k][256+tid];
    #pragma unroll
    for(int k=0;k<4;k++){ZZ[k]=arena[k][512+tid];ZZZ[k]=arena[k][768+tid];}
    qsb_asym_last_add(X,Y,ZZ,ZZZ,cx,cy,anchor);
}
