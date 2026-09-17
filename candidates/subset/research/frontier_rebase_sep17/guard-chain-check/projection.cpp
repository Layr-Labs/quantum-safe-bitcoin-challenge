
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unordered_set>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <openssl/sha.h>
static BN_CTX *ctx;
static BIGNUM *prime;
static int inversions;
static void require(bool ok) { if (!ok) { std::fprintf(stderr,"check failed\n"); std::exit(1); } }
static BIGNUM *read256(const uint64_t *v) {
    return BN_lebin2bn(reinterpret_cast<const unsigned char *>(v),32,nullptr);
}
static void write256(uint64_t *v, const BIGNUM *b) {
    require(BN_bn2lebinpad(b,reinterpret_cast<unsigned char *>(v),32)==32);
}
static void field(uint64_t *r,const uint64_t *a,const uint64_t *b,int op) {
    BIGNUM *aa=read256(a), *bb=read256(b), *rr=BN_new();
    int ok=op==0 ? BN_mod_mul(rr,aa,bb,prime,ctx) :
           op==1 ? BN_mod_add(rr,aa,bb,prime,ctx) : BN_mod_sub(rr,aa,bb,prime,ctx);
    require(ok); write256(r,rr); BN_free(aa); BN_free(bb); BN_free(rr);
}
static void _ModMult(uint64_t *r,const uint64_t *a,const uint64_t *b) { field(r,a,b,0); }
static void _ModMult(uint64_t *r,const uint64_t *b) { field(r,r,b,0); }
static void _ModAdd256(uint64_t *r,const uint64_t *a,const uint64_t *b) { field(r,a,b,1); }
static void _ModSub256(uint64_t *r,const uint64_t *a,const uint64_t *b) { field(r,a,b,2); }
static void _ModSub256(uint64_t *r,const uint64_t *b) { field(r,r,b,2); }
static void _ModSqr(uint64_t *r,const uint64_t *a) { field(r,a,a,0); }
static void _ModInv(uint64_t *r) {
    ++inversions; BIGNUM *a=read256(r), *b=BN_mod_inverse(nullptr,a,prime,ctx);
    require(b!=nullptr); write256(r,b); r[4]=0; BN_free(a); BN_free(b);
}
static const int CHUNK_FIRST_ELEMENT[16] = {
    0,65536,131072,196608,262144,327680,393216,458752,
    524288,589824,655360,720896,786432,851968,917504,983040
};

#include <sys/mman.h>
#define __device__
#define __host__
#define __constant__
#define __forceinline__ inline
#define ZLAB_T14 0
#define ZLAB_DIRDIG 1
static void Load256(uint64_t*r,const uint64_t*a){memcpy(r,a,32);}
static uint64_t QSB_U2R_C[4];
struct alignas(16) ulonglong2 {uint64_t x,y;};
template<class T> static T __ldg(const T*p){return *p;}
static uint64_t cc;
#define UADDO1(x,y) do{__uint128_t t=(__uint128_t)(x)+(y);(x)=(uint64_t)t;cc=t>>64;}while(0)
#define UADDC1(x,y) do{__uint128_t t=(__uint128_t)(x)+(y)+cc;(x)=(uint64_t)t;cc=t>>64;}while(0)
#define UADD1(x,y) do{(x)+=(y)+cc;}while(0)
static EC_GROUP* group;static BIGNUM *ord,*halfbase;static uint8_t* table;static int load_count;static uint64_t norms;
#ifndef ZLAB_T14
#define ZLAB_T14 0
#endif
#if ZLAB_T14
#define GT_CHUNKS 14
#define GT_BIG 4
#define GT_TOTAL_ENTRIES (GT_BIG * (1u << 18) + (GT_CHUNKS - GT_BIG) * (1u << 17))
#define GT_LO 256
#define GT_HI 2048
__host__ __device__ __forceinline__ unsigned gt_entries(int c) {
    return c < GT_BIG ? (1u << 18) : (1u << 17);
}
__host__ __device__ __forceinline__ unsigned gt_offset(int c) {
    return c <= GT_BIG ? (unsigned)c << 18 : ((unsigned)GT_BIG << 18) + ((unsigned)(c - GT_BIG) << 17);
}
__host__ __device__ __forceinline__ int gt_shift(int c) {
    return c <= GT_BIG ? 19*c : 19*GT_BIG + 18*(c - GT_BIG);
}
static_assert(GT_TOTAL_ENTRIES*64ULL == 144ULL*1024*1024,
              "14-term table must contain exactly 144 MiB");
#else
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
#endif

__device__ __constant__ uint64_t GT_ORDER_N[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};
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
__device__ __forceinline__ uint32_t gt_field_bits_v(const uint64_t m[4], unsigned pos) {
    unsigned li = pos >> 6, sh = pos & 63u;
    uint64_t lo = li == 0 ? m[0] : li == 1 ? m[1] : li == 2 ? m[2] : m[3];
    uint64_t hi = li == 0 ? m[1] : li == 1 ? m[2] : li == 2 ? m[3] : 0ULL;
    return (uint32_t)((lo >> sh) | ((hi << 1) << (63u - sh)));
}
__host__ __device__ __forceinline__ unsigned gt_width(int c) {
    return c == GT_CHUNKS-1 ? (unsigned)(gt_shift(c) - gt_shift(c-1))
                            : (unsigned)(gt_shift(c+1) - gt_shift(c));
}
__device__ __forceinline__ void gt_direct_digit(const uint64_t M[4], uint64_t sflag,
                                                unsigned pos, unsigned w, bool last,
                                                uint32_t *idx, uint64_t *neg) {
    uint32_t f = gt_field_bits_v(M, pos) & ((1u << w) - 1u);
    uint32_t t = f >> (w - 1);
    *idx = last ? (f & ((1u << (w - 1)) - 1u)) : ((f ^ (t - 1u)) & ((1u << (w - 1)) - 1u));
    *neg = (last ? 0ULL : (uint64_t)(t ^ 1u)) ^ sflag;
}
__device__ __forceinline__ void actual_gt_load_signed_flat(const uint8_t *__restrict__ gTable,
                                                     uint32_t base, uint32_t idx,
                                                     uint64_t neg,
                                                     uint64_t *__restrict__ gx,
                                                     uint64_t *__restrict__ gy) {
    size_t off = ((size_t)base + idx) * 64;
    const ulonglong2 *tx=(const ulonglong2 *)(gTable+off);
    const ulonglong2 *ty=(const ulonglong2 *)(gTable+off+32);
    ulonglong2 x0=__ldg(tx),x1=__ldg(tx+1),y0=__ldg(ty),y1=__ldg(ty+1);
    gx[0]=x0.x;gx[1]=x0.y;gx[2]=x1.x;gx[3]=x1.y;
    uint64_t m=0ULL-neg;
    uint64_t r0=y0.x^m, r1=y0.y^m, r2=y1.x^m, r3=y1.y^m;
    uint64_t c0=0xFFFFFFFEFFFFFC30ULL&m;
    UADDO1(r0,c0); UADDC1(r1,m); UADDC1(r2,m); UADD1(r3,m);
    gy[0]=r0; gy[1]=r1; gy[2]=r2; gy[3]=r3;
}
static void gt_load_signed_flat(const uint8_t*,uint32_t offset,uint32_t idx,uint64_t neg,uint64_t*x,uint64_t*y){
 int c=load_count++;require(c<15);require(offset==gt_offset(c)&&idx<gt_entries(c)&&neg<=1);
 require(gt_entries(c)==(c==0?131072:65536));require(gt_shift(c)==(c?17*c+1:0));
 BIGNUM *k=BN_new(),*xx=BN_new(),*yy=BN_new();EC_POINT *pt=EC_POINT_new(group);
 require(BN_set_word(k,2ull*idx+1));require(BN_lshift(k,k,gt_shift(c)));require(BN_mod_mul(k,k,halfbase,ord,ctx));require(EC_POINT_mul(group,pt,k,nullptr,nullptr,ctx));require(EC_POINT_get_affine_coordinates(group,pt,xx,yy,ctx));
 uint64_t entry[8];write256(entry,xx);write256(entry+4,yy);size_t off=((size_t)offset+idx)*64;require(off+64<=64ull*1024*1024);memcpy(table+off,entry,64);
 actual_gt_load_signed_flat(table,offset,idx,neg,x,y);
 if(neg)require(BN_mod_sub(yy,prime,yy,prime,ctx));uint64_t expected[8];write256(expected,xx);write256(expected+4,yy);require(!memcmp(x,expected,32)&&!memcmp(y,expected+4,32));
 BN_free(k);BN_free(xx);BN_free(yy);EC_POINT_free(pt);
}
__device__ __forceinline__ void gt_load_signed(const uint8_t *gTable,
                                                int c, uint32_t idx, uint64_t neg,
                                                uint64_t gx[4], uint64_t gy[4]) {
    gt_load_signed_flat(gTable, gt_offset(c), idx, neg, gx, gy);
}template<bool DEFER_Y>
__device__ __forceinline__ void _PointAddXYZZ_def(
    uint64_t *__restrict__ X1, uint64_t *__restrict__ Y1,
    uint64_t *__restrict__ ZZ1, uint64_t *__restrict__ ZZZ1,
    const uint64_t *__restrict__ X2, const uint64_t *__restrict__ Y2,
    const uint64_t *__restrict__ Yoff)
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
  if (DEFER_Y) {
    Load256(Y1, Q);                    // actual Y3 = Y1 - Y2*ZZZ3
  } else {
    _ModMult(S2, (uint64_t *)Y2, ZZZ1);// affine Y2*ZZZ3
    _ModSub256(Y1, Q, S2);             // exact Y3
  }

  Load256(X1, T);                      // X3
}__device__ void _PointAddXYZZ_mm_def(uint64_t *X3, uint64_t *Y3, uint64_t *ZZ3, uint64_t *ZZZ3,
                                     const uint64_t *X1, const uint64_t *Y1,
                                     const uint64_t *X2, const uint64_t *Y2)
{
  uint64_t P[4];
  uint64_t R[4];
  uint64_t Q[4];
  uint64_t T[4];

  _ModSub256(P, (uint64_t *)X2, (uint64_t *)X1);   // P = X2 - X1
  _ModSub256(R, (uint64_t *)Y2, (uint64_t *)Y1);   // R = Y2 - Y1
  _ModSqr(ZZ3, P);                                 // ZZ3  = PP  = P^2
  _ModMult(ZZZ3, ZZ3, P);                          // ZZZ3 = PPP = P*PP
  _ModMult(Q, (uint64_t *)X1, ZZ3);                // Q = X1*PP

  _ModSqr(T, R);                                   // R^2
  _ModSub256(T, T, ZZZ3);
  _ModSub256(T, T, Q);
  _ModSub256(T, T, Q);                             // X3 = R^2 - PPP - 2Q

  _ModSub256(Q, Q, T);                             // Q - X3
  _ModMult(Y3, Q, R);                              // deferred R*(Q-X3)
  Load256(X3, T);                                  // X3
}__device__ __forceinline__ void qsb_asym_last_add(
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

  // The 15-window low-to-high ordering has no exceptional prefix before this final add.
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
}__device__ void _FixedBaseSignedXYZZStream(uint64_t *X, uint64_t *Y, uint64_t *ZZ, uint64_t *ZZZ,
                                           const uint64_t k[4], const uint8_t *gTable) {
    uint64_t M[4]; int sign;
    gt_recode_setup(k, M, &sign);
    uint32_t idx; uint64_t neg;
    uint64_t x0[4],y0[4],x1[4],y1[4];
#if ZLAB_T14
#if ZLAB_DIRDIG
    uint64_t sflag=(uint64_t)(sign<0);
    gt_direct_digit(M,sflag,(unsigned)gt_shift(0)+1u,gt_width(0),false,&idx,&neg);
    gt_load_signed(gTable,0,idx,neg,x0,y0);
    gt_direct_digit(M,sflag,(unsigned)gt_shift(1)+1u,gt_width(1),false,&idx,&neg);
    gt_load_signed(gTable,1,idx,neg,x1,y1);
#else
    int32_t ec=gt_mixed_step<19>(M,sign);
    gt_digit_idx(ec, &idx, &neg); gt_load_signed(gTable,0,idx,neg,x0,y0);
    ec=gt_mixed_step<19>(M,sign);
    gt_digit_idx(ec, &idx, &neg); gt_load_signed(gTable,1,idx,neg,x1,y1);
#endif
    _PointAddXYZZ_mm_def(X,Y,ZZ,ZZZ, x0,y0, x1,y1);
    uint64_t cx[4],cy[4];
    uint32_t table_base=gt_offset(2);
#if ZLAB_DIRDIG
    unsigned pos=(unsigned)gt_shift(2)+1u;
#endif
    #pragma unroll 1
    for (int c=2;c<GT_BIG;c++){
#if ZLAB_DIRDIG
        gt_direct_digit(M,sflag,pos,gt_width(2),false,&idx,&neg); pos+=gt_width(2);
#else
        ec=gt_mixed_step<19>(M,sign);
        gt_digit_idx(ec, &idx, &neg);
#endif
        gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
        _PointAddXYZZ_def<true>(X,Y,ZZ,ZZZ, cx,cy, y0);
        Load256(y0, cy);
        table_base += 1u << 18;
    }
    #pragma unroll 1
    for (int c=GT_BIG;c<GT_CHUNKS-1;c++){
#if ZLAB_DIRDIG
        gt_direct_digit(M,sflag,pos,gt_width(GT_BIG),false,&idx,&neg); pos+=gt_width(GT_BIG);
#else
        ec=gt_mixed_step<18>(M,sign);
        gt_digit_idx(ec, &idx, &neg);
#endif
        gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
        _PointAddXYZZ_def<true>(X,Y,ZZ,ZZZ, cx,cy, y0);
        Load256(y0, cy);
        table_base += 1u << 17;
    }
    {
#if ZLAB_DIRDIG
        gt_direct_digit(M,sflag,pos,gt_width(GT_BIG),true,&idx,&neg);
#else
        ec=sign*(int32_t)M[0];
        gt_digit_idx(ec, &idx, &neg);
#endif
        gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
        _PointAddXYZZ_def<false>(X,Y,ZZ,ZZZ, cx,cy, y0);
    }
#else
#if ZLAB_DIRDIG
    uint64_t sflag=(uint64_t)(sign<0);
    gt_direct_digit(M,sflag,(unsigned)gt_shift(0)+1u,gt_width(0),false,&idx,&neg);
    gt_load_signed(gTable,0,idx,neg,x0,y0);
    gt_direct_digit(M,sflag,(unsigned)gt_shift(1)+1u,gt_width(1),false,&idx,&neg);
    gt_load_signed(gTable,1,idx,neg,x1,y1);
    _PointAddXYZZ_mm_def(X,Y,ZZ,ZZZ, x0,y0, x1,y1);
    uint64_t cx[4],cy[4];
    uint32_t table_base=gt_offset(2);
    unsigned pos=(unsigned)gt_shift(2)+1u;
    #pragma unroll 1
    for (int c=2;c<GT_CHUNKS-1;c++){
        gt_direct_digit(M,sflag,pos,gt_width(2),false,&idx,&neg);
        pos+=gt_width(2);
        gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
        _PointAddXYZZ_def<true>(X,Y,ZZ,ZZZ, cx,cy, y0);
        Load256(y0, cy);                /* current affine y anchors next madd */
        table_base += 1u << 16;
    }
    {
        gt_direct_digit(M,sflag,pos,gt_width(2),true,&idx,&neg);
        gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
        qsb_asym_last_add(X,Y,ZZ,ZZZ, cx,cy, y0);
    }
#else
    int32_t ec=gt_mixed_step<18>(M,sign);
    gt_digit_idx(ec, &idx, &neg); gt_load_signed(gTable,0,idx,neg,x0,y0);
    ec=gt_mixed_step<17>(M,sign);
    gt_digit_idx(ec, &idx, &neg); gt_load_signed(gTable,1,idx,neg,x1,y1);
    _PointAddXYZZ_mm_def(X,Y,ZZ,ZZZ, x0,y0, x1,y1);
    uint64_t cx[4],cy[4];
    uint32_t table_base=gt_offset(2);
    #pragma unroll 1
    for (int c=2;c<GT_CHUNKS-1;c++){
        ec=gt_mixed_step<17>(M,sign);
        gt_digit_idx(ec, &idx, &neg); gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
        _PointAddXYZZ_def<true>(X,Y,ZZ,ZZZ, cx,cy, y0);
        Load256(y0, cy);                /* current affine y anchors next madd */
        table_base += 1u << 16;
    }
    {
        ec=sign*(int32_t)M[0];
        gt_digit_idx(ec, &idx, &neg); gt_load_signed_flat(gTable,table_base,idx,neg,cx,cy);
        qsb_asym_last_add(X,Y,ZZ,ZZZ, cx,cy, y0);
    }
#endif
#endif
}__device__ __forceinline__ void qsb_xyzz_finish_prepare(
    uint64_t *X_D, uint64_t *ZZ, uint64_t *ZZZ, uint64_t *xR, uint64_t *W
) {
    uint64_t t[4];
    _ModMult(t, xR, ZZ);
    _ModSub256(t, t, X_D);
    Load256(X_D, t);             /* X_D becomes d */
    _ModMult(W, ZZZ, X_D);       /* W = ZZZ*d */
    W[4] = 0;
}__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed(
    uint64_t *Y, uint64_t *ZZ, uint64_t *ZZZ,
    uint64_t *inv, uint64_t *xR, uint64_t *yR,
    uint64_t *x1, uint64_t *x2
) {
    uint64_t yb[4], m1[4], m2[4], t[4], s[4];
    uint64_t cc[4]={QSB_U2R_C[0],QSB_U2R_C[1],QSB_U2R_C[2],QSB_U2R_C[3]};

    _ModMult(yb, yR, ZZZ);       /* yR*B */
    _ModMult(ZZ, inv);           /* h = A/(B*d), kept in ZZ */

    _ModSub256(m1, yb, Y);
    _ModMult(m1, ZZ);            /* lambda1 = (yR*B-Y)*h */
    _ModAdd256(m2, yb, Y);
    _ModMult(m2, ZZ);            /* m2 = (yR*B+Y)*h = -lambda2 */
    _ModAdd256(s, m1, m2);       /* lambda1+m2 = 2*yR/(xR-xP) */

    _ModSub256(t, m1, cc);
    _ModMult(x1, s, t);
    _ModAdd256(x1, x1, xR);      /* x1 = (lambda1+m2)*(lambda1-c) + xR */
    _ModSub256(t, xR, x1);
    _ModMult(t, m1);
    _ModSub256(t, yR);           /* y1 = lambda1*(xR-x1) - yR */
    uint32_t parities = (uint32_t)(t[0] & 1ULL);

    _ModSub256(t, m2, cc);
    _ModMult(x2, s, t);
    _ModAdd256(x2, x2, xR);      /* x2 = (lambda1+m2)*(m2-c) + xR */
    _ModSub256(t, xR, x2);
    _ModMult(t, m2);
    _ModSub256(t, yR);           /* y2 = -(m2*(xR-x2) - yR) */
    /* y2=-t. Since p is odd, field negation flips its parity. */
    parities |= (uint32_t)(((t[0] & 1ULL) ^ 1ULL) << 1);
    return parities;
}
extern "C" int audit_chain(const uint64_t*k,const uint64_t*coef){
 ctx=BN_CTX_new();group=EC_GROUP_new_by_curve_name(NID_secp256k1);prime=BN_new();ord=BN_new();halfbase=BN_new();require(EC_GROUP_get_curve(group,prime,nullptr,nullptr,ctx));require(EC_GROUP_get_order(group,ord,ctx));
 table=(uint8_t*)mmap(nullptr,64ull*1024*1024,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);require(table!=MAP_FAILED);load_count=0;
 BIGNUM *two=BN_new(),*bs=read256(coef),*scalar=read256(k),*wk=BN_new();BN_set_word(two,2);require(BN_mod_inverse(halfbase,two,ord,ctx));require(BN_mod_mul(halfbase,halfbase,bs,ord,ctx));require(BN_mod_mul(wk,scalar,bs,ord,ctx));
 EC_POINT *want=EC_POINT_new(group);require(EC_POINT_mul(group,want,wk,nullptr,nullptr,ctx));uint64_t state[16];_FixedBaseSignedXYZZStream(state,state+4,state+8,state+12,k,table);require(load_count==15);
 BIGNUM *xx=read256(state),*yy=read256(state+4),*zz=read256(state+8),*zzz=read256(state+12),*wx=BN_new(),*wy=BN_new();int inf=EC_POINT_is_at_infinity(group,want),ok=1,recovered=0;
 if(inf)ok=BN_is_zero(zz)&&BN_is_zero(zzz);else if(BN_is_zero(zz)||BN_is_zero(zzz))ok=0;else{require(BN_mod_inverse(zz,zz,prime,ctx));require(BN_mod_inverse(zzz,zzz,prime,ctx));require(BN_mod_mul(xx,xx,zz,prime,ctx));require(BN_mod_mul(yy,yy,zzz,prime,ctx));require(EC_POINT_get_affine_coordinates(group,want,wx,wy,ctx));ok=BN_cmp(xx,wx)==0&&BN_cmp(yy,wy)==0;}
 if(ok&&!inf){
 BIGNUM *rk=BN_new(),*rx=BN_new(),*ry=BN_new(),*c=BN_new(),*tmp=BN_new();BN_set_word(rk,17);EC_POINT *R=EC_POINT_new(group),*sum=EC_POINT_new(group);require(EC_POINT_mul(group,R,rk,nullptr,nullptr,ctx));require(EC_POINT_get_affine_coordinates(group,R,rx,ry,ctx));
 require(BN_mod_sqr(c,rx,prime,ctx));require(BN_mul_word(c,3));require(BN_mod_add(tmp,ry,ry,prime,ctx));require(BN_mod_inverse(tmp,tmp,prime,ctx));require(BN_mod_mul(c,c,tmp,prime,ctx));write256(QSB_U2R_C,c);
 uint64_t xR[4],yR[4],W[5],inv[5],x1[4],x2[4];write256(xR,rx);write256(yR,ry);qsb_xyzz_finish_prepare(state,state+8,state+12,xR,W);
 if(W[0]|W[1]|W[2]|W[3]){memcpy(inv,W,40);_ModInv(inv);uint32_t par=qsb_xyzz_finish_precomputed(state+4,state+8,state+12,inv,xR,yR,x1,x2);
 for(int j=0;j<2;j++){if(j)require(EC_POINT_invert(group,R,ctx));require(EC_POINT_add(group,sum,want,R,ctx));require(EC_POINT_get_affine_coordinates(group,sum,rx,ry,ctx));uint64_t ex[4],ey[4];write256(ex,rx);write256(ey,ry);ok=ok&&!memcmp(j?x2:x1,ex,32)&&((par>>j&1)==(ey[0]&1));++recovered;}}
 else require(BN_cmp(wx,rx)==0);
 BN_free(rk);BN_free(rx);BN_free(ry);BN_free(c);BN_free(tmp);EC_POINT_free(R);EC_POINT_free(sum);
 }
 munmap(table,64ull*1024*1024);BN_free(two);BN_free(bs);BN_free(scalar);BN_free(wk);BN_free(xx);BN_free(yy);BN_free(zz);BN_free(zzz);BN_free(wx);BN_free(wy);BN_free(halfbase);BN_free(ord);BN_free(prime);EC_POINT_free(want);EC_GROUP_free(group);BN_CTX_free(ctx);return ok?1+recovered:0;
}
extern "C" uint64_t norm_count(){return norms;}
extern "C" void audit_recode(const uint64_t*k,int32_t*d){uint64_t m[4];int sign;gt_recode_setup(k,m,&sign);for(int c=0;c<15;c++){uint32_t ix;uint64_t neg;gt_direct_digit(m,sign<0,gt_shift(c)+1,gt_width(c),c==14,&ix,&neg);d[c]=(int32_t)(2*ix+1)*(neg?-1:1);}}
