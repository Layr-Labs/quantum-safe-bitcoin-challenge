// GLV608 exact scalar split. No CUDA compile or register-pressure claim.
#pragma once
#include "Recode608.cuh"
template<int A,int B,int O> __device__ __forceinline__ void qsb_wmul(uint32_t *out,const uint32_t *a,const uint32_t *b) {
 #pragma unroll
 for(int i=0;i<O;++i)out[i]=0;
 #pragma unroll
 for(int i=0;i<A;++i){
  uint64_t carry=0;
  #pragma unroll
  for(int j=0;j<B && i+j<O;++j){
   uint64_t t=(uint64_t)a[i]*b[j]+out[i+j]+carry;
   out[i+j]=(uint32_t)t;carry=t>>32;
  }
  if(i+B<O)out[i+B]=(uint32_t)carry;
 }
}
template<int L> __device__ __forceinline__ void qsb_wadd(uint32_t *a,const uint32_t *b) {
 uint64_t carry=0;
 #pragma unroll
 for(int i=0;i<L;++i){uint64_t t=(uint64_t)a[i]+b[i]+carry;a[i]=(uint32_t)t;carry=t>>32;}
}
template<int L> __device__ __forceinline__ void qsb_wsub(uint32_t *a,const uint32_t *b) {
 uint64_t borrow=0;
 #pragma unroll
 for(int i=0;i<L;++i){uint64_t t=(uint64_t)b[i]+borrow;uint32_t ai=a[i];a[i]=ai-(uint32_t)t;borrow=(uint64_t)ai<t;}
}
template<int L> __device__ __forceinline__ bool qsb_wless(const uint32_t *a,const uint32_t *b) {
 #pragma unroll
 for(int i=L-1;i>=0;--i){if(a[i]!=b[i])return a[i]<b[i];}
 return false;
}
__device__ __forceinline__ bool qsb_sless5(const uint32_t *a,const uint32_t *b) {
 uint32_t sa=a[4]>>31,sb=b[4]>>31;
 return sa!=sb?sa>sb:qsb_wless<5>(a,b);
}
// Exact round(b*k/N), for k<N and b in {B1,B2}. N=2^256-D, D<2^129.
// T=k*b=q0*2^256+lo. r=lo+q0*D+floor(N/2).
// h=r>>256 <=2. r2=low256(r)+h*D <2*N; one comparison completes division.
__device__ __forceinline__ void qsb_round_glv(uint32_t *q,const uint32_t *k,const uint32_t *b) {
 const uint32_t dn[5]={0x2fc9bebfu,0x402da173u,0x50b75fc4u,0x45512319u,0x1u};
 const uint32_t half[9]={0x681b20a0u,0xdfe92f46u,0x57a4501du,0x5d576e73u,0xffffffffu,0xffffffffu,0xffffffffu,0x7fffffffu,0x0u};
 const uint32_t n[9]={0xd0364141u,0xbfd25e8cu,0xaf48a03bu,0xbaaedce6u,0xfffffffeu,0xffffffffu,0xffffffffu,0xffffffffu,0x0u};
 uint32_t t[12];qsb_wmul<8,4,12>(t,k,b);
 #pragma unroll
 for(int i=0;i<4;++i)q[i]=t[8+i];
 uint32_t r[9];qsb_wmul<4,5,9>(r,q,dn);
 t[8]=0; // high quotient is already copied; reuse product as zero-extended low256.
 qsb_wadd<9>(r,t);qsb_wadd<9>(r,half);
 uint32_t h=r[8];r[8]=0;
 uint64_t carry=0;
 #pragma unroll
 for(int i=0;i<9;++i){
  uint32_t d=i<5?dn[i]:0u;
  uint64_t v=(uint64_t)h*d+r[i]+carry;r[i]=(uint32_t)v;carry=v>>32;
 }
 qsb_inc_words<4>(q,h+(uint32_t)!qsb_wless<9>(r,n));
}
// k is little-endian u32[8], may be >=N. Outputs signed 160-bit x,y, |x|,|y|<=R.
__device__ __forceinline__ void qsb_glv_split608(uint32_t *x,uint32_t *y,const uint32_t *input) {
 const uint32_t n[8]={0xd0364141u,0xbfd25e8cu,0xaf48a03bu,0xbaaedce6u,0xfffffffeu,0xffffffffu,0xffffffffu,0xffffffffu};
 const uint32_t a1[5]={0x9284eb15u,0xe86c90e4u,0xa7d46bcdu,0x3086d221u,0x0u},a2[5]={0x9d44cfd8u,0x57c1108du,0xa8e2f3f6u,0x14ca50f7u,0x1u},b1[5]={0xabfe4c3u,0x6f547fa9u,0x10e8828u,0xe4437ed6u,0x0u},b2[5]={0x9284eb15u,0xe86c90e4u,0xa7d46bcdu,0x3086d221u,0x0u};
 const uint32_t r[5]={0xcea267ecu,0x2be08846u,0xd47179fbu,0x8a65287bu,0x0u},nr[5]={0x315d9814u,0xd41f77b9u,0x2b8e8604u,0x759ad784u,0xffffffffu},t[5]={0xc3e28329u,0xbc8c089du,0xd362f1d2u,0xa621a9a5u,0xffffffffu},nt[5]={0x3c1d7cd7u,0x4373f762u,0x2c9d0e2du,0x59de565au,0x0u};
 uint32_t k[8];
 #pragma unroll
 for(int i=0;i<8;++i)k[i]=input[i];
 if(!qsb_wless<8>(k,n))qsb_wsub<8>(k,n);
 uint32_t c1[4],c2[4];qsb_round_glv(c1,k,b2);qsb_round_glv(c2,k,b1);
 uint32_t p[5];
 #pragma unroll
 for(int i=0;i<5;++i)x[i]=k[i];
 qsb_wmul<4,5,5>(p,c1,a1);qsb_wsub<5>(x,p);
 qsb_wmul<4,5,5>(p,c2,a2);qsb_wsub<5>(x,p);
 qsb_wmul<4,5,5>(y,c1,b1);
 qsb_wmul<4,5,5>(p,c2,b2);qsb_wsub<5>(y,p);
 if(qsb_sless5(r,x)){
  if(!qsb_sless5(y,t)){qsb_wsub<5>(x,a2);qsb_wsub<5>(y,b2);}
  else {qsb_wsub<5>(x,a1);qsb_wadd<5>(y,b1);}
 }else if(qsb_sless5(x,nr)){
  if(!qsb_sless5(nt,y)){qsb_wadd<5>(x,a2);qsb_wadd<5>(y,b2);}
  else {qsb_wadd<5>(x,a1);qsb_wsub<5>(y,b1);}
 }
}
