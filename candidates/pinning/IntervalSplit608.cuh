// Certified exact rounded GLV coefficient with complete fallback and three-product residual.
#pragma once
#include "Split608.cuh"
__device__ __forceinline__ void qsb_round_glv_interval(uint32_t *q,const uint32_t *k,
 const uint32_t *b,const uint32_t *g) {
 uint64_t acc=0;
 #pragma unroll
 for(int i=3;i<8;++i)acc+=(uint64_t)__umulhi(k[i],g[10-i]);
 uint32_t head[5];
 #pragma unroll
 for(int diagonal=11;diagonal<=14;++diagonal){
  uint64_t high=0;
  #pragma unroll
  for(int i=diagonal-7;i<8;++i){
   uint64_t product=(uint64_t)k[i]*g[diagonal-i];
   acc+=(uint32_t)product;high+=product>>32;
  }
  head[diagonal-11]=(uint32_t)acc;acc=(acc>>32)+high;
 }
 head[4]=(uint32_t)acc;
 uint64_t rounded=(uint64_t)head[0]+0x80000000u;
 // The exact scaled quotient lies in [H,H+13); both rounded interval ends
 // agree unless this low word lies within13 of the next base-2^32 boundary.
 if((uint32_t)rounded>=0xfffffff3u){qsb_round_glv(q,k,b);return;}
 #pragma unroll
 for(int i=0;i<4;++i)q[i]=head[i+1];
 qsb_inc_words<4>(q,(uint32_t)(rounded>>32));
}
__device__ __forceinline__ void qsb_glv_split608_interval(uint32_t *x,uint32_t *y,const uint32_t *input) {
 const uint32_t n[8]={0xd0364141u,0xbfd25e8cu,0xaf48a03bu,0xbaaedce6u,0xfffffffeu,0xffffffffu,0xffffffffu,0xffffffffu};
 const uint32_t a1[5]={0x9284eb15u,0xe86c90e4u,0xa7d46bcdu,0x3086d221u,0x0u},a2[5]={0x9d44cfd8u,0x57c1108du,0xa8e2f3f6u,0x14ca50f7u,0x1u},b1[5]={0xabfe4c3u,0x6f547fa9u,0x10e8828u,0xe4437ed6u,0x0u},b2[5]={0x9284eb15u,0xe86c90e4u,0xa7d46bcdu,0x3086d221u,0x0u};
 const uint32_t r[5]={0xcea267ecu,0x2be08846u,0xd47179fbu,0x8a65287bu,0x0u},nr[5]={0x315d9814u,0xd41f77b9u,0x2b8e8604u,0x759ad784u,0xffffffffu},t[5]={0xc3e28329u,0xbc8c089du,0xd362f1d2u,0xa621a9a5u,0xffffffffu},nt[5]={0x3c1d7cd7u,0x4373f762u,0x2c9d0e2du,0x59de565au,0x0u};
 uint32_t k[8];
 #pragma unroll
 for(int i=0;i<8;++i)k[i]=input[i];
 if(!qsb_wless<8>(k,n))qsb_wsub<8>(k,n);
 const uint32_t g1[8]={0x45dbb030u,0xe893209au,0x71e8ca7fu,0x3daa8a14u,0x9284eb15u,0xe86c90e4u,0xa7d46bcdu,0x3086d221u},g2[8]={0x8ac47f71u,0x1571b4aeu,0x9df506c6u,0x221208acu,0xabfe4c4u,0x6f547fa9u,0x10e8828u,0xe4437ed6u};
 uint32_t c1[4],c2[4];qsb_round_glv_interval(c1,k,b2,g1);qsb_round_glv_interval(c2,k,b1,g2);
 // A2=A1+B1: t=A1*(c1+c2), x=k-t-B1*c2, y=A2*c1-t.
 // Sum c1+c2 needs129 bits; retain its fifth word. Same p[5] scratch allocation.
 uint32_t p[5];uint64_t carry=0;
 #pragma unroll
 for(int i=0;i<4;++i){uint64_t v=(uint64_t)c1[i]+c2[i]+carry;p[i]=(uint32_t)v;carry=v>>32;}
 p[4]=(uint32_t)carry;
 qsb_wmul<5,5,5>(y,p,a1);
 #pragma unroll
 for(int i=0;i<5;++i)x[i]=k[i];
 qsb_wsub<5>(x,y);
 qsb_wmul<4,5,5>(p,c2,b1);qsb_wsub<5>(x,p);
 qsb_wmul<4,5,5>(p,c1,a2);qsb_wsub<5>(p,y);
 #pragma unroll
 for(int i=0;i<5;++i)y[i]=p[i];
 if(qsb_sless5(r,x)){
  if(!qsb_sless5(y,t)){qsb_wsub<5>(x,a2);qsb_wsub<5>(y,b2);}
  else {qsb_wsub<5>(x,a1);qsb_wadd<5>(y,b1);}
 }else if(qsb_sless5(x,nr)){
  if(!qsb_sless5(nt,y)){qsb_wadd<5>(x,a2);qsb_wadd<5>(y,b2);}
  else {qsb_wadd<5>(x,a1);qsb_wsub<5>(y,b1);}
 }
}
