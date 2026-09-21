// GLV608. Corrected GLV inputs use signed 160-bit containers (129 bits needed).
#pragma once
#include "Div19.cuh"
#include "C6Digit.cuh"
template<int W> __device__ __forceinline__ void qsb_neg_words(uint32_t *a) {
 uint64_t c=1;
 #pragma unroll
 for(int i=0;i<W;++i){c=(uint64_t)(~a[i])+c;a[i]=(uint32_t)c;c>>=32;}
}
template<int W> __device__ __forceinline__ void qsb_inc_words(uint32_t *a,uint32_t c) {
 #pragma unroll
 for(int i=0;i<W;++i){uint64_t t=(uint64_t)a[i]+c;a[i]=(uint32_t)t;c=(uint32_t)(t>>32);}
}
template<int W,int S> __device__ __forceinline__ void qsb_sar_words(uint32_t *a) {
 static_assert(S>0 && S<32,"nonzero subword shift");
 #pragma unroll
 for(int i=0;i<W-1;++i)a[i]=(a[i]>>S)|(a[i+1]<<(32-S));
 a[W-1]=(a[W-1]>>S)|((0u-(a[W-1]>>31))<<(32-S));
}
template<int W> __device__ __forceinline__ uint32_t qsb_recode512(uint32_t *x,uint32_t *y) {
 qsb_c6_digit d=qsb_c6_radix<512>(x[0]&511u,y[0]&511u);
 qsb_sar_words<W,9>(x);qsb_sar_words<W,9>(y);
 qsb_inc_words<W>(x,d.dx<0);qsb_inc_words<W>(y,d.dy<0);
 return d.code;
}
// Input absolute bit bound after >>5 is <=32*L. Output floor(x/608), residue [0,607].
template<int L> __device__ __forceinline__ uint32_t qsb_floor608(uint32_t *x) {
 uint32_t lo=x[0]&31u;qsb_sar_words<4,5>(x);
 uint32_t neg=x[3]>>31;
 if(neg)qsb_neg_words<4>(x);
 uint32_t q[4]={0,0,0,0};
 uint32_t rem=qsb_udiv19_words<L>(q,x);
 if(neg){qsb_inc_words<4>(q,rem!=0);qsb_neg_words<4>(q);rem=rem?19u-rem:0;}
 #pragma unroll
 for(int i=0;i<4;++i)x[i]=q[i];
 return 32u*rem+lo;
}
template<int L> __device__ __forceinline__ uint32_t qsb_recode608(uint32_t *x,uint32_t *y) {
 uint32_t a=qsb_floor608<L>(x),b=qsb_floor608<L>(y);
 qsb_c6_digit d=qsb_c6_radix<608>(a,b);
 qsb_inc_words<4>(x,d.dx<0);qsb_inc_words<4>(y,d.dy<0);
 return d.code;
}
__device__ __forceinline__ void qsb_final_offer(int x,int y,uint32_t tag,int &bx,int &by,uint32_t &bt) {
 if(x<bx || (x==bx && y<by)){bx=x;by=y;bt=tag;}
}
// x,y in [-236,236]. Canonicalization is over Z; store its inverse orientation.
__device__ __forceinline__ uint32_t qsb_final236(int x,int y) {
 int a=x,b=y;uint32_t t=0;
 qsb_final_offer(-x,-y,4,a,b,t);
 qsb_final_offer(-y,x-y,1,a,b,t);
 qsb_final_offer(y,y-x,5,a,b,t);
 qsb_final_offer(y-x,-x,2,a,b,t);
 qsb_final_offer(x-y,x,6,a,b,t);
 int u=-a,v=-b;
 uint32_t rank=u==0?0u:(uint32_t)(1+(v-1)*237+(u-v));
 uint32_t j=t&3;j=j?3-j:0;
 return rank|(j<<16)|((t>>2)<<18);
}
// Mutates the two input arrays; each has >=5 little-endian u32 words.
// Output plane stride must be positive; caller reserves 14 planes.
__device__ __forceinline__ bool qsb_recode_glv608(uint32_t *out,int stride,uint32_t *x,uint32_t *y) {
 uint32_t code=qsb_recode512<5>(x,y);out[0]=code;bool regular=(code&65535u)!=0; //signed129 input -> signed120 after first quotient
 #pragma unroll
 for(int i=1;i<4;++i){code=qsb_recode512<4>(x,y);out[i*stride]=code;regular &= (code&65535u)!=0;}
 #pragma unroll
 for(int i=4;i<7;++i){code=qsb_recode608<3>(x,y);out[i*stride]=code;regular &= (code&65535u)!=0;}
 #pragma unroll
 for(int i=7;i<10;++i){code=qsb_recode608<2>(x,y);out[i*stride]=code;regular &= (code&65535u)!=0;}
 #pragma unroll
 for(int i=10;i<13;++i){code=qsb_recode608<1>(x,y);out[i*stride]=code;regular &= (code&65535u)!=0;}
 // Avoid implementation-defined conversion of out-of-range uint32_t to int32_t.
 int a=x[3]>>31?-(int)(0u-x[0]):(int)x[0];
 int b=y[3]>>31?-(int)(0u-y[0]):(int)y[0];
 code=qsb_final236(a,b);out[13*stride]=code;regular &= (code&65535u)!=0;
 return regular;
}
