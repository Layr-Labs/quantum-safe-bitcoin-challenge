// GLV14 dual-x deferred XYZZ chain. New exceptional paths retain all add/sub carries.
#pragma once
#include "GLV608Decode.cuh"
#include "DualXLoad.cuh"
#if !QSB_YOFF || !QSB_FUSE_SQRADDSUB2
#error "GLV608 integration requires promoted offset-Y/fused-square configuration"
#endif
__device__ __forceinline__ bool g14_zero(const uint64_t *a) {
 return !(a[0]|a[1]|a[2]|a[3]) || (a[0]==0xFFFFFFFEFFFFFC2FULL && (a[1]&a[2]&a[3])==UINT64_MAX);
}
__device__ __forceinline__ void g14_norm(uint64_t *a) {
 if((a[1]&a[2]&a[3])==UINT64_MAX && a[0]>=0xFFFFFFFEFFFFFC2FULL){
  a[0]-=0xFFFFFFFEFFFFFC2FULL;a[1]=a[2]=a[3]=0;
 }
}
// Full-carry canonical subtraction. Copies permit arbitrary output/input aliasing.
__device__ __forceinline__ void g14_sub(uint64_t *out,const uint64_t *aa,const uint64_t *bb) {
 uint64_t a[4],b[4];Load256(a,aa);Load256(b,bb);g14_norm(a);g14_norm(b);
 uint64_t r0,r1,r2,r3;
 asm("{\n.reg .u64 x0,x1,x2,x3,m,k;\n"
 "sub.cc.u64 x0,%4,%8; subc.cc.u64 x1,%5,%9; subc.cc.u64 x2,%6,%10; subc.cc.u64 x3,%7,%11;\n"
 "subc.u64 m,0,0; and.b64 k,m,0xFFFFFFFEFFFFFC2F;\n"
 "add.cc.u64 x0,x0,k; addc.cc.u64 x1,x1,m; addc.cc.u64 x2,x2,m; addc.u64 x3,x3,m;\n"
 "mov.u64 %0,x0;mov.u64 %1,x1;mov.u64 %2,x2;mov.u64 %3,x3;\n}"
 :"=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
 :"l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
 out[0]=r0;out[1]=r1;out[2]=r2;out[3]=r3;
}
__device__ __forceinline__ void g14_add(uint64_t *out,const uint64_t *aa,const uint64_t *bb) {
 uint64_t a[4],b[4];Load256(a,aa);Load256(b,bb);g14_norm(a);g14_norm(b);_ModAdd256(out,a,b);
}
__device__ __forceinline__ void g14_unoffset(uint64_t *y) {
 uint64_t c[4]={0x800001E8ULL,0,0,0};g14_sub(y,y,c);
}
__device__ __forceinline__ void g14_inf(uint64_t *X,uint64_t *Y,uint64_t *U,uint64_t *V) {
 #pragma unroll
 for(int i=0;i<4;++i){X[i]=0;Y[i]=i==0?1:0;U[i]=V[i]=0;}
}
// Rare doubling: input Y is the resolved ordinate, output has zero affine anchor.
// Multiplication/squaring remain the inherited promoted primitives; additions are complete.
__device__ __noinline__ int g14_double(uint64_t *X,uint64_t *Y,uint64_t *U,uint64_t *V,uint64_t *anchor) {
 if(g14_zero(Y)){g14_inf(X,Y,U,V);return 0;}
 uint64_t S[4],A[4],B[4],Q[4],M[4],T[4],F[4];
 g14_add(S,Y,Y);_ModSqr(A,S);_ModMult(B,A,S);_ModMult(Q,X,A);
 _ModSqr(M,X);g14_add(T,M,M);g14_add(M,T,M);
 _ModSqr(T,M);g14_sub(T,T,Q);g14_sub(T,T,Q);
 g14_sub(F,Q,T);_ModMult(F,M);_ModMult(S,Y,B);g14_sub(Y,F,S);
 _ModMult(U,A);_ModMult(V,B);Load256(X,T);
 anchor[0]=0x800001E8ULL;anchor[1]=anchor[2]=anchor[3]=0;return 2;
}
// State1 stores one affine point as X and offset Y. State2 is deferred XYZZ.
__device__ __forceinline__ int g14_seed(uint64_t *X,uint64_t *Y,uint64_t *U,uint64_t *V,
 const uint64_t *x,const uint64_t *y,uint64_t *anchor) {
 uint64_t P[4],R[4],Q[4],T[4];
 _ModSub256(P,(uint64_t*)x,X);_ModSub256(R,(uint64_t*)y,Y);
 if(g14_zero(P)){
  if(!g14_zero(R)){g14_inf(X,Y,U,V);return 0;}
  g14_unoffset(Y);U[0]=V[0]=1;U[1]=U[2]=U[3]=V[1]=V[2]=V[3]=0;
  return g14_double(X,Y,U,V,anchor);
 }
 Load256(anchor,Y);_ModSqr(U,P);_ModMult(V,U,P);_ModMult(Q,X,U);
 _ModSqr(T,R);_ModSub256(T,T,V);_ModSub256(T,T,Q);_ModSub256(T,T,Q);
 _ModSub256(Q,Q,T);_ModMult(Y,Q,R);Load256(X,T);return 2;
}
__device__ __forceinline__ int g14_madd(uint64_t *X,uint64_t *Y,uint64_t *U,uint64_t *V,
 const uint64_t *x,const uint64_t *y,uint64_t *anchor) {
 uint64_t U2[4],S2[4],P[4],R[4],PP[4],PPP[4],Q[4],T[4];
 _ModAddLazyOff(S2,y,anchor);_ModMult(S2,V);_ModSub256(R,S2,Y);
 _ModMult(U2,(uint64_t*)x,U);_ModSub256(P,U2,X);
 if(g14_zero(P)){
  if(!g14_zero(R)){g14_inf(X,Y,U,V);return 0;}
  uint64_t a[4];Load256(a,anchor);g14_unoffset(a);_ModMult(S2,a,V);g14_sub(Y,Y,S2);
  return g14_double(X,Y,U,V,anchor);
 }
 _ModSqr(PP,P);_ModMult(PPP,PP,P);_ModMult(Q,U2,PP);
 _ModSqrAddSub2(T,R,PPP,Q);
 _ModMult(V,PPP);_ModMult(U,PP);_ModSub256(Q,Q,T);_ModMult(Q,R);Load256(Y,Q);
 Load256(X,T);Load256(anchor,y);return 2;
}
__device__ void qsb_glv14_chain(uint64_t *X,uint64_t *Y,uint64_t *U,uint64_t *V,
 const uint64_t scalar[4],const uint8_t *table,uint32_t *codes) {
 qsb_glv608_decode(codes,QSB_TREE_N,scalar);
 int state=0;uint64_t x[4],y[4],anchor[4];
 // Scalar-dependent branches handle identities and all zero-denominator point-add cases.
 #pragma unroll 1
 for(int c=0;c<14;++c){
  uint32_t code=((volatile uint32_t*)codes)[(size_t)c*QSB_TREE_N];
  if(!qsb_glv_load(table,c,code,x,y))continue;
  if(state==0){Load256(X,x);Load256(Y,y);state=1;}
  else if(state==1)state=g14_seed(X,Y,U,V,x,y,anchor);
  else state=g14_madd(X,Y,U,V,x,y,anchor);
 }
 if(state==0){g14_inf(X,Y,U,V);return;}
 if(state==1){g14_unoffset(Y);U[0]=V[0]=1;U[1]=U[2]=U[3]=V[1]=V[2]=V[3]=0;return;}
 g14_unoffset(anchor);_ModMult(x,anchor,V);_ModSub256(Y,Y,x);
}
