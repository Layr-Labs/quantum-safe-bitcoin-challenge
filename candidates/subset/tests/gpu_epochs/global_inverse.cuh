// Global inverse pipeline; cofactor traversal derives from the promoted pinning track.
#pragma once
#ifndef QSB_EXTERNAL_INVERSE
#define QSB_EXTERNAL_INVERSE 1
#endif
#ifndef Q4_TREE_N
#define Q4_TREE_N 128
#endif
#ifndef Q4_STREAM_STATE
#define Q4_STREAM_STATE 1
#endif
#if QSB_EXTERNAL_INVERSE
__device__ __forceinline__ void q4_store(uint64_t *p,uint64_t x){
#if Q4_STREAM_STATE
 asm volatile("st.global.cs.u64 [%0],%1;"::"l"(p),"l"(x):"memory");
#else
 *p=x;
#endif
}
__device__ __forceinline__ uint64_t q4_load(const uint64_t*p){
#if Q4_STREAM_STATE
 uint64_t x;asm("ld.global.cs.u64 %0,[%1];":"=l"(x):"l"(p));return x;
#else
 return *p;
#endif
}
#include "pipeline_cofactor.cuh"
static_assert(Q4_TREE_N == 64 || Q4_TREE_N == 128, "tree must partition a 256-candidate epoch");
__device__ __forceinline__ void q4_mul(uint64_t*out,const uint64_t*a,const uint64_t*b){uint64_t tmp[5];qsb_field_mul_raw(tmp,(uint64_t*)a,(uint64_t*)b);qsb_field_normalize(tmp);Load256(out,tmp);}
__global__ void __launch_bounds__(Q4_TREE_N,512/Q4_TREE_N) q4_prepare(const epoch_desc_t *epochs,const uint32_t *first,const uint8_t *gtable,int n,uint64_t *saved,uint64_t *roots){
 int idx=blockIdx.x*Q4_TREE_N+threadIdx.x,ep=idx>>8,lane=idx&255; bool active=idx<n;
 uint64_t X[4],Y[4],U[4],V[4],W[5];
 uint32_t state[8],s2[8],b2[16];
 qsb_scheduled_window_hash(state,epochs+ep,lane,first+(size_t)ep*QSB_FIRST_SLOTS*8);
 for(int k=0;k<8;k++)b2[k]=state[k];b2[8]=0x80000000;for(int k=9;k<15;k++)b2[k]=0;b2[15]=256;
 _SHA256Initialize(s2);_SHA256Transform(s2,b2);
 uint64_t z[4];for(int k=0;k<4;k++)z[k]=((uint64_t)s2[6-2*k]<<32)|s2[7-2*k];
 // The filter is speculative; kernel_verify_pair_hits remains the output authority.
 uint32_t bad=0;qsb_filter_chain_trial(X,Y,U,V,z,gtable,bad);
 uint64_t a[4]={QSB_U2R[0],QSB_U2R[1],QSB_U2R[2],QSB_U2R[3]};
 qsb_xyzz_finish_prepare(X,U,V,a,W);qsb_field_normalize(W);
 bool usable=active && (W[0]|W[1]|W[2]|W[3]);if(!usable){W[0]=1;W[1]=W[2]=W[3]=W[4]=0;}
 __shared__ uint64_t products[4][2*Q4_TREE_N],excluded[4][Q4_TREE_N];
 // W becomes the product of the other denominators in this block.
 q4_cofactor_prepare<Q4_TREE_N>(W,roots,products,excluded);
 uint64_t h[5],vbar[5],tbar[5];qsb_field_mul_raw(h,U,W);qsb_field_mul_raw(vbar,Y,h);qsb_field_mul_raw(tbar,V,h);
 if(!usable)for(int k=0;k<4;k++)vbar[k]=tbar[k]=0;
 if(active)for(int k=0;k<4;k++){q4_store(saved+(size_t)k*n+idx,vbar[k]);q4_store(saved+(size_t)(k+4)*n+idx,tbar[k]);}
}
__global__ void __launch_bounds__(256,2) q4_inverse_roots(uint64_t *roots,int count){
 int i=blockIdx.x*256+threadIdx.x;uint64_t v[5]={1,0,0,0,0};
 if(i<count)for(int k=0;k<4;k++)v[k]=roots[(size_t)i*4+k];
 qsb_block_inverse_tree(v);qsb_field_normalize(v);
 if(i<count){uint64_t b[4]={QSB_U2R[4],QSB_U2R[5],QSB_U2R[6],QSB_U2R[7]},w[4];q4_mul(w,v,b);for(int k=0;k<4;k++){roots[(size_t)i*4+k]=v[k];roots[(size_t)(count+i)*4+k]=w[k];}}
}
__global__ void __launch_bounds__(Q4_TREE_N,768/Q4_TREE_N) q4_finish(const epoch_desc_t *epochs,int n,const uint64_t *saved,const uint64_t *roots,uint32_t *hitcnt,uint32_t *hitidx,uint8_t *hitcombos){
 int idx=blockIdx.x*Q4_TREE_N+threadIdx.x;if(idx>=n)return;int ep=idx>>8,lane=idx&255,count=(n+Q4_TREE_N-1)/Q4_TREE_N;
 uint64_t vb[4],tb[4],ri[4],wi[4];for(int k=0;k<4;k++){vb[k]=q4_load(saved+(size_t)k*n+idx);tb[k]=q4_load(saved+(size_t)(k+4)*n+idx);ri[k]=roots[(size_t)blockIdx.x*4+k];wi[k]=roots[(size_t)(count+blockIdx.x)*4+k];}
 if(!(tb[0]|tb[1]|tb[2]|tb[3]))return;
 uint64_t a[4]={QSB_U2R[0],QSB_U2R[1],QSB_U2R[2],QSB_U2R[3]},b[4]={QSB_U2R[4],QSB_U2R[5],QSB_U2R[6],QSB_U2R[7]},c[4]={QSB_U2R_C[0],QSB_U2R_C[1],QSB_U2R_C[2],QSB_U2R_C[3]};
 uint64_t u[4],v[4],l[4],m[4],sum[4],t[4],x1[4],x2[4];q4_mul(u,tb,wi);q4_mul(v,vb,ri);_ModSub256(l,u,v);_ModAdd256(m,u,v);_ModAdd256(sum,l,m);
 _ModSub256(t,l,c);q4_mul(x1,sum,t);_ModAdd256(x1,x1,a);
 _ModSub256(t,m,c);q4_mul(x2,sum,t);_ModAdd256(x2,x2,a);
 _ModSub256(t,a,x1);q4_mul(t,l,t);_ModSub256(t,t,b);uint32_t parity=t[0]&1;
 _ModSub256(t,a,x2);q4_mul(t,m,t);_ModSub256(t,b,t);parity|=(t[0]&1)<<1;
 int recid=0;if(qsb_k2s_gate(x1,x2,parity,&recid)){
  uint32_t pos=atomicAdd(hitcnt,1);if(pos<1024){hitidx[pos*4]=idx|((uint32_t)recid<<30);for(int k=0;k<6;k++)hitcombos[pos*ZLAB_HIT_REC+k]=epochs[ep].early[k];for(int k=0;k<3;k++)hitcombos[pos*ZLAB_HIT_REC+6+k]=WIN3[lane][k];}
 }
}
#endif
