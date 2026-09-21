// GLV608 loader. The host table must contain canonical x,beta*x and offset y+c.
#pragma once
#include "DualXExact.cuh"
__host__ __device__ __forceinline__ uint32_t qsb_glv_base(int c) {
 return c<4?(uint32_t)c*43692u:174768u+(uint32_t)(c-4)*61612u;
}
__device__ __forceinline__ void qsb_glv_load_nonzero(const uint8_t *__restrict__ table,int c,uint32_t code,
 uint64_t *__restrict__ x,uint64_t *__restrict__ yoff) {
 uint32_t rank=code&65535u,j=(code>>16)&3u;
 // Caller has certified rank != 0.
 const uint8_t *entry=table+(size_t)(qsb_glv_base(c)+rank)*96u;
 const ulonglong2 *xp=(const ulonglong2 *)(entry+(j==1?32u:0u));
 ulonglong2 a=__ldg(xp),b=__ldg(xp+1);
 x[0]=a.x;x[1]=a.y;x[2]=b.x;x[3]=b.y;
 if(j==2){
  const ulonglong2 *bp=(const ulonglong2 *)(entry+32);
  ulonglong2 u=__ldg(bp),v=__ldg(bp+1);
  uint64_t bx[4]={u.x,u.y,v.x,v.y};qsb_dual_x_neg_sum(x,x,bx);
 }
 const ulonglong2 *yp=(const ulonglong2 *)(entry+64);
 a=__ldg(yp);b=__ldg(yp+1);
 uint64_t mask=0ULL-(uint64_t)((code>>18)&1u);
 yoff[0]=a.x^mask;yoff[1]=a.y^mask;yoff[2]=b.x^mask;yoff[3]=b.y^mask;
}
__device__ __forceinline__ bool qsb_glv_load(const uint8_t *__restrict__ table,int c,uint32_t code,
 uint64_t *__restrict__ x,uint64_t *__restrict__ yoff) {
 if((code&65535u)==0)return false;
 qsb_glv_load_nonzero(table,c,code,x,yoff);return true;
}
