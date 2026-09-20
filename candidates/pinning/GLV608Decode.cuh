// GLV608 scalar-to-descriptor entry point; integer semantics audited in Python.
#pragma once
#include "Split608.cuh"
__device__ __forceinline__ void qsb_glv608_decode(uint32_t *out,int stride,const uint64_t scalar[4]) {
 uint32_t k[8];
 #pragma unroll
 for(int i=0;i<4;++i){k[2*i]=(uint32_t)scalar[i];k[2*i+1]=(uint32_t)(scalar[i]>>32);}
 uint32_t x[5],y[5];
 qsb_glv_split608(x,y,k);
 qsb_recode_glv608(out,stride,x,y);
}
