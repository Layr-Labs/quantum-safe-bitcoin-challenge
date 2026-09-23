// SPDX-License-Identifier: GPL-3.0-only
// Shared lower windows and a joint endomorphism top window.
#pragma once
#include "glv_splitter.cuh"
enum { GLVJ_GT_CHUNKS=8, GLVJ_GT_TOTAL_ENTRIES=622592,
       GLVJ_GT_LO=256, GLVJ_GT_HI=1024, GLVJ_JOINT_OFFSET=589824 };
GLV_INLINE unsigned glvj_gt_width(int c){return c<2?18u:17u;}
GLV_INLINE unsigned glvj_gt_entries(int c){return c==7?32768u:(1u<<(glvj_gt_width(c)-1));}
GLV_INLINE unsigned glvj_gt_offset(int c){return c<2?(unsigned)c<<17:(unsigned)(c+2)<<16;}
GLV_INLINE int glvj_gt_shift(int c){return c<2?18*c:36+17*(c-2);}
GLV_INLINE int glvj_abs(uint64_t (&m)[3],const uint64_t (&x)[3]){
    uint64_t mask=0-(x[2]>>63),carry=mask&1;
    #pragma unroll
    for(int i=0;i<3;i++){
        __uint128_t t=(__uint128_t)(x[i]^mask)+carry;
        m[i]=(uint64_t)t;carry=(uint64_t)(t>>64);
    }
    return mask?-1:1;
}
GLV_INLINE int32_t glvj_next(uint64_t (&m)[3],int sign,unsigned width){
    const int32_t d=sign*((int32_t)(m[0]&((1u<<(width+1))-1))-(1<<width));
    m[0]=((m[0]>>width)|(m[1]<<(64-width)))|1;
    m[1]=(m[1]>>width)|(m[2]<<(64-width));m[2]>>=width;
    return d;
}
GLV_INLINE void glvj_recode(int32_t *digits,const uint64_t (&component)[3]){
    uint64_t m[3];const int sign=glvj_abs(m,component);
    for(int c=0;c<7;c++)digits[c]=glvj_next(m,sign,glvj_gt_width(c));
    digits[7]=sign*(int32_t)m[0];
}
GLV_INLINE unsigned glvj_joint_index(int32_t a,int32_t b){
    const unsigned aa=(unsigned)(a<0?-a:a),bb=(unsigned)(b<0?-b:b);
    return (aa>>1)|((bb>>1)<<7)|((unsigned)((a<0)!=(b<0))<<14);
}
