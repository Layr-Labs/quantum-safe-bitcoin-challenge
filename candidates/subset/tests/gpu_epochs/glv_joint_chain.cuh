// SPDX-License-Identifier: GPL-3.0-only
// Fifteen point selections from 38 MiB: seven lower windows per component,
// then one joint top point. Table bases are rebuilt for every fresh instance.
#pragma once
#include "glv_joint_recode.cuh"
template<bool FILTER>
__device__ __forceinline__ void glvj_load(const uint8_t *table,unsigned slot,int32_t digit,
                                         uint64_t *x,uint64_t *y){
    uint32_t idx;uint64_t neg;gt_digit_idx(digit,&idx,&neg);
    if constexpr(FILTER)gt_load_signed_flat_f(table,glvj_gt_offset(slot),idx,neg,x,y);
    else gt_load_signed_flat(table,glvj_gt_offset(slot),idx,neg,x,y);
}
template<bool FILTER>
__device__ __forceinline__ void glvj_fixed_base(
    uint64_t *X,uint64_t *Y,uint64_t *ZZ,uint64_t *ZZZ,
    const uint64_t input[4],const uint8_t *table,uint32_t &bad){
    // Copy first: callers may alias the scalar and point output.
    uint64_t z[4]={input[0],input[1],input[2],input[3]};
    uint64_t u[3],v[3],m[3];glv_split_odd(u,v,z);
    int sign=glvj_abs(m,u),topu=0;
    uint64_t x0[4],y0[4],x1[4],y1[4];
    glvj_load<FILTER>(table,0,glvj_next(m,sign,18),x0,y0);
    glvj_load<FILTER>(table,1,glvj_next(m,sign,18),x1,y1);
    if constexpr(FILTER)qsb_filter_point_seed(X,Y,ZZ,ZZZ,x0,y0,x1,y1,bad);
    else _PointAddXYZZ_mm_def(X,Y,ZZ,ZZZ,x0,y0,x1,y1);
    // Every partial lower component has magnitude <2^121. The inverse
    // GLV lattice basis bounds both relation coefficients below one, so
    // these nonfinal additions cannot double, cancel, or yield infinity.
    #pragma unroll 1
    for(int c=2;c<14;c++){
        if(c==7){
            topu=sign*(int32_t)m[0];
            uint64_t beta2[4]={0x3ec693d68e6afa40ULL,0x630fb68aed0a766aULL,
                               0x919bb86153cbcb16ULL,0x851695d49a83f8efULL};
            _ModMult(X,beta2);sign=glvj_abs(m,v);
        }
        const unsigned slot=c<7?(unsigned)c:(unsigned)(c-7);
        glvj_load<FILTER>(table,slot,glvj_next(m,sign,glvj_gt_width(slot)),x1,y1);
        if constexpr(FILTER){
            qsb_filter_point_add<true>(X,Y,ZZ,ZZZ,x1,y1,y0,bad);
#if !QSB_CHAIN_ANCHOR_UPDATE || !defined(__CUDA_ARCH__)
            Load256(y0,y1);
#endif
        }else{
            _PointAddXYZZ_def<true>(X,Y,ZZ,ZZZ,x1,y1,y0);Load256(y0,y1);
        }
    }
    const int topv=sign*(int32_t)m[0];
    const unsigned ji=glvj_joint_index(topu,topv);
    if constexpr(FILTER)gt_load_signed_flat_f(table,GLVJ_JOINT_OFFSET,ji,topu<0,x1,y1);
    else gt_load_signed_flat(table,GLVJ_JOINT_OFFSET,ji,topu<0,x1,y1);
    if constexpr(FILTER)qsb_filter_last_add(X,Y,ZZ,ZZZ,x1,y1,y0,bad);
    else qsb_complete_last_add(X,Y,ZZ,ZZZ,x1,y1,y0);
    uint64_t beta[4]={0xc1396c28719501eeULL,0x9cf0497512f58995ULL,
                      0x6e64479eac3434e9ULL,0x7ae96a2b657c0710ULL};
    _ModMult(X,beta);
}
