// SPDX-License-Identifier: GPL-3.0-only
// One accumulator, shared odd-digit table and two endomorphism basis changes.
#pragma once

GLV_INLINE int glv_abs_component(uint64_t (&m)[3],const uint64_t (&x)[3]) {
    uint64_t mask=0-(x[2]>>63),carry=mask&1;
    #pragma unroll
    for(int i=0;i<3;i++) {
        __uint128_t t=(__uint128_t)(x[i]^mask)+carry;
        m[i]=(uint64_t)t;carry=(uint64_t)(t>>64);
    }
    return mask?-1:1;
}
GLV_INLINE int32_t glv_next16(uint64_t (&m)[3],int sign) {
    int32_t digit=sign*((int32_t)(m[0]&0x1ffff)-0x10000);
    m[0]=((m[0]>>16)|(m[1]<<48))|1;
    m[1]=(m[1]>>16)|(m[2]<<48);m[2]>>=16;
    return digit;
}
template<bool FILTER>
__device__ __forceinline__ void glv_load(const uint8_t *table,unsigned slot,int32_t digit,
                                        uint64_t *x,uint64_t *y) {
    uint32_t idx;uint64_t neg;gt_digit_idx(digit,&idx,&neg);
    if constexpr(FILTER)gt_load_signed_flat_f(table,slot<<15,idx,neg,x,y);
    else gt_load_signed_flat(table,slot<<15,idx,neg,x,y);
}
template<bool FILTER>
__device__ __forceinline__ void glv_fixed_base(
    uint64_t *X,uint64_t *Y,uint64_t *ZZ,uint64_t *ZZZ,
    const uint64_t input[4],const uint8_t *table,uint32_t &bad) {
    // Intermediate additions cannot be equal/inverse points. In the u half,
    // the accumulated integer is odd while the next point is divisible by
    // 2^16, and their magnitudes are far below n. In the v half before its
    // top digit, any exceptional relation would give a nonzero GLV lattice
    // vector with |u|<2^130 and |v_partial|<2^112. For the pinned basis its
    // inverse bounds give |coefficient_1|<1 and |coefficient_2|<4. The three
    // possible nonzero multiples of (a2,a1) all have |v|>=a1>2^125. Thus no
    // such vector exists. The final top-digit addition remains complete.
    // Copy before writing any output: publication callers may alias input.
    uint64_t z[4]={input[0],input[1],input[2],input[3]};
    uint64_t u[3],v[3],m[3];glv_split_odd(u,v,z);
    int sign=glv_abs_component(m,u);
    uint64_t x0[4],y0[4],x1[4],y1[4];
    glv_load<FILTER>(table,0,glv_next16(m,sign),x0,y0);
    glv_load<FILTER>(table,1,glv_next16(m,sign),x1,y1);
    if constexpr(FILTER)qsb_filter_point_seed(X,Y,ZZ,ZZZ,x0,y0,x1,y1,bad);
    else _PointAddXYZZ_mm_def(X,Y,ZZ,ZZZ,x0,y0,x1,y1);
    #pragma unroll 1
    for(int c=2;c<16;c++) {
        if(c==8) {
            uint64_t beta2[4]={0x3ec693d68e6afa40ULL,0x630fb68aed0a766aULL,
                               0x919bb86153cbcb16ULL,0x851695d49a83f8efULL};
            _ModMult(X,beta2);
            sign=glv_abs_component(m,v);
        }
        const unsigned slot=(unsigned)c&7u;
        const int32_t digit=slot==7?sign*(int32_t)m[0]:glv_next16(m,sign);
        glv_load<FILTER>(table,slot,digit,x1,y1);
        if(c==15) {
            if constexpr(FILTER)qsb_filter_last_add(X,Y,ZZ,ZZZ,x1,y1,y0,bad);
            else qsb_complete_last_add(X,Y,ZZ,ZZZ,x1,y1,y0);
        } else {
            if constexpr(FILTER) {
                qsb_filter_point_add<true>(X,Y,ZZ,ZZZ,x1,y1,y0,bad);
#if !QSB_CHAIN_ANCHOR_UPDATE || !defined(__CUDA_ARCH__)
                Load256(y0,y1);
#endif
            } else {
                _PointAddXYZZ_def<true>(X,Y,ZZ,ZZZ,x1,y1,y0);
                Load256(y0,y1);
            }
        }
    }
    uint64_t beta[4]={0xc1396c28719501eeULL,0x9cf0497512f58995ULL,
                      0x6e64479eac3434e9ULL,0x7ae96a2b657c0710ULL};
    _ModMult(X,beta);
}
