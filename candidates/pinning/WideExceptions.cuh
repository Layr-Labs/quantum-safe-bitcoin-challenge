// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "WideField.cuh"
__device__ __forceinline__ void wide_add(uint64_t *out,const uint64_t *a,const uint64_t *b) {
    uint64_t v[4],carry=0;
    #pragma unroll
    for(int j=0;j<4;j++){
        __uint128_t t=(__uint128_t)a[j]+b[j]+carry;
        v[j]=(uint64_t)t;carry=(uint64_t)(t>>64);
    }
    uint64_t k=carry*0x1000003D1ULL;carry=0;
    #pragma unroll
    for(int j=0;j<4;j++){
        __uint128_t t=(__uint128_t)v[j]+(j==0?k:0)+carry;
        v[j]=(uint64_t)t;carry=(uint64_t)(t>>64);
    }
    Load256(out,v);qsb_field_normalize(out);
}
__device__ __forceinline__ unsigned wide_exception(const uint64_t *input) {
    uint64_t k[4];Load256(k,input);
    if(k[3]==GT_ORDER_N[3] && (k[2]>GT_ORDER_N[2] ||
       (k[2]==GT_ORDER_N[2] && (k[1]>GT_ORDER_N[1] || (k[1]==GT_ORDER_N[1] && k[0]>=GT_ORDER_N[0]))))){
        uint64_t borrow=0;
        #pragma unroll
        for(int j=0;j<4;j++){
            __uint128_t t=(__uint128_t)k[j]-GT_ORDER_N[j]-borrow;
            k[j]=(uint64_t)t;borrow=(uint64_t)(t>>64)&1;
        }
    }
    if((k[0]|k[1]|k[2]|k[3])==0)return 1;
    if((k[0]==0xBFD25E8CD0364141ULL && k[1]==0xBAAEDCE6AF48A03BULL && k[2]==0xFFFFFFFFFFFFFFFEULL && k[3]==0x00000FFFFFFFFFFFULL) || (k[0]==0x0000000000000000ULL && k[1]==0x0000000000000000ULL && k[2]==0x0000000000000000ULL && k[3]==0xFFFFF00000000000ULL))return 2;
    return 0;
}
__device__ __noinline__ void wide_double(uint64_t *x,uint64_t *y) {
    uint64_t a[4],b[4],s[4],inv[5];
    wide_add(inv,y,y);inv[4]=0;_ModInv(inv);
    wide_square(a,x);wide_add(b,a,a);wide_add(a,a,b);wide_mul(s,a,inv);
    wide_square(a,s);wide_sub(a,a,x);wide_sub(a,a,x);
    wide_sub(b,x,a);wide_mul(b,s,b);wide_sub(y,b,y);Load256(x,a);
}
