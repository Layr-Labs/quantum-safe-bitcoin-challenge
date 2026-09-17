/* GPL-3.0-only. Four-owner form of VanitySearch's delayed-divstep inverse.
 * Original arithmetic and decision: Jean Luc Pons, see GPUMath.h.
 * Exactly lanes 0..3 participate; they own U,V,R,S, respectively. */
#pragma once
#include "hm39_divstep.cuh"
#ifndef HM41_QUAD_ROOT
#define HM41_QUAD_ROOT 1
#endif
#ifdef HM41_HOST_ORACLE
uint64_t hm41_exchange(uint64_t value,int source);
#else
__device__ __forceinline__ uint64_t hm41_exchange(uint64_t value,int source){
    return (uint64_t)__shfl_sync(15u,(unsigned long long)value,source);
}
#endif

// Negation of the full signed product, not only its low five words. Equivalent
// to negating both matrix coefficients before _MatrixVecMulHalf, modulo 2^384.
__device__ __forceinline__ void hm41_negate_product(uint64_t t[5],uint64_t *carry){
    Neg(t);
    *carry=~*carry+(uint64_t)_IsZero(t);
}

__device__ __forceinline__ void hm41_quad_inverse(uint64_t result[5],int lane){
    uint64_t state[5]={0,0,0,0,0};
    #pragma unroll
    for(int j=0;j<5;j++){
        const uint64_t root=hm41_exchange(result[j],0);
        if(lane==1)state[j]=root;
    }
    if(lane==0){state[0]=0xFFFFFFFEFFFFFC2FULL;state[1]=state[2]=state[3]=~0ULL;}
    if(lane==3)state[0]=1;
    int32_t pos=4;
    while(true){
        uint64_t left[5],right[5];
        #pragma unroll
        for(int j=0;j<5;j++){
            uint64_t partner=hm41_exchange(state[j],lane^1);
            left[j]=(lane&1)?partner:state[j];
            right[j]=(lane&1)?state[j]:partner;
        }
        // U and V owners have identical decision inputs. They calculate the
        // same matrix in one warp instruction stream, then each broadcasts
        // only its own row to the corresponding R or S owner: two exchanges
        // instead of four coefficients (or four normalized decision inputs).
        int64_t uu=0,uv=0,vu=0,vv=0;
        if(lane<2)hm39_divstep62(left,right,&pos,&uu,&uv,&vu,&vv);
        const int64_t a=(int64_t)hm41_exchange((uint64_t)((lane&1)?vu:uu),lane&1);
        const int64_t b=(int64_t)hm41_exchange((uint64_t)((lane&1)?vv:uv),lane&1);
        uint64_t t[5],carry;
        _MatrixVecMulHalf(t,left,right,a,b,&carry);
        const bool negative=hm41_exchange((uint64_t)_IsNegative(t),lane&1)!=0;
        if(lane<2){
            if(negative)Neg(t);
            _ShiftR62(t);Load(state,t);
        }else{
            if(negative)hm41_negate_product(t,&carry);
            uint64_t correction[5];
            _MulP(correction,(t[0]*MM64)&MSK62);
            carry=_AddCh(t,correction,carry);
            _ShiftR62(state,t,carry);
        }
        if(hm41_exchange((uint64_t)_IsZero(state),1))break;
    }
    const bool invertible=hm41_exchange((uint64_t)_IsOne(state),0)!=0;
    if(lane==2){
        if(!invertible){for(int j=0;j<5;j++)state[j]=0;}
        else{
            while(_IsNegative(state))AddP(state);
            while(!_IsNegative(state))SubP(state);
            AddP(state);
        }
    }
    #pragma unroll
    for(int j=0;j<5;j++)result[j]=hm41_exchange(state[j],2);
}
