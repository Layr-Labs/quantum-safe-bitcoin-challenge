/* GPL-3.0-only. Paired ownership of VanitySearch's delayed-divstep inverse.
 * Synthetic public field values only. Exactly lanes 0 and 1 participate. */
#pragma once
#include "hm39_divstep.cuh"
#ifndef HM39_PAIR_ROOT
#define HM39_PAIR_ROOT 1
#endif
#ifdef HM39_HOST_ORACLE
uint64_t hm39_exchange(uint64_t value,int source);
#else
__device__ __forceinline__ uint64_t hm39_exchange(uint64_t value,int source){
    return (uint64_t)__shfl_sync(3u,(unsigned long long)value,source);
}
#endif

__device__ __forceinline__ void hm39_pair_inverse(uint64_t result[5],int lane){
    uint64_t x[5],residue[5]={0,0,0,0,0};
    uint64_t root[5];
    #pragma unroll
    for(int j=0;j<5;j++)root[j]=hm39_exchange(result[j],0);
    if(lane==0){
        x[0]=0xFFFFFFFEFFFFFC2FULL;x[1]=x[2]=x[3]=~0ULL;x[4]=0;
    }else{Load(x,root);residue[0]=1;}
    int32_t pos=4;
    while(true){
        uint64_t u[5],v[5],partner[5];
        #pragma unroll
        for(int j=0;j<5;j++)partner[j]=hm39_exchange(x[j],lane^1);
        #pragma unroll
        for(int j=0;j<5;j++){u[j]=lane?partner[j]:x[j];v[j]=lane?x[j]:partner[j];}
        int64_t uu=0,uv=0,vu=0,vv=0;
        /* Both lanes have identical u/v/pos. Replicating this scalar decision
         * adds no warp instruction stream and avoids four 64-bit broadcasts. */
        hm39_divstep62(u,v,&pos,&uu,&uv,&vu,&vv);
        int64_t a=lane?vu:uu,b=lane?vv:uv;uint64_t carry;
        _MatrixVecMulHalf(x,u,v,a,b,&carry); // low 320 bits, same as control
        if(_IsNegative(x)){Neg(x);a=-a;b=-b;}
        _ShiftR62(x);
        bool finished=hm39_exchange((uint64_t)_IsZero(x),1)!=0;
        #pragma unroll
        for(int j=0;j<5;j++)partner[j]=hm39_exchange(residue[j],lane^1);
        #pragma unroll
        for(int j=0;j<5;j++){u[j]=lane?partner[j]:residue[j];v[j]=lane?residue[j]:partner[j];}
        uint64_t t[5],correction[5];
        _MatrixVecMulHalf(t,u,v,a,b,&carry);
        _MulP(correction,(t[0]*MM64)&MSK62);carry=_AddCh(t,correction,carry);
        _ShiftR62(residue,t,carry);
        if(finished)break;
    }
    if(lane==0){
        if(!_IsOne(x)){for(int j=0;j<5;j++)residue[j]=0;}
        else{
            while(_IsNegative(residue))AddP(residue);
            while(!_IsNegative(residue))SubP(residue);
            AddP(residue);
        }
    }
    #pragma unroll
    for(int j=0;j<5;j++)result[j]=hm39_exchange(residue[j],0);
}
