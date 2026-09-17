#pragma once
#include "host_warp.hpp"
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#define HM43_HOST_ORACLE 1
#ifndef QSB_HM43_NO_FIELD_STUBS
// Semantic replacements only for CUDA scalar intrinsics/extended carry ops.
// AddP/SubP and constants in field_projection.hpp are extracted unchanged.
inline thread_local uint64_t qsb_host_cc=0;
inline uint64_t qsb_host_add(uint64_t a,uint64_t b,bool consume,bool update){
 __uint128_t t=(__uint128_t)a+b+(consume?qsb_host_cc:0);if(update)qsb_host_cc=(uint64_t)(t>>64);return (uint64_t)t;
}
inline uint64_t qsb_host_sub(uint64_t a,uint64_t b,bool consume,bool update){
 __uint128_t t=(__uint128_t)b+(consume?qsb_host_cc:0);uint64_t r=(uint64_t)((__uint128_t)a-t);if(update)qsb_host_cc=(__uint128_t)a<t;return r;
}
#define UADDO1(r,a) r=qsb_host_add(r,a,false,true)
#define UADDC1(r,a) r=qsb_host_add(r,a,true,true)
#define UADD1(r,a) r=qsb_host_add(r,a,true,false)
#define USUBO1(r,a) r=qsb_host_sub(r,a,false,true)
#define USUBC1(r,a) r=qsb_host_sub(r,a,true,true)
#define USUB1(r,a) r=qsb_host_sub(r,a,true,false)
inline uint32_t _CTZ(uint64_t x){return x?__builtin_ctzll(x):64;}
inline int __clzll(uint64_t x){return x?__builtin_clzll(x):64;}
#include "field_projection.hpp"
#endif
