// Research-only mixed signed-window geometry. No production entry point change.
// Recoder identity derives from PR24; see ../pr24_reference/PROVENANCE.json.
#pragma once
#include <stdint.h>
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
constexpr int WIDE_CHUNKS=10;
__host__ __device__ __forceinline__ int wide_bits(int c){return c<6?26:25;}
__host__ __device__ __forceinline__ unsigned wide_entries(int c){return 1u<<(wide_bits(c)-1);}
__host__ __device__ __forceinline__ unsigned wide_offset(int c){return c<6?(unsigned)c<<25:(6u<<25)+((unsigned)(c-6)<<24);}
__host__ __device__ __forceinline__ int wide_shift(int c){return c<6?26*c:156+25*(c-6);}
constexpr uint64_t WIDE_TOTAL_ENTRIES=(6ull<<25)+(4ull<<24);
static_assert(WIDE_TOTAL_ENTRIES*64==(16ull<<30),"ten-window table must occupy16GiB");
// Input M is odd. e=(M mod 2^(b+1))-2^b; M'=(M-e)/2^b remains odd.
__host__ __device__ __forceinline__ int32_t wide_step(uint64_t M[4],int sign,int bits){
    int32_t e=(int32_t)(M[0]&((1u<<(bits+1))-1))-(1<<bits);
    uint64_t r0=(M[0]>>(bits+1))|(M[1]<<(63-bits));
    uint64_t r1=(M[1]>>(bits+1))|(M[2]<<(63-bits));
    uint64_t r2=(M[2]>>(bits+1))|(M[3]<<(63-bits));
    uint64_t r3=M[3]>>(bits+1);
    M[0]=(r0<<1)|1;M[1]=(r1<<1)|(r0>>63);M[2]=(r2<<1)|(r1>>63);M[3]=(r3<<1)|(r2>>63);
    return sign*e;
}
