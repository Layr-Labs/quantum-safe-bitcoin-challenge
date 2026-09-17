// Research-only asymmetric signed-window geometry. No production entry point change.
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
// Nine small windows cover156bits in48MiB.
// Four25-bit cold windows cover100bits in4GiB: matched storage to14-window9c.
constexpr int MIXED_CHUNKS=13;
__host__ __device__ __forceinline__ int mixed_bits(int c){return c<3?18:(c<9?17:25);}
__host__ __device__ __forceinline__ unsigned mixed_entries(int c){return 1u<<(mixed_bits(c)-1);}
__host__ __device__ __forceinline__ unsigned mixed_offset(int c){
    return c<3?((unsigned)c<<17):(c<9?((3u<<17)+((unsigned)(c-3)<<16)):((12u<<16)+((unsigned)(c-9)<<24)));
}
__host__ __device__ __forceinline__ int mixed_shift(int c){return c<3?18*c:(c<9?54+17*(c-3):156+25*(c-9));}
constexpr uint64_t MIXED_TOTAL_ENTRIES=(12ull<<16)+(4ull<<24);
static_assert(MIXED_TOTAL_ENTRIES*64==((48ull<<20)+(4ull<<30)),"thirteen matched table size");
// Input M is odd. e=(M mod 2^(b+1))-2^b; M'=(M-e)/2^b remains odd.
__host__ __device__ __forceinline__ int32_t mixed_step(uint64_t M[4],int sign,int bits){
    int32_t e=(int32_t)(M[0]&((1u<<(bits+1))-1))-(1<<bits);
    uint64_t r0=(M[0]>>(bits+1))|(M[1]<<(63-bits));
    uint64_t r1=(M[1]>>(bits+1))|(M[2]<<(63-bits));
    uint64_t r2=(M[2]>>(bits+1))|(M[3]<<(63-bits));
    uint64_t r3=M[3]>>(bits+1);
    M[0]=(r0<<1)|1;M[1]=(r1<<1)|(r0>>63);M[2]=(r2<<1)|(r1>>63);M[3]=(r3<<1)|(r2>>63);
    return sign*e;
}
