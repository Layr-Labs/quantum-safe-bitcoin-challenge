#include <cstdint>
#include <cstring>
#include <cstdio>
#include <random>
#include <openssl/sha.h>
#define __host__
#define __device__
#define __constant__
#define __forceinline__ inline __attribute__((always_inline))
#define __umulhi(a,b) ((uint32_t)(((uint64_t)(a)*(uint64_t)(b))>>32))
#define ROR(x,n) (((uint32_t)(x)>>(n))|((uint32_t)(x)<<(32-(n))))
#define S0(x) (ROR(x,2)^ROR(x,13)^ROR(x,22))
#define S1(x) (ROR(x,6)^ROR(x,11)^ROR(x,25))
#define s0(x) (ROR(x,7)^ROR(x,18)^((uint32_t)(x)>>3))
#define s1(x) (ROR(x,17)^ROR(x,19)^((uint32_t)(x)>>10))
#define Ch(x,y,z) (((x)&(y))^(~(x)&(z)))
#define Maj(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
#define QSB_ZEROS_N 24
#define WMIX() { \
w[0] += s1(w[14]) + w[9] + s0(w[1]);\
w[1] += s1(w[15]) + w[10] + s0(w[2]);\
w[2] += s1(w[0]) + w[11] + s0(w[3]);\
w[3] += s1(w[1]) + w[12] + s0(w[4]);\
w[4] += s1(w[2]) + w[13] + s0(w[5]);\
w[5] += s1(w[3]) + w[14] + s0(w[6]);\
w[6] += s1(w[4]) + w[15] + s0(w[7]);\
w[7] += s1(w[5]) + w[0] + s0(w[8]);\
w[8] += s1(w[6]) + w[1] + s0(w[9]);\
w[9] += s1(w[7]) + w[2] + s0(w[10]);\
w[10] += s1(w[8]) + w[3] + s0(w[11]);\
w[11] += s1(w[9]) + w[4] + s0(w[12]);\
w[12] += s1(w[10]) + w[5] + s0(w[13]);\
w[13] += s1(w[11]) + w[6] + s0(w[14]);\
w[14] += s1(w[12]) + w[7] + s0(w[15]);\
w[15] += s1(w[13]) + w[8] + s0(w[0]);\
}
#include "../sha_gate_fma.cuh"
int main() {
    std::mt19937_64 rng(20260923);
    for(int t=0;t<10000;t++) {
        uint8_t in[32]; for(int i=0;i<32;i++) in[i]=(uint8_t)rng();
        uint32_t m[8],got[8]; for(int i=0;i<8;i++) m[i]=((uint32_t)in[4*i]<<24)|((uint32_t)in[4*i+1]<<16)|((uint32_t)in[4*i+2]<<8)|in[4*i+3];
        _SHA256TransformDigest32Q(got,m);
        uint8_t ref[32]; SHA256(in,32,ref);
        for(int i=0;i<8;i++) {
            uint32_t want=((uint32_t)ref[4*i]<<24)|((uint32_t)ref[4*i+1]<<16)|((uint32_t)ref[4*i+2]<<8)|ref[4*i+3];
            if(got[i]!=want) { std::fprintf(stderr,"mismatch t=%d word=%d got=%08x want=%08x\n",t,i,got[i],want); return 1; }
        }
    }
    std::puts("10000 SHA256 digest32 equivalence cases passed");
}
