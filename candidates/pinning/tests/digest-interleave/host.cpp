// SPDX-License-Identifier: GPL-3.0-only
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <openssl/sha.h>
#define __host__
#define __device__
#define __constant__
#define __forceinline__ inline
#define QSB_ZEROS_N 24
#define QSB_SHA_FMA_ADD 0
#define QSB_FIN_KW_IMAD 0
#define ROR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define S0(x) (ROR(x,2)^ROR(x,13)^ROR(x,22))
#define S1(x) (ROR(x,6)^ROR(x,11)^ROR(x,25))
#define s0(x) (ROR(x,7)^ROR(x,18)^((x)>>3))
#define s1(x) (ROR(x,17)^ROR(x,19)^((x)>>10))
#define Maj(x,y,z) (((x)&(y))|((z)&((x)|(y))))
#define Ch(x,y,z) ((z)^((x)&((y)^(z))))
#define WMIX() do { for(int j=0;j<16;j++) w[j]+=s1(w[(j+14)&15])+w[(j+9)&15]+s0(w[(j+1)&15]); } while(0)
static inline uint32_t __umulhi(uint32_t a,uint32_t b) {return uint32_t((uint64_t(a)*b)>>32);}
#include "../../sha_pinsha.cuh"
int main() {
    uint64_t rng=0x202609296ea5ULL;
    for(int n=0;n<100000;n++) {
        uint32_t m[8],out[8]; unsigned char bytes[32],expected[32],actual[32];
        for(int i=0;i<8;i++) {
            rng^=rng<<13; rng^=rng>>7; rng^=rng<<17;
            m[i]=n==0?0:n==1?0xffffffffu:uint32_t(rng);
            for(int j=0;j<4;j++) bytes[4*i+j]=m[i]>>(24-8*j);
        }
        _SHA256TransformDigest32Q(out,m);
        for(int i=0;i<8;i++) for(int j=0;j<4;j++) actual[4*i+j]=out[i]>>(24-8*j);
        SHA256(bytes,32,expected);
        if(memcmp(expected,actual,32)) {printf("Mismatch at %d\n",n); return 1;}
    }
    puts("100000 full SHA256 digests match OpenSSL, including zero and all-one input.");
}
