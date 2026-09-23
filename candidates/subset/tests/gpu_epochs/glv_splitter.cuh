// Exact scalar decomposition for a shared regular odd-digit GLV table.
// Lattice constants and 384-bit rounding come from bitcoin-core/secp256k1,
// scalar_impl.h at 46db787112beabdb5e17e0dc35680716f1057e7b (MIT).
// Signed components use a 192-bit residue and a parity adjustment along the
// lattice basis. No field or group arithmetic is performed here.
#pragma once
#include <stdint.h>
#ifdef __CUDACC__
#define GLV_INLINE __host__ __device__ __forceinline__
#else
#define GLV_INLINE inline
#endif

template<int A,int B,int O>
GLV_INLINE void glv_mul_low(uint64_t (&out)[O],const uint64_t (&a)[A],const uint64_t (&b)[B]){
    #pragma unroll
    for(int i=0;i<O;i++)out[i]=0;
    #pragma unroll
    for(int i=0;i<A;i++){
        uint64_t carry=0;
        #pragma unroll
        for(int j=0;j<B && i+j<O;j++){
            __uint128_t t=(__uint128_t)a[i]*b[j]+out[i+j]+carry;
            out[i+j]=(uint64_t)t;carry=(uint64_t)(t>>64);
        }
        if(i+B<O)out[i+B]=carry;
    }
}

template<int N>
GLV_INLINE uint64_t glv_sub(uint64_t (&out)[N],const uint64_t (&a)[N],const uint64_t (&b)[N]){
    uint64_t borrow=0;
    #pragma unroll
    for(int i=0;i<N;i++){
        __uint128_t t=(__uint128_t)a[i]-b[i]-borrow;
        out[i]=(uint64_t)t;borrow=(uint64_t)(t>>64)&1;
    }
    return borrow;
}

GLV_INLINE void glv_add_masked(uint64_t (&x)[3],const uint64_t (&b)[3],uint64_t mask){
    uint64_t carry=0;
    #pragma unroll
    for(int i=0;i<3;i++){
        __uint128_t t=(__uint128_t)x[i]+(b[i]&mask)+carry;
        x[i]=(uint64_t)t;carry=(uint64_t)(t>>64);
    }
}

GLV_INLINE void glv_round384(uint64_t (&c)[2],const uint64_t (&k)[4],const uint64_t (&g)[4]){
    uint64_t p[8];glv_mul_low(p,k,g);
    __uint128_t lo=(__uint128_t)p[6]+(p[5]>>63);
    c[0]=(uint64_t)lo;c[1]=p[7]+(uint64_t)(lo>>64);
}

GLV_INLINE void glv_twice_mod_order(uint64_t (&k)[4],const uint64_t (&z)[4]){
    const uint64_t n[4]={0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,
                         0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
    uint64_t d[4],r[4];
    uint64_t mask=0-(1-glv_sub(d,z,n));
    #pragma unroll
    for(int i=0;i<4;i++)r[i]=(z[i]&~mask)|(d[i]&mask);
    const uint64_t high=r[3]>>63;
    #pragma unroll
    for(int i=3;i>0;i--)k[i]=(r[i]<<1)|(r[i-1]>>63);
    k[0]=r[0]<<1;
    mask=0-(high|(1-glv_sub(d,k,n)));
    #pragma unroll
    for(int i=0;i<4;i++)k[i]=(k[i]&~mask)|(d[i]&mask);
}

// Return two signed, two's-complement 192-bit odd components. The exact
// decomposition bounds are below 2^129, so low-192-bit products suffice:
// the residue has a unique signed representative in this range. We do not
// assume either output is nonnegative and do not use scalar-n field reduction.
GLV_INLINE void glv_split_odd(uint64_t (&u)[3],uint64_t (&v)[3],const uint64_t (&z)[4]){
    const uint64_t a1[2]={0xe86c90e49284eb15ULL,0x3086d221a7d46bcdULL};
    const uint64_t a2[3]={0x57c1108d9d44cfd8ULL,0x14ca50f7a8e2f3f6ULL,1};
    const uint64_t nb1[2]={0x6f547fa90abfe4c3ULL,0xe4437ed6010e8828ULL};
    const uint64_t g1[4]={0xe893209a45dbb031ULL,0x3daa8a1471e8ca7fULL,
                          0xe86c90e49284eb15ULL,0x3086d221a7d46bcdULL};
    const uint64_t g2[4]={0x1571b4ae8ac47f71ULL,0x221208ac9df506c6ULL,
                          0x6f547fa90abfe4c4ULL,0xe4437ed6010e8828ULL};
    uint64_t k[4],c1[2],c2[2],x[3],y[3];
    glv_twice_mod_order(k,z);
    glv_round384(c1,k,g1);glv_round384(c2,k,g2);
    glv_mul_low(x,c1,a1);glv_mul_low(y,c2,a2);
    uint64_t lowk[3]={k[0],k[1],k[2]};
    glv_sub(u,lowk,x);glv_sub(u,u,y);
    glv_mul_low(x,c1,nb1);glv_mul_low(y,c2,a1);
    glv_sub(v,x,y);
    const uint64_t va1[3]={a1[0],a1[1],0};
    // b1 is negative; these are its exact low 192 bits.
    const uint64_t vb1[3]={0x90ab8056f5401b3dULL,0x1bbc8129fef177d7ULL,~0ULL};
    uint64_t evenmask=0-(1-(u[0]&1));
    glv_add_masked(u,va1,evenmask);glv_add_masked(v,vb1,evenmask);
    evenmask=0-(1-(v[0]&1));
    glv_add_masked(u,a2,evenmask);glv_add_masked(v,va1,evenmask);
}

GLV_INLINE void glv_recode_odd(int32_t *digits,const uint64_t (&component)[3]){
    uint64_t m[3],mask=0-(component[2]>>63),carry=mask&1;
    const int sign=mask?-1:1;
    #pragma unroll
    for(int i=0;i<3;i++){
        __uint128_t t=(__uint128_t)(component[i]^mask)+carry;
        m[i]=(uint64_t)t;carry=(uint64_t)(t>>64);
    }
    #pragma unroll
    for(int i=0;i<7;i++){
        digits[i]=sign*((int32_t)(m[0]&0x1ffff)-0x10000);
        const uint64_t m0=(m[0]>>16)|(m[1]<<48);
        const uint64_t m1=(m[1]>>16)|(m[2]<<48);
        m[0]=m0|1;m[1]=m1;m[2]>>=16;
    }
    digits[7]=sign*(int32_t)m[0];
}

GLV_INLINE void glv_digits(int32_t *digits,const uint64_t (&z)[4]){
    uint64_t u[3],v[3];glv_split_odd(u,v,z);
    glv_recode_odd(digits,u);glv_recode_odd(digits+8,v);
}
