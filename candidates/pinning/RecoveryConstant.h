// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <openssl/bn.h>

// Input point coordinates are little-endian bytes from the runtime problem.
static int qsb_make_recovery_constant(uint64_t out[4],const uint8_t ax[32],
                                      const uint8_t by[32]) {
    BN_CTX *ctx=BN_CTX_new();
    if(!ctx)return 0;
    BN_CTX_start(ctx);
    BIGNUM *p=BN_CTX_get(ctx),*a=BN_CTX_get(ctx),*b=BN_CTX_get(ctx);
    BIGNUM *c=BN_CTX_get(ctx),*d=BN_CTX_get(ctx),*inv=BN_CTX_get(ctx);
    uint8_t encoded[32];
    int ok=inv && BN_set_word(p,1) && BN_lshift(p,p,256) &&
        BN_sub_word(p,0x1000003D1UL) && BN_lebin2bn(ax,32,a) &&
        BN_lebin2bn(by,32,b) && BN_mod_sqr(c,a,p,ctx) &&
        BN_mul_word(c,3) && BN_mod_lshift1(d,b,p,ctx) &&
        BN_mod_inverse(inv,d,p,ctx) && BN_mod_mul(c,c,inv,p,ctx) &&
        BN_bn2lebinpad(c,encoded,32)==32;
    if(ok){
        for(int i=0;i<4;i++){
            out[i]=0;
            for(int j=0;j<8;j++)out[i]|=(uint64_t)encoded[8*i+j]<<(8*j);
        }
    }
    BN_CTX_end(ctx); BN_CTX_free(ctx);
    return ok;
}

// Canonical h=c/2 and A=a-c^2/2, computed once using the same field modulus.
static int qsb_make_square_constants(uint64_t out_h[4],uint64_t out_A[4],
                                     const uint8_t ax[32],const uint64_t cw[4]) {
    BN_CTX *ctx=BN_CTX_new();
    if(!ctx)return 0;
    BN_CTX_start(ctx);
    BIGNUM *p=BN_CTX_get(ctx),*a=BN_CTX_get(ctx),*c=BN_CTX_get(ctx);
    BIGNUM *half=BN_CTX_get(ctx),*h=BN_CTX_get(ctx),*t=BN_CTX_get(ctx),*A=BN_CTX_get(ctx);
    uint8_t cb[32],hb[32],Ab[32];
    for(int i=0;i<4;i++)for(int j=0;j<8;j++)cb[8*i+j]=(uint8_t)(cw[i]>>(8*j));
    int ok=A && BN_set_word(p,1) && BN_lshift(p,p,256) &&
        BN_sub_word(p,0x1000003D1UL) && BN_lebin2bn(ax,32,a) &&
        BN_lebin2bn(cb,32,c) && BN_copy(half,p) && BN_add_word(half,1) &&
        BN_rshift1(half,half) && BN_mod_mul(h,c,half,p,ctx) &&
        BN_mod_sqr(t,c,p,ctx) && BN_mod_mul(t,t,half,p,ctx) &&
        BN_mod_sub(A,a,t,p,ctx) && BN_bn2lebinpad(h,hb,32)==32 &&
        BN_bn2lebinpad(A,Ab,32)==32;
    if(ok)for(int i=0;i<4;i++) {
        out_h[i]=out_A[i]=0;
        for(int j=0;j<8;j++) {
            out_h[i]|=(uint64_t)hb[8*i+j]<<(8*j);
            out_A[i]|=(uint64_t)Ab[8*i+j]<<(8*j);
        }
    }
    BN_CTX_end(ctx); BN_CTX_free(ctx);
    return ok;
}
