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

/* E = a - c^2 mod p (c as four little-endian 64-bit words, canonical), for the chord identity */
/* x2 - a = (sum - c)^2 - x1 + E of the finish recovery. */
static int qsb_make_chord_constant(uint64_t out[4],const uint8_t ax[32],
                                   const uint64_t cw[4]) {
    BN_CTX *ctx=BN_CTX_new();
    if(!ctx)return 0;
    BN_CTX_start(ctx);
    BIGNUM *p=BN_CTX_get(ctx),*a=BN_CTX_get(ctx),*c=BN_CTX_get(ctx),*e=BN_CTX_get(ctx);
    uint8_t encoded[32];
    for(int i=0;i<4;i++)for(int j=0;j<8;j++)encoded[8*i+j]=(uint8_t)(cw[i]>>(8*j));
    int ok=e && BN_set_word(p,1) && BN_lshift(p,p,256) &&
        BN_sub_word(p,0x1000003D1UL) && BN_lebin2bn(ax,32,a) &&
        BN_lebin2bn(encoded,32,c) && BN_mod_sqr(c,c,p,ctx) &&
        BN_mod_sub(e,a,c,p,ctx) && BN_bn2lebinpad(e,encoded,32)==32;
    if(ok){
        for(int i=0;i<4;i++){
            out[i]=0;
            for(int j=0;j<8;j++)out[i]|=(uint64_t)encoded[8*i+j]<<(8*j);
        }
    }
    BN_CTX_end(ctx); BN_CTX_free(ctx);
    return ok;
}
