#include "stubs.h"
#include <cstdio>
#include <cstring>
#ifndef HEADER
#define HEADER "../../CpuGrindSubset.h"
#endif
#include HEADER
using namespace qcpu;
static unsigned lane_cases, chain_cases, fermat_comparisons, oracle_comparisons, zero_comparisons, failures;
Q8T static fe8 load(const fe a[8]) {
    alignas(64) uint64_t w[4][8];
    for(int k=0;k<4;++k)for(int i=0;i<8;++i)w[k][i]=a[i].v[k];
    fe8 x;fe8_from64(x,_mm512_load_si512(w[0]),_mm512_load_si512(w[1]),_mm512_load_si512(w[2]),_mm512_load_si512(w[3]));return x;
}
static fe input(int profile,int lane) {
    if(profile==0)return fe{{1,0,0,0}};
    if(profile==1)return fe{{(uint64_t)(lane+1),0,0,0}};
    if(profile==2)return fe{{0xfffffffefffffc2eULL-(unsigned)lane,~0ULL,~0ULL,~0ULL}};
    uint64_t x=0x89178bba701ULL+(unsigned)lane;fe a;
    for(int i=0;i<4;++i){x^=x<<13;x^=x>>7;x^=x<<17;a.v[i]=x;}
    a.v[3]&=0x7fffffffffffffffULL;return a;
}
static fe oracle(const fe &a,BIGNUM *p,BN_CTX *ctx) {
    BIGNUM *v=BN_lebin2bn((const unsigned char*)a.v,32,nullptr),*inv=BN_mod_inverse(nullptr,v,p,ctx);
    if(!inv){fprintf(stderr,"Unexpected OpenSSL nonzero inverse failure\n");abort();}
    fe out;BN_bn2lebinpad(inv,(unsigned char*)out.v,32);BN_free(inv);BN_free(v);return out;
}
Q8T static void lane_case(const fe src[8],const fe expected[8],unsigned mask) {
    fe in[8];for(int i=0;i<8;++i)in[i]=(mask&(1u<<i))?fe{{0,0,0,0}}:src[i];
    fe8 guarded=load(in),fermat=guarded;fe8_inv_lanes(guarded);fe8_inv1(fermat);
    fe a[8],b[8];fe8_store_canon(a,guarded);fe8_store_canon(b,fermat);++lane_cases;
    for(int i=0;i<8;++i){++fermat_comparisons;failures+=!fe_eq(a[i],b[i]);
        if(mask&(1u<<i)){++zero_comparisons;failures+=!fe_is_zero(a[i]);}
        else {++oracle_comparisons;failures+=!fe_eq(a[i],expected[i]);}}
}
Q8T static void chain_case(const fe src[4][8],const fe expected[4][8],unsigned mask,bool all_chains) {
    fe8 x[4];for(int c=0;c<4;++c){fe in[8];for(int i=0;i<8;++i)
        in[i]=((mask&(1u<<i))&&(all_chains||c==(i%4)))?fe{{0,0,0,0}}:src[c][i];x[c]=load(in);}
    fe8_inv4(x);++chain_cases;
    for(int c=0;c<4;++c){fe out[8];fe8_store_canon(out,x[c]);for(int i=0;i<8;++i)
        if(mask&(1u<<i)){++zero_comparisons;failures+=!fe_is_zero(out[i]);}
        else {++oracle_comparisons;failures+=!fe_eq(out[i],expected[c][i]);}}
}
Q8T static int run() {
    BIGNUM *p=nullptr;BN_hex2bn(&p,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");BN_CTX *ctx=BN_CTX_new();
    fe src[4][8],expected[4][8];
    for(int k=0;k<4;++k){for(int i=0;i<8;++i){src[k][i]=input(k,i);expected[k][i]=oracle(src[k][i],p,ctx);}
        for(unsigned mask=0;mask<256;++mask)lane_case(src[k],expected[k],mask);}
    // In fe8_inv4, a zero in one chain still zeroes all four chains at that lane.
    // Other lane indices must retain all four independent OpenSSL inverses.
    for(unsigned mask=0;mask<256;++mask)chain_case(src,expected,mask,false);
    chain_case(src,expected,255,true);
    BN_free(p);BN_CTX_free(ctx);
    printf("{\"lane_cases\":%u,\"four_chain_cases\":%u,\"fermat_lane_comparisons\":%u,\"openssl_nonzero_comparisons\":%u,\"zero_convention_comparisons\":%u,\"failures\":%u}\n",lane_cases,chain_cases,fermat_comparisons,oracle_comparisons,zero_comparisons,failures);
    return failures?1:0;
}
int main(){__builtin_cpu_init();if(!__builtin_cpu_supports("avx512ifma")){fprintf(stderr,"AVX-512 IFMA required\n");return 77;}return run();}
