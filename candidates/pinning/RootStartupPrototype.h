#pragma once
// Startup arithmetic verification for the independently loaded root library. Caller must first
// run the unchanged primary startup checks and provide a drained 32768-byte workspace.
#include "RootSecondaryPrototype.h"
#include <openssl/bn.h>
#include <stdlib.h>
#include <string.h>
namespace qsb_root_secondary_research {
enum class Route { Fatal, Original, HalfFused, HalfRegister };
struct Probe { bool arithmetic_ok; cudaError_t cuda_error; };
static cudaError_t read32(State &s, const char *name, uint64_t out[4]) {
    void *address=nullptr; size_t bytes=0;
    cudaError_t e=cudaLibraryGetGlobal(&address,&bytes,s.library,name);
    if(e!=cudaSuccess) return e;
    if(bytes!=32) return cudaErrorInvalidValue;
    return cudaMemcpy(out,address,32,cudaMemcpyDeviceToHost);
}
static Probe probe(State &s,bool use_register,uint64_t *device,size_t capacity,
                   cudaStream_t stream,const uint64_t expected_scale[4],
                   const uint64_t expected_weight[4]) {
    if(!s.constants_ready || !device || capacity<32768 ||
       !expected_scale || !expected_weight) return {false,cudaSuccess};
    uint64_t scale_words[4],weight_words[4];
    cudaError_t e=read32(s,"pin_iso_invu_words",scale_words);
    if(e==cudaSuccess)e=read32(s,"pin_iso_u2ry_words",weight_words);
    if(e!=cudaSuccess)return {false,e};
    if(memcmp(scale_words,expected_scale,32)||memcmp(weight_words,expected_weight,32))
        return {false,cudaSuccess};
    constexpr size_t words=1024*4;
    uint64_t *raw=(uint64_t*)calloc(words,sizeof(uint64_t));
    uint64_t *got=(uint64_t*)malloc(words*sizeof(uint64_t));
    BN_CTX *ctx=BN_CTX_new();
    BIGNUM *p=BN_new(),*a=BN_new(),*inv=BN_new(),*weighted=BN_new(),
        *scale=BN_new(),*weight=BN_new(),*observed=BN_new();
    bool ok=raw&&got&&ctx&&p&&a&&inv&&weighted&&scale&&weight&&observed;
    const uint64_t prime[4]={0xfffffffefffffc2fULL,~0ULL,~0ULL,~0ULL};
    if(ok)ok=BN_lebin2bn((const unsigned char*)prime,32,p) &&
             BN_lebin2bn((const unsigned char*)scale_words,32,scale) &&
             BN_lebin2bn((const unsigned char*)weight_words,32,weight);
    const int counts[]={1,31,32,33,63,64,65,95,96,97,127,128,129,255,256,257,383,384,385,511,512};
    constexpr int mixed=sizeof(counts)/sizeof(counts[0]);
    for(int case_id=0;case_id<mixed+4&&ok&&e==cudaSuccess;case_id++) {
        const int count=case_id<mixed?counts[case_id]:(case_id&1?512:1);
        memset(raw,0,words*sizeof(uint64_t));
        for(int i=0;i<count;i++) {
            for(int k=0;k<4;k++) {
                uint64_t v=0x9e3779b97f4a7c15ULL*(uint64_t)(1+i*4+k);
                v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;
                raw[(size_t)i*4+k]=v^(v>>27);
            }
            if(i%31==0)memset(raw+(size_t)i*4,0,32);
            if(i%31==1||i%31==2) {
                raw[(size_t)i*4]=0xfffffffefffffc2fULL-(i%31==2);
                for(int k=1;k<4;k++)raw[(size_t)i*4+k]=~0ULL;
            }
            if(i%31==3)for(int k=0;k<4;k++)raw[(size_t)i*4+k]=~0ULL;
            if(i%31==4){memset(raw+(size_t)i*4,0,32);raw[(size_t)i*4]=1;}
        }
        if(case_id>=mixed)for(int i=0;i<count;i++) {
            memset(raw+(size_t)i*4,0,32);
            if(case_id>=mixed+2){raw[(size_t)i*4]=prime[0];
                for(int k=1;k<4;k++)raw[(size_t)i*4+k]=~0ULL;}
        }
        e=cudaMemcpyAsync(device,raw,words*sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
        if(e==cudaSuccess)e=launch(s,use_register,device,count,stream);
        if(e==cudaSuccess)e=cudaMemcpyAsync(got,device,words*sizeof(uint64_t),cudaMemcpyDeviceToHost,stream);
        if(e==cudaSuccess)e=cudaStreamSynchronize(stream);
        for(int i=0;i<count&&ok&&e==cudaSuccess;i++) {
            ok=BN_lebin2bn((const unsigned char*)(raw+(size_t)i*4),32,a)&&BN_nnmod(a,a,p,ctx);
            if(!ok)break;
            if(BN_is_zero(a))BN_zero(inv);
            else ok=BN_mod_inverse(inv,a,p,ctx)&&BN_mod_mul(inv,inv,scale,p,ctx);
            ok=ok&&BN_mod_mul(weighted,inv,weight,p,ctx)&&
               BN_lebin2bn((const unsigned char*)(got+(size_t)i*4),32,observed)&&BN_cmp(observed,inv)==0&&
               BN_lebin2bn((const unsigned char*)(got+((size_t)count+i)*4),32,observed)&&
               BN_nnmod(observed,observed,p,ctx)&&BN_cmp(observed,weighted)==0;
        }
    }
    // Any queued transfer must drain before pageable host buffers and BN owners are released.
    cudaError_t drain=cudaStreamSynchronize(stream);
    if(e==cudaSuccess)e=drain;
    BN_free(p);BN_free(a);BN_free(inv);BN_free(weighted);BN_free(scale);BN_free(weight);BN_free(observed);BN_CTX_free(ctx);
    free(raw);free(got);
    return {ok,e};
}
static Route startup(State &s,uint64_t *workspace,size_t capacity,cudaStream_t stream,
                      const uint64_t scale[4],const uint64_t weight[4]) {
    // Select before geometry allocation, graph capture or any search launch.
    if(!s.library || !s.constants_ready)return Route::Original;
    Probe reg=probe(s,true,workspace,capacity,stream,scale,weight);
    if(reg.cuda_error!=cudaSuccess){s.launch_failed=true;return Route::Fatal;}
    // Validate fallback independently, even when register roots pass.
    Probe fused=probe(s,false,workspace,capacity,stream,scale,weight);
    if(fused.cuda_error!=cudaSuccess){s.launch_failed=true;return Route::Fatal;}
    if(!fused.arithmetic_ok)return Route::Original;
    return reg.arithmetic_ok?Route::HalfRegister:Route::HalfFused;
}
} // namespace qsb_root_secondary_research
