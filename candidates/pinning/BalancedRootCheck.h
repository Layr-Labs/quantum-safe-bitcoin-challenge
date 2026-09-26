// SPDX-License-Identifier: GPL-3.0-only
// Official-runtime correctness check only; no timing/calibration or local execution.
#pragma once
static bool qsb_balanced_startup_check(uint64_t *device, cudaStream_t stream) {
    constexpr size_t words=2048u*4u;
    uint64_t *raw=(uint64_t*)calloc(words,sizeof(uint64_t));
    uint64_t *got=(uint64_t*)malloc(words*sizeof(uint64_t));
    BN_CTX *ctx=BN_CTX_new();
    BIGNUM *p=BN_new(),*a=BN_new(),*inv=BN_new(),*weighted=BN_new(),
        *scale=BN_new(),*weight=BN_new(),*observed=BN_new();
    bool ok=raw&&got&&ctx&&p&&a&&inv&&weighted&&scale&&weight&&observed;
    uint64_t scale_words[4]={1,0,0,0},weight_words[4]={0,0,0,0};
    cudaError_t error=cudaSuccess;
    if(ok){
#if QSB_ISO_XR
        error=cudaMemcpyFromSymbol(scale_words,pin_iso_invu_words,32);
        if(error==cudaSuccess)error=cudaMemcpyFromSymbol(weight_words,pin_iso_u2ry_words,32);
#else
        error=cudaMemcpyFromSymbol(weight_words,pin_u2ry_words,32);
#endif
        const uint64_t prime[4]={0xfffffffefffffc2fULL,~0ULL,~0ULL,~0ULL};
        ok=BN_lebin2bn((const unsigned char*)prime,32,p)!=nullptr &&
           BN_lebin2bn((const unsigned char*)scale_words,32,scale)!=nullptr &&
           BN_lebin2bn((const unsigned char*)weight_words,32,weight)!=nullptr;
    }
    const int counts[]={1,127,128,129,511,512,513,557,767,768,769,1023,1024};
    for(int count:counts){
        if(!ok||error!=cudaSuccess)break;
        memset(raw,0,words*sizeof(uint64_t));
        for(int i=0;i<count;++i){
            for(int k=0;k<4;++k){
                uint64_t v=0x9e3779b97f4a7c15ULL*(uint64_t)(1+i*4+k);
                v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;
                raw[(size_t)i*4+k]=v^(v>>27);
            }
            if(i%31==0)memset(raw+(size_t)i*4,0,32);
            if(i%31==1||i%31==2){
                raw[(size_t)i*4]=0xfffffffefffffc2fULL-(i%31==2);
                for(int k=1;k<4;++k)raw[(size_t)i*4+k]=~0ULL;
            }
            if(i%31==3)for(int k=0;k<4;++k)raw[(size_t)i*4+k]=~0ULL;
            if(i%31==4){memset(raw+(size_t)i*4,0,32);raw[(size_t)i*4]=1;}
        }
        error=cudaMemcpyAsync(device,raw,words*sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
        if(error==cudaSuccess){
            qsb_root_balanced_warp<<<1,128,0,stream>>>(device,count);
            error=cudaGetLastError();
        }
        if(error==cudaSuccess)error=cudaMemcpyAsync(got,device,words*sizeof(uint64_t),cudaMemcpyDeviceToHost,stream);
        if(error==cudaSuccess)error=cudaStreamSynchronize(stream);
        if(error!=cudaSuccess)break;
        for(int i=0;i<count&&ok;++i){
            ok=BN_lebin2bn((const unsigned char*)(raw+(size_t)i*4),32,a)!=nullptr &&
               BN_nnmod(a,a,p,ctx);
            if(!ok)break;
            if(BN_is_zero(a))BN_zero(inv);
            else ok=BN_mod_inverse(inv,a,p,ctx)!=nullptr && BN_mod_mul(inv,inv,scale,p,ctx);
            ok=ok&&BN_mod_mul(weighted,inv,weight,p,ctx);
            ok=ok&&BN_lebin2bn((const unsigned char*)(got+(size_t)i*4),32,observed)!=nullptr;
            ok=ok&&BN_cmp(observed,inv)==0;
            ok=ok&&BN_lebin2bn((const unsigned char*)(got+((size_t)count+i)*4),32,observed)!=nullptr;
            ok=ok&&BN_nnmod(observed,observed,p,ctx)&&BN_cmp(observed,weighted)==0;
        }
    }
    BN_free(p);BN_free(a);BN_free(inv);BN_free(weighted);BN_free(scale);BN_free(weight);BN_free(observed);BN_CTX_free(ctx);
    free(raw);free(got);
    // An asynchronous CUDA fault can poison the context. Never silently retry
    // work under the baseline after such a fault. Arithmetic/host-allocation
    // mismatch before search can safely retain the promoted root implementation.
    if(error!=cudaSuccess)qsb_subpipe_die("balanced root startup check",error);
    return ok;
}
