// SPDX-License-Identifier: GPL-3.0-only
// Official-runtime correctness check only; no timing/calibration or local execution.
#pragma once
// Direct helper check is startup correctness work, never timed or used to tune.
__global__ void qsb_prefix_field_check_kernel(uint32_t *data){
    const unsigned tid=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned field=tid>>3,d=tid&7u,lane=threadIdx.x&31u;
    const uint32_t a=data[field*8+d],b=data[(256u+field)*8+d];
    data[(512u+field)*8+d]=qsb_prefix_cyclic_research::multiply8(a,b,lane);
}
static bool qsb_register_startup_check(uint64_t *device, cudaStream_t stream) {
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
    // First 64 products are the complete 8x8 edge Cartesian product; remaining
    // 192 use deterministic full-width values. Compare every result to OpenSSL.
    if(ok&&error==cudaSuccess){
        const uint64_t edges[8][4]={
            {0,0,0,0},{1,0,0,0},
            {0xfffffffefffffc2fULL,~0ULL,~0ULL,~0ULL},
            {0xfffffffefffffc2eULL,~0ULL,~0ULL,~0ULL},
            {0xfffffffefffffc30ULL,~0ULL,~0ULL,~0ULL},
            {~0ULL,~0ULL,~0ULL,~0ULL},{~0ULL,0,0,0},
            {0xaaaaaaaaaaaaaaaaULL,0x5555555555555555ULL,
             0xaaaaaaaaaaaaaaaaULL,0x5555555555555555ULL}};
        memset(raw,0,words*sizeof(uint64_t));
        for(unsigned f=0;f<256;++f)for(unsigned side=0;side<2;++side)
            for(unsigned k=0;k<4;++k){
                uint64_t v=0x9e3779b97f4a7c15ULL*(1+f*8+side*4+k);
                v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v^=v>>27;
                raw[(side*256u+f)*4+k]=f<64?edges[side?f%8:f/8][k]:v;
            }
        error=cudaMemcpyAsync(device,raw,words*sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
        if(error==cudaSuccess){
            qsb_prefix_field_check_kernel<<<16,128,0,stream>>>((uint32_t*)device);
            error=cudaGetLastError();
        }
        if(error==cudaSuccess)error=cudaMemcpyAsync(got,device,words*sizeof(uint64_t),cudaMemcpyDeviceToHost,stream);
        if(error==cudaSuccess)error=cudaStreamSynchronize(stream);
        for(unsigned f=0;f<256&&ok&&error==cudaSuccess;++f){
            ok=BN_lebin2bn((const unsigned char*)(raw+f*4),32,a)!=nullptr &&
               BN_lebin2bn((const unsigned char*)(raw+(256u+f)*4),32,inv)!=nullptr &&
               BN_mod_mul(weighted,a,inv,p,ctx) &&
               BN_lebin2bn((const unsigned char*)(got+(512u+f)*4),32,observed)!=nullptr &&
               BN_nnmod(observed,observed,p,ctx) && BN_cmp(observed,weighted)==0;
        }
    }
    const int counts[]={1,31,32,33,63,64,65,95,96,97,127,128,129,255,256,257,511,512,513,557,767,768,769,1023,1024};
    for(int case_id=0;case_id<(int)(sizeof(counts)/sizeof(counts[0]))+4;++case_id){
        const int mixed=(int)(sizeof(counts)/sizeof(counts[0]));
        const int count=case_id<mixed?counts[case_id]:(case_id&1?1024:1);
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
        if(case_id>=mixed){
            for(int i=0;i<count;++i){
                memset(raw+(size_t)i*4,0,32);
                if(case_id>=mixed+2){raw[(size_t)i*4]=0xfffffffefffffc2fULL;
                    for(int k=1;k<4;++k)raw[(size_t)i*4+k]=~0ULL;}
            }
        }
        error=cudaMemcpyAsync(device,raw,words*sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
        if(error==cudaSuccess){
            qsb_root_register<<<1,128,0,stream>>>(device,count);
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
    if(error!=cudaSuccess)qsb_subpipe_die("register root startup check",error);
    return ok;
}
