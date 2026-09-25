#include <vector>
#include <cstring>
#define QSB_SHORT_CARRY3 0
#define QSB_K2S_PARITY_WINDOW 0
#define main qsb_unused_main
#include "candidate.cu"
#undef main
#include "reference.cuh"

static void audit_cuda(cudaError_t e){
    if(e!=cudaSuccess){fprintf(stderr,"CUDA audit error: %s\n",cudaGetErrorString(e));exit(3);}
}
int main(){
    const unsigned sizes[]={1,2,3,4,5,7,8,9,15,16,17,31,32,33,65};
    BN_CTX *ctx=BN_CTX_new();BIGNUM *prime=nullptr;
    BN_hex2bn(&prime,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    BIGNUM *value=BN_new(),*inverse=BN_new();unsigned long checked=0,failures=0;
    for(unsigned epochs:sizes){
        const unsigned count=epochs*128;const size_t bytes=(size_t)4*count*sizeof(uint64_t);
        std::vector<uint64_t> input((size_t)4*count),got((size_t)4*count),reference((size_t)4*count);
        for(unsigned i=0;i<count;i++){
            uint64_t v[4];
            for(unsigned j=0;j<4;j++){
                uint64_t x=0x9e3779b97f4a7c15ULL*(i+1)^0xd1b54a32d192ed03ULL*(j+1);
                x^=x>>12;x^=x<<25;x^=x>>27;v[j]=x*0x2545f4914f6cdd1dULL;
            }
            v[3]&=0x7fffffffffffffffULL;
            if(i%17==0){v[0]=1;v[1]=v[2]=v[3]=0;}
            else if(i%19==0){v[0]=0xfffffffefffffc2eULL;v[1]=v[2]=v[3]=~0ULL;}
            else if(i%23==0){v[0]=0xfffffffefffffc2dULL;v[1]=v[2]=v[3]=~0ULL;}
            else if(i%29==0){v[0]=~0ULL;v[1]=v[2]=v[3]=0;}
            for(unsigned j=0;j<4;j++)input[(size_t)j*count+i]=v[j];
        }
        uint64_t *work=nullptr;audit_cuda(cudaMalloc(&work,bytes));
        audit_cuda(cudaMemcpy(work,input.data(),bytes,cudaMemcpyHostToDevice));
        kernel_split_inverse_ref<<<(epochs+3)/4,256>>>(work,epochs,count);
        audit_cuda(cudaGetLastError());audit_cuda(cudaDeviceSynchronize());
        audit_cuda(cudaMemcpy(reference.data(),work,bytes,cudaMemcpyDeviceToHost));
        audit_cuda(cudaMemcpy(work,input.data(),bytes,cudaMemcpyHostToDevice));
        kernel_split_inverse4<<<(epochs+7)/8,256>>>(work,epochs,count);
        audit_cuda(cudaGetLastError());audit_cuda(cudaDeviceSynchronize());
        audit_cuda(cudaMemcpy(got.data(),work,bytes,cudaMemcpyDeviceToHost));audit_cuda(cudaFree(work));
        for(unsigned i=0;i<count;i++){
            uint64_t v[4],expected[4];
            for(unsigned j=0;j<4;j++)v[j]=input[(size_t)j*count+i];
            BN_lebin2bn((const unsigned char*)v,32,value);
            if(!BN_mod_inverse(inverse,value,prime,ctx)||BN_bn2lebinpad(inverse,(unsigned char*)expected,32)!=32)return 4;
            bool bad=false;
            for(unsigned j=0;j<4;j++)bad|=got[(size_t)j*count+i]!=expected[j]||reference[(size_t)j*count+i]!=expected[j];
            if(bad){if(failures<8)fprintf(stderr,"mismatch epochs=%u index=%u\n",epochs,i);failures++;}
            checked++;
        }
        printf("epochs=%u count=%u cumulative_failures=%lu\n",epochs,count,failures);fflush(stdout);
    }
    BN_free(value);BN_free(inverse);BN_free(prime);BN_CTX_free(ctx);
    printf("canonical_factors_checked=%lu failures=%lu\n",checked,failures);
    return failures?1:0;
}
