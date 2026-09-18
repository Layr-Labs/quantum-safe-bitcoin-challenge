// Audit the actual PR212 producer against OpenSSL's independent compression.
#define main qsb_production_main
#include "tree.cu"
#undef main
#include <openssl/sha.h>
#include <vector>

static uint64_t audit_rng=0x31219af731cc97e1ULL;
static uint32_t audit_word() {
    audit_rng^=audit_rng<<13; audit_rng^=audit_rng>>7; audit_rng^=audit_rng<<17;
    return (uint32_t)(audit_rng>>16);
}
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    fprintf(stderr,"CUDA failure at %d: %s\n",__LINE__,cudaGetErrorString(e));return 2; } } while(0)

int main() {
    constexpr int epochs=67;
    constexpr uint32_t untouched=0xa5a5a5a5U;
    std::vector<epoch_desc_t> input(epochs);
    std::vector<uint32_t> actual((size_t)epochs*QSB_FIRST_SLOTS*8);
    epoch_desc_t *device_epochs=nullptr;
    uint32_t *device_first=nullptr;
    CUDA_CHECK(cudaMalloc(&device_epochs,input.size()*sizeof(epoch_desc_t)));
    CUDA_CHECK(cudaMalloc(&device_first,actual.size()*sizeof(uint32_t)));
    uint64_t states=0,words=0,padding=0,errors=0;
    const int counts[]={1,31,32,33,54,63,64};
    for(int fixture=0;fixture<2;fixture++) {
        uint32_t table[14][256];
        for(int j=0;j<14;j++)for(int c=0;c<256;c++)table[j][c]=audit_word();
        for(auto &ep:input) {
            for(auto &v:ep.mid)v=audit_word();
            ep.remW[0]=audit_word();ep.remW[1]=audit_word();
        }
        CUDA_CHECK(cudaMemcpyToSymbol(QSB_FIRST_UNIQUE,table,sizeof(table)));
        CUDA_CHECK(cudaMemcpy(device_epochs,input.data(),input.size()*sizeof(epoch_desc_t),cudaMemcpyHostToDevice));
        for(int count:counts) {
            CUDA_CHECK(cudaMemset(device_first,0xa5,actual.size()*sizeof(uint32_t)));
            kernel_build_first<<<epochs,count>>>(device_epochs,device_first);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(actual.data(),device_first,actual.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost));
            for(int e=0;e<epochs;e++)for(int c=0;c<QSB_FIRST_SLOTS;c++) {
                uint32_t expected[8];
                if(c<count) {
                    uint8_t block[64];
                    for(int j=0;j<16;j++) {
                        uint32_t w=j<2?input[e].remW[j]:table[j-2][c];
                        for(int k=0;k<4;k++)block[4*j+k]=(uint8_t)(w>>(24-8*k));
                    }
                    SHA256_CTX ctx;
                    SHA256_Init(&ctx);
                    for(int j=0;j<8;j++)ctx.h[j]=input[e].mid[j];
                    SHA256_Transform(&ctx,block);
                    for(int j=0;j<8;j++)expected[j]=ctx.h[j];
                    states++;words+=8;
                } else {
                    for(auto &v:expected)v=untouched;
                    padding+=8;
                }
                for(int j=0;j<8;j++) {
                    uint32_t got=actual[((size_t)e*QSB_FIRST_SLOTS+c)*8+j];
                    if(got!=expected[j]) {
                        if(errors<8)fprintf(stderr,"mismatch fixture=%d classes=%d epoch=%d class=%d word=%d\n",fixture,count,e,c,j);
                        errors++;
                    }
                }
            }
        }
    }
    CUDA_CHECK(cudaFree(device_first));
    CUDA_CHECK(cudaFree(device_epochs));
    printf("{\"compression_states\":%llu,\"active_word_checks\":%llu,\"padding_word_checks\":%llu,\"errors\":%llu}\n",
        (unsigned long long)states,(unsigned long long)words,(unsigned long long)padding,(unsigned long long)errors);
    return errors?1:0;
}
