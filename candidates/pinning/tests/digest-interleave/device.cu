// SPDX-License-Identifier: GPL-3.0-only
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <openssl/sha.h>
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
#include "../../sha_pinsha.cuh"
#define CK(expr) do { cudaError_t e=(expr); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#expr,cudaGetErrorString(e));return 1;} } while(0)
__global__ void transform(const uint32_t* input,uint32_t* output,int count,int rounds) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    uint32_t v[8],h[8];
    #pragma unroll
    for(int k=0;k<8;k++)v[k]=input[8*i+k];
    for(int r=0;r<rounds;r++) {
        _SHA256TransformDigest32Q(h,v);
        #pragma unroll
        for(int k=0;k<8;k++)v[k]=h[k];
    }
    #pragma unroll
    for(int k=0;k<8;k++)output[8*i+k]=v[k];
}
int main() {
    constexpr int count=65536;
    std::vector<uint32_t> in(count*8),out(count*8);uint64_t rng=0x20260929aULL;
    for(int i=0;i<count;i++)for(int j=0;j<8;j++){
        rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;
        in[8*i+j]=i==0?0:i==1?0xffffffffu:uint32_t(rng);
    }
    uint32_t *di,*dout;CK(cudaMalloc(&di,in.size()*4));CK(cudaMalloc(&dout,out.size()*4));
    CK(cudaMemcpy(di,in.data(),in.size()*4,cudaMemcpyHostToDevice));
    const uint32_t zero=0;CK(cudaMemcpyToSymbol(pin_zero_add,&zero,4));
    transform<<<(count+255)/256,256>>>(di,dout,count,1);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out.data(),dout,out.size()*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<count;i++) {
        unsigned char bytes[32],want[32],got[32];
        for(int k=0;k<8;k++)for(int j=0;j<4;j++){
            bytes[k*4+j]=in[i*8+k]>>(24-8*j);got[k*4+j]=out[i*8+k]>>(24-8*j);
        }
        SHA256(bytes,32,want);if(memcmp(want,got,32)){fprintf(stderr,"Mismatch %d\n",i);return 2;}
    }
    cudaEvent_t start,stop;CK(cudaEventCreate(&start));CK(cudaEventCreate(&stop));
    float times[5];
    for(int repeat=0;repeat<6;repeat++){
        CK(cudaEventRecord(start));transform<<<(count+255)/256,256>>>(di,dout,count,128);
        CK(cudaEventRecord(stop));CK(cudaEventSynchronize(stop));CK(cudaGetLastError());
        float ms;CK(cudaEventElapsedTime(&ms,start,stop));if(repeat)times[repeat-1]=ms;
    }
    CK(cudaMemcpy(out.data(),dout,out.size()*4,cudaMemcpyDeviceToHost));
    unsigned char hash[32];SHA256((unsigned char*)out.data(),out.size()*4,hash);
    printf("{\"variant\":%d,\"wmixZ\":%d,\"deviceOracleCases\":%d,\"deviceMismatches\":0,\"timedDigestsPerRun\":%d,\"milliseconds\":[",QSB_DIGEST_INTERLEAVE_Z,QSB_DIGEST_WMIX_Z,count,count*128);
    for(int i=0;i<5;i++)printf("%s%.6f",i?",":"",times[i]);printf("],\"finalOutputSha256\":\"");
    for(int i=0;i<32;i++)printf("%02x",hash[i]);puts("\"}");
    CK(cudaFree(di));CK(cudaFree(dout));CK(cudaEventDestroy(start));CK(cudaEventDestroy(stop));
}
