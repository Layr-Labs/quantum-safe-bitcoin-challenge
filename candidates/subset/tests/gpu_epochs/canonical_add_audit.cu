#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include "../../GPUMath.h"
struct Input { uint64_t a[4],b[4],expected[4]; };
struct Output { uint64_t value[3][4]; };
__global__ void audit_canonical_add(const Input *input,Output *output,int count){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    uint64_t a[4],b[4];
    for(int k=0;k<4;k++){a[k]=input[i].a[k];b[k]=input[i].b[k];}
    _ModAdd256(output[i].value[0],a,b);
    _ModAdd256(a,a,b);for(int k=0;k<4;k++){output[i].value[1][k]=a[k];a[k]=input[i].a[k];}
    _ModAdd256(b,a,b);for(int k=0;k<4;k++)output[i].value[2][k]=b[k];
}
#define CHECK(call) do{cudaError_t e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));return 2;}}while(0)
int main(int argc,char **argv){
    if(argc!=2)return 2;FILE *f=fopen(argv[1],"rb");if(!f)return 2;
    uint32_t n;if(fread(&n,4,1,f)!=1 || !n || n>1000000)return 2;
    std::vector<Input> in(n);std::vector<Output> out(n);
    if(fread(in.data(),sizeof(Input),n,f)!=n || fgetc(f)!=EOF)return 2;fclose(f);
    Input *di;Output *dr;CHECK(cudaMalloc(&di,in.size()*sizeof(Input)));CHECK(cudaMalloc(&dr,out.size()*sizeof(Output)));
    CHECK(cudaMemcpy(di,in.data(),in.size()*sizeof(Input),cudaMemcpyHostToDevice));
    audit_canonical_add<<<(n+127)/128,128>>>(di,dr,n);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(),dr,out.size()*sizeof(Output),cudaMemcpyDeviceToHost));
    int errors=0;
    for(uint32_t i=0;i<n;i++)for(int mode=0;mode<3;mode++){
        bool same=true;for(int k=0;k<4;k++)same &= out[i].value[mode][k]==in[i].expected[k];
        if(!same){if(errors<8)printf("Add mismatch case=%u alias=%d\n",i,mode);errors++;}
    }
    printf("Canonical add GPU audit: cases=%u aliases=3 errors=%d\n",n,errors);
    cudaFree(di);cudaFree(dr);return errors?1:0;
}
