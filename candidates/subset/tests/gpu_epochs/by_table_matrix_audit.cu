// Diagnostic: production normalized-six-step decision function vs independent Python recursion.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#define QSB_ROOT_MAX_BATCHES 16
struct QsbInverseWords {uint64_t a,b,c,d;};
__device__ __forceinline__ QsbInverseWords qsb_root_fermat(QsbInverseWords){
    asm("trap;");return {0,0,0,0}; // Root wrapper is outside this diagnostic's scope.
}
#include "zinv32.cuh"
#ifndef __CUDA_ARCH__
uint32_t zi_x(uint32_t,int){abort();}
#endif
struct Input {int32_t delta;uint32_t f,g;int32_t expected[5];};
struct Output {int32_t value[5];};
static_assert(sizeof(Input)==32,"fixture layout");
__global__ void matrix_audit(const Input*in,Output*out,unsigned n){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    out[i].value[0]=zi_divstep30_by(in[i].delta,in[i].f,in[i].g,
        out[i].value+1,out[i].value+2,out[i].value+3,out[i].value+4);
}
#define CHECK(call) do{cudaError_t e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));return 2;}}while(0)
int main(int argc,char**argv){
    if(argc!=2)return 2;FILE*f=fopen(argv[1],"rb");if(!f)return 2;
    uint32_t n;if(fread(&n,4,1,f)!=1||!n||n>1000000)return 2;
    std::vector<Input>in(n);std::vector<Output>out(n);
    if(fread(in.data(),sizeof(Input),n,f)!=n||fgetc(f)!=EOF)return 2;fclose(f);
    Input*di;Output*dout;CHECK(cudaMalloc(&di,n*sizeof(Input)));CHECK(cudaMalloc(&dout,n*sizeof(Output)));
    CHECK(cudaMemcpy(di,in.data(),n*sizeof(Input),cudaMemcpyHostToDevice));
    matrix_audit<<<(n+127)/128,128>>>(di,dout,n);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(),dout,n*sizeof(Output),cudaMemcpyDeviceToHost));
    unsigned errors=0;
    for(unsigned i=0;i<n;i++)for(unsigned j=0;j<5;j++)if(out[i].value[j]!=in[i].expected[j]){
        if(errors<8)printf("Matrix mismatch case=%u field=%u got=%d expected=%d\n",i,j,out[i].value[j],in[i].expected[j]);errors++;
    }
    printf("Normalized6 BY matrix audit: inputs=%u values=%u errors=%u\n",n,5*n,errors);
    cudaFree(di);cudaFree(dout);return errors?1:0;
}
