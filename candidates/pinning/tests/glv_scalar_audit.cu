#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "../GLVScalar.cuh"
struct Case {uint64_t k[4],r1[2],r2[2];unsigned signs,pad;};
static_assert(sizeof(Case)==72,"case layout");
__global__ void audit(Case *cases,int n,unsigned *errors){
    int i=blockIdx.x*128+threadIdx.x;if(i>=n)return;Case c=cases[i];uint64_t a[2],b[2];unsigned sa,sb;
    q9_glv_split(c.k,a,b,&sa,&sb);
    bool bad=sa+(sb<<1)!=c.signs;
    for(int j=0;j<2;j++)bad|=a[j]!=c.r1[j]||b[j]!=c.r2[j];
    if(bad&&atomicAdd(errors,1)==0)printf("GLV mismatch case=%d signs=%u expected=%u\n",i,sa+(sb<<1),c.signs);
}
int main(int argc,char **argv){
    if(argc!=2)return 2;FILE*f=fopen(argv[1],"rb");if(!f)return 2;fseek(f,0,SEEK_END);long bytes=ftell(f);rewind(f);
    if(bytes%sizeof(Case))return 2;int n=bytes/sizeof(Case);std::vector<Case> h(n);if(fread(h.data(),sizeof(Case),n,f)!=(size_t)n)return 2;fclose(f);
    Case *d;unsigned *errors,e;cudaMalloc(&d,bytes);cudaMalloc(&errors,4);cudaMemcpy(d,h.data(),bytes,cudaMemcpyHostToDevice);cudaMemset(errors,0,4);
    audit<<<(n+127)/128,128>>>(d,n,errors);cudaError_t ce=cudaDeviceSynchronize();cudaMemcpy(&e,errors,4,cudaMemcpyDeviceToHost);
    printf("GLV scalar cases=%d mismatches=%u %s\n",n,e,cudaGetErrorString(ce));return e||ce!=cudaSuccess;
}
