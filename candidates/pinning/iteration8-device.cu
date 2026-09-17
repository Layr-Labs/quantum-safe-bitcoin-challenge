// Diagnostic: exercise the actual included CUDA arithmetic, not a Python model.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "GPUMath.h"
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);} } while(0)
__global__ void probe(const uint64_t *a,uint64_t *out) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=6)return;
  _ModMultCore(out+8*i,a+4*i,a+4*i);
  _ModSqr(out+8*i+4,a+4*i);
}
int main(){
  const uint64_t d[6]={1,2,65536,65537,65538,4294967295ULL};
  uint64_t a[24],out[48];
  for(int i=0;i<6;i++){a[4*i]=0xfffffffefffffc2fULL-d[i];for(int j=1;j<4;j++)a[4*i+j]=~0ULL;}
  uint64_t *da,*dr; CK(cudaMalloc(&da,sizeof(a)));CK(cudaMalloc(&dr,sizeof(out)));
  CK(cudaMemcpy(da,a,sizeof(a),cudaMemcpyHostToDevice));probe<<<1,32>>>(da,dr);
  CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(out,dr,sizeof(out),cudaMemcpyDeviceToHost));
  int bad=0;
  for(int i=0;i<6;i++)for(int op=0;op<2;op++){
    uint64_t *r=out+8*i+4*op,expected=d[i]*d[i];
    // Lazy residues are valid: canonicalize once since output is below 2^256.
    if(r[3]==~0ULL && r[2]==~0ULL && r[1]==~0ULL && r[0]>=0xfffffffefffffc2fULL){r[0]-=0xfffffffefffffc2fULL;r[1]=r[2]=r[3]=0;}
    bool ok=r[0]==expected && r[1]==0 && r[2]==0 && r[3]==0;bad+=!ok;
    printf("delta=%llu op=%s got=%016llx%016llx%016llx%016llx expected=%016llx correct=%d\n",(unsigned long long)d[i],op?"square":"multiply",(unsigned long long)r[3],(unsigned long long)r[2],(unsigned long long)r[1],(unsigned long long)r[0],(unsigned long long)expected,ok);
  }
  printf("actual_device_mismatches=%d/12 (diagnostic, not throughput)\n",bad);
  CK(cudaFree(da));CK(cudaFree(dr));return 0;
}
