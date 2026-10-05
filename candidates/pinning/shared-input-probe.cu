// SPDX-License-Identifier: GPL-3.0-only
#include "SharedInputPool.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);} } while(0)
constexpr int Slots=5, Words=96, Batches=10000;
__global__ void copy_words(const unsigned* in,unsigned* out){int i=threadIdx.x;if(i<Words)out[i]=in[i];}
__global__ void delay_copy(){unsigned long long begin=clock64();while(clock64()-begin<300000ull){}}
int main(){
  unsigned *host[Slots],*device[Slots],*output[Slots],*readback[Slots];
  cudaStream_t streams[Slots];cudaEvent_t done[Slots];bool busy[Slots]={};
  unsigned expected[Slots][Words]={};qsb::SharedInputPool<Slots> pool;
  for(int s=0;s<Slots;s++){
    CHECK(cudaStreamCreateWithFlags(&streams[s],cudaStreamNonBlocking));
    CHECK(cudaEventCreateWithFlags(&done[s],cudaEventDisableTiming));
    CHECK(cudaMallocHost(&host[s],Words*sizeof(unsigned)));
    CHECK(cudaMallocHost(&readback[s],Words*sizeof(unsigned)));
    CHECK(cudaMalloc(&device[s],Words*sizeof(unsigned)));
    CHECK(cudaMalloc(&output[s],Words*sizeof(unsigned)));
    std::memset(host[s],0xa5,Words*sizeof(unsigned));
    CHECK(pool.init(s,device[s],host[s],Words*sizeof(unsigned)));
  }
  const void* ptr=nullptr;
  if(pool.enqueue(-1,expected[0],sizeof(expected[0]),streams[0],&ptr)!=cudaErrorInvalidValue)return 2;
  if(pool.enqueue(0,nullptr,sizeof(expected[0]),streams[0],&ptr)!=cudaErrorInvalidValue)return 3;
  if(pool.enqueue(0,expected[0],4,streams[0],&ptr)!=cudaErrorInvalidValue)return 4;
  int checked=0,uploads=0,reuses=0;
  auto collect=[&](int s){if(!busy[s])return;CHECK(cudaEventSynchronize(done[s]));if(std::memcmp(readback[s],expected[s],sizeof(expected[s]))){fprintf(stderr,"slot %d mismatch\n",s);exit(5);}checked++;busy[s]=false;};
  for(int phase=0;phase<3;phase++){
    int phase_uploads=0,phase_reuses=0;
    for(int b=0;b<Batches;b++){
      int s=b%Slots;collect(s);
      // Actual ranked geometry: five batches share a group; each slot is used once.
      // Further phases force all misses and alternate long shared runs with eviction.
      unsigned generation=phase==0?(unsigned)(b/Slots):phase==1?(unsigned)(b+20000):(unsigned)(b/17+40000);
      for(int i=0;i<Words;i++)expected[s][i]=generation?generation*2654435761u+(unsigned)i:0;
      if(b%Slots==0)delay_copy<<<1,1,0,streams[s]>>>();
      CHECK(cudaGetLastError());
      bool uploaded=false;
      CHECK(pool.enqueue(s,expected[s],sizeof(expected[s]),streams[s],&ptr,&uploaded));
      if(uploaded)phase_uploads++;else phase_reuses++;
      copy_words<<<1,Words,0,streams[s]>>>(static_cast<const unsigned*>(ptr),output[s]);CHECK(cudaGetLastError());
      CHECK(cudaMemcpyAsync(readback[s],output[s],sizeof(expected[s]),cudaMemcpyDeviceToHost,streams[s]));
      CHECK(cudaEventRecord(done[s],streams[s]));busy[s]=true;
    }
    for(int s=0;s<Slots;s++)collect(s);
    printf("phase %d: %d uploads, %d reuses\n",phase,phase_uploads,phase_reuses);
    if(phase==0 && (phase_uploads!=2000 || phase_reuses!=8000))return 6;
    if(phase==1 && (phase_uploads!=10000 || phase_reuses!=0))return 7;
    uploads+=phase_uploads;reuses+=phase_reuses;
  }
  printf("PASS: %d full-payload checks; %d uploads, %d cross-slot reuses; delayed input producer, changed inputs, eviction, invalid arguments\n",checked,uploads,reuses);
  CHECK(cudaDeviceSynchronize());
  for(int s=0;s<Slots;s++){CHECK(cudaFree(device[s]));CHECK(cudaFree(output[s]));CHECK(cudaFreeHost(host[s]));CHECK(cudaFreeHost(readback[s]));CHECK(cudaEventDestroy(done[s]));CHECK(cudaStreamDestroy(streams[s]));}
}
