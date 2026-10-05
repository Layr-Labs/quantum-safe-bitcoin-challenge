// SPDX-License-Identifier: GPL-3.0-only
#include "SlotInputCache.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);} } while(0)
constexpr int Slots=5, Words=96, Batches=10000;
__global__ void copy_words(const unsigned* in,unsigned* out){int i=threadIdx.x;if(i<Words)out[i]=in[i];}
int main(){
  unsigned *host[Slots],*device[Slots],*output[Slots],*readback[Slots];
  cudaStream_t streams[Slots];cudaEvent_t done[Slots];
  qsb::SlotInputCache cache[Slots];bool busy[Slots]={};
  unsigned expected[Slots][Words]={};
  for(int s=0;s<Slots;s++){
    CHECK(cudaStreamCreateWithFlags(&streams[s],cudaStreamNonBlocking));
    CHECK(cudaEventCreateWithFlags(&done[s],cudaEventDisableTiming));
    CHECK(cudaMallocHost(&host[s],Words*sizeof(unsigned)));
    CHECK(cudaMallocHost(&readback[s],Words*sizeof(unsigned)));
    CHECK(cudaMalloc(&device[s],Words*sizeof(unsigned)));
    CHECK(cudaMalloc(&output[s],Words*sizeof(unsigned)));
    std::memset(host[s],0xa5,Words*sizeof(unsigned));
    CHECK(cache[s].init(device[s],host[s],Words*sizeof(unsigned)));
    if(cache[s].init(device[s],host[s],Words*sizeof(unsigned))!=cudaErrorInvalidValue)return 2;
    if(cache[s].enqueue(nullptr,Words*sizeof(unsigned),streams[s])!=cudaErrorInvalidValue)return 3;
    if(cache[s].enqueue(expected[s],4,streams[s])!=cudaErrorInvalidValue)return 4;
  }
  int checked=0;
  auto collect=[&](int s){if(!busy[s])return;CHECK(cudaEventSynchronize(done[s]));if(std::memcmp(readback[s],expected[s],sizeof(expected[s]))){fprintf(stderr,"slot %d mismatch\n",s);exit(5);}checked++;busy[s]=false;};
  for(int b=0;b<Batches;b++){
    int s=b%Slots;collect(s);
    // First cycle is all zeros; long identical runs alternate with all-byte changes.
    unsigned generation=b<Slots?0:(unsigned)(b/97+1);
    for(int i=0;i<Words;i++)expected[s][i]=generation?generation*2654435761u+(unsigned)i:0;
    CHECK(cache[s].enqueue(expected[s],sizeof(expected[s]),streams[s]));
    copy_words<<<1,Words,0,streams[s]>>>(device[s],output[s]);CHECK(cudaGetLastError());
    CHECK(cudaMemcpyAsync(readback[s],output[s],sizeof(expected[s]),cudaMemcpyDeviceToHost,streams[s]));
    CHECK(cudaEventRecord(done[s],streams[s]));busy[s]=true;
  }
  for(int s=0;s<Slots;s++)collect(s);
  printf("PASS: %d full-payload checks over %d slots; zero first input, cache hits, changed inputs, invalid arguments\n",checked,Slots);
  constexpr int Iterations=20000;
  for(int mode=0;mode<2;mode++){
    auto start=std::chrono::steady_clock::now();
    for(int i=0;i<Iterations;i++){
      CHECK(cudaStreamSynchronize(streams[0]));
      if(mode)CHECK(cache[0].enqueue(expected[0],sizeof(expected[0]),streams[0]));
      else {std::memcpy(host[0],expected[0],sizeof(expected[0]));CHECK(cudaMemcpyAsync(device[0],host[0],sizeof(expected[0]),cudaMemcpyHostToDevice,streams[0]));}
    }
    CHECK(cudaStreamSynchronize(streams[0]));
    double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/Iterations;
    printf("%s identical input + completion: %.3f us/iteration\n",mode?"cached":"uncached",us);
  }
  for(int s=0;s<Slots;s++){CHECK(cudaFree(device[s]));CHECK(cudaFree(output[s]));CHECK(cudaFreeHost(host[s]));CHECK(cudaFreeHost(readback[s]));CHECK(cudaEventDestroy(done[s]));CHECK(cudaStreamDestroy(streams[s]));}
}
