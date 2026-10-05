#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

static void check(cudaError_t e) {
  if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); }
}
__global__ void prepare(unsigned *state, unsigned value) { state[0] = value; }
__global__ void root(unsigned *state) { state[1] = state[0] * 3u; }
__global__ void finish(unsigned *state, unsigned *count) { atomicAdd(count, state[1]); }
int main() {
  cudaStream_t input, output, streams[4];
  cudaEvent_t incoming, done[4];
  cudaGraph_t graphs[4]; cudaGraphExec_t exec[4]; cudaGraphNode_t first[4];
  unsigned *state[4], *count, *host;
  check(cudaMalloc(&count, sizeof(unsigned)));
  check(cudaHostAlloc(&host, sizeof(unsigned), cudaHostAllocDefault));
  check(cudaStreamCreateWithFlags(&input, cudaStreamNonBlocking));
  check(cudaStreamCreateWithFlags(&output, cudaStreamNonBlocking));
  check(cudaEventCreateWithFlags(&incoming, cudaEventDisableTiming));
  for (int i=0;i<4;i++) {
    check(cudaMalloc(&state[i], 2*sizeof(unsigned)));
    check(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    check(cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming));
    check(cudaStreamBeginCapture(streams[i], cudaStreamCaptureModeRelaxed));
    prepare<<<1,1,0,streams[i]>>>(state[i], 1);
    root<<<1,1,0,streams[i]>>>(state[i]);
    finish<<<1,1,0,streams[i]>>>(state[i], count);
    check(cudaStreamEndCapture(streams[i], &graphs[i]));
    size_t n=3; cudaGraphNode_t nodes[3]; check(cudaGraphGetNodes(graphs[i],nodes,&n));
    for (size_t j=0;j<n;j++) {
      size_t before=0;
#if CUDART_VERSION >= 13000
      check(cudaGraphNodeGetDependencies(nodes[j],nullptr,nullptr,&before));
#else
      check(cudaGraphNodeGetDependencies(nodes[j],nullptr,&before));
#endif
      if (!before) first[i]=nodes[j];
    }
    check(cudaGraphInstantiateWithFlags(&exec[i],graphs[i],0));
  }
  for (unsigned batch=0; batch<1000; batch++) {
    check(cudaMemsetAsync(count,0,sizeof(unsigned),input));
    check(cudaEventRecord(incoming,input));
    for (int i=0;i<4;i++) check(cudaStreamWaitEvent(streams[i],incoming,0));
    unsigned expected=0;
    for (unsigned j=0;j<8;j++) {
      int i=j%4; unsigned v=batch*8+j+1; expected+=3*v;
      cudaKernelNodeParams p={}; check(cudaGraphKernelNodeGetParams(first[i],&p));
      void *args[]={&state[i],&v}; p.kernelParams=args;
      check(cudaGraphExecKernelNodeSetParams(exec[i],first[i],&p));
      check(cudaGraphLaunch(exec[i],streams[i]));
    }
    for(int i=0;i<4;i++) { check(cudaEventRecord(done[i],streams[i])); check(cudaStreamWaitEvent(output,done[i],0)); }
    check(cudaMemcpyAsync(host,count,sizeof(unsigned),cudaMemcpyDeviceToHost,output));
    check(cudaStreamSynchronize(output));
    if(*host!=expected) { std::fprintf(stderr,"batch=%u got=%u expected=%u\n",batch,*host,expected); return 2; }
  }
  std::puts("PASS: 1000 batches, 8000 graph launches, changed arguments, ring reuse, reset/readback ordering");
  for(int i=0;i<4;i++) { check(cudaGraphExecDestroy(exec[i])); check(cudaGraphDestroy(graphs[i])); check(cudaEventDestroy(done[i])); check(cudaStreamDestroy(streams[i])); check(cudaFree(state[i])); }
  check(cudaEventDestroy(incoming)); check(cudaStreamDestroy(input)); check(cudaStreamDestroy(output)); check(cudaFree(count)); check(cudaFreeHost(host));
}
