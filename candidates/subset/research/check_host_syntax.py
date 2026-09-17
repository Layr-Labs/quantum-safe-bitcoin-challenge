#!/usr/bin/env python3
"""Type-check a temporary C++ projection; explicitly NOT a CUDA compile.

Strip launch configurations, provide declarations for CUDA runtime/intrinsics,
and replace inline PTX with a no-op macro. This catches ordinary name/type/call
errors across the full include closure, but cannot validate any CUDA-specific
semantics, assembly, launch configuration or resource limits.
"""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from preflight import source_files,source_identity
STUB=r'''
#pragma once
#include <stddef.h>
#include <stdint.h>
#define __device__
#define __global__
#define __host__
#define __constant__
#define __shared__ static
#define __forceinline__ inline
#define __noinline__
#define __launch_bounds__(...)
#define asm(...) ((void)0)
#define __align__(n) alignas(n)
struct dim3{unsigned x,y,z;dim3(unsigned x_=1,unsigned y_=1,unsigned z_=1):x(x_),y(y_),z(z_) {}};
extern dim3 threadIdx,blockIdx,blockDim,gridDim;
struct uint4{unsigned x,y,z,w;};
struct ulonglong2{uint64_t x,y;};
inline ulonglong2 make_ulonglong2(uint64_t x,uint64_t y){return {x,y};}
inline uint4 make_uint4(unsigned x,unsigned y,unsigned z,unsigned w){return {x,y,z,w};}
void __syncthreads();
void __syncwarp();
unsigned __byte_perm(unsigned,unsigned,unsigned);
int __clzll(uint64_t);
int __popc(unsigned);
unsigned long long __shfl_sync(unsigned,unsigned long long,int);
unsigned long long __shfl_xor_sync(unsigned,unsigned long long,int);
unsigned atomicAdd(unsigned*,unsigned);
int atomicAdd(int*,int);
using cudaError_t=int;
using cudaStream_t=void*;
using cudaEvent_t=void*;
constexpr int cudaSuccess=0,cudaErrorInvalidValue=1,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2,cudaLimitStackSize=0;
constexpr unsigned cudaStreamNonBlocking=1,cudaEventDisableTiming=2,cudaHostAllocDefault=0;
struct cudaDeviceProp{char name[256];int multiProcessorCount;};
template<class T> int cudaMalloc(T**,size_t);
int cudaFree(void*);int cudaMemset(void*,int,size_t);int cudaMemcpy(void*,const void*,size_t,int);
int cudaMemsetAsync(void*,int,size_t,cudaStream_t);int cudaMemcpyAsync(void*,const void*,size_t,int,cudaStream_t);
template<class T> int cudaHostAlloc(T**,size_t,unsigned);
int cudaFreeHost(void*);
int cudaStreamCreateWithFlags(cudaStream_t*,unsigned);int cudaStreamWaitEvent(cudaStream_t,cudaEvent_t,unsigned);int cudaStreamDestroy(cudaStream_t);
int cudaEventCreateWithFlags(cudaEvent_t*,unsigned);int cudaEventRecord(cudaEvent_t,cudaStream_t);
int cudaEventSynchronize(cudaEvent_t);int cudaEventDestroy(cudaEvent_t);
template<class T> int cudaMemcpyToSymbol(T&,const void*,size_t,size_t=0,int=0);
template<class T> int cudaMemcpyFromSymbol(void*,const T&,size_t,size_t=0,int=0);
int cudaGetDeviceCount(int*);int cudaGetDeviceProperties(cudaDeviceProp*,int);int cudaSetDevice(int);
int cudaDeviceSetLimit(int,size_t);int cudaDeviceSynchronize();int cudaGetLastError();
const char *cudaGetErrorString(int);
'''
def main():
    identity=source_identity();launches=0
    with tempfile.TemporaryDirectory(prefix='qsb-cxx-projection-') as td:
        root=Path(td)
        (root/'cuda_runtime.h').write_text(STUB)
        for src in source_files():
            data=src.read_text();data,n=re.subn(r'<<<.*?>>>','',data,flags=re.S);launches+=n
            # volatile is outside the asm function-like macro.
            data=data.replace('asm volatile (','asm(').replace('asm volatile(','asm(')
            dst=root/src.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(data)
        for entry in ('subset.cu','tests/gpu_epochs/tree_audit.cu'):
            subprocess.run(['c++','-x','c++','-std=c++17','-fsyntax-only','-Wno-deprecated-declarations',
                '-DQSB_ZEROS_N=24','-I'+str(root),'-I/opt/homebrew/opt/openssl@3/include',str(root/entry)],check=True)
    assert source_identity()==identity
    print(json.dumps({'status':'PASS','validation_level':'C++ projection syntax/types only','launch_configurations_stripped':launches,
        'cuda_compile':False,'gpu_executed':False,**identity},indent=2))
if __name__=='__main__':main()
