#!/usr/bin/env python3
"""Audit the active pinning.cu host-transfer implementation.

Compile the actual allocation, upload, result-copy and result-view statements
against CPU CUDA-memory stubs. Check buffer bounds, all hit counts through the
device capacity, stale tails, both slots, and the QSB_TAIL_PRE=0 fallback.
This checks host buffer semantics, not GPU execution, overlap or performance.
SPDX-License-Identifier: GPL-3.0-only
"""
import json
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def between(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i)]


def harness(source):
    allocation = between(source, '    cudaStream_t slot_stream[QSB_SLOTS];',
                         '        if (se != cudaSuccess) {') + '    }\n'
    upload = between(source, '#if !QSB_SLOT_SKIP_MID_UPLOAD || !QSB_TAIL_PRE',
                     '            launch_pinning_pipeline<true>(')
    copies = between(source, '#if QSB_SLOT_PACKED_READBACK\n            cudaMemcpyAsync',
                     '            cudaEventRecord(slot_done[s], st);')
    views = between(source, '#if QSB_SLOT_PACKED_READBACK\n        uint32_t h_hit',
                    '            int nh = (h_hit > 64)')
    return PREFIX + '\nint main(){\n' + allocation + MAIN.replace('@@UPLOAD@@', upload).replace(
        '@@COPIES@@', copies).replace('@@VIEWS@@', views)


PREFIX = r'''
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>
#include <cassert>
using cudaError_t=int;using cudaStream_t=int;using cudaEvent_t=int;
constexpr int cudaSuccess=0,cudaHostAllocDefault=0,cudaStreamNonBlocking=0,
 cudaEventDisableTiming=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
std::vector<void*> allocations;
unsigned h2d=0,d2h=0;
int cudaHostAlloc(void **p,size_t n,int){*p=std::malloc(n);assert(*p);std::memset(*p,0xa5,n);allocations.push_back(*p);return 0;}
template<class T>int cudaMalloc(T **p,size_t n){return cudaHostAlloc((void**)p,n,0);}
int cudaMemcpy(void *d,const void*s,size_t n,int){std::memcpy(d,s,n);return 0;}
int cudaMemcpyAsync(void*d,const void*s,size_t n,int kind,int){
 if(kind==cudaMemcpyHostToDevice)h2d++;else d2h++;
 return cudaMemcpy(d,s,n,kind);
}
int cudaMemsetAsync(void*d,int c,size_t n,int){std::memset(d,c,n);return 0;}
int cudaStreamCreateWithFlags(int*p,int){static int next=0;*p=next++;return 0;}
int cudaEventCreateWithFlags(int*p,int){*p=0;return 0;}
struct {uint32_t midstate[8]={1,2,3,4,5,6,7,8};}pp;
static uint32_t record(unsigned count,unsigned i,unsigned slot){
 return (i*9973+count*17+slot*131)|((i&1u)<<30);
}
'''

MAIN = r'''
 unsigned cases=0,records=0;
 for(unsigned count=0;count<=1030;count++)for(int s=0;s<QSB_SLOTS;s++){
  cudaStream_t st=slot_stream[s];
  uint32_t cur_mid[8];for(int k=0;k<8;k++)cur_mid[k]=count*31+s*7+k;
  h2d=d2h=0;
  @@UPLOAD@@
  assert(*d_hit_cnt_s[s]==0);
#if !QSB_SLOT_SKIP_MID_UPLOAD || !QSB_TAIL_PRE
  assert(std::memcmp(d_mid_slot[s],cur_mid,32)==0);
  assert(h2d==1);
#else
  assert(h2d==0);
#endif
  // Emulate the unchanged device stores. The counter may exceed capacity,
  // but the actual kernel writes indices only while pos<1024.
  *d_hit_cnt_s[s]=count;
  for(unsigned i=0;i<count&&i<1024;i++)d_hit_idx_s[s][i]=record(count,i,s);
  @@COPIES@@
  @@VIEWS@@
   unsigned used=h_hit>64?64:h_hit;
   for(unsigned i=0;i<used;i++){assert(hits[i]==record(count,i,s));records++;}
  }
  assert(h_hit==count);
  assert(d2h==(QSB_SLOT_PACKED_READBACK?1u:2u));
  cases++;
 }
 for(void*p:allocations)std::free(p);
 std::printf("{\"cases\":%u,\"records\":%u,\"h2d_per_batch\":%u,\"d2h_per_batch\":%u}\n",cases,records,h2d,d2h);
}
'''


def main():
    source = (HERE / 'pinning.cu').read_text()
    results = []
    with tempfile.TemporaryDirectory(prefix='qsb-transfers-cpu-') as tmp:
        cpp, exe = Path(tmp)/'test.cpp', Path(tmp)/'test'
        cpp.write_text(harness(source))
        for packed, skip, pre, slots in [(0,0,1,2),(0,1,1,2),(1,0,1,2),(1,1,1,2),(1,1,0,2),(1,1,1,3)]:
            flags = [f'-DQSB_SLOT_PACKED_READBACK={packed}',f'-DQSB_SLOT_SKIP_MID_UPLOAD={skip}',
                     f'-DQSB_TAIL_PRE={pre}',f'-DQSB_SLOTS={slots}']
            subprocess.run(['g++','-std=c++17','-O2','-fsanitize=address,undefined',
                            '-fno-sanitize-recover=all',*flags,str(cpp),'-o',str(exe)],check=True)
            row = json.loads(subprocess.check_output([str(exe)],text=True))
            results.append(dict(packed=packed,skip_upload=skip,tail_pre=pre,slots=slots,**row))
    print(json.dumps(dict(test='actual host statements with CPU memory stubs and ASan/UBSan',
                         configurations=results,gpu_executed=False,measured_speedup=None),indent=2))


if __name__ == '__main__':
    main()
