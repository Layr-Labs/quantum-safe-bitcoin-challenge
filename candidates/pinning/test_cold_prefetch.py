#!/usr/bin/env python3
"""CPU execution of the actual prefetch address/control body with hint stubs.

Checks signed code masking, 64-bit addresses, bounds, and exact hint schedule.
No CUDA execution, cache observation or performance measurement.
"""
from pathlib import Path
import json
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
source=(HERE/'pinning.cu').read_text()
a=source.index('#ifndef QSB_COLD_PREFETCH\n');b=source.index('__device__ void _FixedBaseSignedXYZZScalar(',a)
body=source[a:b]
early=source.split('// BEGIN QSB_COLD_PREFETCH_EARLY_CALLS',1)[1].split('// END QSB_COLD_PREFETCH_EARLY_CALLS',1)[0]
loop=source.split('// BEGIN QSB_COLD_PREFETCH_LOOP_CALL',1)[1].split('// END QSB_COLD_PREFETCH_LOOP_CALL',1)[0]
for arg in ['record','record+32']:
 old='asm volatile("{ .reg .u64 g; cvta.to.global.u64 g, %0; prefetch.global.L2 [g]; }" :: "l"('+arg+') : "memory");'
 assert body.count(old)==1,old
 body=body.replace(old,'capture('+arg+');')
cpp=r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <sys/mman.h>
#define __device__
#define __constant__
#define __forceinline__ inline
#define QSB_BIGTBL 1
#ifndef QSB_FOUR_HOT
#define QSB_FOUR_HOT 1
#endif
#define QSB_TREE_N 128
struct {unsigned x;} threadIdx;
uint64_t arena[12*QSB_TREE_N];
uint64_t *qsb_digit_arena(){return arena;}
std::vector<uintptr_t> actual;
void capture(const uint8_t *p){actual.push_back((uintptr_t)p);}
'''+body+'\nvoid early_hints(const uint8_t *table){\n'+early+'\n}\n'+\
'\nvoid loop_hints(const uint8_t *table,unsigned term){\n'+loop+'\n}\n'+r'''
#if !(QSB_COLD_PREFETCH && QSB_COLD_PREFETCH_TUNE)
int pin_cold_prefetch_enabled=1;
#endif
int main(){
    constexpr size_t size=9803211584ULL;
    uint8_t *table=(uint8_t*)mmap(nullptr,size,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0);
    assert(table!=MAP_FAILED);
    const unsigned offsets[6]={0,262144,524288,655360,786432,67895296};
    const unsigned counts[6]={262144,262144,131072,131072,67108864,85279885};
    uint32_t rng=0x2142ABC9;unsigned lanes=0,hints=0;
    for(unsigned pass=0;pass<10;pass++){
        volatile uint32_t *codes=(volatile uint32_t*)arena;
        for(unsigned term=0;term<12;term++)for(unsigned lane=0;lane<128;lane++){
            rng=rng*1664525u+1013904223u;unsigned c=term%6;
            unsigned index=pass==0?0:pass==1?counts[c]-1:rng%counts[c];
            codes[term*128+lane]=(offsets[c]+index)|(((lane+pass)&1u)<<31);
        }
        for(int enabled: {0,1})for(unsigned lane=0;lane<128;lane++)for(unsigned first: {0u,6u}){
            pin_cold_prefetch_enabled=enabled;
            threadIdx.x=lane;actual.clear();std::vector<uintptr_t> want;
#if QSB_COLD_PREFETCH_EARLY
            early_hints(table);
#else
            for(unsigned term=first+2;term<12;term++)loop_hints(table,term);
#endif
#if QSB_FOUR_HOT
            for(unsigned target: {4u,5u,10u,11u}){
#if QSB_COLD_PREFETCH_TUNE
                if(!enabled)continue;
#endif
#if !QSB_COLD_PREFETCH_EARLY
                if(target<first+2+QSB_COLD_PREFETCH_LEAD)continue;
#endif
                uint32_t code=codes[target*128+lane];
                size_t offset=size_t(code%0x80000000u)*64;
                assert(offset+64<=size);
                for(unsigned hint=0;hint<QSB_COLD_PREFETCH;hint++)want.push_back((uintptr_t)table+offset+32*hint);
            }
#endif
            assert(actual==want);lanes++;hints+=actual.size();
        }
    }
    munmap(table,size);
    printf("{\"mode\":%d,\"lead\":%d,\"four_hot\":%d,\"early\":%d,\"lane_paths\":%u,\"hints\":%u}",QSB_COLD_PREFETCH,QSB_COLD_PREFETCH_LEAD,QSB_FOUR_HOT,QSB_COLD_PREFETCH_EARLY,lanes,hints);
}
'''
results=[]
with tempfile.TemporaryDirectory(prefix='qsb-prefetch-') as td:
    td=Path(td);p=td/'test.cpp';p.write_text(cpp)
    for mode,lead,four,early_mode,tune in [(0,2,1,0,1),(1,1,1,0,1),(1,2,1,0,1),(2,1,1,0,1),(2,2,1,0,1),(2,2,0,0,1),(1,2,1,1,1),(2,2,1,1,1),(0,2,1,1,1),(2,2,1,1,0)]:
        exe=td/f'check{mode}-{lead}-{four}'
        subprocess.run(['g++','-O2','-std=c++17','-fsanitize=undefined','-fno-sanitize-recover=all',f'-DQSB_COLD_PREFETCH={mode}',f'-DQSB_COLD_PREFETCH_LEAD={lead}',f'-DQSB_FOUR_HOT={four}',f'-DQSB_COLD_PREFETCH_EARLY={early_mode}',f'-DQSB_COLD_PREFETCH_TUNE={tune}',str(p),'-o',str(exe)],check=True)
        results.append(json.loads(subprocess.check_output([str(exe)],text=True)))
print(json.dumps({'actual_source_body':True,'ubsan':'pass','gpu_executed':False,'cases':results},indent=2))
