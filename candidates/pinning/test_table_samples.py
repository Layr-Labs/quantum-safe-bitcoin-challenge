#!/usr/bin/env python3
"""Exercise the actual sampler, gather body and OpenSSL check on the CPU.

CUDA allocation/copy/error calls are modeled. The gather's indexing executes
as C++, including inactive tail threads. GPU execution still needs a GPU.
SPDX-License-Identifier: GPL-3.0-only
"""
import json
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def function(source, name):
    index = source.index(name + '(')
    start = source.rfind('\n', 0, index) + 1
    brace = source.index('{', index)
    end, depth = brace + 1, 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


PREFIX = r'''
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <map>
#include <vector>
#include <sys/mman.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include "TableSamplePlan.h"
#define __host__
#define __device__
#define __global__
#define __forceinline__ inline
#define QSB_TABLE_SAMPLE_READBACK 1
struct {unsigned x;} blockIdx, blockDim, threadIdx;
typedef int cudaError_t;
const int cudaSuccess=0, cudaErrorInvalidValue=1;
const int cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2;
static int operation=0, fail_at=0;
static size_t download_bytes=0;
static std::map<void*,size_t> allocations;
static bool fail() {return ++operation==fail_at;}
template<class T> static cudaError_t cudaMalloc(T **out,size_t bytes) {
    if(fail())return 1;
    *out=(T*)malloc(bytes);assert(*out);allocations[*out]=bytes;return 0;
}
static cudaError_t cudaFree(void *ptr) {
    bool error=fail();assert(allocations.erase(ptr)==1);free(ptr);return error?1:0;
}
static cudaError_t cudaMemcpy(void *dst,const void *src,size_t bytes,int direction) {
    if(fail())return 1;
    void *device=direction==cudaMemcpyHostToDevice?dst:const_cast<void*>(src);
    assert(allocations.count(device) && allocations[device]>=bytes);
    if(direction==cudaMemcpyDeviceToHost)download_bytes+=bytes;
    memcpy(dst,src,bytes);return 0;
}
static cudaError_t cudaGetLastError() {return fail()?1:0;}
'''

LAUNCH = r'''
static void launch_gather(const uint64_t *table,const uint32_t *records,
                          uint64_t *samples,int count) {
    blockDim.x=128;
    for(blockIdx.x=0;blockIdx.x<(unsigned)(count*8+127)/128;++blockIdx.x)
        for(threadIdx.x=0;threadIdx.x<128;++threadIdx.x)
            qsb_gather_table_samples(table,records,samples,count);
}
'''

MAIN = r'''
int main() {
    const int count=GT_CHUNKS*4+192;
    const size_t table_bytes=(size_t)GT_TOTAL_ENTRIES*64;
    // Reserve address space; only sampled pages need physical memory.
    uint8_t *table=(uint8_t*)mmap(NULL,table_bytes,PROT_READ|PROT_WRITE,
                                 MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    assert(table!=MAP_FAILED);
    uint8_t nri[32]={1};uint64_t alpha[4]={1},beta[4]={1};
    EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx=BN_CTX_new();BIGNUM *x=BN_new(),*y=BN_new(),*k=BN_new();
    BIGNUM *p=BN_new(),*one=BN_new();BN_one(one);
    EC_GROUP_get_curve_GFp(grp,p,NULL,NULL,ctx);EC_POINT *point=EC_POINT_new(grp);
    unsigned old_seed=0x9e3779b9u,new_seed=old_seed;
    for(int t=0;t<count;++t) {
        // Original committed selector, independent of the new helper.
        int ch,i;
        if(t<GT_CHUNKS*4) {
            ch=t/4;const int corner[4]={0,1,2,(int)gt_entries(ch)-1};i=corner[t%4];
        } else {
            old_seed=old_seed*1664525u+1013904223u;ch=(int)(old_seed>>28)%GT_CHUNKS;
#if QSB_BIGTBL
            old_seed=old_seed*1664525u+1013904223u;i=old_seed%gt_entries(ch);
#else
            i=(old_seed>>4)&(gt_entries(ch)-1);
#endif
        }
        QsbTableSample sample=qsb_table_sample_position(t,new_seed,GT_CHUNKS,gt_entries,QSB_BIGTBL);
        assert(sample.chunk==ch && sample.index==(unsigned)i && old_seed==new_seed);
        assert(sample.index<gt_entries(ch));
        gt_table_scalar(k,ch,i);assert(EC_POINT_mul(grp,point,k,NULL,NULL,ctx));
        uint64_t limbs[8];gt_point_to_limbs(grp,point,x,y,one,one,p,ctx,limbs);
        memcpy(table+((size_t)gt_offset(ch)+i)*64,limbs,64);
    }
    assert(gt_spot_check_legacy(table,count,nri,alpha,beta)==1);
    std::vector<uint64_t> packed(count*8+2,0xa55aa55aa55aa55aULL);
    assert(gt_readback_samples(table,(uint8_t*)&packed[1],count)==cudaSuccess);
    assert(download_bytes==(size_t)count*64);
    assert(packed.front()==0xa55aa55aa55aa55aULL && packed.back()==packed.front());
    assert(allocations.empty());
    assert(gt_spot_check((uint8_t*)&packed[1],count,nri,alpha,beta,true)==1);
    assert(gt_spot_check(table,count,nri,alpha,beta,false)==1);
    unsigned seed=0x9e3779b9u;
    for(int t=0;t<count;++t) {
        QsbTableSample s=qsb_table_sample_position(t,seed,GT_CHUNKS,gt_entries,QSB_BIGTBL);
        assert(memcmp(&packed[1+t*8],table+((size_t)gt_offset(s.chunk)+s.index)*64,64)==0);
    }
    // Both validators reject corruption at the first and last checked record.
    for(int selected : {0,count-1}) {
        seed=0x9e3779b9u;QsbTableSample s;
        for(int t=0;t<=selected;++t)s=qsb_table_sample_position(t,seed,GT_CHUNKS,gt_entries,QSB_BIGTBL);
        size_t off=((size_t)gt_offset(s.chunk)+s.index)*64;
        table[off]^=1;
        assert(gt_readback_samples(table,(uint8_t*)&packed[1],count)==cudaSuccess);
        assert(gt_spot_check_legacy(table,count,nri,alpha,beta)==0);
        assert(gt_spot_check((uint8_t*)&packed[1],count,nri,alpha,beta,true)==0);
        table[off]^=1;
    }
    // Every allocation/copy/launch-status/free failure must propagate and clean up.
    for(int failure=1;failure<=7;++failure) {
        operation=0;fail_at=failure;
        assert(gt_readback_samples(table,(uint8_t*)&packed[1],count)!=cudaSuccess);
        assert(allocations.empty());
    }
    operation=0;fail_at=0;
    assert(gt_readback_samples(table,(uint8_t*)&packed[1],0)==cudaErrorInvalidValue);
    assert(gt_readback_samples(table,(uint8_t*)&packed[1],count+1)==cudaErrorInvalidValue);
    assert(operation==0);
    // Small counts cover all partial-block boundaries in the gather.
    for(int small : {1,15,16,17,count-1}) {
        std::vector<uint64_t> out(small*8+2,0x123456789abcdef0ULL);
        assert(gt_readback_samples(table,(uint8_t*)&out[1],small)==cudaSuccess);
        assert(out.front()==0x123456789abcdef0ULL && out.back()==out.front());
        assert(gt_spot_check((uint8_t*)&out[1],small,nri,alpha,beta,true)==1);
    }
    assert(allocations.empty());
    EC_POINT_free(point);BN_free(x);BN_free(y);BN_free(k);BN_free(p);BN_free(one);
    EC_GROUP_free(grp);BN_CTX_free(ctx);munmap(table,table_bytes);
    printf("{\"big_table\":%d,\"samples\":%d,\"full_bytes\":%zu,"
           "\"sample_bytes\":%zu,\"injected_api_failures\":7,\"passed\":true}\n",
           QSB_BIGTBL,count,table_bytes,(size_t)count*64);
}
'''


def main():
    source = (HERE / 'pinning.cu').read_text()
    scalar = (HERE / 'GLVScalar.cuh').read_text()
    legacy = subprocess.run(['git', 'show',
        '1fe5a8e40008befcd917668ea9b1a23c6ee590c4:candidates/pinning/pinning.cu'],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout
    geometry = source[source.index('#if QSB_BIGTBL\n/* GLV12:'):
                      source.index('/* n = secp256k1 group order')]
    helpers = scalar.split('// BEGIN QSB_BIGTBL_HOST_EXACT')[1].split('// END QSB_BIGTBL_HOST_EXACT')[0]
    cpp = PREFIX + helpers + geometry
    for name in ['gt_point_to_limbs', 'gt_table_scalar', 'gt_spot_check']:
        cpp += function(source, name) + '\n'
    cpp += function(legacy, 'gt_spot_check').replace('gt_spot_check(', 'gt_spot_check_legacy(', 1)
    cpp += '\n' + function(source, 'qsb_gather_table_samples') + LAUNCH
    readback = function(source, 'gt_readback_samples')
    launch = 'qsb_gather_table_samples<<<(count * 8 + 127) / 128, 128>>>'
    assert readback.count(launch) == 1
    cpp += readback.replace(launch, 'launch_gather') + MAIN
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'test.cpp').write_text(cpp)
        for big in [0, 1]:
            binary = tmp / ('samples' + str(big))
            subprocess.run(['g++', '-O2', '-Wno-deprecated-declarations',
                f'-DQSB_BIGTBL={big}', '-I', str(HERE), str(tmp / 'test.cpp'),
                '-lcrypto', '-o', str(binary)], check=True)
            completed = subprocess.run([str(binary)], capture_output=True, text=True)
            assert completed.returncode == 0, completed.stderr
            results.append(json.loads(completed.stdout))
    print(json.dumps({'passed': True, 'variants': results, 'gpu_executed': False}, indent=2))


if __name__ == '__main__':
    main()
