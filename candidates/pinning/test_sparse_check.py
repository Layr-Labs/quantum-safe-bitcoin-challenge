#!/usr/bin/env python3
"""Exercise production sparse OpenSSL check with a CPU CUDA-copy stub.

Verifies sample identity, >4GiB addressing, and fail-closed handling of copy
errors / corrupt bytes. No CUDA execution or performance measurement.
"""
from pathlib import Path
import json
import shlex
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
src=(HERE/'pinning.cu').read_text()
hdr=(HERE/'GLVScalar.cuh').read_text().split('// QSB/VanitySearch')[0].replace('#pragma once','')
def extract(name):
    pos=src.index('static '+name);brace=src.index('{',pos);end=brace+1;depth=1
    while depth:
        depth+=(src[end]=='{')-(src[end]=='}');end+=1
    return src[pos:end]
cpp=r'''
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <sys/mman.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#define __host__
#define __device__
#define __forceinline__ inline
#define QSB_GT_SPARSE_CHECK 1
'''+hdr+r'''
#define GT_CHUNKS 6
unsigned gt_entries(int c){return q9_bigtbl_entries(c);}
unsigned gt_offset(int c){return q9_bigtbl_offset(c);}
int gt_shift(int c){return q9_bigtbl_shift(c);}
constexpr int cudaSuccess=0,cudaMemcpyDeviceToHost=2;
int copies=0,fail_at=-1,corrupt_at=-1;uintptr_t base=0;size_t largest=0;
int cudaMemcpy(void *out,const void *in,size_t bytes,int kind){
    assert(bytes==64 && kind==cudaMemcpyDeviceToHost);
    size_t off=(uintptr_t)in-base;assert(off+bytes<=QSB_GT_TOTAL*64ULL);
    if(off>largest)largest=off;
    int i=copies++;if(i==fail_at)return 1;
    memcpy(out,in,bytes);if(i==corrupt_at)((uint8_t*)out)[63]^=1;
    return 0;
}
'''
for name in ['void gt_point_to_limbs(', 'void gt_table_scalar(', 'int gt_spot_check(']:cpp+=extract(name)+'\n'
cpp+=r'''
int main(){
    const size_t bytes=QSB_GT_TOTAL*64ULL;
    uint8_t *table=(uint8_t*)mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    assert(table!=MAP_FAILED);base=(uintptr_t)table;
    EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx=BN_CTX_new();BIGNUM *x=BN_new(),*y=BN_new(),*k=BN_new(),*order=BN_new(),*f=BN_new(),*nri=BN_new(),*a=BN_new(),*b=BN_new();
    EC_POINT *pt=EC_POINT_new(grp);EC_GROUP_get_order(grp,order,ctx);EC_GROUP_get_curve_GFp(grp,f,nullptr,nullptr,ctx);
    unsigned seed=0x9e3779b9u;uint8_t nr[32]={7};uint64_t alpha[4]={9},beta[4]={27};
    BN_set_word(nri,7);BN_set_word(a,9);BN_set_word(b,27);
    for(int t=0;t<216;t++){
        int ch;unsigned i;
        if(t<24){ch=t/4;unsigned corner[4]={0,1,2,gt_entries(ch)-1};i=corner[t%4];}
        else{seed=seed*1664525u+1013904223u;ch=(seed>>28)%6;seed=seed*1664525u+1013904223u;i=seed%gt_entries(ch);}
        gt_table_scalar(k,ch,i);BN_mod_mul(k,k,nri,order,ctx);EC_POINT_mul(grp,pt,k,nullptr,nullptr,ctx);
        uint64_t want[8];gt_point_to_limbs(grp,pt,x,y,a,b,f,ctx,want);
        memcpy(table+((size_t)gt_offset(ch)+i)*64,want,64);
    }
    assert(gt_spot_check(table,216,nr,alpha,beta)==1);assert(copies==216);
    if(QSB_FOUR_HOT)assert(largest>0x100000000ULL);
    for(int at: {0,13,215}){
        copies=0;fail_at=at;assert(gt_spot_check(table,216,nr,alpha,beta)==0);assert(copies==at+1);
        copies=0;fail_at=-1;corrupt_at=at;assert(gt_spot_check(table,216,nr,alpha,beta)==0);assert(copies==at+1);corrupt_at=-1;
    }
    std::printf("{\"four_hot\":%d,\"valid_samples\":216,\"injected_failures\":6,\"max_byte_offset\":%zu}",QSB_FOUR_HOT,largest);
    munmap(table,bytes);EC_POINT_free(pt);EC_GROUP_free(grp);BN_CTX_free(ctx);
    for(auto p:{x,y,k,order,f,nri,a,b})BN_free(p);
}
'''
cpp='#include <initializer_list>\n'+cpp
flags=shlex.split(subprocess.check_output(['pkg-config','--cflags','--libs','openssl'],text=True))
results=[]
with tempfile.TemporaryDirectory(prefix='qsb-sparse-') as td:
    td=Path(td);p=td/'test.cpp';p.write_text(cpp)
    for mode in [0,1]:
        exe=td/f'check{mode}'
        subprocess.run(['g++','-O2','-std=c++17','-Wno-deprecated-declarations','-fsanitize=undefined','-fno-sanitize-recover=all',f'-DQSB_FOUR_HOT={mode}',str(p),'-o',str(exe),*flags],check=True)
        r=subprocess.run([str(exe)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True)
        assert len(r.stderr.splitlines())==6,r.stderr
        results.append(json.loads(r.stdout))
print(json.dumps({'gpu_executed':False,'actual_source_check':True,'results':results},indent=2))
