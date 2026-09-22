#!/usr/bin/env python3
"""Compile extracted production finish/replay control flow with host arithmetic.

This is a CPU semantic test, not execution of CUDA instructions or a timing.
The parity PTX is audited separately. Generated sources/binaries stay external.
"""
import os
from pathlib import Path
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent


def function(source, needle):
    start=source.index(needle)
    brace=source.index('{',start)
    level=1;i=brace+1
    while level:
        level+=(source[i]=='{')-(source[i]=='}');i+=1
    return source[start:i]


def main():
    pin=(HERE/'pinning.cu').read_text()
    packed=(HERE/'PackedRecovery.cuh').read_text()
    parity=(HERE/'ParityWindow.cuh').read_text()
    finish=function(pin,'template<bool FAST_TAIL,bool EXACT_PARITY>')
    replay=function(pin,'template<bool FAST_TAIL>\n__global__ __launch_bounds__(128,4) void qsb_replay_parity(')
    window=parity[parity.index('template<bool EXACT_PARITY=true>'):]
    core=packed[packed.index('template<bool EXACT_PARITY=true>'):]
    # Keep the source's preprocessor branches; C++ selects the ranked settings.
    header=r'''
#include <openssl/bn.h>
#include <openssl/sha.h>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <algorithm>
#define __device__
#define __global__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define QSB_TREE_N 128
#define QSB_STREAM2 1
#define QSB_ROOT_V2 1
#define QSB_PK_UNROLL 1
#define QSB_SHA_OPT 1
#define QSB_SPARSE_D 1
#define QSB_ZEROS_N 8
#define QSB_LAZY_REC 1
#define QSB_RAW_X 0
#define QSB_PARITY_SUM 1
#define QSB_PARITY_WINDOW 1
#define QSB_PARITY_WINDOW_NARROW 1
struct ulonglong2 {uint64_t x,y;};
struct dim {uint32_t x;};
static dim blockIdx,blockDim,threadIdx,gridDim;
static uint64_t pin_u2rx_words[4]={1},pin_u2ry_words[4]={0x9123748a92ULL},pin_recovery_c[4]={0x179e78};
static uint32_t current_idx=0,force_mode=0;
static BN_CTX *ctx;
static BIGNUM *modulus;
static uint64_t seed=0x12bd177a52ULL;
static uint64_t random64(){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;}
static void field(uint64_t *out,const uint64_t *a,const uint64_t *b,char op){
 BN_CTX_start(ctx);BIGNUM *aa=BN_CTX_get(ctx),*bb=BN_CTX_get(ctx),*z=BN_CTX_get(ctx);
 BN_lebin2bn((const unsigned char*)a,32,aa);BN_lebin2bn((const unsigned char*)b,32,bb);
 if(op=='*')assert(BN_mod_mul(z,aa,bb,modulus,ctx));
 if(op=='+')assert(BN_mod_add(z,aa,bb,modulus,ctx));
 if(op=='-')assert(BN_mod_sub(z,aa,bb,modulus,ctx));
 assert(BN_bn2lebinpad(z,(unsigned char*)out,32)==32);BN_CTX_end(ctx);
}
static void qsb_packed_raw_mul(uint64_t *o,const uint64_t*a,const uint64_t*b){field(o,a,b,'*');}
static void qsb_recovery_mul(uint64_t *o,const uint64_t*a,const uint64_t*b){field(o,a,b,'*');}
static void _ModSub256(uint64_t*o,const uint64_t*a,const uint64_t*b){field(o,a,b,'-');}
static void _ModAdd256(uint64_t*o,const uint64_t*a,const uint64_t*b){field(o,a,b,'+');}
static void _ModAddLazy(uint64_t*o,const uint64_t*a,const uint64_t*b){field(o,a,b,'+');}
static uint32_t qsb_sum_parity(const uint64_t *a,const uint64_t *b,uint32_t neg){
 uint64_t s[4];field(s,a,b,'+');return (s[0]&1u)^((s[0]|s[1]|s[2]|s[3])?neg:0);
}
static void qsb_parity_window_words(uint64_t &mid,uint64_t &top,const uint64_t *aa,const uint64_t *bb){
 if(force_mode==1 || (force_mode==2&&current_idx%17==0) || (force_mode==3&&current_idx%2==0)){
  mid=0xffffffffu;top=0;return;
 }
 uint32_t a[8],b[8];memcpy(a,aa,32);memcpy(b,bb,32);
 auto column=[&](unsigned k){unsigned __int128 s=0;for(unsigned i=0;i<8;i++)if(k>=i&&k-i<8)s+=(uint64_t)a[i]*b[k-i];return s;};
 mid=(uint64_t)((column(6)>>32)+column(7));
 uint32_t bit=0;for(unsigned i=1;i<8;i++)bit^=a[i]&b[8-i];
 mid^=uint64_t(bit&1u)<<32;mid&=0x1ffffffffULL;
 top=(uint64_t)((column(13)>>32)+column(14));
}
static ulonglong2 qsb_ld_v2(const ulonglong2*p){return *p;}
static uint32_t atomicAdd(uint32_t*p,uint32_t n){uint32_t old=*p;*p+=n;return old;}
static uint32_t __byte_perm(uint32_t a,uint32_t b,uint32_t select){
 uint64_t both=uint64_t(a)|(uint64_t(b)<<32);uint32_t out=0;
 for(unsigned i=0;i<4;i++)out|=uint32_t((both>>(((select>>(4*i))&7)*8))&255)<<(8*i);return out;
}
static uint32_t _SHA256Pubkey33H0(const uint32_t *p){
 unsigned char bytes[36],digest[32];for(unsigned i=0;i<9;i++)for(unsigned j=0;j<4;j++)bytes[4*i+j]=p[i]>>(24-8*j);
 SHA256(bytes,33,digest);return uint32_t(digest[0])<<24|uint32_t(digest[1])<<16|uint32_t(digest[2])<<8|digest[3];
}
static bool gpu_bench_valid_h0(uint32_t h){return (h>>24)==0;}
// FAST_TAIL makes these fallback-only declarations unreachable in this test.
static void _SHA256TransformPubkey33(uint32_t*,const uint32_t*){assert(false);}
static bool gpu_is_der_easy(const uint8_t*,int){assert(false);return false;}
static bool gpu_bench_valid_words(const uint32_t*){assert(false);return false;}
static void _SHA256Initialize(uint32_t*){assert(false);}
static void _SHA256Transform(uint32_t*,const uint32_t*){assert(false);}
'''
    # Instrument only the test's replay entry to identify forced exception rows.
    replay=replay.replace('uint32_t idx=replay[1u+pos];','uint32_t idx=replay[1u+pos];current_idx=idx;')
    driver=r'''
int main(){
 ctx=BN_CTX_new();modulus=BN_new();BN_hex2bn(&modulus,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
 unsigned checked=0,replayed=0,hits=0;
 for(unsigned batch: {1u,127u,128u,129u,1023u,1024u,1025u,8193u}){
  unsigned roots_n=(batch+127)/128;
  std::vector<ulonglong2> saved(4*batch);std::vector<uint64_t> roots(8*roots_n);
  for(auto &r:saved)r={random64(),random64()};for(auto &r:roots)r=random64();
  // Exercise inactive/zero-denominator lanes, including boundary lanes.
  for(unsigned i=0;i<batch;i+=113){saved[2*batch+i]={0,0};saved[3*batch+i]={0,0};}
  auto negative_saved=saved;
  for(unsigned i=0;i<batch;i++){
   uint64_t y[4]={saved[i].x,saved[i].y,saved[batch+i].x,saved[batch+i].y},ny[4],zero[4]={0};
   field(ny,zero,y,'-');negative_saved[i]={ny[0],ny[1]};negative_saved[batch+i]={ny[2],ny[3]};
  }
  for(force_mode=0;force_mode<4;force_mode++){
   std::vector<uint32_t> queue(batch+3,0),expected(1024,0),actual(1024,0);
   queue[batch+1]=0x1234;queue[batch+2]=0xabcd;uint32_t ec=0,ac=0;
   for(unsigned i=0;i<batch;i++){
    current_idx=i;
    control::qsb_finish_candidate<true,true>(i,batch,0,1,saved.data(),roots.data(),&ec,expected.data(),nullptr);
    variant::qsb_finish_candidate<true,false>(i,batch,0,1,negative_saved.data(),roots.data(),&ac,actual.data(),queue.data());
   }
   assert(queue[0]<=batch);assert(queue[batch+1]==0x1234&&queue[batch+2]==0xabcd);
   auto queued=std::vector<uint32_t>(queue.begin()+1,queue.begin()+1+queue[0]);
   std::sort(queued.begin(),queued.end());assert(std::adjacent_find(queued.begin(),queued.end())==queued.end());
   gridDim.x=8;blockDim.x=128;
   for(blockIdx.x=0;blockIdx.x<gridDim.x;blockIdx.x++)for(threadIdx.x=0;threadIdx.x<blockDim.x;threadIdx.x++)
    variant::qsb_replay_parity<true>(batch,0,1,negative_saved.data(),roots.data(),&ac,actual.data(),queue.data());
   assert(ec<1024&&ac==ec);expected.resize(ec);actual.resize(ac);std::sort(expected.begin(),expected.end());std::sort(actual.begin(),actual.end());assert(expected==actual);
   checked+=batch;replayed+=queue[0];hits+=ec;
  }
 }
 printf("{\"finish_candidate_executions\":%u,\"replay_records\":%u,\"matching_hits\":%u,\"mismatches\":0}\n",checked*2+replayed,replayed,hits);
 BN_free(modulus);BN_CTX_free(ctx);
}
'''
    with tempfile.TemporaryDirectory(prefix='qsb-parity-replay-') as tmp:
        tmp=Path(tmp);src=tmp/'audit.cpp';binary=tmp/'audit'
        source=header+'\n'+window+'\n'
        for name,flag in [('control',0),('variant',1)]:
            source+=f'\n#undef QSB_NEG_Y_MAC\n#define QSB_NEG_Y_MAC {flag}\nnamespace {name} {{\n'+core+'\n'+finish+'\n'+replay+'\n}\n'
        src.write_text(source+driver)
        openssl=Path('/opt/homebrew/opt/openssl@3')
        cmd=['clang++','-O2','-std=c++17','-Wno-deprecated-declarations',str(src),'-o',str(binary),'-lcrypto']
        if openssl.exists():cmd+=['-I'+str(openssl/'include'),'-L'+str(openssl/'lib')]
        subprocess.run(cmd,check=True)
        subprocess.run([str(binary)],check=True)
    # Stream-order and capacity checks connect the executable model to launch wiring.
    launch=function(pin,'template<bool FAST_TAIL>\nstatic void launch_pinning_pipeline(')
    assert launch.index('cudaMemsetAsync(tree')<launch.index('kernel_pinning_pipeline<FAST_TAIL,2>')<launch.index('qsb_replay_parity<FAST_TAIL>')
    assert pin.count('((size_t)BATCH+1u)*sizeof(uint32_t)')==2
    assert 'pos+=blockDim.x*gridDim.x' in replay
    assert 'if(!EXACT_PARITY)return 0x10000u;' in parity
    print('production queue allocation, reset, dispatch, root indexing, and grid-stride drain checks passed')

if __name__=='__main__':main()
