#!/usr/bin/env python3
"""Execute extracted scalar hashing and producer buffer access on the CPU.

CUDA scheduling, throughput and PTX arithmetic are outside this test's scope.
All generated files live in a temporary directory outside the editable tree.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess
import tempfile

from test_parity_replay import function
from test_host_gate import C, compress_one

HERE = Path(__file__).resolve().parent


def main():
    pin = (HERE / 'pinning.cu').read_text()
    gpu_hash = (HERE / 'GPUHash.h').read_text()
    macros = gpu_hash.split('//Take the last 8 bytes')[0]
    transform = '#define DEF(x,y) uint32_t x = output[y]\n' + function(gpu_hash, '__device__ void _SHA256Transform(')
    sha = (HERE / 'sha_pinsha.cuh').read_text()
    old = function(sha, '__device__ __forceinline__ uint32_t qsb_fadd(')
    sha = sha.replace(old, old[:old.index('{')] + '{ return a*one+b; }')
    pre = function(pin, 'struct qsb_tail_pre {') + ';'
    make = function(pin, 'static void qsb_make_tail_pre(')
    fast = function(pin, '__device__ __forceinline__ void _SHA256TransformFastTail11Q(')
    message = function(pin, '__device__ __forceinline__ void qsb_tail_message(')
    scalar = function(pin, 'template<bool FAST_TAIL>\n__device__ __forceinline__ void qsb_hash_scalar(')
    kernel = function(pin, 'template<bool FAST_TAIL, int STAGE>\n__global__')
    kernel = kernel[:kernel.index('    uint64_t qx[4]')]
    kernel = kernel.replace('kernel_pinning_pipeline', 'transport')
    load_start = pin.index('    uint64_t z[4];\n#if QSB_SHA_PRODUCER\n    if (STAGE==3)')
    load_end = pin.index('    /* neg_r_inv is folded', load_start)
    kernel += pin[load_start:load_end] + '''
    if(active) for(int k=0;k<4;k++) observed[4*idx+k]=z[k];
}
'''
    header = r'''
#include <cstdint>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>
#define __host__
#define __device__
#define __constant__
#define __global__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define QSB_SHA_PRODUCER 1
#define QSB_TAIL_PRE 1
#define QSB_SPARSE_TAIL 1
#define QSB_SHA_OPT 1
#define QSB_SHA_UNIF 1
#define QSB_SPARSE_D 1
#define QSB_ZEROS_N 24
#define QSB_SHA_FMA_ADD 0
#define QSB_SHA_FMA_ROT 0
#define QSB_SHA_SMEM_W1 0
#define QSB_S0_THREADS 128
#define QSB_S2_THREADS 128
#define QSB_S0_BLOCKS 4
#define QSB_S2_BLOCKS 8
#define QSB_SHA_PRODUCER_BLOCKS 12
struct dim {unsigned x;};
static dim threadIdx,blockIdx,blockDim{128};
struct ulonglong2 {uint64_t x,y;};
static uint32_t pin_tail_words[3];
static uint64_t *observed;
static ulonglong2 qsb_ld_v2(const ulonglong2*p){return *p;}
static void qsb_st_v2(ulonglong2*p,uint64_t x,uint64_t y){*p={x,y};}
static uint32_t __umulhi(uint32_t a,uint32_t b){return ((uint64_t)a*b)>>32;}
static uint32_t __byte_perm(uint32_t a,uint32_t b,unsigned s){
 uint64_t ab=(uint64_t)a|((uint64_t)b<<32);uint32_t out=0;
 for(unsigned i=0;i<4;i++){unsigned n=(s>>(i*4))&15;assert(n<8);out|=((ab>>(n*8))&255)<<(i*8);}return out;
}
static uint32_t qsb_h_ror(uint32_t x,int n){return (x>>n)|(x<<(32-n));}
'''
    harness = header + macros + transform + sha + pre + make + fast + message + scalar + kernel
    harness += r'''
extern "C" void hash_one(uint64_t*out,const uint32_t*mid,const uint8_t*suffix,
 const uint32_t*prefixmid,uint32_t seq,uint32_t start,unsigned idx,int fast){
 for(int i=0;i<3;i++)pin_tail_words[i]=((uint32_t)suffix[64+4*i]<<24)|((uint32_t)suffix[65+4*i]<<16)|((uint32_t)suffix[66+4*i]<<8)|suffix[67+4*i];
 // The suffix is 75 bytes plus its first padding byte. Mask the mutable locktime.
 pin_tail_words[0]&=0xffffff00u;pin_tail_words[1]&=0xffu;
 qsb_tail_pre tp;qsb_make_tail_pre(&tp,mid,pin_tail_words[2]);
 blockIdx.x=idx/128;threadIdx.x=idx%128;
 if(fast)qsb_hash_scalar<true>(out,mid,suffix,75,31,67,9995,seq,start,start+idx,0,1,tp);
 else qsb_hash_scalar<false>(out,prefixmid,suffix,75,31,67,9995,seq,start,start+idx,0,1,tp);
}
extern "C" void audit_transport(const uint32_t*mid,const uint8_t*suffix,
 unsigned n,uint32_t start,uint32_t seq,unsigned reverse){
 qsb_tail_pre tp;qsb_make_tail_pre(&tp,mid,pin_tail_words[2]);
 const ulonglong2 guard{0x123456789abcdef0ULL,0xfedcba9876543210ULL};
 std::vector<ulonglong2> buf(4*n+2,guard);auto saved=buf.data()+1;
 std::vector<uint64_t> out(4*n,0),expected(4*n,0);observed=out.data();
 auto run=[&](auto stage,unsigned block){
  blockIdx.x=block;
  for(threadIdx.x=0;threadIdx.x<128;threadIdx.x++)
   transport<true,decltype(stage)::value>(mid,suffix,75,31,67,9995,seq,start,
    nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,n,0,1,saved,nullptr,nullptr,tp);
 };
 unsigned blocks=(n+127)/128;
 // Include one completely inactive block; its entry return must precede stores.
 for(unsigned b=0;b<=blocks;b++)run(std::integral_constant<int,1>{},b);
 for(unsigned idx=0;idx<n;idx++){
  uint64_t z[4];blockIdx.x=idx/128;threadIdx.x=idx%128;
  qsb_hash_scalar<true>(z,mid,suffix,75,31,67,9995,seq,start,start+idx,0,1,tp);
  for(int k=0;k<4;k++)expected[4*idx+k]=z[k];
 }
 for(unsigned b=0;b<blocks;b++){
  unsigned bi=reverse?blocks-1-b:b;run(std::integral_constant<int,3>{},bi);
  // Simulate the EC checkpoint overwriting all four planes of this block.
  for(unsigned idx=bi*128;idx<std::min(n,(bi+1)*128);idx++)
   for(unsigned plane=0;plane<4;plane++)saved[plane*n+idx]=guard;
 }
 assert(out==expected);
 assert(std::memcmp(&buf.front(),&guard,sizeof(guard))==0);
 assert(std::memcmp(&buf.back(),&guard,sizeof(guard))==0);
}
'''
    problem = json.loads((HERE.parents[1] / 'problems/pinning.json').read_text())
    prefix = bytes.fromhex(problem['pin_prefix'])
    prefixmid = C.sha256_midstate(prefix)
    suffix0 = bytes.fromhex(problem['suffix'])
    u32, u64, u8 = ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint8
    rng = random.Random(0x5CA1A2)
    checks = 0
    with tempfile.TemporaryDirectory(prefix='qsb-sha-producer-test-') as tmp:
        cpp, so = Path(tmp)/'test.cpp', Path(tmp)/'test.so'
        cpp.write_text(harness)
        subprocess.run(['clang++','-std=c++17','-O2','-shared','-fPIC',str(cpp),'-o',str(so)],check=True)
        lib = ctypes.CDLL(str(so))
        lib.hash_one.argtypes = [ctypes.POINTER(u64),ctypes.POINTER(u32),ctypes.POINTER(u8),ctypes.POINTER(u32),u32,u32,u32,ctypes.c_int]
        lib.audit_transport.argtypes = [ctypes.POINTER(u32),ctypes.POINTER(u8),u32,u32,u32,u32]
        pm = (u32*8)(*prefixmid)
        for seq in [0, 0x80000000, 0xffffffff]:
            suffix = bytearray(suffix0)
            struct.pack_into('<I',suffix,31,seq)
            mid = (u32*8)(*compress_one(prefixmid,bytes(suffix[:64])))
            buf = (u8*76).from_buffer_copy(suffix+b'\x80')
            starts = [0, 256, 0x1fffff00, 0xfffff000]
            for start in starts:
                ids = [0,1,126,127,128,129,254,255,256,257,1023,2048,4095]
                ids += [rng.randrange(4096) for _ in range(100)]
                for idx in ids:
                    full_suffix = bytearray(suffix)
                    struct.pack_into('<I',full_suffix,67,(start+idx)&0xffffffff)
                    want = hashlib.sha256(hashlib.sha256(prefix+full_suffix).digest()).digest()
                    for fast_path in [0,1]:
                        out = (u64*4)()
                        lib.hash_one(out,mid,buf,pm,seq,start,idx,fast_path)
                        got = sum(int(x)<<(64*k) for k,x in enumerate(out)).to_bytes(32,'big')
                        assert got == want, (seq,start,idx,fast_path)
                        checks += 1
            for n in [0,1,127,128,129,255,256,257,1023,1024,1025]:
                for reverse in [0,1]:
                    lib.audit_transport(mid,buf,n,0x1fffff00,seq,reverse)
    # Launch stays on the slot stream and has an explicit error check before EC.
    launch = function(pin, 'template<bool FAST_TAIL>\nstatic void launch_pinning_pipeline(')
    assert launch.index('<FAST_TAIL,1>') < launch.index('<FAST_TAIL,0>')
    assert launch.index('<FAST_TAIL,1>') < launch.index('<FAST_TAIL,3>')
    assert '<<<(batch_size+127)/128,128 QSB_STREAM_ARG>>>' in launch
    assert 'SHA producer launch failed' in launch
    print(json.dumps({'scalar_digest_checks':checks,'buffer_order_cases':66,
        'mismatches':0,'gpu_execution':False,'throughput_measured':False}))


if __name__ == '__main__':
    main()
