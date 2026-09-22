#!/usr/bin/env python3
"""Execute extracted production host/SHA code on CPU. No GPU rate is inferred."""
import ctypes
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
_ARTIFACTS = tempfile.TemporaryDirectory(prefix='qsb-hostsha-tests-')
OUT = Path(_ARTIFACTS.name)
TREE = (HERE / 'tests/gpu_epochs/tree.cu').read_text()


def braced(text, marker):
    start = text.index(marker)
    brace = text.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def build(name, source, shared=False, crypto=False):
    cpp, binary = OUT / (name + '.cpp'), OUT / (name + ('.so' if shared else ''))
    cpp.write_text(source)
    cmd = ['c++', '-std=c++17', '-O2', '-Wno-deprecated-declarations', '-Wno-unknown-pragmas']
    if shared:
        cmd += ['-shared', '-fPIC']
    if crypto:
        prefix = Path('/opt/homebrew/opt/openssl@3')
        if prefix.exists():
            cmd += ['-I' + str(prefix / 'include'), '-L' + str(prefix / 'lib')]
        cmd += ['-lcrypto']
    cmd += [str(cpp), '-o', str(binary)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    return binary


def sha_test():
    macros = (HERE / 'GPUHash.h').read_text().split('//Take the last 8 bytes')[0]
    shim = '''#include <cstdint>
#define __host__
#define __device__
#define __constant__
#define __forceinline__ inline
#define QSB_ZEROS_N 24
#define QSB_SHA_FMA_ADD 0
#define QSB_SHA_FMA_ROT 0
#define QSB_SHA_ALU_ADD 0
static uint32_t __umulhi(uint32_t a,uint32_t b){return ((uint64_t)a*b)>>32;}
'''
    source = shim + macros + '\n' + (HERE / 'sha_pinsha.cuh').read_text() + '''
extern "C" void digest32(uint32_t *o,uint32_t *i){_SHA256TransformDigest32Q(o,i);}
extern "C" uint32_t key33(uint32_t *i){return _SHA256Pubkey33H0(i);}
'''
    lib = ctypes.CDLL(str(build('sha', source, shared=True)))
    u32 = ctypes.c_uint32
    lib.digest32.argtypes = [ctypes.POINTER(u32), ctypes.POINTER(u32)]
    lib.key33.argtypes = [ctypes.POINTER(u32)]
    lib.key33.restype = u32
    rng = random.Random(0x515342)
    vectors = [bytes([b])*32 for b in (0, 1, 127, 128, 255)]
    vectors += [(1 << bit).to_bytes(32, 'big') for bit in range(256)]
    vectors += [rng.randbytes(32) for _ in range(20000)]
    comparisons = 0
    for msg in vectors:
        w = [int.from_bytes(msg[i:i+4], 'big') for i in range(0, 32, 4)]
        inp, out = (u32*8)(*w), (u32*8)()
        lib.digest32(out, inp)
        want = hashlib.sha256(msg).digest()
        assert b''.join(int(x).to_bytes(4, 'big') for x in out) == want
        lib.digest32(inp, inp)
        assert b''.join(int(x).to_bytes(4, 'big') for x in inp) == want
        comparisons += 2
        for prefix in (2, 3):
            key = bytes([prefix]) + msg
            block = key + b'\x80' + bytes(22) + (264).to_bytes(8, 'big')
            inp = (u32*16)(*[int.from_bytes(block[i:i+4], 'big') for i in range(0, 64, 4)])
            assert lib.key33(inp) == int.from_bytes(hashlib.sha256(key).digest()[:4], 'big')
            comparisons += 1
    return {'vectors': len(vectors), 'comparisons_to_hashlib': comparisons, 'alias': 'pass'}


def pipeline_test():
    drain = braced(TREE, 'auto hp_drain =') + ';\n'
    loop = braced(TREE[TREE.index('auto hp_launch ='):], 'while (1)')
    source = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <ctime>
#include <unistd.h>
#include <vector>
#include <string>
#include <algorithm>
#define ZLAB_HIT_REC 16
#define QSB_SE_LAUNCH_BLOCKS 7
#define QSB_PAIR_MUL 4
#define QSB_SE_PER_EPOCH 128
using cudaError_t = int;
constexpr int cudaSuccess=0;
std::vector<int> drained;
int event_batch[2];
int fail_event=-1;
int cudaEventSynchronize(int slot){
  if(event_batch[slot]==fail_event) return 1;
  drained.push_back(event_batch[slot]); return 0;
}
int cudaGetLastError(){return 0;}
const char *cudaGetErrorString(int){return "injected event error";}
uint64_t g_total_searched, g_hit_counter;
int run(uint64_t n_epochs, unsigned salt){
  constexpr int HP_HOST_BYTES=4+64*ZLAB_HIT_REC;
  // Guard bytes surround both slot mirrors; exact production drain runs below.
  uint8_t mirror[2*HP_HOST_BYTES+32]; memset(mirror,0xA5,sizeof(mirror));
  uint8_t *h_verified=mirror+16;
  int hp_done[2]={0,1},hp_busy[2]={0,0},hp_epochs[2]={0,0};
  uint64_t hp_batch_no=0,epoch_base=0,total_searched=0,hit_counter=0;
  uint64_t global_total=n_epochs*QSB_SE_PER_EPOCH;
  int gpu_index=0; FILE *summary_f=nullptr;
  timespec t0,t_last_se; clock_gettime(CLOCK_MONOTONIC,&t0); t_last_se=t0;
  FILE *output=tmpfile(); assert(output); int zh_fd=fileno(output);
  drained.clear(); g_total_searched=g_hit_counter=0;
  std::string expected;
  auto hp_launch=[&](int s,uint64_t base,int epochs_in_batch)->int {
    assert(!hp_busy[s]); assert(base==hp_batch_no*QSB_SE_LAUNCH_BLOCKS*QSB_PAIR_MUL);
    assert(epochs_in_batch>0 && epochs_in_batch<=QSB_SE_LAUNCH_BLOCKS*QSB_PAIR_MUL);
    unsigned counts[]={0,1,8,9,63,64,65,1024,0xffffffffu};
    uint32_t count=counts[(hp_batch_no+salt)%9];
    uint8_t *h=h_verified+s*HP_HOST_BYTES; memset(h,0xDD,HP_HOST_BYTES); memcpy(h,&count,4);
    for(unsigned j=0;j<std::min(count,64u);j++){
      uint32_t raw=(j&1)<<30; memcpy(h+4+j*ZLAB_HIT_REC,&raw,4);
      uint8_t *combo=h+8+j*ZLAB_HIT_REC;
      for(int k=0;k<9;k++)combo[k]=(hp_batch_no*31+j+k)%256;
      char line[96]; snprintf(line,sizeof(line),"indices=%d,%d,%d,%d,%d,%d,%d,%d,%d recid=%d\n",
        combo[0],combo[1],combo[2],combo[3],combo[4],combo[5],combo[6],combo[7],combo[8],int(j&1));
      expected+=line;
    }
    event_batch[s]=hp_batch_no; hp_busy[s]=1; hp_epochs[s]=epochs_in_batch; return 0;
  };
''' + drain + loop + r'''
  assert(!hp_busy[0]&&!hp_busy[1]); assert(total_searched==global_total);
  assert(g_total_searched==total_searched && g_hit_counter==hit_counter);
  assert(drained.size()==hp_batch_no);
  for(unsigned i=0;i<drained.size();i++)assert(drained[i]==int(i));
  for(int i=0;i<16;i++)assert(mirror[i]==0xA5&&mirror[2*HP_HOST_BYTES+16+i]==0xA5);
  rewind(output); std::string actual; char buf[512]; size_t got;
  while((got=fread(buf,1,sizeof(buf),output)))actual.append(buf,got);
  fclose(output); assert(actual==expected);
  assert(std::count(actual.begin(),actual.end(),'\n')==hit_counter);
  return 0;
}
int main(){
  unsigned tests=0;
  for(uint64_t n=0;n<400;n++)for(unsigned salt=0;salt<9;salt++){assert(run(n,salt)==0);tests++;}
  for(uint64_t n: {1000,10001,28001}){assert(run(n,7)==0);tests++;}
  for(int f=0;f<3;f++){fail_event=f; assert(run(100,0)==1); assert(g_total_searched==uint64_t(f)*28*128);}
  printf("{\"exhaustion_and_bounds_cases\":%u,\"injected_event_errors\":3}\n",tests);
}
'''
    exe = build('pipeline', source)
    result = subprocess.check_output([str(exe)], text=True)
    return json.loads(result.splitlines()[-1])


def ladder_test():
    start = TREE.index('#if ZLAB_T14')
    geometry = '#define ZLAB_T14 0\n' + TREE[start:TREE.index('__device__ __constant__ uint64_t GT_ORDER_N', start)]
    functions = '\n'.join(braced(TREE, 'static '+sig) for sig in (
        'void gt_point_to_limbs(', 'void gt_build_ladders(', 'void gt_spot_sample(', 'int gt_spot_check('))
    baseline = braced(TREE, 'static void gt_build_ladders(').replace('gt_build_ladders','gt_build_ladders_baseline')
    source = '''#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#define __host__
#define __device__
#define __constant__
#define __forceinline__ inline
''' + geometry + '\n#define QSB_STARTUP_TRIM 1\n' + functions + '\n#undef QSB_STARTUP_TRIM\n#define QSB_STARTUP_TRIM 0\n' + baseline + r'''
int main(){
  std::vector<uint64_t> lo(GT_CHUNKS*GT_LO*8),hi(GT_CHUNKS*GT_HI*8),oldlo(lo.size()),oldhi(hi.size());
  constexpr int samples=GT_CHUNKS*4+192;
  for(int seedno=1;seedno<=3;seedno++){
    uint8_t scalar[32]; for(int i=0;i<32;i++)scalar[i]=(seedno*19+i*17)%251;
    gt_build_ladders(lo.data(),hi.data(),scalar);gt_build_ladders_baseline(oldlo.data(),oldhi.data(),scalar);
    assert(lo==oldlo && hi==oldhi);
    std::vector<uint8_t> full(size_t(GT_TOTAL_ENTRIES)*64),gathered(samples*64);
    EC_GROUP *g=EC_GROUP_new_by_curve_name(NID_secp256k1);BN_CTX *ctx=BN_CTX_new();
    EC_POINT *pt=EC_POINT_new(g); BIGNUM *k=BN_new(),*nri=BN_lebin2bn(scalar,32,nullptr),*order=BN_new(),*x=BN_new(),*y=BN_new();
    EC_GROUP_get_order(g,order,ctx); unsigned seed=0x9e3779b9u;
    for(int t=0;t<samples;t++){
      int ch,i;gt_spot_sample(t,&seed,&ch,&i);assert(ch>=0&&ch<GT_CHUNKS&&i>=0&&unsigned(i)<gt_entries(ch));
      if(t<GT_CHUNKS*4){assert(ch==t/4);assert(i==(t%4==3?int(gt_entries(ch))-1:t%4));}
      BN_set_word(k,2*i+1);BN_lshift(k,k,gt_shift(ch));BN_mod_mul(k,k,nri,order,ctx);
      if(BN_is_odd(k))BN_add(k,k,order);BN_rshift1(k,k);
      assert(EC_POINT_mul(g,pt,k,nullptr,nullptr,ctx));uint64_t limbs[8];gt_point_to_limbs(g,pt,x,y,ctx,limbs);
      memcpy(gathered.data()+t*64,limbs,64);memcpy(full.data()+(size_t(gt_offset(ch))+i)*64,limbs,64);
    }
    assert(gt_spot_check(full.data(),samples,scalar,nullptr));assert(gt_spot_check(nullptr,samples,scalar,gathered.data()));
    gathered[0]^=1;assert(!gt_spot_check(nullptr,samples,scalar,gathered.data()));
    full[0]^=1;assert(!gt_spot_check(full.data(),samples,scalar,nullptr));
    BN_free(k);BN_free(nri);BN_free(order);BN_free(x);BN_free(y);EC_POINT_free(pt);EC_GROUP_free(g);BN_CTX_free(ctx);
  }
  puts("{\"ladder_seeds\":3,\"spot_checks_per_seed\":252,\"corruption_rejections\":6}");
}
'''
    exe = build('ladder', source, crypto=True)
    return json.loads(subprocess.check_output([str(exe)], text=True))


def source_test():
    for flag in ('QSB_HOST_PIPE','QSB_STARTUP_TRIM','QSB_SHA_FOLD'):
        assert '#ifndef '+flag+'\n#define '+flag+' 1\n#endif' in TREE
    assert '../../../pinning/' not in TREE
    assert 'QSB_ISO_FAST_X' not in TREE
    launch = braced(TREE,'auto hp_launch =')
    kernels = re.findall(r'(kernel_\w+)<<<(.*?)>>>',launch,re.S)
    assert len(kernels)==7 and all(', st' in args for _,args in kernels)
    assert 'kernel_verify_pair_hits<<<1, 64, 0, st>>>(d_hb, d_verified_s[s], d_ep, d_fi' in launch
    assert 'err = cudaMemcpyAsync(' in launch and 'err = cudaEventRecord(' in launch
    assert 'cudaEventDestroy(hp_done[s])' in TREE and 'cudaStreamDestroy(hp_stream[s])' in TREE
    assert '#if !QSB_STARTUP_TRIM\n    cudaDeviceSetLimit(cudaLimitStackSize, 32768);' in TREE
    baseline=623518629; donor_base=613936599; donor=622587731
    return {'promotion_floor_snapshot':(baseline*101+99)//100,
            'historical_score_ratio':donor/donor_base,
            'unvalidated_composition_projection':baseline*donor/donor_base,
            'projection_is_measurement':False, 'gpu_executed':False}


if __name__ == '__main__':
    results={'source':source_test(),'sha':sha_test(),'pipeline':pipeline_test(),'ladders':ladder_test()}
    print(json.dumps(results,indent=2))
