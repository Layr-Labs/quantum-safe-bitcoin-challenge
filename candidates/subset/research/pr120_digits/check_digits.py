#!/usr/bin/env python3
"""Actual PR120 helper port vs actual peel source and independent integer law."""
import argparse,ctypes as C,hashlib,json,random,re,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research')]
from preflight import source_identity
from ptx_field_model import function
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,default=HERE/'candidate');ap.add_argument('--report',type=Path,default=HERE/'digits-results.json');args=ap.parse_args();base=args.source.resolve();identity=source_identity(base)
tree=(base/'tests/gpu_epochs/tree.cu').read_text();header=(base/'tests/gpu_epochs/compact_table_device.cuh').read_text();geometry=(base/'tests/gpu_epochs/compact_geometry.cuh').read_text();sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
helpers={name:function(source,sig) for name,source,sig in [
 ('setup',tree,'__device__ __forceinline__ void gt_recode_setup('),
 ('bits',header,'__device__ __forceinline__ uint32_t gt_field_bits_v('),
 ('direct',header,'__device__ __forceinline__ void gt_direct_digit('),
 ('step',header,'__device__ __forceinline__ int32_t compact_recode_step('),
 ('reference',header,'__device__ __forceinline__ void compact_recode_signed('),
 ('cold',header,'__device__ __forceinline__ void qsb_asym_cold_digits('),
 ('index',tree,'__device__ __forceinline__ void gt_digit_idx(')]}
assert sha(helpers['bits'])=='6aeb908a93de6d1a4e28d20f41d39838a1ee1cb96e0306826c9f3702cc27f90b'
assert sha(helpers['direct'])=='c291852d21658448963484bc6974568f3dec63b79530b46143ca74b2058081fa'
chain=function(header,'__device__ void compact_fixed_xyzz(')
assert 'compact_recode_step(' not in chain and chain.count('gt_direct_digit(')==2
assert 'qsb_asym_cold_digits(M,sign,&cold12,&cold13);' in chain
assert 'gt_direct_digit(M,(uint64_t)(sign<0),1u,17u,false,&idx,&neg);' in chain
assert 'gt_direct_digit(M,(uint64_t)(sign<0),(unsigned)mixed_shift(c+1)+1u,(unsigned)mixed_bits(c+1),false,&idx,&neg);' in chain
code='#include <stdint.h>\n#define __device__\n#define __host__\n#define __constant__\n#define __forceinline__ inline\n#define COMPACT_CHUNKS 14\n'+geometry+'\n'
code+=re.search(r'uint64_t GT_ORDER_N\[4\] = \{[^}]+\};',tree).group()+'\n'+'\n'.join(helpers.values())
code+=r'''
extern "C" void audit(const uint64_t*k,uint32_t*idx,uint64_t*neg,int32_t*ref,uint32_t*badidx,uint64_t*badneg){
 uint64_t M[4];int sign;gt_recode_setup(k,M,&sign);compact_recode_signed(k,ref);
 for(int c=0;c<14;++c){
  gt_direct_digit(M,sign<0,mixed_shift(c)+1,mixed_bits(c),c==13,idx+c,neg+c);
  // Negative control: an off-by-one bit start in a call to the actual helper.
  gt_direct_digit(M,sign<0,mixed_shift(c),mixed_bits(c),c==13,badidx+c,badneg+c);
 }
 int32_t c12,c13;qsb_asym_cold_digits(M,sign,&c12,&c13);
 gt_digit_idx(c12,idx+14,neg+14);gt_digit_idx(c13,idx+15,neg+15);
 // Negative control: use middle-digit interpretation on the final window.
 gt_direct_digit(M,sign<0,mixed_shift(13)+1,mixed_bits(13),false,badidx+14,badneg+14);
}
'''
N=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141;B=1<<256;MASK=(1<<64)-1
rng=random.Random(202609170745)
cases={0,1,2,N-2,N-1,N,N+1,N+2,B-2,B-1}
for bit in range(256):
 for d in [-2,-1,0,1,2]:
  v=(1<<bit)+d
  if 0<=v<B:cases.add(v)
cases=sorted(cases)+[rng.getrandbits(256) for _ in range(16000)]
counts={'scalars':0,'direct_digits':0,'preserved_cold_digits':0,'wrong_start_controls_caught':0,'wrong_last_controls_caught':0}
with tempfile.TemporaryDirectory(prefix='qsb-pr120-digits-') as td:
 cpp,so=Path(td)/'digits.cpp',Path(td)/'digits.so';cpp.write_text(code)
 subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC','-fsanitize=undefined','-fsanitize-trap=undefined',str(cpp),'-o',str(so)],check=True)
 lib=C.CDLL(str(so));U64=C.c_uint64;U32=C.c_uint32;I32=C.c_int32;lib.audit.argtypes=[C.POINTER(U64),C.POINTER(U32),C.POINTER(U64),C.POINTER(I32),C.POINTER(U32),C.POINTER(U64)]
 for k in cases:
  idx,neg,ref,badidx,badneg=(U32*16)(),(U64*16)(),(I32*14)(),(U32*15)(),(U64*15)()
  lib.audit((U64*4)(*(k>>(64*i)&MASK for i in range(4))),idx,neg,ref,badidx,badneg)
  m=2*(k%N)%N;sign=1 if m&1 else -1
  if sign<0:m=N-m
  for c,w in enumerate([17]*12+[26]*2):
   e=sign*(m if c==13 else (m&((1<<(w+1))-1))-(1<<w))
   assert ref[c]==e and idx[c]==(abs(e)-1)//2 and neg[c]==(e<0) and idx[c]<(1<<(w-1));m=(m>>w)|1
   counts['direct_digits']+=1
  assert idx[12]==idx[14] and neg[12]==neg[14] and idx[13]==idx[15] and neg[13]==neg[15]
  counts['preserved_cold_digits']+=2;counts['scalars']+=1
  counts['wrong_start_controls_caught']+=any(idx[c]!=badidx[c] or neg[c]!=badneg[c] for c in range(14))
  counts['wrong_last_controls_caught']+=(idx[13],neg[13])!=(badidx[14],badneg[14])
assert counts['wrong_start_controls_caught']>0 and counts['wrong_last_controls_caught']>0
assert source_identity(base)==identity
report={'status':'PASS',**identity,'counts':counts,'actual_helper_sha256':{n:sha(b) for n,b in helpers.items()},'actual_chain_sha256':sha(chain),'projection_sha256':sha(code),'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'validation_level':'Actual extracted donor helper, setup, unchanged reference peel and cold helper on CPU with UBSan; independent Python regular recoder','gpu_executed':False,'limits':'Finite boundary/random full-uint256 samples and source algebra; not exhaustive enumeration or CUDA execution. Whole point-chain/recovery checked separately.'}
args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'PASS','source_fingerprint':identity['source_fingerprint'],'counts':counts},indent=2))
