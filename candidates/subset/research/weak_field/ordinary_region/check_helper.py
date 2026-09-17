#!/usr/bin/env python3
"""Actual weak helper/host primitives versus independent synthetic field polynomials."""
import argparse
import ctypes as CT
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research')]
from preflight import source_identity
from ptx_field_model import function
B=1<<256;P=B-(1<<32)-977;U64=CT.c_uint64
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def words(x):return [(x>>(64*i))&((1<<64)-1) for i in range(4)]
def integer(w):return sum(int(v)<<(64*i) for i,v in enumerate(w))
def oracle(v,defer):
    X,Y,ZZ,ZZZ,x,y,off=[z%P for z in v]
    U=x*ZZ%P;S=(y+off)*ZZZ%P;h=(U-X)%P;r=(S-Y)%P
    pp=h*h%P;ppp=h*pp%P;q=U*pp%P
    nx=(r*r+ppp-2*q)%P;nzz=ZZ*pp%P;nzzz=ZZZ*ppp%P
    ny=(r*(q-nx)-(0 if defer else y*nzzz))%P
    return [nx,ny,nzz,nzzz]

PRE=r'''
#include <cstdint>
#include <cstring>
#define __device__
#define __forceinline__ inline
#include "qsb_weak_addsub.cuh"
#include "qsb_weak_product.cuh"
static void Load256(uint64_t*r,const uint64_t*a){memmove(r,a,32);}
// Only called with canonical affine Y2/Yoff. Deliberately explicit shim.
static void _ModAdd256(uint64_t*r,uint64_t*a,uint64_t*b){qsb_weak_add(r,a,b);qsb_weak_normalize(r);}
'''
WRAP=r'''
static uint64_t marker(int row,int slot){return 0x1234567890abcdefULL^((uint64_t)row<<32)^(uint64_t)slot;}
extern "C" int audit(const uint64_t*input,unsigned owner,int defer,int alias,uint64_t*out){
  if(owner>=192)return 10;
  struct Guard{uint64_t before[4],arena[4][768],after[4];}guard;
  for(int i=0;i<4;i++)guard.before[i]=guard.after[i]=0xcafedecafebeeffeULL;
  for(int k=0;k<4;k++)for(int j=0;j<768;j++)guard.arena[k][j]=marker(k,j);
  struct State{uint64_t before[4],X[4],Y[4],after[4];}s;
  for(int i=0;i<4;i++)s.before[i]=s.after[i]=0xaaff55ffbb667788ULL;
  memcpy(s.X,input,32);memcpy(s.Y,input+4,32);
  uint64_t x[4],y[4],off[4];memcpy(x,input+16,32);memcpy(y,input+20,32);memcpy(off,input+24,32);
  const uint64_t *xp=x,*yp=y,*op=off;
  if(alias==1)xp=s.X;
  if(alias==2)yp=s.Y;
  if(alias==3)op=s.Y;
  if(alias==4){xp=s.X;yp=s.Y;}
  if(alias==5)op=y;
  for(int k=0;k<4;k++){guard.arena[k][384+owner]=input[8+k];guard.arena[k][576+owner]=input[12+k];}
  qsb_ec192_PointAddXYZZ_shared_z_weak(s.X,s.Y,guard.arena,owner,xp,yp,op,defer!=0);
  for(int i=0;i<4;i++)if(guard.before[i]!=0xcafedecafebeeffeULL||guard.after[i]!=0xcafedecafebeeffeULL||s.before[i]!=0xaaff55ffbb667788ULL||s.after[i]!=0xaaff55ffbb667788ULL)return 11;
  for(int k=0;k<4;k++)for(unsigned j=0;j<768;j++)
    if(j!=384+owner&&j!=576+owner&&guard.arena[k][j]!=marker(k,j))return 12;
  memcpy(out,s.X,32);memcpy(out+4,s.Y,32);
  for(int k=0;k<4;k++){out[8+k]=guard.arena[k][384+owner];out[12+k]=guard.arena[k][576+owner];}
  memcpy(out+16,out,128);
  qsb_weak_normalize(out+16);qsb_weak_normalize(out+20);
  qsb_weak_normalize(out+24);qsb_weak_normalize(out+28);
  return 0;
}
'''

def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=HERE/'candidate')
    parser.add_argument('--output',type=Path,default=HERE/'helper-results.json')
    args=parser.parse_args();source=args.source.resolve();identity=source_identity(source)
    header=(source/'tests/gpu_epochs/compact_table_device.cuh').read_text()
    helper=function(header,'__device__ void qsb_ec192_PointAddXYZZ_shared_z_weak(')
    assert helper.count('_ModAdd256(')==1 and helper.count('qsb_weak_add(')==1
    dependencies={n:sha(source/n) for n in ['qsb_weak_addsub.cuh','qsb_weak_product.cuh']}
    rng=random.Random(0x192FA)
    edge=[0,1,2,(1<<32)-1,(1<<32)+977,(1<<64)-1,1<<64,1<<128,1<<192,P-65537,P-65536,P-2,P-1,P,P+1,B-2,B-1]
    cases=[[0,1,0,0,0,0,0]] # deferred Y becomes weak p+1: boundary sentinel.
    for a in edge:
        cases.append([a,a,a,a,a%P,a%P,a%P])
        for j in range(4):
            v=[1,2,3,4,5,6,7];v[j]=a;cases.append(v)
    for a in edge:
        for b in edge:cases.append([a,b,b,a,a%P,b%P,(a+b)%P])
    # All 16 lifts of four small residues into their equivalent p+x encodings.
    for mask in range(16):
        cases.append([i+((P if mask>>i&1 else 0)) for i in range(4)]+[5,6,7])
    cases += [[rng.getrandbits(256) for _ in range(4)]+[rng.randrange(P) for _ in range(3)] for _ in range(1200)]
    tests=[]
    for i,v in enumerate(cases):
        for defer in (0,1):tests.append((v,defer,0,len(tests)%192))
        # Alias only affine inputs with canonical initial output values; output
        # X/Y arrays remain distinct, as required by the real helper ABI.
        if i<450:
            for alias in range(1,6):
                w=list(v)
                if alias in (1,4):w[0]%=P;w[4]=w[0]
                if alias in (2,4):w[1]%=P;w[5]=w[1]
                if alias==3:w[1]%=P;w[6]=w[1]
                if alias==5:w[6]=w[5]
                for defer in (0,1):tests.append((w,defer,alias,len(tests)%192))
    sources={'positive':PRE+helper+WRAP,
      'drop_affine_anchor':PRE+helper.replace('_ModAdd256(S2, (uint64_t *)Y2, (uint64_t *)Yoff);','Load256(S2,Y2);')+WRAP,
      'omit_ZZ_update':PRE+helper.replace('qsb_weak_mul(ZZ1, PP);','/* intentionally omitted ZZ update */')+WRAP,
      'omit_deferred_Y_boundary_normalization':PRE+helper+WRAP.replace('qsb_weak_normalize(out+20);','/* omitted Y normalization */')}
    assert len(set(sources.values()))==len(sources)
    results={};weak_output_cases=0;owner_seen=set();bool_seen=set();alias_seen=set()
    with tempfile.TemporaryDirectory(prefix='qsb-weak-helper-') as td:
      for name,cpp in sources.items():
        path=Path(td)/(name+'.cpp');libpath=Path(td)/(name+'.so');path.write_text(cpp)
        compile_command=['c++','-std=c++17','-O2','-shared','-fPIC','-I',str(source),str(path),'-o',str(libpath)]
        built=subprocess.run(compile_command,capture_output=True,text=True,timeout=60)
        assert built.returncode==0,(name,built.stderr)
        lib=CT.CDLL(str(libpath));lib.audit.argtypes=[CT.POINTER(U64),CT.c_uint,CT.c_int,CT.c_int,CT.POINTER(U64)];lib.audit.restype=CT.c_int
        checked=0;failure=None
        for i,(v,defer,alias,owner) in enumerate(tests):
          expected=oracle(v,defer)
          inp=(U64*28)(*[w for x in v for w in words(x)]);out=(U64*32)()
          rc=lib.audit(inp,owner,defer,alias,out)
          raw=[integer(out[j*4:j*4+4]) for j in range(4)]
          normalized=[integer(out[16+j*4:20+j*4]) for j in range(4)]
          if rc or [x%P for x in raw]!=expected or normalized!=expected:
            failure={'case':i,'diagnostic':'CANARY_OR_OWNERSHIP_FAILURE' if rc else 'POLYNOMIAL_OR_BOUNDARY_MISMATCH','returncode':rc,'defer':defer,'alias':alias,'owner':owner,'input':[hex(x) for x in v],'expected':[hex(x) for x in expected],'raw':[hex(x) for x in raw],'normalized':[hex(x) for x in normalized]};break
          checked+=1
          if name=='positive':
            weak_output_cases+=any(x>=P for x in raw);owner_seen.add(owner);bool_seen.add(defer);alias_seen.add(alias)
        if name=='positive':assert failure is None,failure
        else:assert failure is not None,(name,'mutation survived')
        results[name]={'compile_exit':0,'cases_checked_before_failure':checked,'generated_cpp_sha256':hashlib.sha256(cpp.encode()).hexdigest(),'failure':failure}
    assert owner_seen==set(range(192)) and bool_seen=={0,1} and alias_seen==set(range(6)) and weak_output_cases>0
    assert source_identity(source)==identity
    report={'status':'PASS_ACTUAL_HOST_WEAK_HELPER_SYNTHETIC_POLYNOMIAL',**identity,
      'checker_sha256':sha(Path(__file__)),'actual_helper_sha256':hashlib.sha256(helper.encode()).hexdigest(),'actual_header_sha256':dependencies,
      'ordinary_cases':len(cases),'total_calls':len(tests),'owners_checked':192,'defer_modes':[False,True],'alias_modes':{'0':'separate affine/state buffers','1':'X2 aliases X','2':'Y2 aliases Y','3':'Yoff aliases Y','4':'X2/Y2 alias X/Y respectively','5':'Yoff aliases Y2'},
      'weak_output_cases':weak_output_cases,'results':results,
      'actual_host_primitive_scope':'qsb_weak_product.cuh and qsb_weak_addsub.cuh included unchanged; Load256 is memmove and the one canonical affine-Y sum is weak-add+normalize. Synthetic polynomial inputs, not necessarily curve points.',
      'boundary_scope':'Actual qsb_weak_normalize on all four returned state fields, checked against canonical polynomial values. No final guarded helper or recovery execution here.',
      'limitations':['No CUDA/PTX primitive execution in this checker','No full curve or synchronization proof','No GPU throughput','Distinct X/Y output arrays are an ABI precondition; tested aliases concern affine input pointers only'],
      'gpu_executed':False,'candidate_modified':False}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','source_fingerprint','ordinary_cases','total_calls','weak_output_cases','owners_checked']}))
if __name__=='__main__':run()
