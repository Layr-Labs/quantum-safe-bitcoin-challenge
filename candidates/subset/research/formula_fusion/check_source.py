#!/usr/bin/env python3
"""Execute extracted cubic recovery with actual host multiply/square primitives."""
import argparse
import ctypes as CT
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT));sys.path.insert(0, str(ROOT/'research/wide_windows'))
from preflight import source_identity
from audit_support import BACKEND, function
from check_algebra import P, C, G, add, current

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,default=HERE/'candidate')
parser.add_argument('--report',type=Path,help='Defaults to source-results.json for the original candidate; external sources require a separate explicit report path.')
args=parser.parse_args()
BASE=args.source.resolve()
original_source=(HERE/'candidate').resolve()
original_report=(HERE/'source-results.json').resolve()
if args.report is None:
    if BASE!=original_source:parser.error('An external --source requires a separate --report path to preserve the original report.')
    report_path=original_report
else:
    report_path=args.report.resolve()
    if BASE!=original_source and report_path==original_report:
        parser.error('An external --source cannot overwrite the original source-results.json.')
identity = source_identity(BASE)
tree = (BASE/'tests/gpu_epochs/tree.cu').read_text()
math_src = (BASE/'GPUMath.h').read_text()
square = (BASE/'square32.cuh').read_text()
prepare = function(tree,'__device__ __forceinline__ void qsb_xyzz_finish_prepare(')
finish = function(tree,'__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed(')
host = function(tree,'static void qsb_prepare_cubic_constant(')
# The first projection accidentally invented a two-argument addition overload;
# the native compiler caught it. Bind accepted arities to actual declarations.
for name in ['_ModAdd256','_ModSub256','_ModMult','_ModSqr']:
    declarations = re.findall(r'\bvoid\s+'+name+r'\s*\(([^)]*)\)', math_src)
    allowed = {len(params.split(',')) for params in declarations}
    assert allowed, name
    for arguments in re.findall(r'\b'+name+r'\s*\(([^)]*)\)', prepare+finish):
        assert len(arguments.split(',')) in allowed, (name,arguments,allowed)
assert not re.search(r'\bvoid\s+_ModAdd256\s*\([^,]+,[^,]+\)', math_src)
backend = BACKEND
for signature in ['static void _ModMult(uint64_t *r,const uint64_t *a,const uint64_t *b)',
                  'static void _ModMult(uint64_t *r,const uint64_t *b)',
                  'static void _ModSqr(uint64_t *r,const uint64_t *a)']:
    backend = backend.replace(function(backend,signature),'')
code = backend + r'''
#define __device__
#define __forceinline__ inline
static uint64_t QSB_U2R_C3X2[4];
static int multiplications,squares,uploads;
static void Load256(uint64_t*r,const uint64_t*a){memcpy(r,a,32);}
using cudaError_t=int;constexpr int cudaSuccess=0;
template<class T> int cudaMemcpyToSymbol(T& symbol,const void*input,size_t bytes){
 require(bytes==32 && (void*)&symbol==(void*)QSB_U2R_C3X2);memcpy(&symbol,input,bytes);++uploads;return 0;
}
static void compact_require(bool value,const char*){require(value);}
static void wide_cuda_require(int value,const char*){require(value==0);}
'''
code += function(math_src,'__device__ __forceinline__ void _ModMultCore(')
code += function(square,'__device__ __forceinline__ void qsb_square32(')
code += r'''
static void _ModMult(uint64_t*r,const uint64_t*a,const uint64_t*b){++multiplications;_ModMultCore(r,a,b);}
static void _ModMult(uint64_t*r,const uint64_t*b){++multiplications;_ModMultCore(r,r,b);}
static void _ModSqr(uint64_t*r,const uint64_t*a){++squares;qsb_square32(r,a);}
'''
bad_marker = '_ModSub256(x1, W, m);'
assert finish.count(bad_marker)==1
mutant = finish.replace('qsb_xyzz_finish_precomputed(', 'mutant_finish(').replace(bad_marker,'_ModAdd256(x1, W, m);')
code += host + prepare + finish + mutant
code += r'''
extern "C" int recover(const uint64_t *state,const uint64_t *point,uint64_t *out,int *cost,int mutation){
 ctx=BN_CTX_new();prime=nullptr;
 require(BN_hex2bn(&prime,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F")!=0);
 multiplications=squares=inversions=uploads=0;
 qsb_prepare_cubic_constant((const uint8_t*)point);
 uint64_t expected_constant[4];Load256(expected_constant,QSB_U2R_C3X2);
 uint64_t values[16],r[8],w[5],inv[5],scratch[4]={};
 memcpy(values,state,sizeof(values));memcpy(r,point,sizeof(r));
 qsb_xyzz_finish_prepare(values,values+8,values+12,r,w);
 int result=-1;
 if(w[0]|w[1]|w[2]|w[3]){
  memcpy(inv,w,sizeof(w));_ModInv(inv);
  result=mutation?mutant_finish(values+8,values+4,scratch,values+12,inv,r,r+4,out,out+4):
                  qsb_xyzz_finish_precomputed(values+8,values+4,scratch,values+12,inv,r,r+4,out,out+4);
 }
 cost[0]=multiplications;cost[1]=squares;cost[2]=inversions;cost[3]=uploads;
 memcpy(out+8,expected_constant,32);
 BN_free(prime);BN_CTX_free(ctx);return result;
}
'''

U64 = CT.c_uint64
def words(values): return (U64*(4*len(values)))(*(x>>(64*i)&((1<<64)-1) for x in values for i in range(4)))
def integers(values): return [sum(int(values[4*j+i])<<(64*i) for i in range(4)) for j in range(len(values)//4)]
counts = {'curve_recoveries':0,'affine_keys':0,'singular_cases':0,'mutations_caught':0,'runtime_constants':0}
rng = random.Random(2026091733)
points=[G]
for _ in range(255):points.append(add(points[-1],G))
with tempfile.TemporaryDirectory(prefix='qsb-cubic-source-') as td:
    cpp=Path(td)/'source.cpp';libfile=Path(td)/'source.so';cpp.write_text(code)
    subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC','-fsanitize=undefined',
                    '-fsanitize-trap=undefined','-I/opt/homebrew/opt/openssl@3/include',
                    '-L/opt/homebrew/opt/openssl@3/lib',str(cpp),'-lcrypto','-o',str(libfile)],check=True)
    lib=CT.CDLL(str(libfile));lib.recover.argtypes=[CT.POINTER(U64),CT.POINTER(U64),CT.POINTER(U64),CT.POINTER(CT.c_int),CT.c_int]
    def run(state,r,mutation=0):
        out=(U64*12)();cost=(CT.c_int*4)();parity=lib.recover(words(state),words(r),out,cost,mutation)
        x1,x2,constant=integers(out)
        assert constant==3*r[0]*r[0]%P and cost[3]==1
        return parity,x1,x2,list(cost)
    for i in range(4096):
        p=rng.choice(points);r=rng.choice(points)
        if i&1:p=(p[0],-p[1]%P)
        if i&2:r=(r[0],-r[1]%P)
        z=[1,P-1,P-2,P-C,rng.randrange(1,P)][i%5]
        a=z*z%P;b=a*z%P;state=(p[0]*a%P,p[1]*b%P,a,b)
        actual,x1,x2,cost=run(state,r);counts['runtime_constants']+=1
        reference,_=current(*state,r)
        if reference is None:
            assert actual==-1 and cost[:3]==[2,0,0];counts['singular_cases']+=1;continue
        q1=add(p,r);q2=add(p,(r[0],-r[1]%P))
        want=((q1[1]&1)|((q2[1]&1)<<1),q1[0],q2[0])
        assert (actual,x1,x2)==want and reference==(q1,q2)
        assert cost[:3]==[10,1,1]
        bad=run(state,r,1)
        assert bad[:3]!=want
        counts['mutations_caught']+=1;counts['curve_recoveries']+=1;counts['affine_keys']+=2
    for r in points[:32]:
        for state in [(r[0],r[1],1,1),(r[0],-r[1]%P,1,1),(0,0,0,0)]:
            result=run(state,r);assert result[0]==-1 and result[3][:3]==[2,0,0]
            counts['singular_cases']+=1;counts['runtime_constants']+=1
assert identity==source_identity(BASE)
report={'status':'PASS',**identity,**counts,
        'validation_level':'Actual extracted recovery and host constant helper; actual corrected host multiply/square; OpenSSL add/sub/inverse; independent affine oracle',
        'per_recovery_cost':{'M':10,'S':1,'inversions':1},
        'helper_sha256':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in [('prepare',prepare),('finish',finish),('host_constant',host)]},
        'projection_sha256':hashlib.sha256(code.encode()).hexdigest(),
        'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sanitizer':'CPU UBSan trap mode; no dynamic sanitizer runtime',
        'gpu_executed':False,'limits':'No CUDA collective execution, full hash pipeline, PTX scheduling or throughput. Frozen base covers unchanged guarded chain and field carry regression suites.'}
report_path.parent.mkdir(parents=True,exist_ok=True)
report_path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
