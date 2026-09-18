#!/usr/bin/env python3
"""Execute actual three-field pipeline expressions and collectives on the CPU.

OpenSSL replaces field primitives; a controlled XYZZ producer replaces EC/SHA
coupling. Actual SHA, hit packing and source kernel bodies remain. No GPU claim.
"""
import ctypes as C,hashlib,json,random,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;BASE=HERE/'full-rebuild-candidate';PIN=HERE.parents[1]
sys.path.insert(0,str(PIN.parent/'subset'))
from check_candidate import BACKEND,function
from check_deferred_source import FIELD
import audit_integrated as ref
P=ref.P
U64=C.c_uint64;U32=C.c_uint32
def buf(values):return (U64*(4*len(values)))(*[x>>(64*i)&((1<<64)-1) for x in values for i in range(4)])
def integer(v):return sum(int(x)<<(64*i) for i,x in enumerate(v))
STUBS=r'''
#define __launch_bounds__(...)
#define QSB_ZEROS_N 6
#define QSB_CHECKPOINT_NODES 254
#define QSB_CHECKPOINT_STRIDE 256
struct alignas(16) ulonglong2{uint64_t x,y;};
static ulonglong2 make_ulonglong2(uint64_t x,uint64_t y){return {x,y};}
static void _ModSub256(uint64_t*r,uint64_t*b){field_op(r,r,b,2);}
static uint32_t atomicAdd(uint32_t*r,uint32_t n){return __atomic_fetch_add(r,n,__ATOMIC_RELAXED);}
static uint32_t pin_tail_words[3]={0,0,0x01800000};
static uint64_t pin_u2rx_words[4],pin_u2ry_words[4];
static const uint64_t *input_points;
static void _FixedBaseSignedXYZZScalar(uint64_t*X,uint64_t*Y,uint64_t*A,uint64_t*B,const uint64_t*,const uint8_t*){
 const uint64_t*p=input_points+(blockIdx.x*256+threadIdx.x)*16;
 Load256(X,p);Load256(Y,p+4);Load256(A,p+8);Load256(B,p+12);
}
'''
WRAPPER=r'''
extern "C" uint32_t recover(const uint64_t*point,const uint64_t*rx,const uint64_t*ry,uint64_t*out){
 uint64_t X[4],Y[4],A[4],B[4],W[5],D[5],xR[4],yR[4];
 Load256(X,point);Load256(Y,point+4);Load256(A,point+8);Load256(B,point+12);Load256(xR,rx);Load256(yR,ry);
 qsb_xyzz_finish_prepare(X,A,xR,W);_ModMult(D,W,B);D[4]=0;_ModInv(D);
 return qsb_xyzz_finish_three(Y,W,B,D,xR,yR,out,out+4);
}
extern "C" int pipeline(const uint64_t *points,const uint64_t *rx,const uint64_t *ry,int count,uint32_t *hits,uint32_t *hit_count){
 input_points=points;memcpy(pin_u2rx_words,rx,32);memcpy(pin_u2ry_words,ry,32);
 int blocks=(count+255)/256,groups=(blocks+255)/256;
 std::vector<ulonglong2> saved(count*6+2);saved.back()={0x12345678,0x87654321};
 std::vector<uint64_t> roots(blocks*4),super(groups*4),root_tree(groups*4*256);
 uint32_t mid[8]={};*hit_count=0;inverse_count=0;
 for(int b=0;b<blocks;b++)launch(256,[&](int){blockIdx.x=b;kernel_pinning_pipeline<true,0>(mid,nullptr,0,0,0,0,0x80000000,500000000,nullptr,rx,ry,nullptr,nullptr,nullptr,hit_count,hits,count,0,1,0,saved.data(),roots.data(),nullptr);});
 for(int b=0;b<groups;b++)launch(256,[&](int){blockIdx.x=b;qsb_root_group_prepare(roots.data(),blocks,super.data(),root_tree.data());});
 launch(256,[&](int){qsb_invert_super_roots(super.data(),groups);});
 for(int b=0;b<groups;b++)launch(256,[&](int){blockIdx.x=b;qsb_root_group_finish(roots.data(),blocks,super.data(),root_tree.data());});
 for(int b=0;b<blocks;b++)launch(256,[&](int){blockIdx.x=b;kernel_pinning_pipeline<true,2>(mid,nullptr,0,0,0,0,0x80000000,500000000,nullptr,rx,ry,nullptr,nullptr,nullptr,hit_count,hits,count,0,1,0,saved.data(),roots.data(),nullptr);});
 require(saved.back().x==0x12345678&&saved.back().y==0x87654321);
 return inverse_count;
}
extern "C" int tree(const uint64_t *in,uint64_t*out,int *counts){
 uint64_t root[5]={};multiply_count=0;inverse_count=0;
 launch(256,[&](int i){uint64_t v[5]={};Load256(v,in+4*i);qsb_block_product_only(v,root);});
 counts[0]=multiply_count;_ModInv(root);multiply_count=0;
 launch(256,[&](int i){uint64_t v[5]={};Load256(v,in+4*i);qsb_block_inverse_rebuild(v,root);memcpy(out+4*i,v,32);});
 counts[1]=multiply_count;return inverse_count;
}
'''
src=(BASE/'pinning.cu').read_text();helper=(BASE/'three_state.cuh').read_text()
identities={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.iterdir() if p.suffix in ('.h','.cuh','.cu')}
functions=[function(src,sig) for sig in (
 '__device__ __forceinline__ void qsb_field_normalize(',
 '__device__ __forceinline__ void qsb_block_inverse(',
 '__device__ __forceinline__ void qsb_block_product_checkpoint(',
 '__device__ __forceinline__ void qsb_block_inverse_checkpoint(',
 '__global__ void __launch_bounds__(256,2) qsb_root_group_prepare(',
 '__global__ void __launch_bounds__(256,1) qsb_invert_super_roots(',
 '__global__ void __launch_bounds__(256,2) qsb_root_group_finish(',
 '__device__ __forceinline__ void qsb_xyzz_finish_prepare(',
 '__device__ int gpu_is_der_easy(',
 '__device__ __forceinline__ int gpu_bench_valid_words(')]
sha=(BASE/'GPUHash.h').read_text().split('//Modified SHA256 function')[0]
kernel='template<bool FAST_TAIL,int STAGE>\n'+function(src,'__global__ void __launch_bounds__(256, STAGE == 0 ? 2 : 3) kernel_pinning_pipeline(')
code='\n'.join([BACKEND,FIELD,STUBS,sha,*functions,helper,kernel,WRAPPER])
rng=random.Random(9617);counts={}
with tempfile.TemporaryDirectory(prefix='qsb-three-state-') as td:
 p=Path(td);cpp=p/'audit.cpp';so=p/'audit.so';cpp.write_text(code)
 subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC','-pthread','-Wno-pragma-once-outside-header','-I/opt/homebrew/opt/openssl@3/include','-L/opt/homebrew/opt/openssl@3/lib',str(cpp),'-lcrypto','-o',str(so)],check=True)
 lib=C.CDLL(str(so));ptr=C.POINTER(U64)
 lib.recover.argtypes=[ptr,ptr,ptr,ptr];lib.recover.restype=U32
 lib.tree.argtypes=[ptr,ptr,C.POINTER(C.c_int)]
 lib.pipeline.argtypes=[ptr,ptr,ptr,C.c_int,C.POINTER(U32),C.POINTER(U32)]
 # Includes arbitrary affine pairs, zero parities and near-p field values.
 for i in range(5000):
  x,y,xr,yr,z=[rng.randrange(1,P) for _ in range(5)]
  if i<12:y=[0,1,P-1][i%3];yr=[0,1,P-1][(i//3)%3]
  if x==xr:continue
  a=z*z%P;b=a*z%P;point=[x*a%P,y*b%P,a,b];out=(U64*8)()
  parity=lib.recover(buf(point),buf([xr]),buf([yr]),out)
  want=[]
  for rY in (yr,-yr%P):
   m=(rY-y)*pow(xr-x,-1,P)%P;xx=(m*m-x-xr)%P;yy=(m*(xr-xx)-rY)%P;want.append((xx,yy))
  assert [integer(out[:4]),integer(out[4:])]==[q[0] for q in want]
  assert parity==((want[0][1]&1)|((want[1][1]&1)<<1))
 counts['arbitrary_scaled_recovery_cases']=5000
 for i in range(8):
  vals=[rng.randrange(1,P) for _ in range(256)];vals[0]=1;vals[-1]=P-1
  if i==0:vals=[1]*256
  out=(U64*1024)();ops=(C.c_int*2)();assert lib.tree(buf(vals),out,ops)==1
  assert list(ops)==[255,764],list(ops)
  assert all(integer(out[4*j:4*j+4])==pow(v,-1,P) for j,v in enumerate(vals))
 counts['leaf_inverse_cases']=8*256;counts['tree_product_counts']={'prepare':255,'finish':764}
 R=ref.scalar_mult(1984321);pool=[ref.scalar_mult(rng.randrange(1,ref.N)) for _ in range(32)]
 import hashlib as h
 for n in (1,31,32,255,256,257,511,513):
  points=[];expected=[];padded=(n+255)//256*256
  for idx in range(padded):
   q=R if idx%31==0 else pool[idx%32];z=rng.randrange(1,P);a=z*z%P;b=a*z%P
   points.extend([q[0]*a%P,q[1]*b%P,a,b])
   if idx>=n or q==R:continue
   for ri,rr in enumerate((R,(R[0],-R[1]%P))):
    result=ref.affine_add(q,rr);digest=h.sha256(bytes([2+(result[1]&1)])+result[0].to_bytes(32,'big')).digest()
    if digest[0]>>2==0:expected.append((500000000+idx)|(ri<<31));break
  hits=(U32*1024)();nh=U32();assert lib.pipeline(buf(points),buf([R[0]]),buf([R[1]]),n,hits,C.byref(nh))==1
  assert sorted(hits[:nh.value])==sorted(expected),(n,list(hits[:nh.value]),expected)
 counts['pipeline_sizes']=[1,31,32,255,256,257,511,513]
 counts['pipeline_active_candidates']=sum(counts['pipeline_sizes'])
assert all(hashlib.sha256((BASE/n).read_bytes()).hexdigest()==v for n,v in identities.items())
report={'status':'PASS','gpu_executed':False,'source_sha256':identities,'counts':counts,
 'limits':'Actual source kernel/control/collectives on CPU threads with OpenSSL fields and controlled EC producer. SHA uses CPU header path. No CUDA/PTX arithmetic, GPU ordering, memory traffic or speed result.'}
(HERE/'full-rebuild-cpu.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
