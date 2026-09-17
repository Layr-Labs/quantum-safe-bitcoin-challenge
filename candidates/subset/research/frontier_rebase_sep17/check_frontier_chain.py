#!/usr/bin/env python3
"""Execute actual selected 15-window source chain and loader against OpenSSL.
Weak headers, when present, execute their actual host branches. Canonical field
operations and inverse use OpenSSL. No CUDA, PTX, synchronization or speed test.
"""
import argparse,ctypes as C,hashlib,json,random,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'research/wide_windows')]
from preflight import source_identity
from audit_support import BACKEND,function
N=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
U=1<<256;P=U-(1<<32)-977;MASK=(1<<64)-1
limbs=lambda x:(C.c_uint64*4)(*(x>>(64*i)&MASK for i in range(4)))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--mutation',choices=['unguarded-final','weak-zz']);a=ap.parse_args();src=a.source.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 ident=source_identity(src);tree=(src/'tests/gpu_epochs/tree.cu').read_text();math=(src/'GPUMath.h').read_text();parts={}
 def get(text,sig):
  f=function(text,sig);parts[sig]=hashlib.sha256(f.encode()).hexdigest();return f
 geom=tree[tree.index('#ifndef ZLAB_T14'):tree.index('/* n = secp256k1 group order')]
 order=tree[tree.index('__device__ __constant__ uint64_t GT_ORDER_N'):tree.index('};',tree.index('__device__ __constant__ uint64_t GT_ORDER_N'))+2]
 base=BACKEND+'''\n#include <sys/mman.h>
#define __device__
#define __host__
#define __constant__
#define __forceinline__ inline
#define ZLAB_T14 0
#define ZLAB_DIRDIG 1
static void Load256(uint64_t*r,const uint64_t*a){memcpy(r,a,32);}
static uint64_t QSB_U2R_C[4];
struct alignas(16) ulonglong2 {uint64_t x,y;};
template<class T> static T __ldg(const T*p){return *p;}
static uint64_t cc;
#define UADDO1(x,y) do{__uint128_t t=(__uint128_t)(x)+(y);(x)=(uint64_t)t;cc=t>>64;}while(0)
#define UADDC1(x,y) do{__uint128_t t=(__uint128_t)(x)+(y)+cc;(x)=(uint64_t)t;cc=t>>64;}while(0)
#define UADD1(x,y) do{(x)+=(y)+cc;}while(0)
static EC_GROUP* group;static BIGNUM *ord,*halfbase;static uint8_t* table;static int load_count;static uint64_t norms;
'''+geom+order
 for sig in ['__device__ __forceinline__ void gt_recode_setup(','__device__ __forceinline__ uint32_t gt_field_bits_v(','__host__ __device__ __forceinline__ unsigned gt_width(','__device__ __forceinline__ void gt_direct_digit(']:base+='\n'+get(tree,sig)
 # Execute exact vector/address/signed loader. Point entries are supplied sparsely
 # by independent EC multiplication, not by any candidate table-builder formula.
 loader=get(tree,'__device__ __forceinline__ void gt_load_signed_flat(')
 base+='\n'+loader.replace('gt_load_signed_flat(','actual_gt_load_signed_flat(',1)
 base+='''
static void gt_load_signed_flat(const uint8_t*,uint32_t offset,uint32_t idx,uint64_t neg,uint64_t*x,uint64_t*y){
 int c=load_count++;require(c<15);require(offset==gt_offset(c)&&idx<gt_entries(c)&&neg<=1);
 require(gt_entries(c)==(c==0?131072:65536));require(gt_shift(c)==(c?17*c+1:0));
 BIGNUM *k=BN_new(),*xx=BN_new(),*yy=BN_new();EC_POINT *pt=EC_POINT_new(group);
 require(BN_set_word(k,2ull*idx+1));require(BN_lshift(k,k,gt_shift(c)));require(BN_mod_mul(k,k,halfbase,ord,ctx));require(EC_POINT_mul(group,pt,k,nullptr,nullptr,ctx));require(EC_POINT_get_affine_coordinates(group,pt,xx,yy,ctx));
 uint64_t entry[8];write256(entry,xx);write256(entry+4,yy);size_t off=((size_t)offset+idx)*64;require(off+64<=64ull*1024*1024);memcpy(table+off,entry,64);
 actual_gt_load_signed_flat(table,offset,idx,neg,x,y);
 if(neg)require(BN_mod_sub(yy,prime,yy,prime,ctx));uint64_t expected[8];write256(expected,xx);write256(expected+4,yy);require(!memcmp(x,expected,32)&&!memcmp(y,expected+4,32));
 BN_free(k);BN_free(xx);BN_free(yy);EC_POINT_free(pt);
}
'''+get(tree,'__device__ __forceinline__ void gt_load_signed(')
 # Keep actual primitive bodies; canonical API shims have exact available arities.
 weak=(src/'qsb_weak_product.cuh').exists()
 if weak:
  add=(src/'qsb_weak_addsub.cuh').read_text();prod=(src/'qsb_weak_product.cuh').read_text();base+=add.replace('qsb_weak_normalize(','qsb_weak_normalize_actual(')+'\n'+prod+'\nstatic void qsb_weak_normalize(uint64_t*r){++norms;qsb_weak_normalize_actual(r);}\n'
 # Template plus any boolean overload (both source spellings occur in controls).
 pos=math.index('void _PointAddXYZZ_def(');start=math.rfind('template<bool DEFER_Y>',0,pos)
 if start>=0 and pos-start<150:
  f=get(math,math[start:pos]+'void _PointAddXYZZ_def(');base+=f
 else:base+=get(math,'__device__ void _PointAddXYZZ_def(')
 if 'qsb_frontier_PointAddXYZZ_weak' in math:
  pos=math.index('void qsb_frontier_PointAddXYZZ_weak(');start=math.rfind('template<bool DEFER_Y>',0,pos);f=get(math,math[start:pos]+'void qsb_frontier_PointAddXYZZ_weak(')
  if a.mutation=='weak-zz':
   assert 'qsb_weak_mul(ZZ1, PP);' in f;f=f.replace('qsb_weak_mul(ZZ1, PP);','/* invalid omitted ZZ update */',1)
  base+=f
 seed_sig='__device__ void _PointAddXYZZ_mm_def('
 if seed_sig not in math:seed_sig='__device__ __forceinline__ void _PointAddXYZZ_mm_def('
 base+=get(math,seed_sig)
 guard_src=math if 'qsb_asym_last_add(' in math else tree
 if 'qsb_asym_last_add(' in guard_src:base+=get(guard_src,'__device__ __forceinline__ void qsb_asym_last_add(')
 chain=get(tree,'__device__ void _FixedBaseSignedXYZZStream(')
 if a.mutation=='unguarded-final':
  assert 'qsb_asym_last_add(' in chain;chain=chain.replace('qsb_asym_last_add(', '_PointAddXYZZ_def<false>(')
 base+=chain
 for sig in ['__device__ __forceinline__ void qsb_xyzz_finish_prepare(','__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed(']:base+=get(tree,sig)
 base+=r'''
extern "C" int audit_chain(const uint64_t*k,const uint64_t*coef){
 ctx=BN_CTX_new();group=EC_GROUP_new_by_curve_name(NID_secp256k1);prime=BN_new();ord=BN_new();halfbase=BN_new();require(EC_GROUP_get_curve(group,prime,nullptr,nullptr,ctx));require(EC_GROUP_get_order(group,ord,ctx));
 table=(uint8_t*)mmap(nullptr,64ull*1024*1024,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);require(table!=MAP_FAILED);load_count=0;
 BIGNUM *two=BN_new(),*bs=read256(coef),*scalar=read256(k),*wk=BN_new();BN_set_word(two,2);require(BN_mod_inverse(halfbase,two,ord,ctx));require(BN_mod_mul(halfbase,halfbase,bs,ord,ctx));require(BN_mod_mul(wk,scalar,bs,ord,ctx));
 EC_POINT *want=EC_POINT_new(group);require(EC_POINT_mul(group,want,wk,nullptr,nullptr,ctx));uint64_t state[16];_FixedBaseSignedXYZZStream(state,state+4,state+8,state+12,k,table);require(load_count==15);
 BIGNUM *xx=read256(state),*yy=read256(state+4),*zz=read256(state+8),*zzz=read256(state+12),*wx=BN_new(),*wy=BN_new();int inf=EC_POINT_is_at_infinity(group,want),ok=1,recovered=0;
 if(inf)ok=BN_is_zero(zz)&&BN_is_zero(zzz);else if(BN_is_zero(zz)||BN_is_zero(zzz))ok=0;else{require(BN_mod_inverse(zz,zz,prime,ctx));require(BN_mod_inverse(zzz,zzz,prime,ctx));require(BN_mod_mul(xx,xx,zz,prime,ctx));require(BN_mod_mul(yy,yy,zzz,prime,ctx));require(EC_POINT_get_affine_coordinates(group,want,wx,wy,ctx));ok=BN_cmp(xx,wx)==0&&BN_cmp(yy,wy)==0;}
 if(ok&&!inf){
 BIGNUM *rk=BN_new(),*rx=BN_new(),*ry=BN_new(),*c=BN_new(),*tmp=BN_new();BN_set_word(rk,17);EC_POINT *R=EC_POINT_new(group),*sum=EC_POINT_new(group);require(EC_POINT_mul(group,R,rk,nullptr,nullptr,ctx));require(EC_POINT_get_affine_coordinates(group,R,rx,ry,ctx));
 require(BN_mod_sqr(c,rx,prime,ctx));require(BN_mul_word(c,3));require(BN_mod_add(tmp,ry,ry,prime,ctx));require(BN_mod_inverse(tmp,tmp,prime,ctx));require(BN_mod_mul(c,c,tmp,prime,ctx));write256(QSB_U2R_C,c);
 uint64_t xR[4],yR[4],W[5],inv[5],x1[4],x2[4];write256(xR,rx);write256(yR,ry);qsb_xyzz_finish_prepare(state,state+8,state+12,xR,W);
 if(W[0]|W[1]|W[2]|W[3]){memcpy(inv,W,40);_ModInv(inv);uint32_t par=qsb_xyzz_finish_precomputed(state+4,state+8,state+12,inv,xR,yR,x1,x2);
 for(int j=0;j<2;j++){if(j)require(EC_POINT_invert(group,R,ctx));require(EC_POINT_add(group,sum,want,R,ctx));require(EC_POINT_get_affine_coordinates(group,sum,rx,ry,ctx));uint64_t ex[4],ey[4];write256(ex,rx);write256(ey,ry);ok=ok&&!memcmp(j?x2:x1,ex,32)&&((par>>j&1)==(ey[0]&1));++recovered;}}
 else require(BN_cmp(wx,rx)==0);
 BN_free(rk);BN_free(rx);BN_free(ry);BN_free(c);BN_free(tmp);EC_POINT_free(R);EC_POINT_free(sum);
 }
 munmap(table,64ull*1024*1024);BN_free(two);BN_free(bs);BN_free(scalar);BN_free(wk);BN_free(xx);BN_free(yy);BN_free(zz);BN_free(zzz);BN_free(wx);BN_free(wy);BN_free(halfbase);BN_free(ord);BN_free(prime);EC_POINT_free(want);EC_GROUP_free(group);BN_CTX_free(ctx);return ok?1+recovered:0;
}
extern "C" uint64_t norm_count(){return norms;}
extern "C" void audit_recode(const uint64_t*k,int32_t*d){uint64_t m[4];int sign;gt_recode_setup(k,m,&sign);for(int c=0;c<15;c++){uint32_t ix;uint64_t neg;gt_direct_digit(m,sign<0,gt_shift(c)+1,gt_width(c),c==14,&ix,&neg);d[c]=(int32_t)(2*ix+1)*(neg?-1:1);}}
'''
 (out/'projection.cpp').write_text(base);cp=subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC','-I/opt/homebrew/opt/openssl@3/include','-L/opt/homebrew/opt/openssl@3/lib',str(out/'projection.cpp'),'-lcrypto','-o',str(out/'projection.so')],capture_output=True,text=True);(out/'compile.log').write_text(cp.stdout+cp.stderr)
 if cp.returncode:raise RuntimeError('C++ projection compilation failed: '+cp.stderr[:1600])
 lib=C.CDLL(str(out/'projection.so'));u64=C.POINTER(C.c_uint64);lib.audit_chain.argtypes=[u64,u64];lib.norm_count.restype=C.c_uint64
 lib.audit_recode.argtypes=[u64,C.POINTER(C.c_int32)]
 H=U-(1<<239);edge={0,1,2,3,N-1,N,N+1,U-1,H,N-H}
 for bit in [0,1,17,18,35,52,69,86,103,120,137,154,171,188,205,222,239,255]:
  edge.update(x for x in [(1<<bit)-1,1<<bit,(1<<bit)+1] if 0<=x<U)
 rng=random.Random(189217)
 recode_cases=sorted({0,N,N-1,N+1,U-1,H,N-H}|{x for b in range(256) for x in [(1<<b)-1,1<<b,(1<<b)+1] if 0<=x<U})+[rng.getrandbits(256) for _ in range(12000)]
 signs=set()
 for k in recode_cases:
  d=(C.c_int32*15)();lib.audit_recode(limbs(k),d);terms=[]
  twice=2*(k%N)%N;sign=1 if twice&1 else -1;odd=twice if twice&1 else N-twice
  for c,x in enumerate(d):
   width=18 if c==0 else 17;shift=17*c+1 if c else 0;rem=(odd>>shift)|1
   expected=sign*(rem if c==14 else (rem&((1<<(width+1))-1))-(1<<width))
   assert x==expected and x&1 and abs(x)<(1<<width);signs.add(x<0);terms.append(int(x)<<shift)
  assert sum(terms)*pow(2,-1,N)%N==k%N
  for c in range(1,14):
   assert (sum(terms[:c])-terms[c])%N and (sum(terms[:c])+terms[c])%N
 assert signs=={False,True}
 cases=[H,N-H]+sorted(edge)+[rng.getrandbits(256) for _ in range(256)];recovered=0;fail=None
 for j,k in enumerate(cases):
  coef=[1,N-1,2][j%3] if j<12 else rng.randrange(1,N);v=lib.audit_chain(limbs(k),limbs(coef))
  if not v:fail={'case':j,'scalar':hex(k),'runtime_base_scalar':hex(coef)};break
  recovered+=v-1
 status='MUTATION_DETECTED' if fail and a.mutation else 'FAIL_CHAIN' if fail else 'PASS_ACTUAL_FRONTIER_CHAIN'
 if a.mutation and not fail:raise AssertionError('Mutation escaped')
 assert source_identity(src)==ident
 result={'status':status,**ident,'compiled':True,'checker_sha256':sha(Path(__file__)),'backend_sha256':sha(ROOT/'research/wide_windows/audit_support.py'),'projection_sha256':sha(out/'projection.cpp'),'actual_helper_sha256':parts,'mutation':a.mutation,'failure':fail,'recoder_cases':len(recode_cases),'ordinary_prefix_equal_opposite_checks':13*len(recode_cases)*2,'curve_chains':j+1,'recovered_keys_compared':recovered,'actual_sparse_loader_calls':15*(j+1),'actual_weak_host_headers':weak,'boundary_normalizations':lib.norm_count(),'scope':__doc__,'canonical_seed_final_recovery_field':'Independent OpenSSL operations; not actual canonical device arithmetic','geometry':{'widths':[18]+[17]*14,'order':list(range(15)),'bytes':64*1024*1024},'gpu_executed':False,'source_modified':False}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','source_fingerprint','curve_chains','recovered_keys_compared','boundary_normalizations','failure']}));sys.exit(0 if not fail or a.mutation else 2)
if __name__=='__main__':main()
