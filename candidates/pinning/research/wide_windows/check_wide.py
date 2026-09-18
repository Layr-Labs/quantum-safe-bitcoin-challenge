#!/usr/bin/env python3
"""Audit actual wide recoding and point chain without allocating a 16 GiB table.

Source expressions run on the CPU; field and sparse table values use OpenSSL.
Native CUDA and GPU performance remain separate checks.
"""
import ctypes as C
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent.parent))
from check_projective import BACKEND,function
N=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
MASK=(1<<64)-1
P=(1<<256)-(1<<32)-977

def limbs(x):return (C.c_uint64*4)(*(x>>(64*i)&MASK for i in range(4)))
def integer(x):return sum(int(v)<<(64*i) for i,v in enumerate(x))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',choices=('pr24_corrected','pr64_corrected','wide_windows/candidate'),default='pr24_corrected')
    args=ap.parse_args()
    base=HERE.parent/args.base
    integrated=args.base=='wide_windows/candidate'
    geometry_path=base/'wide_geometry.cuh' if integrated else HERE/'wide_geometry.cuh'
    paths=[base/'pinning.cu',base/'GPUMath.h',geometry_path]
    if not integrated:paths.append(HERE/'wide_probe.cu')
    hashes={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    src=(base/'pinning.cu').read_text();math=(base/'GPUMath.h').read_text()
    geometry=geometry_path.read_text()
    setup=function(src,'__device__ __forceinline__ void gt_recode_setup(')
    order=src[src.index('__device__ __constant__ uint64_t GT_ORDER_N'):src.index('};',src.index('__device__ __constant__ uint64_t GT_ORDER_N'))+2]
    prelude=BACKEND+'\n#define __forceinline__ inline\n#define __constant__\n#define __device__\n'+geometry+'\n'+order+'\n'+setup+r'''
static void Load256(uint64_t*r,const uint64_t*a){memcpy(r,a,32);}
static EC_GROUP *group;
static BIGNUM *group_order,*half_base;
static void gt_load_signed_flat(const uint8_t*,uint32_t base,uint32_t idx,uint64_t neg,uint64_t*x,uint64_t*y){
    int ch=-1;for(int c=0;c<WIDE_CHUNKS;++c)if(wide_offset(c)==base)ch=c;
    require(ch>=0 && idx<wide_entries(ch) && (neg<=1 || neg==UINT64_MAX));
    BIGNUM *k=BN_new();EC_POINT *pt=EC_POINT_new(group);
    require(BN_set_word(k,2ull*idx+1));require(BN_lshift(k,k,wide_shift(ch)));
    require(BN_mod_mul(k,k,half_base,group_order,ctx));
    require(EC_POINT_mul(group,pt,k,nullptr,nullptr,ctx));
    if(neg)require(EC_POINT_invert(group,pt,ctx));
    BIGNUM *xx=BN_new(),*yy=BN_new();require(EC_POINT_get_affine_coordinates(group,pt,xx,yy,ctx));
    write256(x,xx);write256(y,yy);BN_free(xx);BN_free(yy);BN_free(k);EC_POINT_free(pt);
}
'''
    wide_body=(function(src,'__device__ void _FixedBaseSignedXYZZScalar(').replace('_FixedBaseSignedXYZZScalar','wide_fixed')
        if integrated else function((HERE/'wide_probe.cu').read_text(),'__device__ void wide_fixed('))
    code='\n'.join([prelude,function(src,'__device__ __forceinline__ void gt_digit_idx('),
        function(math,'template<bool DEFER_Y>'),function(math,'__device__ void _PointAddXYZZ_mm('),
        function(src,'__device__ __forceinline__ void qsb_xyzz_finish_prepare('),
        function(src,'__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed('),
        wide_body,r'''
extern "C" void recode(const uint64_t*k,int32_t*digits){
 uint64_t M[4];int sign;gt_recode_setup(k,M,&sign);
 for(int c=0;c<WIDE_CHUNKS-1;++c)digits[c]=wide_step(M,sign,wide_bits(c));
 digits[WIDE_CHUNKS-1]=sign*(int32_t)M[0];
}
extern "C" void geometry_out(uint64_t*out){
 for(int c=0;c<WIDE_CHUNKS;++c){out[3*c]=wide_entries(c);out[3*c+1]=wide_offset(c);out[3*c+2]=wide_shift(c);}
}
extern "C" int audit_chain(const uint64_t*k,const uint64_t*base_scalar){
 ctx=BN_CTX_new();group=EC_GROUP_new_by_curve_name(NID_secp256k1);
 prime=BN_new();group_order=BN_new();half_base=BN_new();
 require(EC_GROUP_get_curve(group,prime,nullptr,nullptr,ctx));require(EC_GROUP_get_order(group,group_order,ctx));
 BIGNUM *two=BN_new(),*coef=read256(base_scalar),*scalar=read256(k),*wantk=BN_new();
 BN_set_word(two,2);require(BN_mod_inverse(half_base,two,group_order,ctx)!=nullptr);
 require(BN_mod_mul(half_base,half_base,coef,group_order,ctx));
 uint64_t out[16];wide_fixed(out,out+4,out+8,out+12,k,nullptr);
 require(BN_mod_mul(wantk,scalar,coef,group_order,ctx));
 EC_POINT *want=EC_POINT_new(group);require(EC_POINT_mul(group,want,wantk,nullptr,nullptr,ctx));
 BIGNUM *zz=read256(out+8),*zzz=read256(out+12),*xx=read256(out),*yy=read256(out+4),*wx=BN_new(),*wy=BN_new();
 int inf=EC_POINT_is_at_infinity(group,want),ok,recovered=0;
 if(inf)ok=BN_is_zero(zz)&&BN_is_zero(zzz);
 else {
  require(BN_mod_inverse(zz,zz,prime,ctx)!=nullptr);require(BN_mod_inverse(zzz,zzz,prime,ctx)!=nullptr);
  require(BN_mod_mul(xx,xx,zz,prime,ctx));require(BN_mod_mul(yy,yy,zzz,prime,ctx));
  require(EC_POINT_get_affine_coordinates(group,want,wx,wy,ctx));ok=BN_cmp(xx,wx)==0&&BN_cmp(yy,wy)==0;
 }
 if(ok&&!inf){
  // Exercise the actual selected source's direct recovery expressions as well.
  BIGNUM *rk=BN_new(),*rx=BN_new(),*ry=BN_new();BN_set_word(rk,17);
  EC_POINT *R=EC_POINT_new(group),*sum=EC_POINT_new(group);
  require(EC_POINT_mul(group,R,rk,nullptr,nullptr,ctx));
  require(EC_POINT_get_affine_coordinates(group,R,rx,ry,ctx));
  uint64_t rxw[4],ryw[4],state[16],W[5],inv[5],Cs[4],x1[4],x2[4];
  write256(rxw,rx);write256(ryw,ry);memcpy(state,out,sizeof(state));
  qsb_xyzz_finish_prepare(state,state+8,rxw,W);
  if(W[0]|W[1]|W[2]|W[3]){
   _ModSqr(Cs,state);_ModMult(Cs,state+8);memcpy(inv,W,sizeof(W));_ModInv(inv);
   uint32_t parity=qsb_xyzz_finish_precomputed(Cs,state+4,W,state+12,inv,rxw,ryw,x1,x2);
   for(int recid=0;recid<2;++recid){
    if(recid)require(EC_POINT_invert(group,R,ctx));
    require(EC_POINT_add(group,sum,want,R,ctx));
    require(EC_POINT_get_affine_coordinates(group,sum,rx,ry,ctx));
    uint64_t ex[4],ey[4];write256(ex,rx);write256(ey,ry);
    ok=ok&&!memcmp(recid?x2:x1,ex,32)&&(((parity>>recid)&1)==(ey[0]&1));
    ++recovered;
   }
  }else{
   // The inherited candidate drops this singular recovery lane; verify cause.
   require(BN_cmp(wx,rx)==0);
  }
  BN_free(rk);BN_free(rx);BN_free(ry);EC_POINT_free(R);EC_POINT_free(sum);
 }
 BN_free(zz);BN_free(zzz);BN_free(xx);BN_free(yy);BN_free(wx);BN_free(wy);EC_POINT_free(want);
 BN_free(two);BN_free(coef);BN_free(scalar);BN_free(wantk);BN_free(half_base);BN_free(group_order);BN_free(prime);EC_GROUP_free(group);BN_CTX_free(ctx);
 return ok?1+recovered:0;
}
'''])
    rng=random.Random(2026091610)
    edge={0,1,2,3,N-1,N,N+1,(1<<256)-1}
    for b in range(256):
        for d in (-1,0,1):
            x=(1<<b)+d
            if 0<=x<1<<256:edge.add(x)
    trials=sorted(edge)+[rng.getrandbits(256) for _ in range(12000)]
    expected_bits=[26]*6+[25]*4
    with tempfile.TemporaryDirectory(prefix='qsb-wide-source-') as tmp:
        cpp=Path(tmp)/'audit.cpp';libp=Path(tmp)/'audit.so';cpp.write_text(code)
        subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC','-I/opt/homebrew/opt/openssl@3/include','-L/opt/homebrew/opt/openssl@3/lib',str(cpp),'-lcrypto','-o',str(libp)],check=True)
        lib=C.CDLL(str(libp));u64=C.POINTER(C.c_uint64)
        lib.recode.argtypes=[u64,C.POINTER(C.c_int32)];lib.geometry_out.argtypes=[u64];lib.audit_chain.argtypes=[u64,u64]
        gout=(C.c_uint64*30)();lib.geometry_out(gout)
        offset=shift=0
        for c,bits in enumerate(expected_bits):
            assert list(gout[3*c:3*c+3])==[1<<(bits-1),offset,shift]
            offset+=1<<(bits-1);shift+=bits
        assert shift==256 and offset*64==16<<30
        digit_signs=set()
        for k in trials:
            ds=(C.c_int32*10)();lib.recode(limbs(k),ds)
            total=0;shift=0
            for c,d in enumerate(ds):
                assert d&1 and 0<abs(d)<1<<expected_bits[c],(k,c,d)
                assert (abs(d)-1)//2<gout[3*c]
                digit_signs.add(d<0);total+=d<<shift;shift+=expected_bits[c]
            assert total*pow(2,-1,N)%N==k%N,(k,list(ds))
        assert digit_signs=={False,True}
        # Source-derived point accumulation and virtual table addresses against an
        # independent OpenSSL scalar multiplication, varying the runtime base.
        chain_cases=sorted(edge)[::5]+[0,N,N-1,(1<<256)-1]+[rng.getrandbits(256) for _ in range(256)]
        recovered=0
        for i,k in enumerate(chain_cases):
            coef=[1,N-1,2][i%3] if i<12 else rng.randrange(1,N)
            outcome=lib.audit_chain(limbs(k),limbs(coef))
            assert outcome,(i,k,coef)
            recovered+=outcome-1
    assert all(hashlib.sha256((HERE.parent/name).read_bytes()).hexdigest()==h for name,h in hashes.items())
    result={'status':'PASS','source_base':args.base,'validation_level':'CPU-extracted wide chain with selected base recoder and point expressions; OpenSSL field/table/oracle','recoding_cases':len(trials),'curve_chains':len(chain_cases),'geometry_cases':10,'table_bytes':16<<30,'point_chain_operations':{'multiplications':60,'squares':18},'source_sha256':hashes,'gpu_executed':False,'limits':'No 16GiB table allocation, GPU memory traffic, GPU execution or performance measured. Virtual table loader checks bases/indices and generates only referenced entries. PR64 selection tests composition with its new recoder/sign decoder; the current native wide probe still includes PR24.'}
    filename='cpu-results-integrated.json' if integrated else ('cpu-results.json' if args.base=='pr24_corrected' else 'cpu-results-pr64.json')
    result['recovered_keys_compared']=recovered
    if integrated:result['limits']='Exact integrated scalar-entry and recovery expressions with OpenSSL arithmetic and sparse virtual table. No GPU execution, CUDA field emulation or performance result; full table builder is audited separately.'
    (HERE/filename).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
