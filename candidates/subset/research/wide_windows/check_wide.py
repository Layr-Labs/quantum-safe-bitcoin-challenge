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
import re
import subprocess
import sys
import tempfile
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
from audit_support import BACKEND,function
from preflight import source_identity
N=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
MASK=(1<<64)-1
P=(1<<256)-(1<<32)-977

def limbs(x):return (C.c_uint64*4)(*(x>>(64*i)&MASK for i in range(4)))
def integer(x):return sum(int(v)<<(64*i) for i,v in enumerate(x))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mutation-anchor',action='store_true',help='negative control: apply the incorrect proposed seed-anchor change in extracted test code only')
    parser.add_argument('--source',type=Path,default=HERE/'candidate',help='isolated candidate include closure')
    parser.add_argument('--report',type=Path,help='write separate source-bound result instead of the default report')
    parser.add_argument('--small',action='store_true',help='audit the adaptive candidate\'s actual 16-window planar branch')
    parser.add_argument('--mixed',action='store_true',help='audit the isolated mixed15-window interleaved geometry')
    parser.add_argument('--compact',action='store_true',help='audit the adaptive candidate\'s compact64MiB branch')
    parser.add_argument('--widths',type=str,help='explicit comma-separated signed-window widths, independently checked against actual source')
    parser.add_argument('--prefetch-chunks',type=int,help='only windows below this index issue prefetches')
    parser.add_argument('--prefetch-start',type=int,default=2,help='first window that must have a checked prefetch')
    parser.add_argument('--chain-order',type=str,help='independent expected permutation of table windows in the actual point chain')
    parser.add_argument('--extra-chain-scalars',type=Path,help='JSON list of additional scalar integers or hex strings for exceptional-domain checks')
    args=parser.parse_args()
    if args.compact:args.mixed=True
    if args.small and args.mixed:parser.error('Select exactly one compact geometry')
    base=args.source.resolve()
    identity=source_identity(base)
    src=(base/'tests/gpu_epochs/tree.cu').read_text()+'\n'+(base/'tests/gpu_epochs/ranked_pipeline.cuh').read_text()
    math=(base/'GPUMath.h').read_text()
    geometry=(base/'tests/gpu_epochs/wide_geometry.cuh').read_text()
    if args.compact:
        compact=(base/'tests/gpu_epochs/compact_table_device.cuh').read_text()
        compact=compact.replace('#include "compact_geometry.cuh"','')
        for before,after in [('compact_recode_step','gt_recode_step'),('compact_recode_signed','gt_recode_signed'),
            ('compact_load_signed','gt_load_signed'),('compact_prefetch_point','gt_prefetch_point'),
            ('compact_fixed_xyzz','_FixedBaseSignedXYZZ'),('compact_entries','gt_entries'),
            ('compact_offset','gt_offset'),('compact_shift','gt_shift')]:
            compact=re.sub(r'\b'+before+r'\b',after,compact)
        compact=compact.replace('COMPACT_','GT_').replace('MIXED_','WIDE_').replace('mixed_','wide_')
        src=compact+'\n'+src
        geometry=(base/'tests/gpu_epochs/compact_geometry.cuh').read_text().replace('MIXED_','WIDE_').replace('mixed_','wide_')
    if args.small:
        small=(base/'tests/gpu_epochs/small_table_device.cuh').read_text()
        defines=small[:small.index('__device__')]
        for before,after in [('small_recode_step','gt_recode_step'),('small_recode_signed','gt_recode_signed'),
                             ('small_load_signed','gt_load_signed'),('small_fixed_xyzz','_FixedBaseSignedXYZZ')]:
            small=re.sub(r'\b'+before+r'\b',after,small)
        src=small+'\n'+src
        geometry=defines+r'''
#define WIDE_CHUNKS SMALL_CHUNKS
#define WIDE_TOTAL_ENTRIES ((uint64_t)SMALL_CHUNKS*SMALL_ENTRIES)
static unsigned wide_entries(int){return SMALL_ENTRIES;}
static unsigned wide_offset(int c){return c*SMALL_ENTRIES;}
static unsigned wide_shift(int c){return c*16;}
'''
    setup=function(src,'__device__ __forceinline__ void gt_recode_setup(')
    order=src[src.index('__device__ __constant__ uint64_t GT_ORDER_N'):src.index('};',src.index('__device__ __constant__ uint64_t GT_ORDER_N'))+2]
    prelude=BACKEND+'\n#define __forceinline__ inline\n#define __constant__\n#define __device__\n#define __host__\n'+geometry+'\n#define GT_CHUNKS WIDE_CHUNKS\n'+order+'\n'+setup+r'''
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
    wide_body=function(src,'__device__ void _FixedBaseSignedXYZZ(').replace('_FixedBaseSignedXYZZ','wide_fixed')
    if args.mutation_anchor:
        marker='_PointAddXYZZ_mm(X,Y,ZZ,ZZZ, x0,y0, x1,y1);'
        assert marker in wide_body
        wide_body=wide_body.replace(marker,marker+'\n    Load256(y0,y1); // intentionally wrong negative control')
    has_prefetch=not args.small and 'void gt_prefetch_point(' in src
    prefetch_limit=args.prefetch_chunks if args.prefetch_chunks is not None else 1000
    if args.chain_order:
        order=[int(c) for c in args.chain_order.split(',')]
        assert sorted(order)==list(range(len(order)))
        declaration='\nstatic unsigned audit_load_count=0;\nstatic const int audit_load_order[]={'+','.join(map(str,order))+'};\n'
        prelude=prelude.replace('static void gt_load_signed_flat(',declaration+'static void gt_load_signed_flat(',1)
        marker='    int ch=-1;for(int c=0;c<WIDE_CHUNKS;++c)if(wide_offset(c)==base)ch=c;'
        assert prelude.count(marker)==1
        prelude=prelude.replace(marker,marker+'\n    require(WIDE_CHUNKS=='+str(len(order))+');\n    require(ch==audit_load_order[(audit_load_count++)%WIDE_CHUNKS]);')
    if 'qsb_asym_cold_digits(' in wide_body:
        prelude+=function(src,'__device__ __forceinline__ void qsb_asym_cold_digits(')
    if has_prefetch:
        prelude += 'static uintptr_t prefetched[WIDE_CHUNKS];\nstatic unsigned prefetched_halves[WIDE_CHUNKS];\nstatic uint64_t prefetch_checks=0;\n'
    prelude += r'''
static void gt_load_signed(const uint8_t *tx,const uint8_t*,int c,uint32_t idx,uint64_t neg,uint64_t*x,uint64_t*y){
    gt_load_signed_flat(tx,wide_offset(c),idx,neg,x,y);
}
'''
    if has_prefetch:
        marker='    gt_load_signed_flat(tx,wide_offset(c),idx,neg,x,y);'
        prelude=prelude.replace(marker,'''
    if(c>=PREFETCH_START && c<PREFETCH_LIMIT){
      require(prefetched_halves[c]==3);
      require(prefetched[c]==((uintptr_t)wide_offset(c)+idx)*64);
      prefetched_halves[c]=0;++prefetch_checks;
    }
'''+marker)
        prelude=prelude.replace('PREFETCH_LIMIT',str(prefetch_limit))
        prelude=prelude.replace('PREFETCH_START',str(args.prefetch_start))
    prelude += function(src,'__device__ __forceinline__ int32_t gt_recode_step(')
    # Reserve address space only; touch one 64-byte entry per case. This executes
    # the actual AoS vector loader at offsets above 4 GiB without a resident table.
    prelude += r'''
#include <sys/mman.h>
struct alignas(16) ulonglong2 { uint64_t x,y; };
static void _ModNeg256(uint64_t*r,const uint64_t*a){
 const uint64_t p[4]={0xFFFFFFFEFFFFFC2FULL,UINT64_MAX,UINT64_MAX,UINT64_MAX};
 uint64_t borrow=0;for(int k=0;k<4;++k){__uint128_t d=(__uint128_t)p[k]-a[k]-borrow;r[k]=(uint64_t)d;borrow=(uint64_t)(d>>64)&1;}
}
'''
    cache_header=base/'tests/gpu_epochs/cache_ops.cuh'
    if cache_header.exists():prelude+=cache_header.read_text()
    prelude += function(src,'__host__ __device__ __forceinline__ unsigned gt_offset(')
    if has_prefetch:
        # Execute actual address expressions; replace only PTX cache hints with
        # a range/sector oracle. This checks addresses, not device prefetching.
        prelude += r'''
static void audit_prefetch(uintptr_t off){
 require(off<WIDE_TOTAL_ENTRIES*64 && !(off&31));
 int c=-1;
 for(int i=0;i<WIDE_CHUNKS;++i)
   if(off>=(uintptr_t)wide_offset(i)*64 && off<((uintptr_t)wide_offset(i)+wide_entries(i))*64)c=i;
 require(c>=2);
 if(!(off&63)){prefetched[c]=off;prefetched_halves[c]=1;}
 else {require(prefetched_halves[c]==1 && prefetched[c]+32==off);prefetched_halves[c]|=2;}
}
extern "C" uint64_t audit_prefetch_checks(){return prefetch_checks;}
'''
        prelude=prelude.replace('require(c>=2);','require(c>='+str(args.prefetch_start)+' && c<'+str(prefetch_limit)+');')
        helper=function(src,'__device__ __forceinline__ void gt_prefetch_point(')
        helper=helper.replace('asm volatile("prefetch.global.L2 [%0];" :: "l"(table+off));','audit_prefetch((uintptr_t)table+off);')
        helper=helper.replace('asm volatile("prefetch.global.L2 [%0];" :: "l"(table+off+32));','audit_prefetch((uintptr_t)table+off+32);')
        assert 'asm' not in helper
        prelude += helper
    prelude += function(src,'__device__ __forceinline__ void gt_load_signed(').replace('gt_load_signed','actual_gt_load_signed')
    prelude += r'''
extern "C" void audit_loader(uint64_t offset,int c,uint32_t idx,uint64_t neg,const uint64_t*input,uint64_t*out){
 void *mapping=mmap(nullptr,WIDE_TOTAL_ENTRIES*64,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
 require(mapping!=MAP_FAILED);uint8_t*table=(uint8_t*)mapping;
 require(offset+64<=WIDE_TOTAL_ENTRIES*64);memcpy(table+offset,input,64);
 actual_gt_load_signed(table,nullptr,c,idx,neg,out,out+4);
 require(munmap(mapping,WIDE_TOTAL_ENTRIES*64)==0);
}
'''
    if args.small:
        prelude=prelude.replace('require(offset+64<=WIDE_TOTAL_ENTRIES*64);memcpy(table+offset,input,64);',
            'require(offset+32<=WIDE_TOTAL_ENTRIES*32);memcpy(table+offset,input,32);memcpy(table+WIDE_TOTAL_ENTRIES*32+offset,input+4,32);')
        prelude=prelude.replace('actual_gt_load_signed(table,nullptr,c,idx,neg,out,out+4);',
            'actual_gt_load_signed(table,table+WIDE_TOTAL_ENTRIES*32,c,idx,neg,out,out+4);')
    code='\n'.join([prelude,function(src,'__device__ __forceinline__ void gt_digit_idx('),
        function(math,'template<bool DEFER_Y>'),function(math,'__device__ void _PointAddXYZZ_mm('),
        function(src,'__device__ __forceinline__ void qsb_xyzz_finish_prepare('),
        function(src,'__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed('),
        function(src,'__device__ __forceinline__ void qsb_asym_last_add(') if 'qsb_asym_last_add(' in wide_body else '',
        wide_body,function(src,'__device__ __forceinline__ void gt_recode_signed('),r'''
extern "C" void recode(const uint64_t*k,int32_t*digits){
 gt_recode_signed(k,digits);
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
 uint64_t out[16];wide_fixed(out,out+4,out+8,out+12,k,nullptr,nullptr);
 require(BN_mod_mul(wantk,scalar,coef,group_order,ctx));
 EC_POINT *want=EC_POINT_new(group);require(EC_POINT_mul(group,want,wantk,nullptr,nullptr,ctx));
 BIGNUM *zz=read256(out+8),*zzz=read256(out+12),*xx=read256(out),*yy=read256(out+4),*wx=BN_new(),*wy=BN_new();
 int inf=EC_POINT_is_at_infinity(group,want),ok,recovered=0;
 if(inf)ok=BN_is_zero(zz)&&BN_is_zero(zzz);
 else if(BN_is_zero(zz)||BN_is_zero(zzz))ok=0;
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
    expected_bits=[16]*16 if args.small else ([18]+[17]*14 if args.mixed else [26]*6+[25]*4)
    if args.widths:
        expected_bits=[int(b) for b in args.widths.split(',')]
        assert sum(expected_bits)==256 and all(2<=b<=30 for b in expected_bits)
    chunks=len(expected_bits)
    table_bytes=sum(1<<(b-1) for b in expected_bits)*64
    with tempfile.TemporaryDirectory(prefix='qsb-wide-source-') as tmp:
        cpp=Path(tmp)/'audit.cpp';libp=Path(tmp)/'audit.so';cpp.write_text(code)
        subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC','-I/opt/homebrew/opt/openssl@3/include','-L/opt/homebrew/opt/openssl@3/lib',str(cpp),'-lcrypto','-o',str(libp)],check=True)
        lib=C.CDLL(str(libp));u64=C.POINTER(C.c_uint64)
        lib.recode.argtypes=[u64,C.POINTER(C.c_int32)];lib.geometry_out.argtypes=[u64];lib.audit_chain.argtypes=[u64,u64]
        lib.audit_loader.argtypes=[C.c_uint64,C.c_int,C.c_uint32,C.c_uint64,u64,u64]
        gout=(C.c_uint64*(3*chunks))();lib.geometry_out(gout)
        offset=shift=0
        for c,bits in enumerate(expected_bits):
            assert list(gout[3*c:3*c+3])==[1<<(bits-1),offset,shift]
            offset+=1<<(bits-1);shift+=bits
        assert shift==256 and offset*64==table_bytes
        loader_cases=0
        for c,bits in enumerate(expected_bits):
            indices=[0,1,(1<<(bits-1))-1]+[rng.randrange(1<<(bits-1)) for _ in range(16)]
            for idx in indices:
                x,y=rng.randrange(P),rng.randrange(1,P)
                values=(C.c_uint64*8)(*limbs(x),*limbs(y));out=(C.c_uint64*8)()
                for neg in (0,1):
                    address=(sum(1<<(b-1) for b in expected_bits[:c])+idx)*(32 if args.small else 64)
                    lib.audit_loader(address,c,idx,neg,values,out)
                    assert integer(out[:4])==x and integer(out[4:])==((-y)%P if neg else y)
                    loader_cases+=1
        digit_signs=set()
        for k in trials:
            ds=(C.c_int32*chunks)();lib.recode(limbs(k),ds)
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
        if args.extra_chain_scalars:
            extra=json.loads(args.extra_chain_scalars.read_text())
            extra=[int(k,0) if isinstance(k,str) else int(k) for k in extra]
            assert all(0<=k<1<<256 for k in extra)
            chain_cases=extra+chain_cases
        if args.mutation_anchor:chain_cases=[1]+[k for k in chain_cases if k!=1]
        recovered=0
        for i,k in enumerate(chain_cases):
            coef=[1,N-1,2][i%3] if i<12 else rng.randrange(1,N)
            outcome=lib.audit_chain(limbs(k),limbs(coef))
            if not outcome and args.mutation_anchor:
                assert source_identity(base)==identity
                result={'status':'MUTATION_DETECTED','mutation':'replace seed anchor y0 with y1',
                        'case_index':i,'scalar_hex':hex(k),'runtime_base_scalar_hex':hex(coef),
                        'source_fingerprint':identity['source_fingerprint'],'production_modified':False,
                        'limits':'Mutation exists only in temporary extracted CPU code; mismatch against independent OpenSSL scalar multiplication.'}
                (args.report or HERE/'anchor-mutation-results.json').write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps(result,indent=2))
                return
            if not outcome:
                assert source_identity(base)==identity
                result={'status':'FAIL','validation_level':'CPU-extracted point chain versus independent OpenSSL scalar multiplication',
                        'case_index':i,'scalar_hex':hex(k),'runtime_base_scalar_hex':hex(coef),
                        'source_fingerprint':identity['source_fingerprint'],'gpu_executed':False,
                        'reason':'Extracted chain does not represent the required finite/infinite point.'}
                (args.report or HERE/'cpu-results.json').write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps(result,indent=2))
                sys.exit(1)
            recovered+=outcome-1
        prefetch_count=0
        if has_prefetch:
            lib.audit_prefetch_checks.restype=C.c_uint64
            prefetch_count=lib.audit_prefetch_checks()
            assert prefetch_count==len(chain_cases)*(min(chunks,prefetch_limit)-args.prefetch_start)
    assert not args.mutation_anchor,'Incorrect anchor mutation escaped the curve oracle'
    assert source_identity(base)==identity
    result={'status':'PASS','validation_level':'CPU-extracted subset point chain with OpenSSL field and sparse table oracle',
            'recoding_cases':len(trials),'curve_chains':len(chain_cases),'recovered_keys_compared':recovered,
            'geometry_cases':chunks,'actual_loader_cases':loader_cases,'table_bytes':table_bytes,
            'branch':'small planar' if args.small else 'signed-window interleaved',
            'geometry_bits':expected_bits,
            'point_chain_operations':{'multiplications':7*chunks-10,'squares':2*chunks-2},
            'prefetch_next_load_checks':prefetch_count,
            'checked_chain_order':list(map(int,args.chain_order.split(','))) if args.chain_order else None,
            'cache_helper_projection':'ordinary host branch; no cache simulation' if cache_header.exists() else None,
            **identity,'gpu_executed':False,
            'limits':'Actual recoder, chain and recovery use OpenSSL field/table oracle. Actual vector loader runs on CPU against sparse virtual mappings with independently calculated offsets. No full resident table, CUDA arithmetic execution or GPU throughput measurement.'}
    (args.report or HERE/'cpu-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
