#!/usr/bin/env python3
"""Execute actual wide-table CPU ladders and extracted affine finish via OpenSSL.

Checks address decomposition, H=identity handling and runtime base variation.
Does not execute CUDA collectives or allocate the16GiB table.
"""
import ctypes as C
import hashlib,json,random,re,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;BASE=HERE/'candidate'
sys.path.insert(0,str(HERE.parent.parent))
from check_projective import BACKEND,function
N=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

def main():
    paths=list(BASE.glob('*.cu'))+list(BASE.glob('*.cuh'))+list(BASE.glob('*.h'))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    src=(BASE/'pinning.cu').read_text();dev=(BASE/'wide_table_kernels.cuh').read_text()
    decl=src[src.index('#define GT_CHUNKS'):src.index('/* n = secp256k1')]
    code='\n'.join([BACKEND,'#include <vector>\n#define __host__\n#define __device__\n#define __forceinline__ inline',
        (BASE/'wide_geometry.cuh').read_text(),decl,
        function(src,'static void gt_point_to_limbs('),function(src,'static void gt_require('),function(src,'static void gt_build_ladders('),
        function(dev,'__host__ __device__ __forceinline__ void wide_decode_entry('),function(dev,'__device__ __forceinline__ void wide_finish_affine('),r'''
extern "C" void decode(uint32_t t,uint64_t*out){int ch;uint32_t hi,lo;wide_decode_entry(t,&ch,&hi,&lo);out[0]=ch;out[1]=hi;out[2]=lo;}
extern "C" int audit_builder(const uint64_t*coef,const uint32_t*entries,int count){
 std::vector<uint64_t>L((size_t)GT_CHUNKS*GT_LO*8),H((size_t)GT_CHUNKS*GT_HI*8);
 gt_build_ladders(L.data(),H.data(),(const uint8_t*)coef);
 ctx=BN_CTX_new();EC_GROUP*g=EC_GROUP_new_by_curve_name(NID_secp256k1);prime=BN_new();BIGNUM*n=BN_new(),*scale=read256(coef),*k=BN_new(),*half=BN_new(),*wx=BN_new(),*wy=BN_new();
 require(EC_GROUP_get_curve(g,prime,nullptr,nullptr,ctx));require(EC_GROUP_get_order(g,n,ctx));BN_set_word(k,2);require(BN_mod_inverse(half,k,n,ctx)!=nullptr);
 require(BN_mod_mul(scale,scale,half,n,ctx));EC_POINT*want=EC_POINT_new(g);int ok=1;
 for(int i=0;i<count&&ok;++i){
  int ch;uint32_t hi,lo;wide_decode_entry(entries[i],&ch,&hi,&lo);
  const uint64_t*hp=H.data()+((size_t)ch*GT_HI+hi)*8,*lp=L.data()+((size_t)ch*GT_LO+lo)*8;
  uint64_t x[4],y[4];
  if(!hi){memcpy(x,lp,32);memcpy(y,lp+4,32);}
  else{uint64_t inv[5]={0,0,0,0,0};_ModSub256(inv,lp,hp);_ModInv(inv);wide_finish_affine(x,y,hp,lp,inv);}
  BN_set_word(k,2ull*(entries[i]-gt_offset(ch))+1);require(BN_lshift(k,k,gt_shift(ch)));require(BN_mod_mul(k,k,scale,n,ctx));
  require(EC_POINT_mul(g,want,k,nullptr,nullptr,ctx));require(EC_POINT_get_affine_coordinates(g,want,wx,wy,ctx));
  uint64_t expected[8];write256(expected,wx);write256(expected+4,wy);
  ok=memcmp(x,expected,32)==0&&memcmp(y,expected+4,32)==0;
  if(!ok)fprintf(stderr,"Builder mismatch at entry %u\n",entries[i]);
 }
 EC_POINT_free(want);EC_GROUP_free(g);BN_free(n);BN_free(scale);BN_free(k);BN_free(half);BN_free(wx);BN_free(wy);BN_free(prime);BN_CTX_free(ctx);return ok;
}
'''])
    rng=random.Random(2026091613);widths=[26]*6+[25]*4;offsets=[sum(1<<(b-1) for b in widths[:c]) for c in range(10)]
    entries=[]
    for c,bits in enumerate(widths):
        limit=1<<(bits-1)
        local={0,1,2,4094,4095,4096,4097,limit-2,limit-1}
        for j in range(25):
            for delta in (-1,0,1):
                v=(1<<j)+delta
                if 0<=v<limit:local.add(v)
        entries.extend(offsets[c]+x for x in sorted(local))
    entries.extend(rng.randrange(1<<28) for _ in range(768))
    with tempfile.TemporaryDirectory(prefix='qsb-wide-builder-') as tmp:
        cpp=Path(tmp)/'audit.cpp';libp=Path(tmp)/'audit.so';cpp.write_text(code)
        subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC','-Wno-deprecated-declarations','-Wno-pragma-once-outside-header','-I/opt/homebrew/opt/openssl@3/include','-L/opt/homebrew/opt/openssl@3/lib',str(cpp),'-lcrypto','-o',str(libp)],check=True)
        lib=C.CDLL(str(libp));lib.decode.argtypes=[C.c_uint32,C.POINTER(C.c_uint64)]
        lib.audit_builder.argtypes=[C.POINTER(C.c_uint64),C.POINTER(C.c_uint32),C.c_int]
        maps=entries+[rng.randrange(1<<28) for _ in range(100000)]
        got=(C.c_uint64*3)()
        for t in maps:
            lib.decode(t,got);ch=max(c for c,o in enumerate(offsets) if t>=o);m=2*(t-offsets[ch])+1
            assert list(got)==[ch,m>>13,m&8191]
            assert got[2]&1 and got[1]<8192
            if got[1]:assert 0<got[1]*8192-got[2]<N and 0<got[1]*8192+got[2]<N
        for coef in (1,N-1,rng.randrange(1,N)):
            limbs=(C.c_uint64*4)(*(coef>>(64*k)&((1<<64)-1) for k in range(4)))
            assert lib.audit_builder(limbs,(C.c_uint32*len(entries))(*entries),len(entries))
    assert all(hashlib.sha256((BASE/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    # These textual checks bind the allocation/lifetime contract to inspected source.
    host=(BASE/'wide_table_host.cuh').read_text()
    assert 'const uint32_t chunk=1u<<20' in host and 'qsb_invert_super_roots<<<1,256>>>' in host
    assert host.index('cudaDeviceSynchronize()')<host.index('cudaFree(L)')<host.index('gt_spot_check(')
    assert 'memcpy(table+(size_t)t*64,d,32)' in dev and 'if(!active)return;' in dev
    assert dev.index('qsb_block_inverse_checkpoint(inv,roots,tree);')<dev.index('if(!active)return;')
    result={'status':'PASS','validation_level':'CPU source-derived host ladders and builder affine formulas with OpenSSL','entry_decode_cases':len(maps),'runtime_bases':3,'table_entries_compared':len(entries)*3,'full_ladder_points_per_base':106486,'source_sha256':hashes,'gpu_executed':False,'limits':'No CUDA collective execution, no16GiB allocation, no timing. CUDA scratch lifetimes checked from source; inverse primitives covered separately.'}
    # exact ladder sizes:4096 odd low entries and(high_count-1) high entries/window.
    result['full_ladder_points_per_base']=sum(4096+((1<<b)//8192)-1 for b in widths)
    (HERE/'builder-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
