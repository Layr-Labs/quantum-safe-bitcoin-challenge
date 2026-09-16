#!/usr/bin/env python3
"""Execute extracted pipeline kernels on CPU with OpenSSL and simulated CTAs.

The EC producer and scheduled first-hash have controlled inputs here; their
actual helpers are checked separately. No GPU scheduling/codegen/timing claim.
"""
import argparse
import ctypes as CT
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
sys.dont_write_bytecode=True
from check_candidate import BACKEND,function
from check_deferred_source import FIELD
from preflight import source_identity
import audit_integrated as ref
HERE=Path(__file__).resolve().parent
U64,U32,U8=CT.c_uint64,CT.c_uint32,CT.c_uint8
P=ref.P
STUBS=r'''
#define __launch_bounds__(...)
#define MAX_T 16
struct ulonglong2{uint64_t x,y;};
static ulonglong2 make_ulonglong2(uint64_t x,uint64_t y){return {x,y};}
struct epoch_desc_t{uint32_t mid[8];uint8_t early[6];};
static uint8_t WIN3[256][3];
static const uint64_t *input_xyzz;
static uint64_t *observed_scalar;
static void _ModSub256(uint64_t *r,uint64_t *b){field_op(r,r,b,2);}
static uint32_t atomicAdd(uint32_t *r,uint32_t n){return __atomic_fetch_add(r,n,__ATOMIC_RELAXED);}
static int gpu_bench_valid_words(const uint32_t *hs){return (hs[0]&7)==0;}
static void qsb_scheduled_window_hash(uint32_t *h,const epoch_desc_t *desc,int lane){
    for(int k=0;k<8;k++)h[k]=desc->mid[k] ^ ((uint32_t)lane*0x10001u+(uint32_t)k);
}
static void _FixedBaseSignedXYZZ(uint64_t *X,uint64_t *Y,uint64_t *ZZ,uint64_t *ZZZ,
    const uint64_t *z,const uint8_t*,const uint8_t*){
    int idx=blockIdx.x*256+threadIdx.x;
    memcpy(observed_scalar+idx*4,z,32);
    const uint64_t *p=input_xyzz+idx*16;
    Load256(X,p);Load256(Y,p+4);Load256(ZZ,p+8);Load256(ZZZ,p+12);
}
'''
WRAP=r'''
extern "C" int check_roots(const uint64_t *in,uint64_t *out,int n){
    int groups=(n+255)/256;std::vector<uint64_t> roots(in,in+n*4),super(groups*4),tree(groups*4*256);
    inverse_count=0;
    for(int b=0;b<groups;b++)launch(256,[&](int){blockIdx.x=b;qsb_root_group_prepare(roots.data(),n,super.data(),tree.data());});
    launch(256,[&](int){qsb_invert_super_roots(super.data(),groups);});
    for(int b=0;b<groups;b++)launch(256,[&](int){blockIdx.x=b;qsb_root_group_finish(roots.data(),n,super.data(),tree.data());});
    memcpy(out,roots.data(),n*32);return inverse_count;
}
extern "C" int pipeline(const uint64_t *points,const uint64_t *rx,const uint64_t *ry,
    int blocks,uint64_t *scalars,uint32_t *hits,uint8_t *combos,uint32_t *nh){
    int n=blocks*256,groups=(blocks+255)/256;
    std::vector<epoch_desc_t> epochs(blocks);
    for(int b=0;b<blocks;b++){
        for(int k=0;k<8;k++)epochs[b].mid[k]=0x12345678u+(uint32_t)b*12345u+k;
        for(int k=0;k<6;k++)epochs[b].early[k]=b*6+k;
    }
    for(int i=0;i<256;i++)for(int k=0;k<3;k++)WIN3[i][k]=100+(i+k)%50;
    std::vector<ulonglong2> state(n*8);
    std::vector<uint64_t> roots(blocks*4),tree(blocks*4*256),super(groups*4),root_tree(groups*4*256);
    input_xyzz=points;observed_scalar=scalars;*nh=0;inverse_count=0;
    for(int b=0;b<blocks;b++)launch(256,[&](int){blockIdx.x=b;qsb_ranked_prepare(epochs.data(),nullptr,nullptr,rx,state.data(),roots.data(),tree.data(),n);});
    for(int b=0;b<groups;b++)launch(256,[&](int){blockIdx.x=b;qsb_root_group_prepare(roots.data(),blocks,super.data(),root_tree.data());});
    launch(256,[&](int){qsb_invert_super_roots(super.data(),groups);});
    for(int b=0;b<groups;b++)launch(256,[&](int){blockIdx.x=b;qsb_root_group_finish(roots.data(),blocks,super.data(),root_tree.data());});
    for(int b=0;b<blocks;b++)launch(256,[&](int){blockIdx.x=b;qsb_ranked_finish(epochs.data(),rx,ry,state.data(),roots.data(),tree.data(),n,nh,hits,combos);});
    return inverse_count;
}
extern "C" void recover(const uint64_t *in,const uint64_t *rx,const uint64_t *ry,uint64_t *out,uint32_t *parity){
    uint64_t C[4],Y[4],ZZ[4],ZZZ[4],W[5],inv[5],xR[4],yR[4];
    Load256(C,in);Load256(Y,in+4);Load256(ZZ,in+8);Load256(ZZZ,in+12);
    Load256(xR,rx);Load256(yR,ry);
    qsb_xyzz_finish_prepare(C,ZZ,xR,W);_ModSqr(C,C);_ModMult(C,ZZ);
    Load256(inv,W);inv[4]=0;_ModInv(inv);
    *parity=qsb_xyzz_finish_precomputed(C,Y,W,ZZZ,inv,xR,yR,out,out+4);
}
'''
def words(v):return [v>>(64*i)&((1<<64)-1) for i in range(4)]
def buf(vals):return (U64*(4*len(vals)))(*[w for v in vals for w in words(v)])
def integer(vals):return sum(int(v)<<(64*i) for i,v in enumerate(vals))
def expected(x,y,R):return [ref.affine_add((x,y),R),ref.affine_add((x,y),(R[0],(-R[1])%P))]
def main():
    global HERE
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=HERE)
    ap.add_argument('--compact',action='store_true',help='exercise the compact prepare kernel of a dual-table candidate')
    ap.add_argument('--report',type=Path)
    args=ap.parse_args();HERE=args.source.resolve()
    identity=source_identity(HERE);h=(HERE/'tests/gpu_epochs/ranked_pipeline.cuh').read_text()
    h=h[:h.index('static cudaError_t qsb_launch_ranked_pipeline')]
    sha=(HERE/'GPUHash.h').read_text().split('//Modified SHA256 function')[0]
    if args.compact:assert 'qsb_ranked_prepare_compact(' in h,'Candidate has no compact prepare'
    # Both EC producers receive the same controlled points in this pipeline-only
    # audit; actual geometry/curve helpers are checked by check_wide separately.
    stubs=STUBS+'\n#define compact_fixed_xyzz _FixedBaseSignedXYZZ\n'
    wrapper=WRAP.replace('qsb_ranked_prepare(', 'qsb_ranked_prepare_compact(') if args.compact else WRAP
    source='\n'.join([BACKEND,FIELD,stubs,sha,(HERE/'tests/gpu_epochs/tree_inverse.cuh').read_text(),h,wrapper])
    rng=random.Random(260916925);counts={}
    with tempfile.TemporaryDirectory(prefix='qsb-pipeline-source-') as td:
        cpp,so=Path(td)/'pipeline.cpp',Path(td)/'pipeline.so';cpp.write_text(source)
        subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC','-pthread','-I/opt/homebrew/opt/openssl@3/include','-L/opt/homebrew/opt/openssl@3/lib',str(cpp),'-lcrypto','-o',str(so)],check=True)
        lib=CT.CDLL(str(so));ptr=CT.POINTER(U64)
        lib.check_roots.argtypes=[ptr,ptr,CT.c_int]
        lib.recover.argtypes=[ptr,ptr,ptr,ptr,CT.POINTER(U32)]
        lib.pipeline.argtypes=[ptr,ptr,ptr,CT.c_int,ptr,CT.POINTER(U32),CT.POINTER(U8),CT.POINTER(U32)]
        root_count=0
        for n in (1,255,256,257,513):
            values=[rng.randrange(1,P) for _ in range(n)];values[0]=1;values[-1]=P-1
            out=(U64*(n*4))();assert lib.check_roots(buf(values),out,n)==1
            for i,v in enumerate(values):assert integer(out[i*4:i*4+4])==pow(v,-1,P)
            root_count+=n
        counts['hierarchy_leaf_inverses']=root_count
        R=ref.scalar_mult(1984321);pool=[ref.scalar_mult(rng.randrange(1,ref.N)) for _ in range(64)]
        for i in range(2000):
            point=pool[i%len(pool)];z=rng.randrange(1,P);zz=z*z%P;zzz=zz*z%P
            inp=[point[0]*zz%P,point[1]*zzz%P,zz,zzz];out=(U64*8)();parity=U32()
            lib.recover(buf(inp),buf([R[0]]),buf([R[1]]),out,CT.byref(parity))
            exp=expected(*point,R)
            assert [integer(out[0:4]),integer(out[4:8])]==[q[0] for q in exp]
            assert parity.value==(exp[0][1]&1)|((exp[1][1]&1)<<1)
        counts['direct_xyzz_recoveries']=2000
        blocks=3;n=256*blocks;points=[];want={}
        for idx in range(n):
            point=R if idx%256==255 else pool[idx%64]
            z=rng.randrange(1,P);zz=z*z%P;zzz=zz*z%P
            points += [point[0]*zz%P,point[1]*zzz%P,zz,zzz]
            if point==R:continue
            for ri,q in enumerate(expected(*point,R)):
                hs=hashlib.sha256(bytes([2+(q[1]&1)])+q[0].to_bytes(32,'big')).digest()
                if int.from_bytes(hs[:4],'big')&7==0:
                    want[idx]=ri;break
        scalars=(U64*(n*4))();hits=(U32*1024)();combos=(U8*(1024*16))();nh=U32()
        assert lib.pipeline(buf(points),buf([R[0]]),buf([R[1]]),blocks,scalars,hits,combos,CT.byref(nh))==1
        got={}
        for p in range(nh.value):
            idx=hits[p]&0x3fffffff;got[idx]=hits[p]>>30
            assert list(combos[p*16:p*16+9])==[idx//256*6+k for k in range(6)]+[100+((idx%256+k)%50) for k in range(3)]
        assert got==want,(got,want)
        for idx in range(n):
            state=[(0x12345678+(idx//256)*12345+k)^((idx%256)*0x10001+k) for k in range(8)]
            digest=hashlib.sha256(struct.pack('>8I',*state)).digest()
            assert integer(scalars[idx*4:idx*4+4])==int.from_bytes(digest,'big')
        counts.update(pipeline_candidates=n,isolated_singular_lanes=3,verified_hit_records=len(got),pipeline_scalar_inversions=1)
    assert source_identity(HERE)==identity
    result={'status':'PASS',**counts,**identity,'gpu_executed':False,'prepare_branch':'compact' if args.compact else 'default',
        'validation_level':'extracted pipeline kernels, CPU CTAs and OpenSSL field arithmetic',
        'limits':'Controlled first-hash/point-producer inputs, reduced gate for hit coverage; no device arithmetic or GPU execution.'}
    if args.report:args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
