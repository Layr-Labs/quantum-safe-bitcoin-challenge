#!/usr/bin/env python3
"""Execute extracted pipeline kernels on CPU with OpenSSL and simulated CTAs.

The EC producer and scheduled first-hash have controlled inputs here; their
actual helpers are checked separately. No GPU scheduling/codegen/timing claim.
"""
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
    const uint64_t *z,const uint8_t*){
    int idx=blockIdx.x*256+threadIdx.x;
    memcpy(observed_scalar+idx*4,z,32);
    const uint64_t *p=input_xyzz+idx*16;
    Load256(X,p);Load256(Y,p+4);Load256(ZZ,p+8);Load256(ZZZ,p+12);
}
'''
WRAP=r'''
static constexpr uint64_t checkpoint_canary=0xd4a19e63827bf005ULL;
struct GuardedWords{
    size_t n;std::vector<uint64_t> storage;
    explicit GuardedWords(size_t size):n(size),storage(size+64,checkpoint_canary){}
    uint64_t *data(){return storage.data()+32;}
    void check(){
        for(int i=0;i<32;i++)require(storage[i]==checkpoint_canary && storage[32+n+i]==checkpoint_canary);
    }
};
extern "C" void checkpoint_roundtrip(const uint64_t *in,uint64_t *out,
    uint64_t *saved,uint64_t *root,int block,int *counts){
    const size_t stride=4*QSB_CHECKPOINT_STRIDE;
    GuardedWords roots((block+1)*4),tree((block+1)*stride);
    std::vector<int> barriers(256),warps(256);
    multiply_count=0;
    launch(256,[&](int i){
        blockIdx.x=block;uint64_t value[5]={};memcpy(value,in+4*i,32);
        qsb_block_product_checkpoint(value,roots.data(),tree.data());barriers[i]=block_syncs;warps[i]=warp_syncs;
    });
    for(int i=1;i<256;i++)require(barriers[i]==barriers[0]);
    counts[0]=multiply_count;counts[1]=barriers[0];counts[4]=warps[0];
    memcpy(root,roots.data()+4*block,32);memcpy(saved,tree.data()+block*stride,stride*8);
    uint64_t inv[5]={};memcpy(inv,root,32);_ModInv(inv);memcpy(roots.data()+4*block,inv,32);
    multiply_count=0;
    launch(256,[&](int i){
        blockIdx.x=block;uint64_t value[5]={};memcpy(value,in+4*i,32);
        qsb_block_inverse_checkpoint(value,roots.data(),tree.data());memcpy(out+4*i,value,32);
        barriers[i]=block_syncs;warps[i]=warp_syncs;
    });
    for(int i=1;i<256;i++)require(barriers[i]==barriers[0]);
    counts[2]=multiply_count;counts[3]=barriers[0];counts[5]=warps[0];
    roots.check();tree.check();
    for(size_t i=0;i<block*stride;i++)require(tree.data()[i]==checkpoint_canary);
    for(int i=0;i<block*4;i++)require(roots.data()[i]==checkpoint_canary);
    for(int k=0;k<4;k++)for(int i=QSB_CHECKPOINT_NODES;i<QSB_CHECKPOINT_STRIDE;i++)
        require(tree.data()[block*stride+k*QSB_CHECKPOINT_STRIDE+i]==checkpoint_canary);
    require(memcmp(saved,tree.data()+block*stride,stride*8)==0);
}
extern "C" int check_roots(const uint64_t *in,uint64_t *out,int n){
    int groups=(n+255)/256;
    GuardedWords roots(n*4),super(groups*4),tree(groups*4*QSB_CHECKPOINT_STRIDE);
    memcpy(roots.data(),in,n*32);
    inverse_count=0;
    for(int b=0;b<groups;b++)launch(256,[&](int){blockIdx.x=b;qsb_root_group_prepare(roots.data(),n,super.data(),tree.data());});
    launch(256,[&](int){qsb_invert_super_roots(super.data(),groups);});
    for(int b=0;b<groups;b++)launch(256,[&](int){blockIdx.x=b;qsb_root_group_finish(roots.data(),n,super.data(),tree.data());});
    roots.check();super.check();tree.check();
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
    require(qsb_prepare_recovery_point((const uint8_t*)rx,(const uint8_t*)ry)==0);
    require(memcmp(QSB_U2R,rx,32)==0 && memcmp(QSB_U2R+4,ry,32)==0);
    require(qsb_select_window_triples(WIN3)==0);
    std::vector<ulonglong2> state(n*8);
    std::vector<uint64_t> roots(blocks*4),tree(blocks*4*QSB_CHECKPOINT_STRIDE),super(groups*4),root_tree(groups*4*QSB_CHECKPOINT_STRIDE);
    input_xyzz=points;observed_scalar=scalars;*nh=0;inverse_count=0;
    for(int b=0;b<blocks;b++)launch(256,[&](int){blockIdx.x=b;qsb_ranked_prepare(epochs.data(),nullptr,nullptr,state.data(),roots.data(),tree.data(),n);});
    for(int b=0;b<groups;b++)launch(256,[&](int){blockIdx.x=b;qsb_root_group_prepare(roots.data(),blocks,super.data(),root_tree.data());});
    launch(256,[&](int){qsb_invert_super_roots(super.data(),groups);});
    for(int b=0;b<groups;b++)launch(256,[&](int){blockIdx.x=b;qsb_root_group_finish(roots.data(),blocks,super.data(),root_tree.data());});
    for(int b=0;b<blocks;b++)launch(256,[&](int){blockIdx.x=b;qsb_ranked_finish(epochs.data(),nullptr,nullptr,state.data(),roots.data(),tree.data(),n,nh,hits,combos);});
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
    identity=source_identity();h=(HERE/'tests/gpu_epochs/ranked_pipeline.cuh').read_text()
    h=h[:h.index('static cudaError_t qsb_launch_ranked_pipeline')]
    sha=(HERE/'GPUHash.h').read_text().split('//Modified SHA256 function')[0]
    tree=(HERE/'tests/gpu_epochs/tree.cu').read_text()
    assert 'qsb_prepare_recovery_point(dp.u2r_x,dp.u2r_y)' in tree
    assert 'qsb_select_window_triples(h_win3)' in tree
    window=(HERE/'tests/gpu_epochs/window_schedule_shared.cuh').read_text()
    select='\n'.join(function(window,s) for s in (
        'static uint32_t qsb_window_second_key(', 'static uint32_t qsb_window_first_key(',
        'static int qsb_select_window_triples('))
    source='\n'.join([BACKEND,FIELD,STUBS,sha,select,(HERE/'tests/gpu_epochs/tree_inverse.cuh').read_text(),h,WRAP])
    rng=random.Random(260916925);counts={}
    with tempfile.TemporaryDirectory(prefix='qsb-pipeline-source-') as td:
        cpp,so=Path(td)/'pipeline.cpp',Path(td)/'pipeline.so';cpp.write_text(source)
        subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC','-pthread','-I/opt/homebrew/opt/openssl@3/include','-L/opt/homebrew/opt/openssl@3/lib',str(cpp),'-lcrypto','-o',str(so)],check=True)
        lib=CT.CDLL(str(so));ptr=CT.POINTER(U64)
        lib.check_roots.argtypes=[ptr,ptr,CT.c_int]
        lib.recover.argtypes=[ptr,ptr,ptr,ptr,CT.POINTER(U32)]
        lib.pipeline.argtypes=[ptr,ptr,ptr,CT.c_int,ptr,CT.POINTER(U32),CT.POINTER(U8),CT.POINTER(U32)]
        lib.checkpoint_roundtrip.argtypes=[ptr,ptr,ptr,ptr,CT.c_int,CT.POINTER(CT.c_int)]
        # Independent node oracle: a node is the product of leaves at a fixed
        # residue modulo its width. Check the entire compact wire format,
        # untouched padding and guard regions, then all 256 leaf inverses.
        checkpoint_cases=0
        for block in (0,1,255):
            patterns=[[1]*256,[P-1]*256,
                      [1 if i%3==0 else P-65537 if i%3==1 else P-1 for i in range(256)],
                      [rng.randrange(1,P) for _ in range(256)]]
            for values in patterns:
                out=(U64*1024)();saved=(U64*(4*64))();root=(U64*4)();ops=(CT.c_int*6)()
                lib.checkpoint_roundtrip(buf(values),out,saved,root,block,ops)
                assert list(ops)==[255,3,702,5,5,4],list(ops)
                product=1
                for v in values:product=product*v%P
                assert integer(root)==product
                expected_saved=[]
                for width in (32,16,8,4,2):
                    for lane in range(width):
                        product=1
                        for i in range(lane,256,width):product=product*values[i]%P
                        expected_saved.append(product)
                assert len(expected_saved)==62
                for j,v in enumerate(expected_saved):
                    assert integer([saved[k*64+j] for k in range(4)])==v
                for i,v in enumerate(values):assert integer(out[i*4:i*4+4])==pow(v,-1,P)
                checkpoint_cases+=1
        counts.update(checkpoint_roundtrips=checkpoint_cases,checkpoint_leaf_inverses=checkpoint_cases*256,
                      checkpoint_canaries=True,prepare_multiplies=255,finish_multiplies=702,
                      prepare_block_barriers=3,finish_block_barriers=5,prepare_warp_barriers=5,finish_warp_barriers=4)
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
        import itertools
        windows=[w for w in itertools.combinations(range(137,150),3)
                 if not (w[0]>=138 and w[2]<=144 and w[:2]!=(138,139))]
        def key(w):
            kept=[i for i in range(137,150) if i not in w]
            return tuple(reversed(kept[-5:])),tuple(kept[:6])
        windows.sort(key=key)
        got={}
        for p in range(nh.value):
            idx=hits[p]&0x3fffffff;got[idx]=hits[p]>>30
            assert list(combos[p*16:p*16+9])==[idx//256*6+k for k in range(6)]+list(windows[idx%256])
        assert got==want,(got,want)
        for idx in range(n):
            state=[(0x12345678+(idx//256)*12345+k)^((idx%256)*0x10001+k) for k in range(8)]
            digest=hashlib.sha256(struct.pack('>8I',*state)).digest()
            assert integer(scalars[idx*4:idx*4+4])==int.from_bytes(digest,'big')
        counts.update(pipeline_candidates=n,isolated_singular_lanes=3,verified_hit_records=len(got),pipeline_scalar_inversions=1,runtime_constant_upload=True,real_window_mapping=True)
    assert source_identity()==identity
    print(json.dumps({'status':'PASS',**counts,**identity,'gpu_executed':False,
        'validation_level':'extracted pipeline kernels, CPU CTAs and OpenSSL field arithmetic',
        'limits':'Controlled first-hash/point-producer inputs, reduced gate for hit coverage; no device arithmetic or GPU execution.'},indent=2))
if __name__=='__main__':main()
