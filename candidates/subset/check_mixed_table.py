#!/usr/bin/env python3
"""CPU audit of extracted mixed table construction, loads, chain and recovery.

Executes actual source with OpenSSL field replacements and carry-chain stubs.
Independent EC_POINT_mul supplies the oracle. This is not CUDA execution.
"""
import ctypes as CT
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
from check_candidate import BACKEND, function
from check_deferred_source import FIELD
from preflight import source_identity

HERE = Path(__file__).resolve().parent
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
U64 = CT.c_uint64

SUPPORT = r'''
#include <algorithm>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
static uint64_t hot_muls,hot_squares,carry_bit;
#define UADDO1(c,a) do { __uint128_t t=(__uint128_t)(c)+(a); (c)=(uint64_t)t; carry_bit=t>>64; } while(0)
#define UADDC1(c,a) do { __uint128_t t=(__uint128_t)(c)+(a)+carry_bit; (c)=(uint64_t)t; carry_bit=t>>64; } while(0)
#define UADD1(c,a) do { (c)=(uint64_t)((__uint128_t)(c)+(a)+carry_bit); } while(0)
struct ulonglong2{uint64_t x,y;};
'''
WRAPPER = r'''
static const uint64_t canary=0xb4316782d9acfe05ULL;
static std::vector<uint64_t> table_words((size_t)GT_TOTAL_ENTRIES*8+8,canary);
static uint8_t *table=(uint8_t*)(table_words.data()+4);
static std::vector<uint8_t> ready(GT_TOTAL_ENTRIES);
static std::vector<uint64_t> ladder_l(GT_CHUNKS*GT_LO*8),ladder_h(GT_CHUNKS*GT_HI*8);
static uint64_t factor_words[4];
static EC_GROUP *ec_group=EC_GROUP_new_by_curve_name(NID_secp256k1);
static BN_CTX *ec_ctx=BN_CTX_new();
static BIGNUM *ec_order=BN_new(),*ec_factor=BN_new();
static void check_guards(){
    for(int i=0;i<4;i++)require(table_words[i]==canary && table_words[table_words.size()-1-i]==canary);
}
static void ensure_entry(int c,unsigned index){
    require(c>=0 && c<GT_CHUNKS && index<gt_entries(c));
    unsigned flat=gt_offset(c)+index;
    require(flat<GT_TOTAL_ENTRIES);
    if(!ready[flat]){
        blockDim.x=256;blockIdx.x=flat/256;threadIdx.x=flat%256;
        kernel_build_gtable(ladder_l.data(),ladder_h.data(),table);
        ready[flat]=1;
    }
    check_guards();
}
static bool affine(const BIGNUM *scalar,uint64_t *x,uint64_t *y){
    EC_POINT *point=EC_POINT_new(ec_group);BN_CTX_start(ec_ctx);
    BIGNUM *ax=BN_CTX_get(ec_ctx),*ay=BN_CTX_get(ec_ctx);
    require(EC_POINT_mul(ec_group,point,scalar,nullptr,nullptr,ec_ctx));
    bool finite=!EC_POINT_is_at_infinity(ec_group,point);
    if(finite){
        require(EC_POINT_get_affine_coordinates(ec_group,point,ax,ay,ec_ctx));
        require(BN_bn2lebinpad(ax,(uint8_t*)x,32)==32);
        require(BN_bn2lebinpad(ay,(uint8_t*)y,32)==32);
    }
    BN_CTX_end(ec_ctx);EC_POINT_free(point);return finite;
}
extern "C" void set_base(const uint64_t *factor){
    memcpy(factor_words,factor,32);EC_GROUP_get_order(ec_group,ec_order,ec_ctx);
    BN_lebin2bn((uint8_t*)factor,32,ec_factor);
    gt_build_ladders(ladder_l.data(),ladder_h.data(),(uint8_t*)factor);
    std::fill(ready.begin(),ready.end(),0);
    std::fill(table_words.begin(),table_words.end(),canary);
    // An out-of-range builder thread must not touch the table or its guards.
    blockDim.x=256;blockIdx.x=GT_TOTAL_ENTRIES/256;threadIdx.x=0;
    kernel_build_gtable(ladder_l.data(),ladder_h.data(),table);check_guards();
}
extern "C" void table_probe(int c,unsigned index,int neg,uint64_t *out){
    ensure_entry(c,index);gt_load_signed(table,c,index,neg,out,out+4);
    BN_CTX_start(ec_ctx);BIGNUM *k=BN_CTX_get(ec_ctx),*half=BN_CTX_get(ec_ctx);
    BN_copy(half,ec_order);BN_add_word(half,1);BN_rshift1(half,half);
    // Independent layout oracle: do not use gt_shift/gt_offset here.
    BN_set_word(k,2*index+1);BN_lshift(k,k,c==0?0:18+17*(c-1));
    BN_mod_mul(k,k,half,ec_order,ec_ctx);BN_mod_mul(k,k,ec_factor,ec_order,ec_ctx);
    if(neg)BN_sub(k,ec_order,k);
    require(affine(k,out+8,out+12));BN_CTX_end(ec_ctx);
}
extern "C" int chain_probe(const uint64_t *scalar,const uint64_t *r_scalar,uint64_t *out,uint64_t *counts){
    int32_t digits[GT_CHUNKS];gt_recode_signed(scalar,digits);
    for(int c=0;c<GT_CHUNKS;c++){
        uint32_t index;uint64_t neg;gt_digit_idx(digits[c],&index,&neg);ensure_entry(c,index);
    }
    uint64_t C[4],Y[4],ZZ[4],ZZZ[4],W[5],xR[4],yR[4];
    BN_CTX_start(ec_ctx);
    BIGNUM *rs=BN_CTX_get(ec_ctx),*k=BN_CTX_get(ec_ctx),*sum=BN_CTX_get(ec_ctx),*diff=BN_CTX_get(ec_ctx);
    BN_lebin2bn((const uint8_t*)r_scalar,32,rs);require(affine(rs,xR,yR));
    hot_muls=hot_squares=0;
    _FixedBaseSignedXYZZ(C,Y,ZZ,ZZZ,scalar,table);
    qsb_xyzz_finish_prepare(C,ZZ,xR,W);
    _ModSqr(C,C);_ModMult(C,ZZ);
    if(!(W[0]|W[1]|W[2]|W[3])){BN_CTX_end(ec_ctx);return 0;}
    uint64_t inv[5]={};Load256(inv,W);_ModInv(inv);
    out[8]=qsb_xyzz_finish_precomputed(C,Y,W,ZZZ,inv,xR,yR,out,out+4);
    counts[0]=hot_muls;counts[1]=hot_squares;
    BN_lebin2bn((const uint8_t*)scalar,32,k);BN_mod_mul(k,k,ec_factor,ec_order,ec_ctx);
    BN_mod_add(sum,k,rs,ec_order,ec_ctx);BN_mod_sub(diff,k,rs,ec_order,ec_ctx);
    uint64_t y1[4],y2[4];
    require(affine(sum,out+9,y1) && affine(diff,out+13,y2));
    out[17]=(y1[0]&1)|((y2[0]&1)<<1);
    BN_CTX_end(ec_ctx);return 1;
}
'''

def words(x):
    return (U64*4)(*[x >> (64*i) & ((1 << 64)-1) for i in range(4)])

def main():
    identity = source_identity()
    tree = (HERE/'tests/gpu_epochs/tree.cu').read_text()
    math = (HERE/'GPUMath.h').read_text()
    pipeline = (HERE/'tests/gpu_epochs/ranked_pipeline.cuh').read_text()
    counted_field = FIELD.replace('field_op(r,a,b,0);', '++hot_muls;field_op(r,a,b,0);')
    counted_field = counted_field.replace('field_op(r,r,b,0);', '++hot_muls;field_op(r,r,b,0);')
    counted_field = counted_field.replace('field_op(r,a,a,0);', '++hot_squares;field_op(r,a,a,0);')
    source = '\n'.join([BACKEND, SUPPORT, counted_field,
        'static void _ModSub256(uint64_t*r,uint64_t*b){field_op(r,r,b,2);}',
        function(math, 'template<bool DEFER_Y>'),
        function(math, '__device__ void _PointAddXYZZ_mm('),
        function(math, '__device__ void _PointAddSecp256k1('),
        tree[tree.index('#define GT_CHUNKS'):tree.index('/* _FixedBaseSignedAffine: removed')],
        function(tree, '__global__ void kernel_build_gtable('),
        function(tree, 'static void gt_point_to_limbs('),
        function(tree, 'static void gt_build_ladders('),
        function(pipeline, '__device__ __forceinline__ void qsb_xyzz_finish_prepare('),
        function(pipeline, '__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed('), WRAPPER])
    rng = random.Random(260917111)
    counts = {'runtime_bases': 0, 'table_loads': 0, 'recovered_pairs': 0, 'singular_skips': []}
    flags = []
    prefix = os.environ.get('OPENSSL_PREFIX')
    if prefix: flags = [f'-I{prefix}/include', f'-L{prefix}/lib']
    with tempfile.TemporaryDirectory(prefix='qsb-mixed-table-') as td:
        cpp, so = Path(td)/'check.cpp', Path(td)/'check.so'
        cpp.write_text(source)
        subprocess.run(shlex.split(os.environ.get('CXX', 'c++')) +
            ['-std=c++17', '-O2', '-shared', '-fPIC', '-pthread',
             '-Wno-deprecated-declarations', *flags, str(cpp), '-lcrypto', '-o', str(so)], check=True)
        lib = CT.CDLL(str(so));ptr = CT.POINTER(U64)
        lib.set_base.argtypes = [ptr]
        lib.table_probe.argtypes = [CT.c_int, CT.c_uint, CT.c_int, ptr]
        lib.chain_probe.argtypes = [ptr, ptr, ptr, ptr]
        for base in (1, N-1, rng.randrange(1, N)):
            lib.set_base(words(base));counts['runtime_bases'] += 1
            for c in range(15):
                entries = 1 << (17 if c == 0 else 16)
                probes = [0,1,2,127,128,129,entries//2-1,entries//2,entries-2,entries-1]
                probes += [rng.randrange(entries) for _ in range(8)]
                for index in probes:
                    for neg in (0,1):
                        out = (U64*16)();lib.table_probe(c,index,neg,out)
                        assert list(out[:8]) == list(out[8:]), (base,c,index,neg)
                        counts['table_loads'] += 1
            edges = [0,1,2,N-1,N,N+1,(1<<256)-1,0xffff<<240,((1<<17)-1)<<239]
            scalars = edges + [1 << b for b in range(256)] + [rng.getrandbits(256) for _ in range(400)]
            for scalar in scalars:
                out, ops = (U64*18)(), (U64*2)()
                ok = lib.chain_probe(words(scalar),words(1984321),out,ops)
                if not ok:
                    assert scalar in edges, ('unexpected singular random/power scalar',base,scalar)
                    counts['singular_skips'].append({'base':str(base),'scalar':str(scalar)})
                    continue
                assert list(out[:8]) == list(out[9:17]) and out[8] == out[17], (base,scalar)
                assert list(ops) == [105,32], list(ops)
                counts['recovered_pairs'] += 1
    assert source_identity() == identity
    print(json.dumps({'status':'PASS', **counts, 'hot_multiplications':105,'hot_squares':32,
        'gpu_executed':False, **identity,
        'audit_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limits':'CPU execution of extracted builder, ladders, table loads, recoder, point chain and recovery; OpenSSL field arithmetic, emulated carry macros and individual inverse. No GPU execution or timing.'}, indent=2))

if __name__ == '__main__':
    main()
