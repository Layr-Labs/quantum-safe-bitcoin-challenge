#!/usr/bin/env python3
"""Execute extracted deferred point helpers with CPU/OpenSSL field operations.

This checks production helper ordering/aliasing, not CUDA field primitives.
The independent integer model also compares on-curve sums to affine addition.
"""
import ctypes as CT
import hashlib
import json
import os
import re
from pathlib import Path
import random
import shlex
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
import audit_integrated as ref
from check_candidate import BACKEND, function
from preflight import source_identity

HERE = Path(__file__).resolve().parent
FIELD = r'''
static void field_op(uint64_t *r,const uint64_t *a,const uint64_t *b,int op){
    BIGNUM *aa=BN_lebin2bn((const uint8_t*)a,32,nullptr);
    BIGNUM *bb=BN_lebin2bn((const uint8_t*)b,32,nullptr),*rr=BN_new();
    int ok=op==0 ? BN_mod_mul(rr,aa,bb,field.p,field.ctx) :
           op==1 ? BN_mod_add(rr,aa,bb,field.p,field.ctx) :
                   BN_mod_sub(rr,aa,bb,field.p,field.ctx);
    require(ok && BN_bn2lebinpad(rr,(uint8_t*)r,32)==32);
    BN_free(aa);BN_free(bb);BN_free(rr);
}
static void _ModMult(uint64_t *r,uint64_t *a,uint64_t *b){field_op(r,a,b,0);}
static void _ModMult(uint64_t *r,uint64_t *b){field_op(r,r,b,0);}
static void _ModSqr(uint64_t *r,uint64_t *a){field_op(r,a,a,0);}
static void _ModAdd256(uint64_t *r,uint64_t *a,uint64_t *b){field_op(r,a,b,1);}
static void _ModSub256(uint64_t *r,uint64_t *a,uint64_t *b){field_op(r,a,b,2);}
static void Load256(uint64_t *r,const uint64_t *a){memcpy(r,a,32);}
'''
RAW_STUBS = r'''
static const uint64_t *raw_points;
static void gt_recode_setup(const uint64_t *,uint64_t *,int *sign){*sign=1;}
static int32_t gt_recode_step(uint64_t *,int,int){return 1;}
static void gt_load_signed(const uint8_t*,const uint8_t*,int c,uint32_t,uint64_t,uint64_t*x,uint64_t*y){
    Load256(x,raw_points+8*c);Load256(y,raw_points+8*c+4);
}
'''

WRAPPER = r'''
extern "C" void chain(const uint64_t *points,uint64_t *states){
    uint64_t X[4],Y[4],ZZ[4],ZZZ[4],anchor[4];
    _PointAddXYZZ_mm(X,Y,ZZ,ZZZ,points,points+4,points+8,points+12);
    Load256(anchor,points+4);
    for(int c=1;c<16;c++){
        if(c>1){
            if(c<15)_PointAddXYZZ<true>(X,Y,ZZ,ZZZ,points+8*c,points+8*c+4,anchor);
            else _PointAddXYZZ<false>(X,Y,ZZ,ZZZ,points+8*c,points+8*c+4,anchor);
            Load256(anchor,points+8*c+4);
        }
        uint64_t *out=states+16*(c-1);
        memcpy(out,X,32);memcpy(out+4,Y,32);memcpy(out+8,ZZ,32);memcpy(out+12,ZZZ,32);
    }
}
'''


def main():
    identity = source_identity()
    m = (HERE/'GPUMath.h').read_text()
    tree = (HERE/'tests/gpu_epochs/tree.cu').read_text()
    source = '\n'.join([BACKEND, FIELD, re.search(r'^#define GT_CHUNKS.*$',tree,re.M)[0], function(m, 'template<bool DEFER_Y>'),
                        function(m, '__device__ void _PointAddXYZZ_mm('), RAW_STUBS,
                        function(tree,'__device__ __forceinline__ void gt_digit_idx('),
                        function(tree,'__device__ void _FixedBaseSignedXYZZ('), WRAPPER, r'''
extern "C" void raw_chain(const uint64_t *points,uint64_t *out){
    raw_points=points;uint64_t scalar[4]={0,0,0,0};
    _FixedBaseSignedXYZZ(out,out+4,out+8,out+12,scalar,nullptr,nullptr);
}
'''])
    rng = random.Random(260916404)
    flags = []
    prefix = os.environ.get('OPENSSL_PREFIX')
    if not prefix and Path('/opt/homebrew/opt/openssl@3').exists():
        prefix = '/opt/homebrew/opt/openssl@3'
    if prefix:
        flags = [f'-I{prefix}/include', f'-L{prefix}/lib']
    with tempfile.TemporaryDirectory(prefix='qsb-deferred-source-') as tmp:
        cpp, libfile = Path(tmp)/'chain.cpp', Path(tmp)/'chain.so'
        cpp.write_text(source)
        subprocess.run(shlex.split(os.environ.get('CXX', 'c++')) +
                       ['-std=c++17', '-O2', '-shared', '-fPIC', '-pthread', *flags,
                        str(cpp), '-lcrypto', '-o', str(libfile)], check=True)
        lib = CT.CDLL(str(libfile))
        lib.chain.argtypes = [CT.POINTER(CT.c_uint64), CT.POINTER(CT.c_uint64)]
        lib.raw_chain.argtypes = lib.chain.argtypes
        pool = [ref.scalar_mult(rng.randrange(1, ref.N)) for _ in range(64)]
        counts = {'arbitrary_field_chains': 0, 'curve_chains': 0, 'intermediate_states': 0}
        for curve, limit in ((False, 2000), (True, 128)):
            done = 0
            while done < limit:
                points = tuple(rng.choice(pool) if curve else
                               (rng.randrange(ref.P), rng.randrange(ref.P)) for _ in range(16))
                if not ref.denominators_nonzero(points):
                    continue
                raw = [v >> (64*k) & ((1 << 64)-1) for point in points for v in point for k in range(4)]
                out = (CT.c_uint64*(15*16))()
                lib.chain((CT.c_uint64*128)(*raw), out)
                exact = ref.mm(points[0], points[1])
                anchor = points[0][1]
                for c in range(1, 16):
                    if c > 1:
                        exact = ref.madd_old(exact, points[c])
                        anchor = points[c][1]
                    expected = list(exact)
                    if c < 15:
                        expected[1] = (expected[1] + anchor*expected[3]) % ref.P
                    got = tuple(sum(out[16*(c-1)+4*j+k] << (64*k) for k in range(4)) for j in range(4))
                    assert got == tuple(expected), (curve, done, c)
                    counts['intermediate_states'] += 1
                raw_out = (CT.c_uint64*16)()
                lib.raw_chain((CT.c_uint64*128)(*raw),raw_out)
                assert list(raw_out) == list(out[-16:]), "raw producer differs from helper chain"
                if curve:
                    affine = None
                    for point in points:
                        affine = ref.affine_add(affine, point)
                    assert ref.normalize(got) == affine
                done += 1
            counts['curve_chains' if curve else 'arbitrary_field_chains'] = done
    assert source_identity() == identity, 'Source changed during audit'
    print(json.dumps({'status': 'PASS', 'validation_level': 'cpu_source_with_openssl',
                      **counts, 'raw_producer_chains':2128, **identity, 'gpu_executed': False,
                      'audit_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'limitations': 'Extracted point helpers and raw producer with controlled table lookups; no CUDA primitives, GPU timing or singular cases.'}, indent=2))


if __name__ == '__main__':
    main()
