#!/usr/bin/env python3
"""Actual host field/finish source, with PTX add/sub macros emulated on CPU.

Tests exact compressed-key coordinates rather than equality only modulo p.
The inverse is supplied independently by Python. No GPU execution is implied.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import random
import subprocess
import tempfile

def definition(source, signature):
    start=source.index(signature); opening=source.index('{',start)
    depth=1; i=opening+1
    while depth:
        depth+=(source[i]=='{')-(source[i]=='}'); i+=1
    return source[start:i]


ROOT = Path(__file__).resolve().parents[2]
P = 2**256 - 2**32 - 977
U64 = ct.c_uint64

PREAMBLE = r'''
#include <cstdint>
#include <cstring>
#define __device__
#define __forceinline__ inline
#define Load256(a,b) memcpy(a,b,32)
#define _IsPositive(x) (((int64_t)(x[4]))>=0LL)
static unsigned carry;
static uint64_t add64(uint64_t a,uint64_t b,unsigned cin,bool cc){
 __uint128_t n=(__uint128_t)a+b+cin;
 if(cc)carry=n>>64;
 return (uint64_t)n;
}
static uint64_t sub64(uint64_t a,uint64_t b,unsigned bin,bool cc){
 __uint128_t sub=(__uint128_t)b+bin;
 if(cc)carry=(__uint128_t)a<sub;
 return (uint64_t)((__uint128_t)a-sub);
}
#define UADDO(c,a,b) c=add64(a,b,0,true)
#define UADDC(c,a,b) c=add64(a,b,carry,true)
#define UADD(c,a,b) c=add64(a,b,carry,false)
#define UADDO1(c,a) UADDO(c,c,a)
#define UADDC1(c,a) UADDC(c,c,a)
#define UADD1(c,a) UADD(c,c,a)
#define USUBO(c,a,b) c=sub64(a,b,0,true)
#define USUBC(c,a,b) c=sub64(a,b,carry,true)
#define USUB(c,a,b) c=sub64(a,b,carry,false)
#define USUBO1(c,a) USUBO(c,c,a)
#define USUBC1(c,a) USUBC(c,c,a)
#define USUB1(c,a) USUB(c,c,a)
#define SubP(r) {USUBO1(r[0],0xFFFFFFFEFFFFFC2FULL); \
 USUBC1(r[1],UINT64_MAX);USUBC1(r[2],UINT64_MAX); \
 USUBC1(r[3],UINT64_MAX);USUB1(r[4],0ULL);}
'''

WRAPPERS = r'''
extern "C" void prepare(const uint64_t *point,const uint64_t *rx,uint64_t *state){
 uint64_t C[4],Y[4],ZZ[4],ZZZ[4],W[5],xR[4];
 Load256(C,point);Load256(Y,point+4);Load256(ZZ,point+8);Load256(ZZZ,point+12);
 Load256(xR,rx);
 qsb_xyzz_finish_prepare(C,ZZ,xR,W);_ModSqr(C,C);_ModMult(C,ZZ);
 Load256(state,C);Load256(state+4,Y);Load256(state+8,W);Load256(state+12,ZZZ);
}
extern "C" unsigned finish(const uint64_t *state,const uint64_t *inverse,
 const uint64_t *rx,const uint64_t *ry,uint64_t *out){
 uint64_t C[4],Y[4],W[4],ZZZ[4],inv[5]={0},xR[4],yR[4];
 Load256(C,state);Load256(Y,state+4);Load256(W,state+8);Load256(ZZZ,state+12);
 Load256(inv,inverse);Load256(xR,rx);Load256(yR,ry);
 return qsb_xyzz_finish_precomputed(C,Y,W,ZZZ,inv,xR,yR,out,out+4);
}
extern "C" void square(uint64_t *r,const uint64_t *a){_ModSqr(r,a);}
'''


def buf(*values):
    return (U64*(4*len(values)))(*[
        (v >> (64*i)) & ((1 << 64)-1) for v in values for i in range(4)])


def integer(words):
    return sum(int(w) << (64*i) for i, w in enumerate(words))


def add(a, b):
    x, y = a
    u, v = b
    if x == u:
        if (y+v) % P == 0:
            return None
        slope = 3*x*x*pow(2*y, -1, P) % P
    else:
        slope = (v-y)*pow((u-x) % P, -1, P) % P
    xx = (slope*slope-x-u) % P
    return xx, (slope*(x-xx)-y) % P


def witnesses():
    """Three collinear curve points; slope -k makes square near p noncanonical."""
    result = []
    for xq in range(1, 9):
        yy = (xq**3+7) % P
        y = pow(yy, (P+1)//4, P)
        if y*y % P != yy:
            continue
        for yq in [y, -y % P]:
            for k in range(1, 9):
                if k*k < xq:
                    continue
                m = -k % P
                disc = (m**4-6*m*m*xq-3*xq*xq+8*m*yq) % P
                sd = pow(disc, (P+1)//4, P)
                if sd == 0 or sd*sd % P != disc:
                    continue
                xs = [((m*m-xq+sign*sd)*pow(2, -1, P)) % P for sign in [1, -1]]
                points = [(x, (m*(x-xq)+yq) % P) for x in xs]
                assert all(b*b % P == (a*a*a+7) % P for a, b in points)
                assert add(*points) == (xq, -yq % P)
                result.append((*points, k))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=str(ROOT))
    ap.add_argument('--output', type=Path)
    ap.add_argument('--expect-noncanonical', action='store_true')
    args = ap.parse_args()
    repo = ROOT / args.repo
    math = (repo/'candidates/pinning/GPUMath.h').read_text()
    source = (repo/'candidates/pinning/pinning.cu').read_text()
    functions = [
        '__device__ void _ModAdd256(',
        '__device__ void _ModSub256(uint64_t *r, uint64_t *a, uint64_t *b)',
        '__device__ void _ModSub256(uint64_t *r, uint64_t *b)',
        '__device__ __forceinline__ void _ModMultCore',
        '__device__ void _ModMult(uint64_t *r, uint64_t *a, uint64_t *b)',
        '__device__ void _ModMult(uint64_t *r, uint64_t *a)',
        '__device__ __forceinline__ void _ModSqr',
    ]
    code = PREAMBLE + '\n'.join(definition(math, sig) for sig in functions)
    code += '\n'+definition(source, '__device__ __forceinline__ void qsb_xyzz_finish_prepare(')
    code += '\n'+definition(source, '__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed(')
    code += WRAPPERS
    cases = witnesses()
    assert cases
    rng = random.Random(2026091810)
    scales = [1, 2, P-1, P-2, P-65537, rng.randrange(1, P)]
    failures = []
    with tempfile.TemporaryDirectory(prefix='qsb-canonical-recovery-') as td:
        cpp = Path(td)/'audit.cpp'
        so = Path(td)/'audit.so'
        cpp.write_text(code)
        subprocess.run(['g++', '-std=c++17', '-O2', '-shared', '-fPIC',
                        '-fsanitize=undefined', '-fno-sanitize-recover=all',
                        str(cpp), '-o', str(so)], check=True)
        lib = ct.CDLL(str(so))
        ptr = ct.POINTER(U64)
        lib.prepare.argtypes = [ptr, ptr, ptr]
        lib.finish.argtypes = [ptr]*5
        lib.finish.restype = ct.c_uint
        for point, R, k in cases:
            expected = [add(point, R), add(point, (R[0], -R[1] % P))]
            assert all(q is not None for q in expected)
            for z in scales:
                zz, zzz = z*z % P, z*z*z % P
                projective = buf(point[0]*zz % P, point[1]*zzz % P, zz, zzz)
                state = (U64*16)()
                lib.prepare(projective, buf(R[0]), state)
                W = integer(state[8:12]) % P
                assert W
                inv = buf(pow(W, -1, P))
                out = (U64*8)()
                parities = lib.finish(state, inv, buf(R[0]), buf(R[1]), out)
                got = [integer(out[:4]), integer(out[4:])]
                want = [q[0] for q in expected]
                bits = (expected[0][1] & 1) | ((expected[1][1] & 1) << 1)
                if got != want or parities != bits:
                    failures.append({'slope_abs': k, 'P': point, 'R': R, 'scale': z,
                                     'got_x': got, 'expected_x': want,
                                     'parities': parities, 'expected_parities': bits})
    report = {'repo': str(repo), 'valid_curve_pairs': len(cases),
              'cases': len(cases)*len(scales), 'failures': len(failures),
              'first_failures': failures[:3], 'gpu_executed': False, 'ubsan': True,
              'source_sha256': {'GPUMath.h': hashlib.sha256(math.encode()).hexdigest(),
                                'pinning.cu': hashlib.sha256(source.encode()).hexdigest()},
              'scope': 'Actual host mul/sqr, actual add/sub bodies with fixed-width PTX macro semantics; actual recovery; independent affine curve oracle'}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    assert bool(failures) == args.expect_noncanonical


if __name__ == '__main__':
    main()
