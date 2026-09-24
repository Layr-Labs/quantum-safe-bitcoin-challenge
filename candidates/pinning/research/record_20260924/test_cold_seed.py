#!/usr/bin/env python3
"""Source-derived GLV consumption-order audit against independent OpenSSL points.

Compiles the production geometry, slot mapping, decoders, full chain, and both
point-add bodies. Inline PTX field operations and the GLV split are replaced by
exact host arithmetic; the latter uses independently rounded Python integers.
No GPU timing, rare-carry equivalence, or CUDA memory-safety claim is made.
The sparse-table/OpenSSL approach follows inherited wide_windows/check_wide.py;
its missing check_projective module is not imported or copied. Backend is new.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
P = (1 << 256) - (1 << 32) - 977
LAMBDA = 0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72
BOUND = 0xA2A8918CA85BAFE22016D0B917E4DD77
G1 = 0x3086D221A7D46BCDE86C90E49284EB153DAA8A1471E8CA7FE893209A45DBB031
G2 = 0xE4437ED6010E88286F547FA90ABFE4C4221208AC9DF506C61571B4AE8AC47F71
A = 0x3086D221A7D46BCDE86C90E49284EB15
B = 0xE4437ED6010E88286F547FA90ABFE4C3
MASK = (1 << 64) - 1


def function(source, name):
    """Extract definition, ignoring braces in comments and string literals."""
    start = source.index(name)
    brace = source.index('{', start)
    depth = 0
    token = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\{|\}', re.S)
    for match in token.finditer(source, brace):
        item = match.group()
        if item == '{':
            depth += 1
        elif item == '}':
            depth -= 1
            if depth == 0:
                return source[start:match.end()]
    raise ValueError(name)


def limbs(value, count=4):
    return (C.c_uint64 * count)(*(value >> (64 * i) & MASK for i in range(count)))


def split(scalar):
    k = scalar % N
    c1 = (k * G1 + (1 << 383)) >> 384
    c2 = (k * G2 + (1 << 383)) >> 384
    r1, r2 = k - c1 * A - c2 * (A + B), c1 * B - c2 * A
    assert (r1 + LAMBDA * r2 - k) % N == 0
    assert max(abs(r1), abs(r2)) <= BOUND
    return r1, r2


BACKEND = r'''
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#define __host__
#define __device__
#define __forceinline__ inline
#define QSB_TREE_N 128
#define GT_CHUNKS 6
#define GT_GLV_TERMS 12
#define QSB_GLV_SEED_REG 1
#define QSB_YOFF 1
#define QSB_LAZY 1
#define QSB_NEG_Y_MAC 1
#define QSB_XY_DIRECT 1
#define QSB_FUSE_SQRADDSUB2 1
static struct { unsigned x; } threadIdx={0};
static BN_CTX *ctx;
static EC_GROUP *group;
static BIGNUM *prime,*order,*base,*scale_x,*scale_y;
static uint64_t components[2][2];static unsigned signs[2];
static uint64_t arena[12*QSB_TREE_N];
static void require(int value){if(!value)abort();}
static BIGNUM *read256(const uint64_t *a){return BN_lebin2bn((const unsigned char*)a,32,nullptr);}
static void write256(uint64_t *out,const BIGNUM *a){require(BN_bn2lebinpad(a,(unsigned char*)out,32)==32);}
static void Load256(uint64_t *r,const uint64_t *a){memcpy(r,a,32);}
static void field(uint64_t *r,const uint64_t *a,const uint64_t *b,int mode){
 BIGNUM *x=read256(a),*y=read256(b),*z=BN_new();
 if(mode==0)require(BN_mod_mul(z,x,y,prime,ctx));
 else if(mode==1)require(BN_mod_add(z,x,y,prime,ctx));
 else require(BN_mod_sub(z,x,y,prime,ctx));
 write256(r,z);BN_free(x);BN_free(y);BN_free(z);
}
static void _ModMult(uint64_t*r,const uint64_t*a,const uint64_t*b){field(r,a,b,0);}
static void _ModMult(uint64_t*r,const uint64_t*a){field(r,r,a,0);}
static void _ModSqr(uint64_t*r,const uint64_t*a){field(r,a,a,0);}
static void _ModSub256(uint64_t*r,const uint64_t*a,const uint64_t*b){field(r,a,b,2);}
static void _ModAdd256(uint64_t*r,const uint64_t*a,const uint64_t*b){field(r,a,b,1);}
static void _ModAddLazy(uint64_t*r,const uint64_t*a,const uint64_t*b){field(r,a,b,1);}
static void _ModAddLazyOff(uint64_t*r,const uint64_t*a,const uint64_t*b){
 uint64_t off[4]={0x1000003d0ULL,0,0,0};field(r,a,b,1);field(r,r,off,2);
}
static void qsb_yoff_to_y(uint64_t*y){uint64_t off[4]={0x800001e8ULL,0,0,0};field(y,y,off,2);}
static void qsb_negate_residue(uint64_t*y){uint64_t zero[4]={};field(y,zero,y,2);}
static void qsb_muladd_seed(uint64_t*r,const uint64_t*a,const uint64_t*b,const uint64_t*c){
 uint64_t product[4];field(product,a,b,0);field(r,product,c,1);
}
static void _ModSqrAddSub2(uint64_t*r,const uint64_t*a,const uint64_t*b,const uint64_t*c){
 uint64_t t[4];field(t,a,a,0);field(t,t,b,1);field(t,t,c,2);field(r,t,c,2);
}
static uint64_t *qsb_digit_arena(){return arena;}
static void q9_glv_split(const uint64_t*,uint64_t*r1,uint64_t*r2,unsigned*s1,unsigned*s2){
 memcpy(r1,components[0],16);memcpy(r2,components[1],16);*s1=signs[0];*s2=signs[1];
}
'''

LOADER = r'''
// Sparse substitute for the real 9.8GB table: absolute record index is decoded
// using production geometry, then OpenSSL derives that exact stored point.
static void gt_load_signed_flat_m(const uint8_t*,unsigned base_idx,unsigned idx,uint64_t sign,
                                  uint64_t*x,uint64_t*y){
 require(base_idx==0 && (sign==0 || sign==UINT64_MAX));
 int bank=-1;
 for(int c=0;c<6;c++)if(idx>=q9_bigtbl_offset(c) && idx-q9_bigtbl_offset(c)<q9_bigtbl_entries(c))bank=c;
 require(bank>=0);unsigned d=idx-q9_bigtbl_offset(bank);
 BIGNUM*k=BN_new(),*xx=BN_new(),*yy=BN_new();EC_POINT*point=EC_POINT_new(group);
 if(bank==0){
  require(BN_set_word(k,QSB_GT_TOP_CENTER+1u));require(BN_lshift(k,k,QSB_GT_TOP_SHIFT-1u));
  require(BN_sub_word(k,1u<<17));require(BN_add_word(k,d));
 }else{require(BN_set_word(k,2ull*d+1));require(BN_lshift(k,k,q9_bigtbl_shift(bank)-1));}
 require(BN_mod_mul(k,k,base,order,ctx));require(EC_POINT_mul(group,point,k,nullptr,nullptr,ctx));
 require(EC_POINT_get_affine_coordinates(group,point,xx,yy,ctx));
 require(BN_mod_mul(xx,xx,scale_x,prime,ctx));require(BN_mod_mul(yy,yy,scale_y,prime,ctx));
 // Real signed loader XORs the offset-y representation, including all limbs.
 require(BN_add_word(yy,0x800001e8ULL));write256(x,xx);write256(y,yy);
 for(int i=0;i<4;i++)y[i]^=sign;
 BN_free(k);BN_free(xx);BN_free(yy);EC_POINT_free(point);
}
'''

WRAPPER = r'''
extern "C" int audit(const uint64_t *r1,unsigned s1,const uint64_t*r2,unsigned s2,
                     const uint64_t*scalar,const uint64_t*base_words,unsigned iso){
 ctx=BN_CTX_new();group=EC_GROUP_new_by_curve_name(NID_secp256k1);
 prime=BN_new();order=BN_new();base=read256(base_words);scale_x=BN_new();scale_y=BN_new();
 require(EC_GROUP_get_curve(group,prime,nullptr,nullptr,ctx));require(EC_GROUP_get_order(group,order,ctx));
 BIGNUM*u=BN_new();require(BN_set_word(u,iso));
 require(BN_mod_sqr(scale_x,u,prime,ctx));require(BN_mod_mul(scale_y,scale_x,u,prime,ctx));
 memcpy(components[0],r1,16);memcpy(components[1],r2,16);signs[0]=s1;signs[1]=s2;
 memset(arena,0xa5,sizeof(arena));
 uint64_t out[16],dummy[4]={};_FixedBaseSignedXYZZScalar(out,out+4,out+8,out+12,dummy,nullptr,nullptr);
 BIGNUM*k=read256(scalar),*want_x=BN_new(),*want_y=BN_new();EC_POINT*want=EC_POINT_new(group);
 require(BN_mod_mul(k,k,base,order,ctx));require(EC_POINT_mul(group,want,k,nullptr,nullptr,ctx));
 BIGNUM*x=read256(out),*y=read256(out+4),*zz=read256(out+8),*zzz=read256(out+12);
 int infinity=EC_POINT_is_at_infinity(group,want),ok=0;
 if(infinity)ok=BN_is_zero(zz)&&BN_is_zero(zzz);
 else if(!BN_is_zero(zz)&&!BN_is_zero(zzz)){
  require(BN_mod_inverse(zz,zz,prime,ctx)!=nullptr);require(BN_mod_inverse(zzz,zzz,prime,ctx)!=nullptr);
  require(BN_mod_mul(x,x,zz,prime,ctx));require(BN_mod_mul(y,y,zzz,prime,ctx));
  // Production chain returns negative actual ordinate with NEG_Y_MAC enabled.
  BN_zero(k);require(BN_mod_sub(y,k,y,prime,ctx));
  require(EC_POINT_get_affine_coordinates(group,want,want_x,want_y,ctx));
  require(BN_mod_mul(want_x,want_x,scale_x,prime,ctx));require(BN_mod_mul(want_y,want_y,scale_y,prime,ctx));
  ok=BN_cmp(x,want_x)==0&&BN_cmp(y,want_y)==0;
 }
 BN_free(x);BN_free(y);BN_free(zz);BN_free(zzz);BN_free(k);BN_free(want_x);BN_free(want_y);
 BN_free(u);BN_free(prime);BN_free(order);BN_free(base);BN_free(scale_x);BN_free(scale_y);
 EC_POINT_free(want);EC_GROUP_free(group);BN_CTX_free(ctx);return ok;
}
extern "C" void codes(const uint64_t*mag,unsigned sign,uint32_t*out){
 for(int c=0;c<6;c++)out[c]=q9_bigtbl_code(mag,sign,c);
}
'''


def make_code():
    source = (ROOT / 'pinning.cu').read_text()
    math = (ROOT / 'GPUMath.h').read_text()
    glv = (ROOT / 'GLVScalar.cuh').read_text()
    geometry = glv[:glv.index('// QSB/VanitySearch')].replace('#pragma once', '')
    chain = function(source, '__device__ void _FixedBaseSignedXYZZScalar(')
    # This audit intentionally uses the exact field-level formulas extracted
    # from production, with a mathematical backend in place of inline PTX.
    parts = [BACKEND, geometry,
             function(source, '__host__ __device__ __forceinline__ unsigned qsb_glv_chain_position('),
             LOADER,
             function(source, 'template<int SIDE>\n__device__ __forceinline__ void qsb_decode_glv_side('),
             function(source, '__device__ __forceinline__ unsigned qsb_decode_glv('),
             function(source, '__device__ __forceinline__ void qsb_load_glv('),
             math[math.index('// Compile-time twin'):math.index('// Direct-three-affine prefix')],
             function(math, '__device__ void _PointAddXYZZ_mm('),
             chain, WRAPPER]
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--random', type=int, default=512)
    args = parser.parse_args()
    rng = random.Random(2026092401)
    source_hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                     for name in ('pinning.cu', 'GPUMath.h', 'GLVScalar.cuh')}
    edge = {0, 1, 2, 3, N-1, N, N+1, (1 << 256)-1}
    for bit in (0, 1, 17, 18, 36, 37, 54, 55, 72, 73, 99, 100, 127, 128, 255):
        edge.update((1 << bit)+delta for delta in (-1, 0, 1) if (1 << bit)+delta >= 0)
    scalars = sorted(edge) + [rng.getrandbits(256) for _ in range(args.random)]
    cases = [('scalar', k, *split(k)) for k in scalars]
    # Exercise Q-zero fallback, P-zero, both-zero, both signs, and radix edges.
    # Ascending-hot order [4,5,0,1,2,3] has a final mixed-add doubling
    # at this magnitude; the corrected [4,5,3,2,1,0] must handle it.
    magnitudes = {0, 1, 2, BOUND-1, BOUND, (1 << 73)-(1 << 55)}
    for bit in (17, 18, 36, 37, 54, 55, 72, 73, 99, 100, 126, 127):
        magnitudes.update((1 << bit)+delta for delta in (-1, 0, 1)
                          if 0 <= (1 << bit)+delta <= BOUND)
    for m in sorted(magnitudes):
        for s in (-1, 1):
            cases.append(('Q_zero', (s*m) % N, s*m, 0))
            cases.append(('P_zero', (s*m*LAMBDA) % N, 0, s*m))
    for _ in range(64):
        r1, r2 = rng.randrange(-BOUND, BOUND+1), rng.randrange(-BOUND, BOUND+1)
        cases.append(('synthetic', (r1+LAMBDA*r2) % N, r1, r2))
    code = make_code()
    counts = {}
    with tempfile.TemporaryDirectory(prefix='qsb-cold-seed-audit-') as td:
        cpp = Path(td)/'audit.cpp'
        cpp.write_text(code)
        for mode in (0, 1):
            library = Path(td)/f'audit{mode}.so'
            subprocess.run(['g++', '-std=c++17', '-O2', '-shared', '-fPIC',
                            f'-DQSB_GLV_COLD_SEED={mode}', str(cpp), '-lcrypto', '-o', str(library)], check=True)
            lib = C.CDLL(str(library))
            u64 = C.POINTER(C.c_uint64)
            lib.audit.argtypes = [u64, C.c_uint, u64, C.c_uint, u64, u64, C.c_uint]
            lib.codes.argtypes = [u64, C.c_uint, C.POINTER(C.c_uint32)]
            # Check actual decoded signed record scalars telescope independently
            # of every point-chain formula, including boundary magnitudes.
            recode_cases = sorted(magnitudes) + [rng.randrange(BOUND+1) for _ in range(2000)]
            entries = [262144, 262144, 131072, 131072, 67108864, 85279885]
            offsets = [0, 262144, 524288, 655360, 786432, 67895296]
            shifts = [0, 18, 37, 55, 73, 100]
            for mag in recode_cases:
                for sign in (0, 1):
                    encoded = (C.c_uint32*6)()
                    lib.codes(limbs(mag, 2), sign, encoded)
                    total = 0
                    for bank, value in enumerate(encoded):
                        idx = (value & 0x7fffffff)-offsets[bank]
                        assert 0 <= idx < entries[bank], (mode, bank, idx)
                        term = (((170559769+1) << 99)-(1 << 17)+idx if bank == 0
                                else (2*idx+1) << (shifts[bank]-1))
                        total += -term if value >> 31 else term
                    assert total == (-mag if sign else mag), (mode, mag, sign)
            for index, (kind, scalar, r1, r2) in enumerate(cases):
                base = [1, 2, N-1][index % 3] if index < 12 else rng.randrange(1, N)
                iso = [1, 2, 7][index % 3]
                ok = lib.audit(limbs(abs(r1), 2), r1 < 0, limbs(abs(r2), 2), r2 < 0,
                               limbs(scalar % N), limbs(base), iso)
                assert ok, {'mode': mode, 'index': index, 'kind': kind, 'scalar': hex(scalar),
                            'r1': hex(r1), 'r2': hex(r2), 'base': hex(base), 'iso': iso}
            counts[str(mode)] = {'full_chains': len(cases), 'signed_recodes': 2*len(recode_cases)}
        # A deliberate bad permutation must fail the exact same source-derived
        # chain oracle. This protects against accidentally testing only the
        # commutative scalar sum while missing incomplete-add exceptions.
        mapping = function(code, '__host__ __device__ __forceinline__ unsigned qsb_glv_chain_position(')
        bad_mapping = ('__host__ __device__ __forceinline__ unsigned qsb_glv_chain_position(unsigned bank)'
                       '{return bank < 4u ? bank + 2u : bank - 4u;}')
        cpp.write_text(code.replace(mapping, bad_mapping, 1))
        bad_library = Path(td)/'bad.so'
        subprocess.run(['g++', '-std=c++17', '-O2', '-shared', '-fPIC', '-DQSB_GLV_COLD_SEED=1',
                        str(cpp), '-lcrypto', '-o', str(bad_library)], check=True)
        bad = C.CDLL(str(bad_library))
        bad.audit.argtypes = [u64, C.c_uint, u64, C.c_uint, u64, u64, C.c_uint]
        m = (1 << 73)-(1 << 55)
        assert not bad.audit(limbs(m, 2), 0, limbs(0, 2), 0, limbs(m), limbs(1), 1), \
            'Oracle failed to detect known doubling in the ascending-hot permutation'
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
               for name, digest in source_hashes.items()), 'Production source changed during audit; rerun.'
    print(json.dumps({'status': 'PASS', 'modes': counts, 'source_sha256': source_hashes,
                      'negative_control': 'Known ascending-hot incomplete-add doubling correctly rejected',
                      'gpu_executed': False,
                      'scope': 'Actual CPU-extracted mapping/decoder/fullchain/point formulas; OpenSSL field and sparse table; independent integer split and EC_POINT_mul oracle',
                      'limits': 'Does not test CUDA PTX arithmetic, rare-carry behavior, GPU memory scheduling, or performance'}, indent=2))


if __name__ == '__main__':
    main()
