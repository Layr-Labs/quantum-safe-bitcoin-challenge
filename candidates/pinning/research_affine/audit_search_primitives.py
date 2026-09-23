#!/usr/bin/env python3
"""Host execution of actual affine-search SHA, digit and final-hash helpers.

Fresh synthetic instances and independent hashlib/Python recovery are used.
Only PTX mad.lo, byte_perm, and CUDA qualifiers are mapped to their integer
semantics. The point oracle uses OpenSSL; no CUDA execution is claimed.
"""
from pathlib import Path
import ctypes
import hashlib
import json
import random
import re
import struct
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
CANDIDATE=HERE.parent
ROOT=CANDIDATE.parents[1]
sys.path.insert(0,str(ROOT/'harness'))
import crypto as C
import gen_problem as GEN
import problem as PROBLEM


def extract(source,name):
    match=re.search(r'^[^\n]*\b'+re.escape(name)+r'\s*\(',source,re.M)
    assert match,name
    start=match.start();brace=source.index('{',match.end());depth=1;end=brace+1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start:end]


def main():
    source=(CANDIDATE/'pinning.cu').read_text()
    search_path=CANDIDATE/'affine_search.cuh'
    search=search_path.read_text()
    sha=(CANDIDATE/'GPUHash.h').read_text().split('//Take the last 8 bytes')[0]
    optimized=(CANDIDATE/'sha_pinsha.cuh').read_text()
    old=extract(optimized,'qsb_fadd')
    optimized=optimized.replace(old,'inline uint32_t qsb_fadd(uint32_t a,uint32_t one,uint32_t b){return a*one+b;}')
    gate=extract(optimized,'gpu_bench_valid_h0')
    optimized=optimized.replace(gate,gate.replace('{','{ audit_h0=h0;',1))
    struct_start=source.index('struct qsb_tail_pre {')
    tail_type=source[struct_start:source.index('};',struct_start)+2]
    order_start=source.index('__device__ __constant__ uint64_t GT_ORDER_N[4]')
    order=source[order_start:source.index('};',order_start)+2]
    # These names are the public, independently auditable primitive boundary.
    helper_names=['qsb_affine_search_hash_scalar','qsb_affine_search_recode','qsb_affine_search_pubkey_matches']
    helpers='\n'.join(extract(search,name) for name in helper_names)
    prefix=r'''
#include <cstdint>
#include <cstring>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#define __device__
#define __host__
#define __constant__
#define __forceinline__ inline
#define QSB_SHA_OPT 1
#define QSB_SPARSE_D 1
#define QSB_ZEROS_N 4
#define GT_CHUNKS 15
constexpr unsigned QSB_AFFINE_SEARCH_N=128;
struct Index {unsigned x;}; static Index threadIdx{0};
static uint32_t audit_h0;
static uint32_t pin_tail_words[3];
inline uint32_t __umulhi(uint32_t a,uint32_t b){return uint32_t((uint64_t(a)*b)>>32);}
inline uint32_t __byte_perm(uint32_t a,uint32_t b,uint32_t selector){
 uint64_t both=uint64_t(a)|(uint64_t(b)<<32);uint32_t out=0;
 for(int i=0;i<4;i++){unsigned s=(selector>>(4*i))&15;unsigned v=(both>>(8*(s&7)))&255;
 if(s&8)v=(v&128)?255:0;out|=v<<(8*i);}return out;
}
'''
    body=prefix+sha+'\n'+(CANDIDATE/'sha_schedule_interleaved.cuh').read_text()+'\n'+optimized+'\n'+tail_type+'\n'+order+'\n'
    for name in ['qsb_h_ror','qsb_make_tail_pre','_SHA256TransformFastTail11Q','qsb_signed_recode_setup']:
        body+=extract(source,name)+'\n'
    body+=helpers+r'''
extern "C" void scalar(uint64_t out[4],const uint32_t mid[8],const uint32_t tail[3],uint32_t lt){
 for(int i=0;i<3;i++)pin_tail_words[i]=tail[i];qsb_tail_pre tp;qsb_make_tail_pre(&tp,mid,tail[2]);
 qsb_affine_search_hash_scalar(out,lt,tp);
}
extern "C" void digits(uint32_t out[15],const uint64_t scalar[4]){
 uint32_t codes[15][128];qsb_affine_search_recode(scalar,codes);
 for(unsigned w=0;w<15;w++)out[w]=codes[w][0];
}
extern "C" uint32_t pubkey(uint32_t *h0,const uint64_t x[4],uint32_t parity){
 uint32_t hit=qsb_affine_search_pubkey_matches(x,parity);*h0=audit_h0;return hit;
}
extern "C" int recover(uint8_t pub[66],const uint8_t zbytes[32],const uint8_t nribytes[32],const uint8_t rxy[64]){
 EC_GROUP*g=EC_GROUP_new_by_curve_name(NID_secp256k1);BN_CTX*ctx=BN_CTX_new();
 BIGNUM*z=BN_lebin2bn(zbytes,32,nullptr),*nri=BN_lebin2bn(nribytes,32,nullptr),*n=BN_new();
 BIGNUM*x=BN_lebin2bn(rxy,32,nullptr),*y=BN_lebin2bn(rxy+32,32,nullptr),*u=BN_new();
 EC_POINT*p=EC_POINT_new(g),*r=EC_POINT_new(g),*q=EC_POINT_new(g);
 bool ok=g&&ctx&&z&&nri&&n&&x&&y&&u&&p&&r&&q;
 if(ok)ok=EC_GROUP_get_order(g,n,ctx)==1&&BN_mod_mul(u,z,nri,n,ctx)==1&&
 EC_POINT_mul(g,p,u,nullptr,nullptr,ctx)==1&&EC_POINT_set_affine_coordinates(g,r,x,y,ctx)==1;
 for(unsigned arm=0;ok&&arm<2;arm++){
 if(arm)ok=EC_POINT_invert(g,r,ctx)==1;
 ok=ok&&EC_POINT_add(g,q,p,r,ctx)==1&&EC_POINT_point2oct(g,q,POINT_CONVERSION_COMPRESSED,pub+33*arm,33,ctx)==33;
 }
 EC_POINT_free(p);EC_POINT_free(r);EC_POINT_free(q);BN_free(z);BN_free(nri);BN_free(n);BN_free(x);BN_free(y);BN_free(u);BN_CTX_free(ctx);EC_GROUP_free(g);
 return ok;
}
'''
    U8,U32,U64=ctypes.c_uint8,ctypes.c_uint32,ctypes.c_uint64
    counts={'sha256d_cases':0,'digit_cases':0,'recoded_recovery_pairs':0,'pubkey_h0_cases':0,
            'both_recids_hit':0,'recid_1_only_hit':0,'recid_0_only_hit':0}
    rng=random.Random(0xAFF1CE2026)
    def words(value):return (U64*4)(*((value>>(64*i))&((1<<64)-1) for i in range(4)))
    def bytes32(value):return (U8*32).from_buffer_copy(value.to_bytes(32,'little'))
    with tempfile.TemporaryDirectory(prefix='.audit-search-primitives-',dir=HERE) as directory:
        temporary=Path(directory);cpp=temporary/'audit.cpp';so=temporary/'audit.so';cpp.write_text(body)
        result=subprocess.run(['rtk','proxy','g++','-std=c++17','-O2','-shared','-fPIC',str(cpp),'-lcrypto','-o',str(so)],capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr)
        lib=ctypes.CDLL(str(so))
        lib.scalar.argtypes=[ctypes.POINTER(U64),ctypes.POINTER(U32),ctypes.POINTER(U32),U32]
        lib.digits.argtypes=[ctypes.POINTER(U32),ctypes.POINTER(U64)]
        lib.pubkey.argtypes=[ctypes.POINTER(U32),ctypes.POINTER(U64),U32]
        lib.recover.argtypes=[ctypes.POINTER(U8)]*4
        def check_digits(z):
            codes=(U32*15)();lib.digits(codes,words(z));total=0
            for w,code in enumerate(codes):
                index=code&0x1ffff;negative=code>>31
                assert negative in (0,1) and index<(131072 if w==0 else 65536),(w,hex(code))
                digit=(2*index+1)*(-1 if negative else 1)
                total+=digit<<(0 if w==0 else 17*w+1)
            assert total==2*(z%C.N)-C.N,(hex(z),hex(total))
            counts['digit_cases']+=1
            return total*pow(2,-1,C.N)%C.N
        def check_pub(pub):
            h=hashlib.sha256(pub).digest();x=int.from_bytes(pub[1:],'big');h0=U32()
            hit=lib.pubkey(ctypes.byref(h0),words(x),pub[0]&1)
            assert h0.value==int.from_bytes(h[:4],'big')
            assert bool(hit)==(h[0]<16)
            counts['pubkey_h0_cases']+=1
            return bool(hit)
        for z in [0,1,2,C.N//2-1,C.N//2,C.N//2+1,C.N-2,C.N-1,C.N,C.N+1,2**256-1]+[rng.getrandbits(256) for _ in range(4096)]:check_digits(z)
        for parity in (0,1):
            for x in [0,1,C.P-1,C.P,2**256-1]+[1<<i for i in range(256)]+[rng.getrandbits(256) for _ in range(1024)]:
                check_pub(bytes([2+parity])+x.to_bytes(32,'big'))
        for seed in [0,20260923]:
            problem,_=GEN.gen_pinning(random.Random(seed))
            prefix=bytes.fromhex(problem['pin_prefix']);suffix=bytes.fromhex(problem['suffix'])
            mid=C.sha256_midstate(prefix);nri=int(problem['neg_r_inv'],16)
            rxy=(U8*64).from_buffer_copy(int(problem['u2r_x'],16).to_bytes(32,'little')+int(problem['u2r_y'],16).to_bytes(32,'little'))
            tail=(U32*3)(int.from_bytes(suffix[64:67]+b'\0','big'),suffix[71],int.from_bytes(suffix[72:75]+b'\x80','big'))
            locktimes=[0,1,127,128,255,256,257,65535,65536,0x7fffffff,0x80000000,0xfffffeff,0xffffff00,0xfffffffe,0xffffffff]+[rng.getrandbits(32) for _ in range(241)]
            for sequence in [0,1,0x80000000,0xffffffff]:
                first=bytearray(suffix[:64]);struct.pack_into('<I',first,problem['seq_offset'],sequence)
                # One generic SHA transform is independent of the optimized tail.
                first_mid=C.sha256_compress(mid,bytes(first)) if hasattr(C,'sha256_compress') else None
                if first_mid is None:
                    sys.path.insert(0,str(CANDIDATE));from test_host_gate import compress_one
                    first_mid=compress_one(mid,bytes(first))
                for lt in locktimes:
                    expected=hashlib.sha256(hashlib.sha256(PROBLEM.pin_preimage(problem,sequence,lt)).digest()).digest()
                    output=(U64*4)();lib.scalar(output,(U32*8)(*first_mid),tail,lt)
                    z=sum(int(output[i])<<(64*i) for i in range(4));assert z==int.from_bytes(expected,'big')
                    reconstructed=check_digits(z);assert reconstructed==z%C.N
                    pubs=(U8*66)();assert lib.recover(pubs,bytes32(reconstructed),bytes32(nri),rxy)
                    hits=[check_pub(bytes(pubs[33*r:33*r+33])) for r in range(2)]
                    if all(hits):counts['both_recids_hit']+=1
                    elif hits[1]:counts['recid_1_only_hit']+=1
                    elif hits[0]:counts['recid_0_only_hit']+=1
                    # Independent ECDSA recovery derives R from (r,s), not cached u2R.
                    if counts['sha256d_cases']%31==0:
                        for arm in range(2):
                            q=C.ecdsa_recover(int(problem['r'],16),int(problem['s'],16),z,arm)
                            assert C.compress_pubkey(q)==bytes(pubs[33*arm:33*arm+33])
                    counts['sha256d_cases']+=1;counts['recoded_recovery_pairs']+=1
    assert counts['both_recids_hit']>0 and counts['recid_1_only_hit']>0
    report={'checks':counts,'mismatches':0,'gpu_execution':False,
            'scope':'Actual SHA, digit extraction and pubkey filter helpers; OpenSSL recovery; independent hashlib and ECDSA recovery; N=4 only in this test.',
            'source_sha256':{name:hashlib.sha256((CANDIDATE/name).read_bytes()).hexdigest() for name in ['pinning.cu','affine_search.cuh','GPUHash.h','sha_pinsha.cuh','sha_schedule_interleaved.cuh']},
            'extracted_translation_sha256':hashlib.sha256(body.encode()).hexdigest()}
    (HERE/'search_primitives_result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
