#!/usr/bin/env python3
"""Check actual candidate recovery expressions against OpenSSL, without CUDA.

Extracts the point-add, G-table multiplication and production recovery code.
Only the field arithmetic backend is replaced with OpenSSL BIGNUM. This tests
the coordinate transformation, not CUDA/PTX arithmetic or GPU performance.
Requires a C++ compiler and OpenSSL development headers (or OPENSSL_PREFIX).
Generated source and executable live in a temporary directory.
"""
import os
import hashlib
import json
import sys
from pathlib import Path
import shlex
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def function(source, signature):
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end].replace("__device__", "")


BACKEND = r'''
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unordered_set>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <openssl/sha.h>
static BN_CTX *ctx;
static BIGNUM *prime;
static int inversions;
static void require(bool ok) { if (!ok) { std::fprintf(stderr,"check failed\n"); std::exit(1); } }
static BIGNUM *read256(const uint64_t *v) {
    return BN_lebin2bn(reinterpret_cast<const unsigned char *>(v),32,nullptr);
}
static void write256(uint64_t *v, const BIGNUM *b) {
    require(BN_bn2lebinpad(b,reinterpret_cast<unsigned char *>(v),32)==32);
}
static void field(uint64_t *r,const uint64_t *a,const uint64_t *b,int op) {
    BIGNUM *aa=read256(a), *bb=read256(b), *rr=BN_new();
    int ok=op==0 ? BN_mod_mul(rr,aa,bb,prime,ctx) :
           op==1 ? BN_mod_add(rr,aa,bb,prime,ctx) : BN_mod_sub(rr,aa,bb,prime,ctx);
    require(ok); write256(r,rr); BN_free(aa); BN_free(bb); BN_free(rr);
}
static void _ModMult(uint64_t *r,const uint64_t *a,const uint64_t *b) { field(r,a,b,0); }
static void _ModMult(uint64_t *r,const uint64_t *b) { field(r,r,b,0); }
static void _ModAdd256(uint64_t *r,const uint64_t *a,const uint64_t *b) { field(r,a,b,1); }
static void _ModSub256(uint64_t *r,const uint64_t *a,const uint64_t *b) { field(r,a,b,2); }
static void _ModSub256(uint64_t *r,const uint64_t *b) { field(r,r,b,2); }
static void _ModSqr(uint64_t *r,const uint64_t *a) { field(r,a,a,0); }
static void _ModInv(uint64_t *r) {
    ++inversions; BIGNUM *a=read256(r), *b=BN_mod_inverse(nullptr,a,prime,ctx);
    require(b!=nullptr); write256(r,b); r[4]=0; BN_free(a); BN_free(b);
}
static const int CHUNK_FIRST_ELEMENT[16] = {
    0,65536,131072,196608,262144,327680,393216,458752,
    524288,589824,655360,720896,786432,851968,917504,983040
};
'''

TEST = r'''
static void coords(const EC_GROUP *g,const EC_POINT *p,uint64_t *x,uint64_t *y) {
    BIGNUM *bx=BN_new(),*by=BN_new();
    require(EC_POINT_get_affine_coordinates(g,p,bx,by,ctx));
    write256(x,bx); write256(y,by); BN_free(bx); BN_free(by);
}
int main() {
    uint16_t endian=1; require(*reinterpret_cast<uint8_t *>(&endian)==1);
    ctx=BN_CTX_new(); EC_GROUP *g=EC_GROUP_new_by_curve_name(NID_secp256k1);
    prime=BN_new(); BIGNUM *order=BN_new();
    require(EC_GROUP_get_curve(g,prime,nullptr,nullptr,ctx));
    require(EC_GROUP_get_order(g,order,ctx));
    std::vector<uint8_t> tx(16ULL*65536*32),ty(tx.size());
    std::unordered_set<int> populated;
    EC_POINT *entry=EC_POINT_new(g),*a=EC_POINT_new(g),*neg2a=EC_POINT_new(g),*expected=EC_POINT_new(g);
    BIGNUM *k=BN_new(),*digit=BN_new(),*ak=BN_new();
    const int edge_cases=35, random_cases=512;
    for(int trial=0;trial<edge_cases+random_cases;trial++) {
        if(trial<32) { BN_one(k); require(BN_lshift(k,k,trial*8)); }
        else if(trial==32) BN_set_word(k,65535);
        else if(trial==33) { BN_copy(k,order); BN_sub_word(k,1); }
        else if(trial==34) { BN_copy(k,order); BN_sub_word(k,2); }
        else {
            char seed[80]; std::snprintf(seed,sizeof(seed),"qsb-projective-test-%d",trial);
            unsigned char hash[32]; SHA256(reinterpret_cast<unsigned char *>(seed),std::strlen(seed),hash);
            BN_bin2bn(hash,32,k); require(BN_mod(k,k,order,ctx));
        }
        require(!BN_is_zero(k));
        uint64_t scalar[4]; write256(scalar,k); uint16_t pk[16]; std::memcpy(pk,scalar,32);
        for(int ch=0;ch<16;ch++) if(pk[ch]) {
            int slot=ch*65536+pk[ch]-1;
            if(populated.insert(slot).second) {
                BN_set_word(digit,pk[ch]); require(BN_lshift(digit,digit,16*ch));
                require(EC_POINT_mul(g,entry,digit,nullptr,nullptr,ctx));
                uint64_t x[4],y[4]; coords(g,entry,x,y);
                std::memcpy(tx.data()+slot*32,x,32); std::memcpy(ty.data()+slot*32,y,32);
            }
        }
        // A varies independently of the candidate scalar, as u2*R does per problem.
        BN_set_word(ak,1234567+trial*7919);
        require(EC_POINT_mul(g,a,ak,nullptr,nullptr,ctx));
        require(EC_POINT_dbl(g,neg2a,a,ctx)); require(EC_POINT_invert(g,neg2a,ctx));
        uint64_t ax[4],ay[4],nx[4],ny[4],out[16],want[8];
        coords(g,a,ax,ay); coords(g,neg2a,nx,ny);
        inversions=0;
        candidate_recover(scalar,ax,ay,nx,ny,tx.data(),ty.data(),out);
        require(inversions==1);
        for(int recid=0;recid<2;recid++) {
            if(recid==0) require(BN_mod_add(digit,k,ak,order,ctx));
            else require(BN_mod_sub(digit,k,ak,order,ctx));
            require(EC_POINT_mul(g,expected,digit,nullptr,nullptr,ctx));
            coords(g,expected,want,want+4);
            if(std::memcmp(out+8*recid,want,64)) {
                std::fprintf(stderr,"Mismatch: trial=%d recid=%d\n",trial,recid); return 1;
            }
        }
        // The diagnostic wrapper must continue returning affine u1*G.
        uint64_t x[4],y[4]; _PointMultiSecp256k1(x,y,pk,tx.data(),ty.data());
        require(EC_POINT_mul(g,expected,k,nullptr,nullptr,ctx)); coords(g,expected,want,want+4);
        require(!std::memcmp(x,want,32) && !std::memcmp(y,want+4,32));
    }
    std::printf("PASS: %d scalars, %d recovered keys, affine debug wrapper; one recovery inversion/candidate\n",
                edge_cases+random_cases,2*(edge_cases+random_cases));
    BN_free(k); BN_free(digit); BN_free(ak); BN_free(order); BN_free(prime);
    EC_POINT_free(entry); EC_POINT_free(a); EC_POINT_free(neg2a); EC_POINT_free(expected);
    EC_GROUP_free(g); BN_CTX_free(ctx);
}
'''


def main():
    source = (HERE / "pinning.cu").read_text()
    if '#include "wide_geometry.cuh"' in source:
        # The current exact-source audit lives beside the reproducible research
        # candidate. Bind every copied production file before delegating to it.
        staged=HERE/'research/wide_windows/candidate'
        provenance=json.loads((staged/'PROVENANCE.json').read_text())
        for name,want in provenance['candidate_sha256'].items():
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==want, name
        subprocess.run([sys.executable,str(HERE/'research/wide_windows/check_wide.py'),
                        '--base','wide_windows/candidate'],check=True)
        return
    math = (HERE / "GPUMath.h").read_text()
    point_add = function(math, "__device__ void _PointAddSecp256k1(")
    projective = function(source, "__device__ void _PointMultiSecp256k1Projective(")
    affine = function(source, "__device__ void _PointMultiSecp256k1(")
    production = source[source.index("kernel_pinning_real("):]
    start = production.index("    uint16_t pk[16];")
    end = production.index("    /* Check both pubkeys")
    recovery = """
static void candidate_recover(uint64_t *u1,
    const uint64_t *d_u2rx,const uint64_t *d_u2ry,
    const uint64_t *d_neg2u2rx,const uint64_t *d_neg2u2ry,
    uint8_t *d_gtX,uint8_t *d_gtY,uint64_t *out) {
""" + production[start:end] + """
    memcpy(out,q1x,32); memcpy(out+4,q1y,32);
    memcpy(out+8,q2x,32); memcpy(out+12,q2y,32);
}
"""
    flags = []
    prefix = os.environ.get("OPENSSL_PREFIX")
    if not prefix and Path("/opt/homebrew/opt/openssl@3").exists():
        prefix = "/opt/homebrew/opt/openssl@3"
    if prefix:
        flags = [f"-I{prefix}/include", f"-L{prefix}/lib"]
    with tempfile.TemporaryDirectory(prefix="qsb-projective-") as tmp:
        cpp = Path(tmp) / "check.cpp"
        exe = Path(tmp) / "check"
        cpp.write_text("\n".join([BACKEND, point_add, projective, affine, recovery, TEST]))
        subprocess.run(shlex.split(os.environ.get("CXX", "c++")) +
                       ["-std=c++17", "-O2", *flags, str(cpp), "-lcrypto", "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
