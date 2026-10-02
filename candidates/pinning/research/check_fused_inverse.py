#!/usr/bin/env python3
"""STALE — DO NOT RUN. Superseded by check_fused_finish.py (2026-09-20, v7).

This oracle targeted the historical `qsb_fused_root_inverse` (cross-CTA
atomic root inversion over the split pipeline), which was REMOVED when
QSB_FUSED landed: the block-root inverse now lives inside
`qsb_fused_finish` in the STAGE-0 kernel. Extraction raises
ValueError: substring not found. Kept for the harness pattern
(std::thread + std::barrier CTA emulation, OpenSSL backend).

Concurrency-faithful CPU check of qsb_fused_root_inverse from pinning.cu.

Extracts the production function, replaces CUDA intrinsics with a 128-thread
std::thread/barrier harness and OpenSSL field arithmetic, fills roots[] with
random group roots, and verifies:
  roots[i]            == r_i^{-1} mod p  (canonical)
  roots[root_count+i] == r_i^{-1} * u2ry mod p
for every lane, including the tail lanes that received identity leaves.

This tests the packed-tree indexing, barrier placement and write layout of
the production code path.  It does not test fences/atomics (simulated by the
single-CTA driver), PTX arithmetic, occupancy or speed.
"""
import os, shlex, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def function(source, signature):
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


CPP_PREFIX = r'''
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <barrier>
#include <atomic>
#include <mutex>
#include <vector>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <openssl/sha.h>

static BN_CTX *ctx;
static BIGNUM *prime;
static void require(bool ok){ if(!ok){ std::fprintf(stderr,"require failed\n"); std::exit(1);} }
static BIGNUM *read256(const uint64_t *v){
    return BN_lebin2bn(reinterpret_cast<const unsigned char*>(v),32,nullptr);
}
static void write256(uint64_t *v,const BIGNUM *b){
    require(BN_bn2lebinpad(b,reinterpret_cast<unsigned char*>(v),32)==32);
}

/* OpenSSL-backed equivalents of the device helpers the function uses.
 * BN_CTX is not thread-safe, so the verification backend serializes. */
static std::mutex field_mtx;
static void qsb_field_mul(uint64_t *out,const uint64_t *a,const uint64_t *b){
    std::lock_guard<std::mutex> lk(field_mtx);
    BIGNUM *aa=read256(a),*bb=read256(b),*rr=BN_new();
    require(BN_mod_mul(rr,aa,bb,prime,ctx));
    write256(out,rr); out[4]=0;
    BN_free(aa); BN_free(bb); BN_free(rr);
}
static void qsb_field_normalize(uint64_t *r){
    std::lock_guard<std::mutex> lk(field_mtx);
    /* Device version only handles r==p (small residues); emulate exactly:
       subtract p when the canonical value was hit. */
    BIGNUM *a=read256(r),*m=BN_new();
    require(BN_nnmod(m,a,prime,ctx));
    write256(r,m); r[4]=0;
    BN_free(a); BN_free(m);
}
static void _ModInv(uint64_t *r){
    std::lock_guard<std::mutex> lk(field_mtx);
    BIGNUM *a=read256(r),*b=BN_mod_inverse(nullptr,a,prime,ctx);
    require(b!=nullptr); write256(r,b); r[4]=0;
    BN_free(a); BN_free(b);
}
static uint64_t pin_u2ry_words[4];

/* Concurrency shims: one CTA of 128 std::threads with a real barrier. */
static constexpr int NTH=128;
static thread_local int sim_tid=0;
static std::barrier<> sim_bar(NTH);
static uint32_t sim_atomicAdd(uint32_t *p,uint32_t v){
    return reinterpret_cast<std::atomic<uint32_t>*>(p)->fetch_add(v);
}
'''

DRIVER = r'''
static uint64_t sim_products[4][256];
static uint64_t sim_inverses[4][128];
static uint64_t sim_roots[2*128*4];

int main(){
    ctx=BN_CTX_new();
    EC_GROUP *g=EC_GROUP_new_by_curve_name(NID_secp256k1);
    prime=BN_new();
    require(EC_GROUP_get_curve(g,prime,nullptr,nullptr,ctx));

    /* u2ry: arbitrary nonzero field element. */
    {
        unsigned char h[32];
        SHA256(reinterpret_cast<const unsigned char*>("qsb-u2ry"),8,h);
        BIGNUM *b=BN_bin2bn(h,32,nullptr);
        require(BN_nnmod(b,b,prime,ctx));
        write256(pin_u2ry_words,b); BN_free(b);
    }

    /* Fill 128 group roots with distinct nonzero residues. */
    int root_count=128;
    for(int i=0;i<root_count;i++){
        char seed[64]; std::snprintf(seed,sizeof(seed),"qsb-root-%d",i);
        unsigned char h[32];
        SHA256(reinterpret_cast<unsigned char*>(seed),std::strlen(seed),h);
        BIGNUM *b=BN_bin2bn(h,32,nullptr);
        require(BN_nnmod(b,b,prime,ctx));
        require(!BN_is_zero(b));
        write256(&sim_roots[i*4],b); BN_free(b);
    }

    /* Launch the collective: each thread runs the production function. */
    {
        std::vector<std::thread> ts;
        for(int t=0;t<NTH;t++)
            ts.emplace_back([t,root_count]{
                sim_tid=t;
                qsb_fused_root_inverse<128>(
                    sim_roots,root_count,0u,sim_products,sim_inverses);
            });
        for(auto &th:ts) th.join();
    }

    /* Verify: roots[i]=1/r_i, roots[count+i]=(1/r_i)*u2ry. */
    for(int i=0;i<root_count;i++){
        char seed[64]; std::snprintf(seed,sizeof(seed),"qsb-root-%d",i);
        unsigned char h[32];
        SHA256(reinterpret_cast<unsigned char*>(seed),std::strlen(seed),h);
        BIGNUM *r=BN_bin2bn(h,32,nullptr);
        require(BN_nnmod(r,r,prime,ctx));
        BIGNUM *inv=BN_mod_inverse(nullptr,r,prime,ctx); require(inv);
        BIGNUM *got=read256(&sim_roots[i*4]);
        require(BN_cmp(got,inv)==0);
        BIGNUM *u=read256(pin_u2ry_words),*w=BN_new();
        require(BN_mod_mul(w,inv,u,prime,ctx));
        BIGNUM *gotw=read256(&sim_roots[(root_count+i)*4]);
        require(BN_cmp(gotw,w)==0);
        BN_free(r); BN_free(inv); BN_free(got); BN_free(u); BN_free(w); BN_free(gotw);
    }
    /* Partial group: 73 live roots, rest identity -- must still invert. */
    root_count=73;
    for(int i=0;i<root_count;i++){
        char seed[64]; std::snprintf(seed,sizeof(seed),"qsb-proot-%d",i);
        unsigned char h[32];
        SHA256(reinterpret_cast<unsigned char*>(seed),std::strlen(seed),h);
        BIGNUM *b=BN_bin2bn(h,32,nullptr);
        require(BN_nnmod(b,b,prime,ctx));
        require(!BN_is_zero(b));
        write256(&sim_roots[i*4],b); BN_free(b);
    }
    {
        std::vector<std::thread> ts;
        for(int t=0;t<NTH;t++)
            ts.emplace_back([t,root_count]{
                sim_tid=t;
                qsb_fused_root_inverse<128>(
                    sim_roots,root_count,0u,sim_products,sim_inverses);
            });
        for(auto &th:ts) th.join();
    }
    for(int i=0;i<root_count;i++){
        char seed[64]; std::snprintf(seed,sizeof(seed),"qsb-proot-%d",i);
        unsigned char h[32];
        SHA256(reinterpret_cast<unsigned char*>(seed),std::strlen(seed),h);
        BIGNUM *r=BN_bin2bn(h,32,nullptr);
        require(BN_nnmod(r,r,prime,ctx));
        BIGNUM *inv=BN_mod_inverse(nullptr,r,prime,ctx); require(inv);
        BIGNUM *got=read256(&sim_roots[i*4]);
        require(BN_cmp(got,inv)==0);
        BN_free(r); BN_free(inv); BN_free(got);
    }
    std::printf("PASS: fused root inverse correct for 128-root and 73-root groups\n");
    BN_free(prime); EC_GROUP_free(g); BN_CTX_free(ctx);
}
'''


def main():
    src = (HERE.parent / "pinning.cu").read_text()
    body = function(src, "template<int N>\n__device__ __forceinline__ void qsb_fused_root_inverse(")
    # Strip CUDA tokens, retarget intrinsics to the shims.
    for tok in ("__device__", "__forceinline__", "__shared__"):
        body = body.replace(tok, "")
    body = body.replace("threadIdx.x", "sim_tid")
    body = body.replace("blockIdx.x", "0u")
    body = body.replace("__syncthreads()", "sim_bar.arrive_and_wait()")
    body = body.replace("__threadfence()", "std::atomic_thread_fence(std::memory_order_seq_cst)")
    body = body.replace("atomicAdd", "sim_atomicAdd")

    prefix = os.environ.get("OPENSSL_PREFIX", "")
    flags = []
    if prefix:
        flags = [f"-I{prefix}/include", f"-L{prefix}/lib"]
    with tempfile.TemporaryDirectory(prefix="qsb-fused-") as tmp:
        cpp = Path(tmp) / "check.cpp"
        exe = Path(tmp) / "check"
        cpp.write_text("\n".join([CPP_PREFIX, body, DRIVER]))
        cxx = shlex.split(os.environ.get("CXX", "c++"))
        subprocess.run(cxx + ["-std=c++20", "-O2", *flags, str(cpp),
                              "-lcrypto", "-lpthread", "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
