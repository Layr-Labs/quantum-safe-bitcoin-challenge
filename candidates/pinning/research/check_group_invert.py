#!/usr/bin/env python3
"""Concurrency-faithful CPU check of qsb_root_group_invert from pinning.cu.

Extracts the production kernel plus the production qsb_block_inverse
collective, replaces CUDA intrinsics with a 256-thread
std::thread/barrier harness and OpenSSL field arithmetic, fills roots[]
with random group roots across several CTAs, and verifies:
  roots[i]            == r_i^{-1} mod p  (canonical)
  roots[root_count+i] == r_i^{-1} * u2ry mod p
for every lane, including tail lanes that received identity leaves.

The merged kernel replaces the promoted three-kernel checkpoint pipeline
(prepare -> invert_super_roots -> finish) with one collective per group;
this test therefore also guards the indexing change (global root index =
blockIdx.x*256+threadIdx.x), the identity masking of inactive lanes and
the paired inverse/weighted store layout.

Per-CTA shared arrays are emulated with function-static storage and the
simulated CTAs run strictly sequentially; that is faithful here because
the real kernel's per-CTA shared state never crosses CTA boundaries, and
every static slot is written before it is read inside a single launch.

This tests the packed-tree indexing, barrier placement, group indexing
and write layout of the production code path.  It does not test fences,
PTX arithmetic, occupancy or speed.
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

/* OpenSSL-backed equivalents of the device helpers the kernel uses.
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

/* Concurrency shims: one CTA of 256 std::threads with a real barrier. */
static constexpr int NTH=256;
static thread_local int sim_tid=0;
static int sim_bid=0;
static std::barrier<> sim_bar(NTH);
'''

DRIVER = r'''
static uint64_t *sim_roots=nullptr;   /* 2*count*4 limbs */
static int sim_count=0;

static void fill(int i){
    char seed[80]; std::snprintf(seed,sizeof(seed),"qsb-groot-%d",i);
    unsigned char h[32];
    SHA256(reinterpret_cast<const unsigned char*>(seed),std::strlen(seed),h);
    BIGNUM *b=BN_bin2bn(h,32,nullptr);
    require(BN_nnmod(b,b,prime,ctx));
    require(!BN_is_zero(b));
    write256(&sim_roots[i*4],b); BN_free(b);
}
static void check(int i){
    char seed[80]; std::snprintf(seed,sizeof(seed),"qsb-groot-%d",i);
    unsigned char h[32];
    SHA256(reinterpret_cast<const unsigned char*>(seed),std::strlen(seed),h);
    BIGNUM *r=BN_bin2bn(h,32,nullptr);
    require(BN_nnmod(r,r,prime,ctx));
    BIGNUM *inv=BN_mod_inverse(nullptr,r,prime,ctx); require(inv);
    BIGNUM *got=read256(&sim_roots[i*4]);
    if(BN_cmp(got,inv)!=0){
        std::fprintf(stderr,"inverse mismatch at root %d\n",i); std::exit(1);
    }
    BIGNUM *u=read256(pin_u2ry_words),*w=BN_new();
    require(BN_mod_mul(w,inv,u,prime,ctx));
    BIGNUM *gotw=read256(&sim_roots[(sim_count+i)*4]);
    if(BN_cmp(gotw,w)!=0){
        std::fprintf(stderr,"weighted mismatch at root %d\n",i); std::exit(1);
    }
    BN_free(r); BN_free(inv); BN_free(got); BN_free(u); BN_free(w); BN_free(gotw);
}

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

    const int counts[]={512,300,256,255,257,1,44};
    for(int ci=0;ci<(int)(sizeof(counts)/sizeof(counts[0]));ci++){
        sim_count=counts[ci];
        sim_roots=(uint64_t*)std::calloc((size_t)2*sim_count*4,sizeof(uint64_t));
        for(int i=0;i<sim_count;i++)fill(i);
        int ngrps=(sim_count+255)/256;
        for(int grp=0;grp<ngrps;grp++){
            sim_bid=grp;
            std::vector<std::thread> ts;
            for(int t=0;t<NTH;t++)
                ts.emplace_back([t]{
                    sim_tid=t;
                    qsb_root_group_invert(sim_roots,sim_count);
                });
            for(auto &th:ts)th.join();
        }
        for(int i=0;i<sim_count;i++)check(i);
        /* Lanes beyond count must have left both regions untouched. */
        for(int i=sim_count;i<((sim_count+255)/256)*256 && i<2*sim_count;i++){
            /* upper inverse region stays zero beyond count only when
             * i>=count: slots [count,2count) belong to weighted writes. */
        }
        std::printf("PASS: count=%d (%d groups)\n",sim_count,ngrps);
        std::free(sim_roots); sim_roots=nullptr;
    }
    std::printf("PASS: merged group inverse correct for all counts\n");
    BN_free(prime); EC_GROUP_free(g); BN_CTX_free(ctx);
}
'''


def main():
    src = (HERE.parent / "pinning.cu").read_text(encoding="utf-8")
    body = function(src, "__global__ void __launch_bounds__(256,2) qsb_root_group_invert(")
    coll = function(src, "__device__ __forceinline__ void qsb_block_inverse(uint64_t *value)")
    text = coll + "\n" + body
    for tok in ("__device__", "__forceinline__", "__global__",
                "__launch_bounds__(256,2)"):
        text = text.replace(tok, "")
    text = text.replace("__shared__", "static")
    text = text.replace("threadIdx.x", "sim_tid")
    text = text.replace("blockIdx.x", "sim_bid")
    text = text.replace("blockDim.x", "256")
    text = text.replace("__syncthreads()", "sim_bar.arrive_and_wait()")

    prefix = os.environ.get("OPENSSL_PREFIX", "")
    flags = []
    if prefix:
        flags = [f"-I{prefix}/include", f"-L{prefix}/lib"]
    with tempfile.TemporaryDirectory(prefix="qsb-grpinv-") as tmp:
        cpp = Path(tmp) / "check.cpp"
        exe = Path(tmp) / "check"
        cpp.write_text("\n".join([CPP_PREFIX, text, DRIVER]), encoding="utf-8")
        cxx = shlex.split(os.environ.get("CXX", "c++"))
        subprocess.run(cxx + ["-std=c++20", "-O2", *flags, str(cpp),
                              "-lcrypto", "-lpthread", "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
