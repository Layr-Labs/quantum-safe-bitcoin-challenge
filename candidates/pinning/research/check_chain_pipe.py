#!/usr/bin/env python3
"""Structural CPU check of the QSB_CHAIN_PIPE rotating-buffer gather pipeline
inside _FixedBaseSignedXYZZScalar (pinning.cu).

Extracts the production chain verbatim -- qsb_signed_recode_setup,
qsb_decode_to_shared, qsb_load_decoded, gt_load_signed_flat_m,
_PointAddXYZZ_mm, _PointAddXYZZT, _PointAddXYZZT_pipe and the scalar driver --
and compiles it twice under the names chain_ref (QSB_CHAIN_PIPE=0, the
promoted load-at-head loop) and chain_pipe (QSB_CHAIN_PIPE=1, the rotating
x/y/o prefetch). Field arithmetic is OpenSSL-backed, so any divergence
between the two variants is a bug in the buffer rotation, not in the field
code (which is identical in both).

Test A (fake table, 64 MiB of PRNG bytes): chain_pipe must produce BITWISE
identical XYZZ outputs to chain_ref on every scalar -- catches wrong role
rotation, wrong chunk index, off-by-one prefetch, anchor corruption.

Test B (lazy real table, OpenSSL EC points): both chains must produce the
affine point z*A where A = neg_r_inv*G -- anchors the structure to real
group arithmetic including the deferred-Y resolve.

This tests the loop structure and buffer lifetimes only. It does not test
PTX scheduling, registers, occupancy or speed; those are measured by the
ranked build on the actual GPU.
"""
import os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC  = HERE.parent / "pinning.cu"

def function(source, signature):
    """Extract a function definition starting at `signature`. Skips forward
    declarations (signature followed by ';' before any '{')."""
    start = 0
    while True:
        start = source.index(signature, start)
        brace = source.index("{", start)
        semi  = source.find(";", start, brace)
        if semi == -1:
            break
        start = semi + 1
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]

def block(source, start_marker, end_marker):
    i = source.index(start_marker)
    j = source.index(end_marker, i)
    return source[i:j]

def main():
    src = SRC.read_text(encoding="utf-8")
    gm  = (SRC.parent / "GPUMath.h").read_text(encoding="utf-8")

    order_n   = block(src, "__device__ __constant__ uint64_t GT_ORDER_N", ";") + ";"
    entries   = function(src, "__host__ __device__ __forceinline__ unsigned gt_entries")
    offset    = function(src, "__host__ __device__ __forceinline__ unsigned gt_offset")
    shift     = function(src, "__host__ __device__ __forceinline__ int gt_shift")
    arena     = function(src, "__device__ __forceinline__ uint64_t *qsb_digit_arena")
    recode    = function(src, "__device__ __forceinline__ void qsb_signed_recode_setup")
    decode    = function(src, "__device__ __forceinline__ void qsb_decode_to_shared")
    loader    = function(src, "__device__ __forceinline__ void gt_load_signed_flat_m")
    load_dec  = function(src, "__device__ __forceinline__ void qsb_load_decoded")
    mm        = function(gm, "__device__ void _PointAddXYZZ_mm")
    maddT     = "template<bool DEFER_Y>\n" + function(gm, "__device__ __forceinline__ void _PointAddXYZZT")
    maddPipe  = "template<bool DEFER_Y>\n" + function(src, "__device__ __forceinline__ void _PointAddXYZZT_pipe")
    scalar    = function(src, "__device__ void _FixedBaseSignedXYZZScalar")

    cpp = r'''
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>

/* ---- CUDA -> CPU shims ------------------------------------------------ */
#define __device__
#define __forceinline__ inline
#define __host__
#define __shared__ static
#define __constant__ static
#define __ldg(p) (*(p))
struct ulonglong2 { unsigned long long x, y; };
static struct { int x, y, z; } threadIdx = {0,0,0};
static struct { int x, y, z; } blockIdx  = {0,0,0};
static struct { int x, y, z; } blockDim  = {128,1,1};

/* carry-flag emulation for the PTX add macros used by the verbatim loader */
static uint64_t __carry;
#define UADDO1(c,a) do{ __uint128_t s=(__uint128_t)(c)+(a); (c)=(uint64_t)s; __carry=(uint64_t)(s>>64);}while(0)
#define UADDC1(c,a) do{ __uint128_t s=(__uint128_t)(c)+(a)+__carry; (c)=(uint64_t)s; __carry=(uint64_t)(s>>64);}while(0)
#define UADD1(c,a)  do{ (c)=(c)+(a)+__carry; }while(0)
#define Load256(r,a) do{ (r)[0]=(a)[0];(r)[1]=(a)[1];(r)[2]=(a)[2];(r)[3]=(a)[3]; }while(0)

#define GT_CHUNKS 15
#define GT_LO 256
#define GT_HI 1024
#define GT_TOTAL_ENTRIES (1u << 20)
#define QSB_TREE_N 128
#define QSB_LAZY 1
#define QSB_FUSE_SQRADDSUB2 1

/* ---- OpenSSL field backend (canonical mod p) --------------------------- */
static BN_CTX *CTX;
static BIGNUM *PRIME, *ORDER_N;
static EC_GROUP *GRP;
static BIGNUM *HALF_NRI;          /* (2^-1 * neg_r_inv) mod n = A/2 scalar   */
static void require(int ok){ if(!ok){ std::fprintf(stderr,"require failed\n"); std::exit(1);} }
static BIGNUM *rd(const uint64_t *v){ return BN_lebin2bn((const unsigned char*)v,32,NULL); }
static void wr(uint64_t *v,const BIGNUM *b){ require(BN_bn2lebinpad(b,(unsigned char*)v,32)==32); }
static void fadd(BIGNUM *r,const BIGNUM*a,const BIGNUM*b){ BN_mod_add(r,a,b,PRIME,CTX); }
static void fsub(BIGNUM *r,const BIGNUM*a,const BIGNUM*b){ BN_mod_sub(r,a,b,PRIME,CTX); }
static void fmul(BIGNUM *r,const BIGNUM*a,const BIGNUM*b){ BN_mod_mul(r,a,b,PRIME,CTX); }

void _ModMult(uint64_t *r, uint64_t *a, uint64_t *b){
    BIGNUM *A=rd(a),*B=rd(b),*R=BN_new(); fmul(R,A,B); wr(r,R);
    BN_free(A);BN_free(B);BN_free(R);
}
void _ModMult(uint64_t *r, uint64_t *a){ _ModMult(r,r,a); }
void _ModSub256(uint64_t *r, const uint64_t *a, const uint64_t *b){
    BIGNUM *A=rd(a),*B=rd(b),*R=BN_new(); fsub(R,A,B); wr(r,R);
    BN_free(A);BN_free(B);BN_free(R);
}
void _ModSub256(uint64_t *r, uint64_t *b){ _ModSub256(r,r,b); }
void _ModAdd256(uint64_t *r, const uint64_t *a, const uint64_t *b){
    BIGNUM *A=rd(a),*B=rd(b),*R=BN_new(); fadd(R,A,B); wr(r,R);
    BN_free(A);BN_free(B);BN_free(R);
}
void _ModAddLazy(uint64_t *r, const uint64_t *a, const uint64_t *b){
    /* canonical shim: identical in both variants, so rotation bugs still show */
    _ModAdd256(r,a,b);
}
void _ModSqr(uint64_t *r, const uint64_t *a){
    BIGNUM *A=rd(a),*R=BN_new(); BN_mod_sqr(R,A,PRIME,CTX); wr(r,R);
    BN_free(A);BN_free(R);
}
void _ModSqrAddSub2(uint64_t *out, const uint64_t *a, const uint64_t *e, const uint64_t *q){
    /* out = a^2 + e - 2q mod p */
    BIGNUM *A=rd(a),*E=rd(e),*Q=rd(q),*R=BN_new(),*T=BN_new();
    BN_mod_sqr(R,A,PRIME,CTX); fadd(R,R,E);
    fadd(T,Q,Q); fsub(R,R,T); wr(out,R);
    BN_free(A);BN_free(E);BN_free(Q);BN_free(R);BN_free(T);
}
void _ModX3Fused(uint64_t *r, const uint64_t *a, const uint64_t *b, const uint64_t *c){
    /* r = a + b - 2c mod p */
    BIGNUM *A=rd(a),*B=rd(b),*C=rd(c),*R=BN_new(),*T=BN_new();
    fadd(R,A,B); fadd(T,C,C); fsub(R,R,T); wr(r,R);
    BN_free(A);BN_free(B);BN_free(C);BN_free(R);BN_free(T);
}

/* ---- verbatim production code ----------------------------------------- */
@@ORDER_N@@

@@ENTRIES@@
@@OFFSET@@
@@SHIFT@@

@@ARENA@@
@@RECODE@@
@@DECODE@@

#ifdef FILL_LOADER
/* Test B loader: compute the real table point lazily with OpenSSL instead of
 * reading the flat table. Same signature and negate semantics as
 * gt_load_signed_flat_m: y is replaced by p-y when m is all-ones. */
static void gt_load_signed_flat_m(const uint8_t*, uint32_t base, uint32_t idx,
                                  uint64_t m, uint64_t *gx, uint64_t *gy) {
    int c = base ? (int)(base>>16)-1 : 0;
    BIGNUM *k=BN_new(); BN_one(k);
    BN_lshift(k,k,gt_shift(c));
    BN_mul_word(k,(BN_ULONG)(2u*idx+1u));
    BN_mod_mul(k,k,HALF_NRI,ORDER_N,CTX);
    EC_POINT *pt=EC_POINT_new(GRP);
    require(EC_POINT_mul(GRP,pt,k,NULL,NULL,CTX));
    BIGNUM *x=BN_new(),*y=BN_new();
    require(EC_POINT_get_affine_coordinates_GFp(GRP,pt,x,y,CTX));
    wr(gx,x); wr(gy,y);
    if(m){ BIGNUM *ny=BN_new(); fsub(ny,PRIME,y); wr(gy,ny); BN_free(ny); }
    BN_free(k);BN_free(x);BN_free(y);EC_POINT_free(pt);
}
#else
@@LOADER@@
#endif

@@LOAD_DEC@@
@@MM@@
@@MADDT@@
@@MADDPIPE@@

/* The verbatim scalar driver is included twice under different names. */
static void dummy_unused(void){}
#define _FixedBaseSignedXYZZScalar chain_pipe
#define QSB_CHAIN_PIPE 1
@@SCALAR@@
#undef _FixedBaseSignedXYZZScalar
#undef QSB_CHAIN_PIPE
#define _FixedBaseSignedXYZZScalar chain_ref
#define QSB_CHAIN_PIPE 0
@@SCALAR@@
#undef _FixedBaseSignedXYZZScalar
#undef QSB_CHAIN_PIPE

/* ---- harness ----------------------------------------------------------- */
static void to_affine(const uint64_t *X,const uint64_t *Y,const uint64_t *ZZ,const uint64_t *ZZZ,
                      BIGNUM *ax,BIGNUM *ay){
    BIGNUM *x=rd(X),*y=rd(Y),*zz=rd(ZZ),*zzz=rd(ZZZ),*i1=BN_new(),*i2=BN_new();
    require(BN_mod_inverse(i1,zz,PRIME,CTX)!=NULL);  fmul(ax,x,i1);
    require(BN_mod_inverse(i2,zzz,PRIME,CTX)!=NULL); fmul(ay,y,i2);
    BN_free(x);BN_free(y);BN_free(zz);BN_free(zzz);BN_free(i1);BN_free(i2);
}

int main(int argc,char**argv){
    int cases = argc>1?atoi(argv[1]):400;
    CTX=BN_CTX_new(); GRP=EC_GROUP_new_by_curve_name(NID_secp256k1);
    PRIME=BN_new(); ORDER_N=BN_new(); HALF_NRI=BN_new();
    BN_hex2bn(&PRIME,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    require(EC_GROUP_get_order(GRP,ORDER_N,CTX));

#ifdef FILL_LOADER
    /* choose a fixed nri; A/2 = inv2*nri mod n */
    BIGNUM *two=BN_new(),*inv2=BN_new(),*nri=BN_new();
    BN_set_word(two,2); require(BN_mod_inverse(inv2,two,ORDER_N,CTX)!=NULL);
    BN_hex2bn(&nri,"5A17B3C9D8E2F4061B2C3D4E5F60718293A4B5C6D7E8F90123456789ABCDEF01");
    BN_mod_mul(HALF_NRI,inv2,nri,ORDER_N,CTX);
    BN_free(two);BN_free(inv2);/* nri kept for the expected-point check */
#else
    BIGNUM *nri=NULL;
    static uint8_t fake_table[(size_t)GT_TOTAL_ENTRIES*64];
    std::mt19937_64 trng(0x5eedu);
    for(size_t i=0;i<sizeof(fake_table);i+=8){ uint64_t v=trng(); memcpy(fake_table+i,&v,8); }
#endif

    std::mt19937_64 rng(0xC0FFEEu);
    uint64_t k[4],X1[4],Y1[4],Z1[4],W1[4],X2[4],Y2[4],Z2[4],W2[4];
    int fails=0;
    for(int t=0;t<cases;t++){
        for(int j=0;j<4;j++)k[j]=rng();
        chain_ref (X1,Y1,Z1,W1,k,(const uint8_t*)
#ifdef FILL_LOADER
                   nullptr
#else
                   fake_table
#endif
                   ,NULL);
        chain_pipe(X2,Y2,Z2,W2,k,(const uint8_t*)
#ifdef FILL_LOADER
                   nullptr
#else
                   fake_table
#endif
                   ,NULL);
        uint64_t *A[4]={X1,Y1,Z1,W1},*B[4]={X2,Y2,Z2,W2};
        for(int j=0;j<4;j++)if(memcmp(A[j],B[j],32)){ fails++;
            fprintf(stderr,"case %d: coord %d diverges (ref vs pipe)\n",t,j);break;}
#ifdef FILL_LOADER
        /* semantic anchor: expected point = (k mod n)*nri*G */
        BIGNUM *kk=rd(k),*s=BN_new(),*ax=BN_new(),*ay=BN_new();
        BN_nnmod(kk,kk,ORDER_N,CTX); BN_mod_mul(s,kk,nri,ORDER_N,CTX);
        EC_POINT *want=EC_POINT_new(GRP);
        require(EC_POINT_mul(GRP,want,s,NULL,NULL,CTX));
        require(EC_POINT_get_affine_coordinates_GFp(GRP,want,ax,ay,CTX));
        BIGNUM *gx=BN_new(),*gy=BN_new();
        to_affine(X2,Y2,Z2,W2,gx,gy);
        if(BN_cmp(gx,ax)||BN_cmp(gy,ay)){ fails++;
            fprintf(stderr,"case %d: pipe result != (k mod n)*nri*G\n",t);}
        BN_free(kk);BN_free(s);BN_free(ax);BN_free(ay);BN_free(gx);BN_free(gy);
        EC_POINT_free(want);
#endif
    }
    if(fails){ fprintf(stderr,"FAIL: %d divergence(s) in %d cases\n",fails,cases); return 1; }
    printf("OK: %d cases, chain_pipe bitwise-identical to chain_ref%s\n",cases,
#ifdef FILL_LOADER
           " and affine-equal to OpenSSL z*A");
#else
           " (fake table)");
#endif
    return 0;
}
'''
    repl = {"@@ORDER_N@@":order_n,"@@ENTRIES@@":entries,"@@OFFSET@@":offset,
            "@@SHIFT@@":shift,"@@ARENA@@":arena,"@@RECODE@@":recode,
            "@@DECODE@@":decode,"@@LOADER@@":loader,"@@LOAD_DEC@@":load_dec,
            "@@MM@@":mm,"@@MADDT@@":maddT,"@@MADDPIPE@@":maddPipe,"@@SCALAR@@":scalar}
    for k_,v_ in repl.items(): cpp = cpp.replace(k_,v_)

    work = Path(tempfile.mkdtemp(prefix="qsb_chain_pipe_"))
    srcf = work/"check_chain_pipe.cpp"; srcf.write_text(cpp, encoding="utf-8")
    exeA = work/"check_fake.exe"; exeB = work/"check_real.exe"
    inc = r"C:/Strawberry/c/include"; lib = r"C:/Strawberry/c/lib"
    ok = True
    for exe,defs,cases in ((exeA,[],1500),(exeB,["-DFILL_LOADER"],120)):
        cmd = ["g++","-O2","-std=c++17","-I",inc]+defs+[str(srcf),"-o",str(exe),
               "-L",lib,"-lcrypto"]
        r = subprocess.run(cmd,capture_output=True,text=True)
        if r.returncode:
            sys.stderr.write(r.stderr); print("BUILD FAIL",exe.name); ok=False; continue
        r = subprocess.run([str(exe),str(cases)],capture_output=True,text=True)
        sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
        ok = ok and (r.returncode==0)
    print("chain-pipe oracle:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
