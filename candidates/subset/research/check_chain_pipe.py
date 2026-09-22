#!/usr/bin/env python3
"""Structural CPU check of the ZLAB_CHAIN_PIPE depth-1 gather pipeline inside
qsb_filter_chain_trial (tests/gpu_epochs/tree.cu).

Extracts the production chain verbatim -- gt_recode_setup, gt_direct_digit /
the funnel-shift digit window, gt_load_signed_flat_f, the qsb_filter_* field
helpers' host fallbacks, qsb_filter_point_seed, qsb_filter_point_add,
qsb_filter_last_add and the scalar driver -- and compiles it twice under the
names chain_ref (ZLAB_CHAIN_PIPE=0, the promoted load-at-head loop) and
chain_pipe (ZLAB_CHAIN_PIPE=1, the (cx,cy)/(x1,y1) double-buffer prefetch).
Field arithmetic is OpenSSL-backed, so any divergence between the two variants
is a bug in the buffer rotation / decode schedule, not in the field code
(which is identical in both).

Test A (fake table, 64 MiB of PRNG bytes): chain_pipe must produce BITWISE
identical X,Y,ZZ,ZZZ outputs to chain_ref on every scalar -- catches wrong
buffer rotation, wrong chunk index, off-by-one prefetch, anchor corruption.

Test B (lazy real table, OpenSSL EC points): both chains must produce the
affine point (k mod n)*nri*G -- anchors the structure to real group
arithmetic including the deferred-Y resolve and the filter negation formula.

The anchor copy is exercised for real by building with
-DQSB_CHAIN_ANCHOR_UPDATE=0, which selects the explicit Load256(y0, <consumed
y>) arms in both loop bodies -- semantically identical to the device path
where the asm publishes Yoff=Y2. A second build with =1 checks the flag
combination that ships (host fallback keeps a stale-but-identical anchor in
both variants; buffer rotation is still fully exercised).

This tests the loop structure and buffer lifetimes only. It does not test
PTX scheduling, registers, occupancy or speed; those are measured by the
ranked build on the actual GPU.
"""
import os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC  = HERE.parent / "tests" / "gpu_epochs" / "tree.cu"
HDR  = HERE.parent / "hit_filter_field_sc.cuh"

def function(source, signature, occurrence=0):
    """Extract a function definition starting at `signature`. Skips forward
    declarations (signature followed by ';' before any '{'). The brace scan
    is string- and comment-aware: the inline-PTX bodies contain unbalanced
    '{' or '}' characters inside quoted strings, which a naive counter
    mistakes for real scope boundaries."""
    start = 0
    for _ in range(occurrence + 1):
        start = source.index(signature, start)
        brace = source.index("{", start)
        semi  = source.find(";", start, brace)
        if semi == -1:
            break
        start = semi + 1
    depth = 1
    end = brace + 1
    instr = esc = incmt = inlcm = False
    while depth and end < len(source):
        c = source[end]
        n = source[end + 1] if end + 1 < len(source) else ''
        if inlcm:
            if c == '\n':
                inlcm = False
        elif incmt:
            if c == '*' and n == '/':
                incmt = False
                end += 1
        elif instr:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                instr = False
        else:
            if c == '/' and n == '/':
                inlcm = True
                end += 1
            elif c == '/' and n == '*':
                incmt = True
                end += 1
            elif c == '"':
                instr = True
            else:
                depth += (c == '{') - (c == '}')
        end += 1
    return source[start:end]

def block(source, start_marker, end_marker):
    i = source.index(start_marker)
    j = source.index(end_marker, i)
    return source[i:j]

def main():
    src = SRC.read_text(encoding="utf-8")
    hf  = HDR.read_text(encoding="utf-8")

    order_n   = block(src, "__device__ __constant__ uint64_t GT_ORDER_N", ";") + ";"
    # ZLAB_T14=0 geometry: the #else arm defines GT_CHUNKS 15 ... gt_shift.
    # Grab from the 15-chunk #define up to (not incl.) the static_assert.
    geometry  = block(src, "#define GT_CHUNKS 15", "static_assert")
    recode    = function(src, "void gt_recode_setup")
    fieldbits = function(src, "uint32_t gt_field_bits_v")
    width     = function(src, "unsigned gt_width")
    digit     = function(src, "void gt_direct_digit")
    loader    = function(src, "void gt_load_signed_flat(")
    loader_f  = function(src, "void gt_load_signed_flat_f(")
    prefetch  = function(src, "void gt_prefetch_flat_f(")
    lastadd   = function(src, "void qsb_filter_last_add(")
    chain     = function(src, "void qsb_filter_chain_trial(")

    fadd      = function(hf, "void qsb_filter_add(")
    fmul3     = function(hf, "void qsb_filter_mul(uint64_t *r, const uint64_t *a, const uint64_t *b, uint32_t &bad)")
    fmul1     = function(hf, "void qsb_filter_mul(uint64_t*r,const uint64_t*a,uint32_t &bad)")
    fsqr      = function(hf, "void qsb_filter_sqr(")
    faddpt    = "template<bool DEFER_Y>\n" + function(hf, "void qsb_filter_point_add(")
    seedx3    = function(hf, "void qsb_filter_seed_x3(")
    pseed     = function(hf, "void qsb_filter_point_seed(")

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
static uint32_t __funnelshift_r(uint32_t lo, uint32_t hi, uint32_t s){
    return (uint32_t)(((((uint64_t)hi)<<32)|(uint64_t)lo) >> s);
}

/* carry-flag emulation for the PTX add macros used by the verbatim loader */
static uint64_t __carry;
#define UADDO1(c,a) do{ __uint128_t s=(__uint128_t)(c)+(a); (c)=(uint64_t)s; __carry=(uint64_t)(s>>64);}while(0)
#define UADDC1(c,a) do{ __uint128_t s=(__uint128_t)(c)+(a)+__carry; (c)=(uint64_t)s; __carry=(uint64_t)(s>>64);}while(0)
#define UADD1(c,a)  do{ (c)=(c)+(a)+__carry; }while(0)
#define Load256(r,a) do{ (r)[0]=(a)[0];(r)[1]=(a)[1];(r)[2]=(a)[2];(r)[3]=(a)[3]; }while(0)

/* ---- ranked flag configuration ---------------------------------------- */
#define ZLAB_T14 0
#define ZLAB_DIRDIG 1
#define QSB_DIGIT_SHIFT 1
#define QSB_CHAIN_UNROLL 1
#define QSB_NEG_SHORT 1
#define QSB_SPEC_LAST_RESOLVE 1
#ifndef QSB_CHAIN_ANCHOR_UPDATE
#define QSB_CHAIN_ANCHOR_UPDATE 1
#endif
/* filter_tail_sc.cuh is not extracted; give qsb_filter_last_add a canonical
 * QSB_FSUB so the resolve is exact on host. */
#define QSB_FSUB(r,a,b) _ModSub256(r,(uint64_t*)(a),(uint64_t*)(b))
/* field-helper knobs (their #if arms live inside __CUDA_ARCH__ asm bodies,
 * dead on host; defaults kept for fidelity) */
#define QSB_SHORT_CARRY2 1
#define QSB_FINAL_CARRY 1
#define QSB_CHAIN_MUL_LEAN 1

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

void _ModMultCore(uint64_t *r, const uint64_t *a, const uint64_t *b){
    BIGNUM *A=rd(a),*B=rd(b),*R=BN_new(); fmul(R,A,B); wr(r,R);
    BN_free(A);BN_free(B);BN_free(R);
}
void _ModMult(uint64_t *r, uint64_t *a, uint64_t *b){ _ModMultCore(r,a,b); }
void _ModMult(uint64_t *r, uint64_t *a){ _ModMultCore(r,r,a); }
void _ModSub256(uint64_t *r, const uint64_t *a, const uint64_t *b){
    BIGNUM *A=rd(a),*B=rd(b),*R=BN_new(); fsub(R,A,B); wr(r,R);
    BN_free(A);BN_free(B);BN_free(R);
}
void _ModSub256(uint64_t *r, uint64_t *b){ _ModSub256(r,r,b); }
void _ModAdd256(uint64_t *r, const uint64_t *a, const uint64_t *b){
    BIGNUM *A=rd(a),*B=rd(b),*R=BN_new(); fadd(R,A,B); wr(r,R);
    BN_free(A);BN_free(B);BN_free(R);
}
void _ModSqr(uint64_t *r, const uint64_t *a){
    BIGNUM *A=rd(a),*R=BN_new(); BN_mod_sqr(R,A,PRIME,CTX); wr(r,R);
    BN_free(A);BN_free(R);
}

/* ---- verbatim production code ----------------------------------------- */
@@ORDER_N@@

@@GEOMETRY@@

@@RECODE@@
@@FIELDBITS@@
@@WIDTH@@
@@DIGIT@@

@@LOADER@@
#ifdef FILL_LOADER
/* Test B loader: compute the real table point lazily with OpenSSL instead of
 * reading the flat table. Same signature and negate semantics as
 * gt_load_signed_flat_f under QSB_NEG_SHORT: gy = ~y - (K-1) with the limb-0
 * borrow dropped (exact whenever y0 <= 2^64 - K). */
static void gt_load_signed_flat_f(const uint8_t*, uint32_t base, uint32_t idx,
                                  uint64_t neg, uint64_t *gx, uint64_t *gy) {
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
    uint64_t m=0ULL-neg;
    gy[0]=(gy[0]^m)+(0xFFFFFFFEFFFFFC30ULL&m); gy[1]^=m; gy[2]^=m; gy[3]^=m;
    BN_free(k);BN_free(x);BN_free(y);EC_POINT_free(pt);
}
#else
@@LOADER_F@@
#endif

@@PREFETCH@@

@@FADD@@
@@FMUL3@@
@@FSQR@@
@@FMUL1@@
@@SEEDX3@@
@@PSEED@@
@@FADDPT@@
@@LASTADD@@

/* The verbatim scalar driver is included twice under different names. */
static void dummy_unused(void){}
#define qsb_filter_chain_trial chain_ref
#define ZLAB_CHAIN_PIPE 0
@@CHAIN@@
#undef qsb_filter_chain_trial
#undef ZLAB_CHAIN_PIPE
#define qsb_filter_chain_trial chain_pipe
#define ZLAB_CHAIN_PIPE 1
@@CHAIN@@
#undef qsb_filter_chain_trial
#undef ZLAB_CHAIN_PIPE

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
    uint32_t b1=0,b2=0;
    int fails=0;
    for(int t=0;t<cases;t++){
        for(int j=0;j<4;j++)k[j]=rng();
        b1=b2=0;
        chain_ref (X1,Y1,Z1,W1,k,(const uint8_t*)
#ifdef FILL_LOADER
                   nullptr
#else
                   fake_table
#endif
                   ,b1);
        chain_pipe(X2,Y2,Z2,W2,k,(const uint8_t*)
#ifdef FILL_LOADER
                   nullptr
#else
                   fake_table
#endif
                   ,b2);
        uint64_t *A[4]={X1,Y1,Z1,W1},*B[4]={X2,Y2,Z2,W2};
        for(int j=0;j<4;j++)if(memcmp(A[j],B[j],32)){ fails++;
            fprintf(stderr,"case %d: coord %d diverges (ref vs pipe)\n",t,j);break;}
        if(b1!=b2){ fails++; fprintf(stderr,"case %d: bad flag diverges\n",t); }
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
            if(fails<=2){ char *hx=BN_bn2hex(gx),*hy=BN_bn2hex(gy),*ex=BN_bn2hex(ax),*ey=BN_bn2hex(ay);
                fprintf(stderr,"case %d: pipe result != expected\n  got x=%s\n      y=%s\n  exp x=%s\n      y=%s\n",t,hx,hy,ex,ey);}
            else fprintf(stderr,"case %d: pipe result != (k mod n)*nri*G\n",t);}
        BN_free(kk);BN_free(s);BN_free(ax);BN_free(ay);BN_free(gx);BN_free(gy);
        EC_POINT_free(want);
#endif
    }
    if(fails){ fprintf(stderr,"FAIL: %d divergence(s) in %d cases\n",fails,cases); return 1; }
    printf("OK: %d cases, chain_pipe bitwise-identical to chain_ref%s\n",cases,
#ifdef FILL_LOADER
           " and affine-equal to OpenSSL (k mod n)*nri*G");
#else
           " (fake table)");
#endif
    return 0;
}
'''
    repl = {"@@ORDER_N@@":order_n,"@@GEOMETRY@@":geometry,"@@RECODE@@":recode,
            "@@FIELDBITS@@":fieldbits,"@@WIDTH@@":width,"@@DIGIT@@":digit,
            "@@LOADER@@":loader,"@@LOADER_F@@":loader_f,"@@PREFETCH@@":prefetch,
            "@@FADD@@":fadd,"@@FMUL3@@":fmul3,"@@FMUL1@@":fmul1,"@@FSQR@@":fsqr,
            "@@SEEDX3@@":seedx3,"@@PSEED@@":pseed,"@@FADDPT@@":faddpt,
            "@@LASTADD@@":lastadd,"@@CHAIN@@":chain}
    for k_,v_ in repl.items(): cpp = cpp.replace(k_,v_)

    work = Path(tempfile.mkdtemp(prefix="qsb_subset_chain_pipe_"))
    srcf = work/"check_chain_pipe.cpp"; srcf.write_text(cpp, encoding="utf-8")

    # Compiler detection: Windows Strawberry g++ first, then a plain g++
    # (WSL/Linux). OpenSSL is required for -lcrypto.
    compilers = []
    if os.name == "nt":
        compilers.append(["g++","-O2","-std=c++17","-w","-I",r"C:/Strawberry/c/include"])
        libdirs = ["-L",r"C:/Strawberry/c/lib"]
    else:
        compilers.append(["g++","-O2","-std=c++17","-w"])
        libdirs = []
    ok = True
    built = False
    for base in compilers:
        exeA = work/"check_fake"; exeB = work/"check_real"; exeC = work/"check_fake_anchor"
        jobs = [
            (exeA, ["-DQSB_CHAIN_ANCHOR_UPDATE=0"], 1500, "fake table, ANCHOR_UPDATE=0"),
            (exeC, ["-DQSB_CHAIN_ANCHOR_UPDATE=1"],  500, "fake table, ANCHOR_UPDATE=1"),
            (exeB, ["-DFILL_LOADER","-DQSB_CHAIN_ANCHOR_UPDATE=0"], 120, "real EC table"),
        ]
        # Probe this compiler with the first job; on build failure try the next.
        out0 = str(exeA) + (".exe" if os.name == "nt" else "")
        r = subprocess.run(base + jobs[0][1] + [str(srcf),"-o",out0] + libdirs + ["-lcrypto"],
                           capture_output=True,text=True)
        if r.returncode:
            continue
        built = True
        for exe,defs,cases,label in jobs:
            out = str(exe) + (".exe" if os.name == "nt" else "")
            if out != out0:
                r = subprocess.run(base + defs + [str(srcf),"-o",out] + libdirs + ["-lcrypto"],
                                   capture_output=True,text=True)
                if r.returncode:
                    sys.stderr.write(r.stderr); print("BUILD FAIL",label); ok=False; continue
            r = subprocess.run([out,str(cases)],capture_output=True,text=True)
            sys.stdout.write("[%s] %s"%(label,r.stdout)); sys.stderr.write(r.stderr)
            ok = ok and (r.returncode==0)
        break
    if not built:
        print("chain-pipe oracle: NO WORKING COMPILER"); sys.exit(2)
    print("chain-pipe oracle:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
