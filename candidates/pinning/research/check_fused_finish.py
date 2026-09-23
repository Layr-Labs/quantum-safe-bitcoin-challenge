#!/usr/bin/env python3
"""Concurrency-faithful CPU oracle for the QSB_FUSED single-kernel finish
(qsb_fused_finish inside kernel_pinning_pipeline STAGE==0, pinning.cu).

The fused path replaces the saved[]/root-hierarchy/S2 round-trip with an
in-block _ModInv of the cofactor tree root.  This oracle extracts the new
code VERBATIM -- qsb_digit_arena, qsb_field_normalize, qsb_recovery_mul,
qsb_recovery_denominator, the packed-recovery helpers,
qsb_cofactor_prepare, qsb_fused_finish itself (including the emit tail)
and gpu_bench_valid_words -- and runs it on 128 real std::threads with
real std::barrier synchronization, so collective indexing and barrier
placement are exercised exactly as written.  __shared__ maps to static
storage shared by all 128 threads = one emulated CTA.

Field arithmetic is OpenSSL-backed (canonical mod p), so any divergence
between the fused path and the independent EC reference is a bug in the
new wiring, not in the field primitives.

Checks per lane of each simulated CTA:
  A) recovered x-coords: x1 == x(P+R), x2 == x(P-R) for the true affine
     point P=(X/ZZ, Y/ZZZ), verified through OpenSSL EC_POINT_add -- the
     whole denominator -> exclusion-tree -> root-inverse -> packed-finish
     chain, anchored to real curve arithmetic.
  B) pubkey33 block bytes: the pb[] words reconstruct to compress(Q)
     exactly (0x02|parity || x big-endian), for both recids.
  C) hit records: emitted raw = idx | ri<<30 (hc<<31 for the dead second
     hash path) for lanes whose digest passes the gate; unusable lanes
     never emit and never poison siblings.
  D) fault injection: lanes forced inactive (active=false -> leaf=1) and
     lanes with a real zero denominator (P=R -> prod=0 -> leaf=1) do not
     corrupt usable siblings; all-dead and 73-lane tail CTAs are clean.

Gate note: QSB_ZEROS_N is set to 4 (not 24) so the emit path is exercised
(~1/16 lanes/recid hit instead of ~1/2^23).

Not covered (declared): PTX scheduling, register pressure, real GPU
barrier timing, carry-edge behaviour of the raw muls (canonical OpenSSL
backend), and the sparse-vs-generic SHA equivalence -- the pubkey33
transform is shimmed to OpenSSL SHA-256 because it is unchanged by this
diff and already covered elsewhere.
"""
import os, shlex, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC  = HERE.parent / "pinning.cu"
LEAF = HERE.parent / "LeafRecovery.cuh"
PACK = HERE.parent / "PackedRecovery.cuh"
COF  = HERE.parent / "cofactor_checkpoint.h"

def function(source, signature):
    """Extract a function definition starting at `signature`. Skips forward
    declarations (signature followed by ';' before any '{'). The brace scan
    is string- and comment-aware: inline-PTX bodies contain unbalanced
    '{' or '}' inside quoted strings, which a naive counter mistakes for
    real scope boundaries."""
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

#define QSB_TREE_N 128
#define QSB_RECOVERY_N 128
#define QSB_ZEROS_N 4            /* low gate so the emit path is exercised */
#define QSB_PK_UNROLL 1
#define QSB_SPARSE_D 1
#define GT_CHUNKS 15

#define __ldg(p) (*(p))
#define Load256(r,a) do{ (r)[0]=(a)[0];(r)[1]=(a)[1];(r)[2]=(a)[2];(r)[3]=(a)[3]; }while(0)

static BN_CTX *ctx;
static BIGNUM *prime;
static void require_(bool ok,int ln,const char*what){
    if(!ok){ std::fprintf(stderr,"require failed @%d (%s)\n",ln,what); std::exit(1);} }
#define require(ok) require_((ok),__LINE__,#ok)
static BIGNUM *read256(const uint64_t *v){
    return BN_lebin2bn(reinterpret_cast<const unsigned char*>(v),32,nullptr);
}
static void write256(uint64_t *v,const BIGNUM *b){
    require(BN_bn2lebinpad(b,reinterpret_cast<unsigned char*>(v),32)==32);
}

/* OpenSSL-backed field backend (canonical mod p).  BN_CTX is not
 * thread-safe -> serialize behind a mutex; correctness over speed.
 * Raw inputs may carry non-canonical residues -> reduce on read. */
static std::mutex field_mtx;
static void fmul256(uint64_t *out,const uint64_t *a,const uint64_t *b){
    std::lock_guard<std::mutex> lk(field_mtx);
    BIGNUM *aa=read256(a),*bb=read256(b),*rr=BN_new();
    require(BN_nnmod(aa,aa,prime,ctx));
    require(BN_nnmod(bb,bb,prime,ctx));
    require(BN_mod_mul(rr,aa,bb,prime,ctx));
    write256(out,rr); out[4]=0;
    BN_free(aa); BN_free(bb); BN_free(rr);
}
static void qsb_field_mul(uint64_t *out,uint64_t *a,uint64_t *b){ fmul256(out,a,b); }
static void qsb_field_mul_sc(uint64_t *out,uint64_t *a,uint64_t *b){ fmul256(out,a,b); }
static void _ModMult(uint64_t *r,uint64_t *a,uint64_t *b){ fmul256(r,a,b); }
static void _ModSub256(uint64_t *r,const uint64_t *a,const uint64_t *b){
    std::lock_guard<std::mutex> lk(field_mtx);
    BIGNUM *aa=read256(a),*bb=read256(b),*rr=BN_new();
    require(BN_nnmod(aa,aa,prime,ctx));
    require(BN_nnmod(bb,bb,prime,ctx));
    require(BN_mod_sub(rr,aa,bb,prime,ctx));
    write256(r,rr);
    BN_free(aa); BN_free(bb); BN_free(rr);
}
static void _ModSub256(uint64_t *r,const uint64_t *b){
    uint64_t tmp[4]; Load256(tmp,r);
    _ModSub256(r,tmp,b);
}
static void _ModAdd256(uint64_t *r,const uint64_t *a,const uint64_t *b){
    std::lock_guard<std::mutex> lk(field_mtx);
    BIGNUM *aa=read256(a),*bb=read256(b),*rr=BN_new();
    require(BN_nnmod(aa,aa,prime,ctx));
    require(BN_nnmod(bb,bb,prime,ctx));
    require(BN_mod_add(rr,aa,bb,prime,ctx));
    write256(r,rr);
    BN_free(aa); BN_free(bb); BN_free(rr);
}
static void _ModInv(uint64_t *r){
    std::lock_guard<std::mutex> lk(field_mtx);
    BIGNUM *a=read256(r);
    require(BN_nnmod(a,a,prime,ctx));
    BIGNUM *b=BN_mod_inverse(nullptr,a,prime,ctx);
    require(b!=nullptr); write256(r,b); r[4]=0;
    BN_free(a); BN_free(b);
}
/* The 33-byte compress() block: pb[0..8] live, rest zero, W[15]=0x108.
 * Shim computes the real SHA-256 of the assembled block so the verbatim
 * gpu_bench_valid_words gate sees the true digest. */
static void _SHA256TransformPubkey33(uint32_t out[8], const uint32_t m[9]){
    uint8_t blk[64];
    for(int i=0;i<9;i++){ blk[4*i]=(m[i]>>24)&0xFF; blk[4*i+1]=(m[i]>>16)&0xFF;
        blk[4*i+2]=(m[i]>>8)&0xFF; blk[4*i+3]=m[i]&0xFF; }
    memset(blk+36,0,24);
    blk[60]=0; blk[61]=0; blk[62]=1; blk[63]=8;   /* W[15]=0x108 */
    uint8_t d[32];
    SHA256(blk,64,d);
    for(int i=0;i<8;i++)
        out[i]=((uint32_t)d[4*i]<<24)|((uint32_t)d[4*i+1]<<16)|((uint32_t)d[4*i+2]<<8)|d[4*i+3];
}
/* Dead-code stubs: FAST_TAIL=true never reaches the second hash, but the
 * branch must still compile. */
static void _SHA256Initialize(uint32_t out[8]){
    const uint32_t iv[8]={0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                          0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    for(int i=0;i<8;i++) out[i]=iv[i];
}
static void _SHA256Transform(uint32_t st[8], const uint32_t m[16]){
    uint8_t blk[64];
    for(int i=0;i<16;i++){ blk[4*i]=(m[i]>>24)&0xFF; blk[4*i+1]=(m[i]>>16)&0xFF;
        blk[4*i+2]=(m[i]>>8)&0xFF; blk[4*i+3]=m[i]&0xFF; }
    uint8_t d[32]; SHA256(blk,64,d);
    for(int i=0;i<8;i++)
        st[i]+=((uint32_t)d[4*i]<<24)|((uint32_t)d[4*i+1]<<16)|((uint32_t)d[4*i+2]<<8)|d[4*i+3];
}
static uint32_t __byte_perm(uint32_t a, uint32_t b, uint32_t s){
    uint8_t src[8];
    for(int i=0;i<4;i++){ src[i]=(a>>(8*i))&0xFF; src[4+i]=(b>>(8*i))&0xFF; }
    uint32_t r=0;
    for(int i=0;i<4;i++){ unsigned sel=(s>>(4*i))&0x7u; r|=(uint32_t)src[sel]<<(8*i); }
    return r;
}
static int gpu_is_der_easy(const uint8_t*,int){ return 0; }
static int gpu_bench_valid_words(const uint32_t *hs);   /* defined verbatim below */

/* Concurrency shims: one CTA of 128 std::threads; __syncthreads is a full
 * barrier and __syncwarp a per-warp barrier (4 groups of 32). */
static constexpr int NTH=128;
static thread_local int sim_tid=0;
static std::barrier<> sim_bar(NTH);
static std::barrier<> *sim_wbar[4];   /* allocated in main (barrier isn't movable) */
static uint32_t sim_atomicAdd(uint32_t *p,uint32_t v){
    return reinterpret_cast<std::atomic<uint32_t>*>(p)->fetch_add(v);
}

/* Device constants -> harness globals, set by the driver. */
static uint64_t pin_u2rx_words[4];
static uint64_t pin_u2ry_words[4];
static uint64_t pin_recovery_c[4];

/* Instrumentation globals: the extracted fused body writes through these
 * via surgical hooks, so they must be declared before it. */
enum LaneMode { LANE_LIVE=0, LANE_INACTIVE=1, LANE_ZERODEN=2 };
struct LaneIn {
    uint64_t X[4],Y[4],ZZ[4],ZZZ[4];   /* XYZZ point state */
    LaneMode mode;
};
static LaneIn g_in[NTH];
static uint64_t g_roots[4*128];
static uint32_t g_hit_cnt;
static uint32_t g_hit_idx[1024];
static uint64_t g_x1[NTH][4], g_x2[NTH][4];
static uint8_t  g_pub[NTH][2][33];     /* pubkey block bytes 0..32 */
static int      g_emitted[NTH][2];
'''

CPP_TAIL = r'''
/* ---- harness ---------------------------------------------------------- */
static void run_cta(void){
    std::vector<std::thread> ts;
    for(int t=0;t<NTH;t++)
        ts.emplace_back([t]{
            sim_tid=t;
            uint64_t prod[5]={0};
            bool active = g_in[t].mode!=LANE_INACTIVE;
            if(active){
                qsb_recovery_denominator(
                    g_in[t].X,g_in[t].ZZ,g_in[t].Y,g_in[t].ZZZ,
                    pin_u2rx_words,prod);
            }
            /* production mask: usable = active && prod != 0; leaf=1 else */
            bool usable = active && ((prod[0]|prod[1]|prod[2]|prod[3])!=0);
            if(!usable){prod[0]=1;prod[1]=prod[2]=prod[3]=prod[4]=0;}
            qsb_fused_finish<true>(prod,g_in[t].ZZ,g_in[t].Y,g_in[t].ZZZ,
                usable,(uint32_t)t,0,1,&g_hit_cnt,g_hit_idx,g_roots);
        });
    for(auto &th:ts) th.join();
}
'''

def main():
    src  = SRC.read_text(encoding="utf-8")
    leaf = LEAF.read_text(encoding="utf-8")
    pack = PACK.read_text(encoding="utf-8")
    cof  = COF.read_text(encoding="utf-8")

    # Dependency order: each extracted function only calls what precedes it
    # or a prefix shim.
    ext = []
    ext.append(function(src,  "__device__ __forceinline__ void qsb_field_normalize"))
    ext.append(function(leaf, "__device__ __forceinline__ void qsb_recovery_mul"))
    ext.append(function(src,  "__device__ __forceinline__ uint64_t *qsb_digit_arena"))
    ext.append(function(pack, "__device__ __forceinline__ void qsb_parity_boundary"))
    ext.append(function(pack, "__device__ __forceinline__ void qsb_add_boundary"))
    ext.append(function(pack, "__device__ __forceinline__ uint32_t qsb_difference_parity"))
    ext.append(function(pack, "__device__ __forceinline__ void qsb_packed_raw_mul"))
    ext.append(function(pack, "__device__ __forceinline__ uint32_t qsb_packed_finish"))
    ext.append(function(leaf, "__device__ __forceinline__ void qsb_recovery_denominator"))
    ext.append("template<int N> " + function(cof, "void qsb_cofactor_prepare"))
    ext.append(function(src,  "__device__ __forceinline__ int gpu_bench_valid_words"))
    ext.append("template<bool FAST_TAIL>\n" +
               function(src, "void qsb_fused_finish"))
    body = "\n".join(ext)

    # Retarget CUDA tokens/intrinsics to the harness shims.
    body = body.replace("__device__", "")
    body = body.replace("__forceinline__", "")
    body = body.replace("__host__", "")
    body = body.replace("__shared__", "static")
    body = body.replace("threadIdx.x", "sim_tid")
    body = body.replace("blockIdx.x", "0u")
    body = body.replace("blockDim.x", "128u")
    body = body.replace("__syncthreads()", "sim_bar.arrive_and_wait()")
    body = body.replace("__syncwarp()", "sim_wbar[sim_tid>>5]->arrive_and_wait()")
    body = body.replace("atomicAdd", "sim_atomicAdd")

    # Instrumentation hooks on the verbatim fused body (surgical, marked).
    hook_marker = ("uint32_t y_parities = qsb_packed_finish(\n"
                   "        vbar,tbar,rinv,winv,u2rx,u2ry,recovery_c,q1x,q2x);")
    assert hook_marker in body, "packed_finish anchor missing"
    body = body.replace(hook_marker, hook_marker + r'''
    memcpy(g_x1[sim_tid],q1x,32); memcpy(g_x2[sim_tid],q2x,32);''')

    pub_anchor = "pb[7]=__byte_perm(x1,x0,0x0765);pb[8]=__byte_perm(x0,0x80,0x0456);"
    assert pub_anchor in body, "pubkey anchor missing"
    body = body.replace(pub_anchor, pub_anchor + r'''
        { const uint32_t wv[9]={pb[0],pb[1],pb[2],pb[3],pb[4],pb[5],pb[6],pb[7],pb[8]};
          for(int wi=0;wi<9;wi++){ g_pub[sim_tid][ri][4*wi]=(wv[wi]>>24)&0xFF;
            if(4*wi+1<33)g_pub[sim_tid][ri][4*wi+1]=(wv[wi]>>16)&0xFF;
            if(4*wi+2<33)g_pub[sim_tid][ri][4*wi+2]=(wv[wi]>>8)&0xFF;
            if(4*wi+3<33)g_pub[sim_tid][ri][4*wi+3]=(wv[wi])&0xFF; } }''')

    emit_anchor = "if(pos<1024)hit_idx[pos]=idx|(ri<<30);"
    assert emit_anchor in body, "emit anchor missing"
    body = body.replace(emit_anchor, emit_anchor + r'''
            g_emitted[sim_tid][ri]=1;''')
    emit_anchor2 = "if(pos<1024)hit_idx[pos]=idx|(ri<<30)|(1u<<31);"
    assert emit_anchor2 in body, "emit-hc anchor missing"
    body = body.replace(emit_anchor2, emit_anchor2 + r'''
            g_emitted[sim_tid][ri]=2;''')

    driver = r'''
static BIGNUM *order_n;
static EC_GROUP *grp;
static EC_POINT *Rpt;
static BIGNUM *R_scalar;    /* k such that R = k*G, for zero-denominator lanes */

/* Expected recoveries for lane t: x(P+R), x(P-R), y parities. */
static void expected(int t, BIGNUM *ex1, BIGNUM *ex2, int *par){
    BIGNUM *x=read256(g_in[t].X),*y=read256(g_in[t].Y),
           *zz=read256(g_in[t].ZZ),*zzz=read256(g_in[t].ZZZ);
    BIGNUM *i1=BN_new(),*i2=BN_new(),*ax=BN_new(),*ay=BN_new();
    require(BN_mod_inverse(i1,zz,prime,ctx)!=NULL);
    require(BN_mod_inverse(i2,zzz,prime,ctx)!=NULL);
    require(BN_mod_mul(ax,x,i1,prime,ctx));
    require(BN_mod_mul(ay,y,i2,prime,ctx));
    EC_POINT *P=EC_POINT_new(grp);
    require(EC_POINT_set_affine_coordinates_GFp(grp,P,ax,ay,ctx));
    EC_POINT *Q=EC_POINT_new(grp);
    BIGNUM *qx=BN_new(),*qy=BN_new();
    require(EC_POINT_add(grp,Q,P,Rpt,ctx));              /* P+R */
    require(EC_POINT_get_affine_coordinates_GFp(grp,Q,qx,qy,ctx));
    require(BN_nnmod(qx,qx,prime,ctx)); require(BN_nnmod(qy,qy,prime,ctx));
    BN_copy(ex1,qx); par[0]=BN_is_odd(qy);
    EC_POINT *nR=EC_POINT_new(grp); EC_POINT_copy(nR,Rpt);
    BIGNUM *rx=BN_new(),*ry=BN_new();
    require(EC_POINT_get_affine_coordinates_GFp(grp,nR,rx,ry,ctx));
    require(BN_sub(ry,prime,ry));
    require(EC_POINT_set_affine_coordinates_GFp(grp,nR,rx,ry,ctx));
    require(EC_POINT_add(grp,Q,P,nR,ctx));               /* P-R */
    require(EC_POINT_get_affine_coordinates_GFp(grp,Q,qx,qy,ctx));
    require(BN_nnmod(qx,qx,prime,ctx)); require(BN_nnmod(qy,qy,prime,ctx));
    BN_copy(ex2,qx); par[1]=BN_is_odd(qy);
    BN_free(x);BN_free(y);BN_free(zz);BN_free(zzz);
    BN_free(i1);BN_free(i2);BN_free(ax);BN_free(ay);
    BN_free(qx);BN_free(qy);BN_free(rx);BN_free(ry);
    EC_POINT_free(P);EC_POINT_free(Q);EC_POINT_free(nR);
}

/* Fill lane t's XYZZ state with P scaled by random w^2/w^3.  If P is the
 * provided point directly (zden), the denominator V*(xR*U-X) == 0. */
static void fill_lane(int t, EC_POINT *P){
    unsigned char seed[40];
    memcpy(seed,"lane",4);
    uint64_t tv=(uint64_t)t; memcpy(seed+4,&tv,8);
    uint64_t rv; { unsigned char h8[8]; SHA256(seed,12,h8); memcpy(&rv,h8,8); }
    memcpy(seed+12,&rv,8);
    unsigned char h[64];
    SHA512(seed,20,h);
    BIGNUM *w=BN_bin2bn(h+32,32,nullptr);
    require(BN_nnmod(w,w,prime,ctx));
    if(BN_is_zero(w)) BN_set_word(w,7);
    BIGNUM *px=BN_new(),*py=BN_new();
    require(EC_POINT_get_affine_coordinates_GFp(grp,P,px,py,ctx));
    BIGNUM *w2=BN_new(),*w3=BN_new(),*X=BN_new(),*Y=BN_new();
    require(BN_mod_sqr(w2,w,prime,ctx));
    require(BN_mod_mul(w3,w2,w,prime,ctx));
    require(BN_mod_mul(X,px,w2,prime,ctx));
    require(BN_mod_mul(Y,py,w3,prime,ctx));
    write256(g_in[t].X,X); write256(g_in[t].Y,Y);
    write256(g_in[t].ZZ,w2); write256(g_in[t].ZZZ,w3);
    BN_free(w);BN_free(px);BN_free(py);BN_free(w2);BN_free(w3);BN_free(X);BN_free(Y);
}

int main(int argc,char**argv){
    int ctas = argc>1?atoi(argv[1]):24;
    for(int i=0;i<4;i++) sim_wbar[i]=new std::barrier<>(32);
    ctx=BN_CTX_new();
    grp=EC_GROUP_new_by_curve_name(NID_secp256k1);
    prime=BN_new(); order_n=BN_new();
    require(EC_GROUP_get_curve(grp,prime,nullptr,nullptr,ctx));
    require(EC_GROUP_get_order(grp,order_n,ctx));

    /* R = kR*G; constants a=xR, b=yR, c=3xR^2/(2yR). */
    {
        unsigned char h[32];
        SHA256(reinterpret_cast<const unsigned char*>("qsb-R"),5,h);
        R_scalar=BN_bin2bn(h,32,nullptr);
        require(BN_nnmod(R_scalar,R_scalar,order_n,ctx));
        Rpt=EC_POINT_new(grp);
        require(EC_POINT_mul(grp,Rpt,R_scalar,NULL,NULL,ctx));
        BIGNUM *x=BN_new(),*y=BN_new();
        require(EC_POINT_get_affine_coordinates_GFp(grp,Rpt,x,y,ctx));
        write256(pin_u2rx_words,x); write256(pin_u2ry_words,y);
        BIGNUM *t=BN_new(),*u=BN_new();
        require(BN_mod_sqr(t,x,prime,ctx));
        require(BN_mul_word(t,3));
        require(BN_lshift1(u,y));
        require(BN_nnmod(u,u,prime,ctx));
        require(BN_mod_inverse(u,u,prime,ctx)!=NULL);
        require(BN_mod_mul(t,t,u,prime,ctx));
        write256(pin_recovery_c,t);
        BN_free(x);BN_free(y);BN_free(t);BN_free(u);
    }

    int fails=0, emitted_total=0, checked=0, zden_total=0;

    for(int cta=0;cta<ctas;cta++){
        bool mode_zden     = (cta%3==1);   /* mixed inactive + zero-den lanes */
        bool mode_dead_all = (cta%11==10); /* all lanes inactive */
        bool mode_tail     = (cta%7==6);   /* tail block: only 73 usable */
        for(int t=0;t<NTH;t++){
            LaneMode m = LANE_LIVE;
            if(mode_dead_all) m=LANE_INACTIVE;
            else if(mode_tail && t>=73) m=LANE_INACTIVE;
            else if(mode_zden && (t==0||t==1||t==63||t==127)) m=LANE_INACTIVE;
            else if(mode_zden && (t==5||t==77)) m=LANE_ZERODEN;
            if(m==LANE_INACTIVE){
                memset(g_in[t].X,0,32); memset(g_in[t].Y,0,32);
                memset(g_in[t].ZZ,0,32); memset(g_in[t].ZZZ,0,32);
                g_in[t].mode=m; continue;
            }
            if(m==LANE_ZERODEN){ fill_lane(t,Rpt); g_in[t].mode=m; zden_total++; continue; }
            g_in[t].mode=m;
            unsigned char seed[40];
            memcpy(seed,"lane",4);
            uint64_t tv=(uint64_t)t, cv=(uint64_t)cta;
            memcpy(seed+4,&tv,8); memcpy(seed+12,&cv,8);
            unsigned char h[64];
            SHA512(seed,20,h);
            BIGNUM *s=BN_bin2bn(h,32,nullptr);
            require(BN_nnmod(s,s,order_n,ctx));
            EC_POINT *P=EC_POINT_new(grp);
            require(EC_POINT_mul(grp,P,s,NULL,NULL,ctx));
            fill_lane(t,P);
            EC_POINT_free(P); BN_free(s);
        }
        g_hit_cnt=0; memset(g_hit_idx,0,sizeof(g_hit_idx));
        memset(g_emitted,0,sizeof(g_emitted));
        memset(g_x1,0,sizeof(g_x1)); memset(g_x2,0,sizeof(g_x2));
        memset(g_pub,0,sizeof(g_pub));
        run_cta();

        for(int t=0;t<NTH;t++){
            LaneMode m=g_in[t].mode;
            bool live = (m==LANE_LIVE);   /* ZERODEN lanes are unusable */
            if(!live){
                if(g_emitted[t][0]||g_emitted[t][1]){
                    std::fprintf(stderr,"cta %d lane %d (mode %d): unusable lane emitted\n",
                                 cta,t,(int)m);
                    fails++;
                }
                continue;
            }
            BIGNUM *ex1=BN_new(),*ex2=BN_new(); int par[2];
            expected(t,ex1,ex2,par);
            BIGNUM *g1=read256(g_x1[t]),*g2=read256(g_x2[t]);
            if(BN_cmp(g1,ex1)||BN_cmp(g2,ex2)){
                std::fprintf(stderr,"cta %d lane %d: x-coord mismatch\n",cta,t);
                fails++;
            }
            for(int ri=0;ri<2;ri++){
                /* A recid-0 hit returns before ri=1 is hashed: the second
                 * pubkey was never materialized.  Same early-return as the
                 * split finish, so only verify what production computed. */
                if(ri==1 && g_emitted[t][0]) continue;
                uint8_t exp_pub[33];
                const BIGNUM *xq = ri?ex2:ex1;
                uint8_t xb[32];
                require(BN_bn2binpad(xq,xb,32)==32);
                exp_pub[0]=0x02u|(uint8_t)par[ri];
                memcpy(exp_pub+1,xb,32);
                if(memcmp(g_pub[t][ri],exp_pub,33)){
                    std::fprintf(stderr,"cta %d lane %d recid %d: pubkey bytes mismatch "
                                 "(got %02x exp %02x)\n",
                                 cta,t,ri,g_pub[t][ri][0],exp_pub[0]);
                    fails++;
                }
            }
            BN_free(ex1);BN_free(ex2);BN_free(g1);BN_free(g2);
            checked++;
        }
        for(uint32_t p=0;p<g_hit_cnt && p<1024;p++){
            uint32_t raw=g_hit_idx[p];
            unsigned lane=raw&0x3FFFFFFFu, ri=(raw>>30)&1u;
            if(lane>=(unsigned)NTH || g_in[lane].mode!=LANE_LIVE ||
               !g_emitted[lane][ri]){
                std::fprintf(stderr,"cta %d: bad record raw=%08x\n",cta,raw);
                fails++;
            } else emitted_total++;
        }
        if(g_hit_cnt>1024){ std::fprintf(stderr,"cta %d: hit cap overflow\n",cta); fails++; }
    }
    std::printf("checked %d usable lanes across %d CTAs, %d hit records, %d zero-den lanes\n",
                checked,ctas,emitted_total,zden_total);
    if(fails){ std::fprintf(stderr,"FAIL: %d divergence(s)\n",fails); return 1; }
    std::printf("PASS: fused finish oracle clean\n");
    return 0;
}
'''
    cpp = "\n".join([CPP_PREFIX, body, CPP_TAIL, driver])
    work = Path(tempfile.mkdtemp(prefix="qsb_fused_finish_"))
    srcf = work/"check_fused_finish.cpp"; srcf.write_text(cpp, encoding="utf-8")
    exe  = work/"check_fused_finish.exe"
    inc = os.environ.get("OPENSSL_INCLUDE", r"C:/Strawberry/c/include")
    lib = os.environ.get("OPENSSL_LIB", r"C:/Strawberry/c/lib")
    cxx = shlex.split(os.environ.get("CXX", "g++"))
    cmd = cxx + ["-O2","-std=c++20","-I",inc,str(srcf),"-o",str(exe),
                 "-L",lib,"-lcrypto","-lpthread"]
    r = subprocess.run(cmd,capture_output=True,text=True)
    if r.returncode:
        sys.stderr.write(r.stderr[-8000:]); print("BUILD FAIL"); sys.exit(1)
    r = subprocess.run([str(exe),"24"],capture_output=True,text=True)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
    print("fused-finish oracle:", "PASS" if r.returncode==0 else "FAIL")
    sys.exit(0 if r.returncode==0 else 1)

if __name__ == "__main__":
    main()
