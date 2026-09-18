#!/usr/bin/env python3
"""Build a host reference .so out of pin-merge's SHIPPED device source text.

Nothing below is retyped: every routine is sliced out of
/root/qsb/exp/pin-merge/{GPUMath.h,pinning.cu} by brace matching or by an
explicit line range, and the wrappers only call the sliced code.  The host
prelude (limb carry/borrow adapters for the UADDO/USUBO PTX macros) is the one
from xlib's candidates/pinning/audit_fused_point_source.py @ 8afe4bd.
"""
import re, sys
from pathlib import Path

ROOT   = Path("/root/qsb/exp/pin-merge")
GPUMATH = ROOT / "GPUMath.h"
PINNING = ROOT / "pinning.cu"

HOST_PRELUDE = r"""
#include <cstdint>
#include <cstring>
#include <cstdio>
#define __device__
#define __forceinline__ inline
#define __constant__
static uint64_t cc;
static uint64_t addw(uint64_t a,uint64_t b,bool use,bool set){__uint128_t t=(__uint128_t)a+b+(use?cc:0);if(set)cc=(uint64_t)(t>>64);return (uint64_t)t;}
static uint64_t subw(uint64_t a,uint64_t b,bool use,bool set){__uint128_t t=(__uint128_t)a-b-(use?cc:0);if(set)cc=(uint64_t)(t>>127);return (uint64_t)t;}
#define UADDO(c,a,b) c=addw(a,b,false,true)
#define UADDC(c,a,b) c=addw(a,b,true,true)
#define UADD(c,a,b) c=addw(a,b,true,false)
#define UADDO1(c,a) c=addw(c,a,false,true)
#define UADDC1(c,a) c=addw(c,a,true,true)
#define UADD1(c,a) c=addw(c,a,true,false)
#define USUBO(c,a,b) c=subw(a,b,false,true)
#define USUBC(c,a,b) c=subw(a,b,true,true)
#define USUB(c,a,b) c=subw(a,b,true,false)
#define USUBO1(c,a) c=subw(c,a,false,true)
#define USUBC1(c,a) c=subw(c,a,true,true)
#define USUB1(c,a) c=subw(c,a,true,false)
static void Load256(uint64_t *r,const uint64_t *a){memmove(r,a,32);}
#ifndef QSB_LAZY
#define QSB_LAZY 1
#endif
#ifndef QSB_SQFREE
#define QSB_SQFREE 1
#endif
#ifndef QSB_COFACTOR
#define QSB_COFACTOR 1
#endif
uint64_t pin_u2rx_words[4];
#ifndef GT_CHUNKS
#define GT_CHUNKS 15
#endif
"""

# Wrappers: every device statement they execute comes from the slices above.
WRAPPERS = r"""
extern "C" {

/* The live fixed-base chain shape of _FixedBaseSignedXYZZScalar under the
 * shipped defaults (QSB_FINAL_TEMPLATE=1): one deferred-Y mmadd seed on the
 * first two points, GT_CHUNKS-3 deferred _PointAddXYZZT<true>, and one
 * resolving _PointAddXYZZT<false>.  Table indexing is omitted: it is identical
 * in both fused and unfused arms and produces the affine points fed in here. */
void h_chain(const uint64_t *pts, uint64_t *out){
  uint64_t X[4],Y[4],ZZ[4],ZZZ[4],y0[4],cx[4],cy[4];
  _PointAddXYZZ_mm(X,Y,ZZ,ZZZ, pts+0,pts+4, pts+8,pts+12);
  Load256(y0, pts+4);   /* _FixedBaseSignedXYZZScalar anchors the first madd on point 0's affine y */
  for(int c=2;c<GT_CHUNKS-1;c++){
    Load256(cx, pts+8*c); Load256(cy, pts+8*c+4);
    _PointAddXYZZT<true>(X,Y,ZZ,ZZZ, cx,cy, y0);
    Load256(y0, cy);
  }
  Load256(cx, pts+8*(GT_CHUNKS-1)); Load256(cy, pts+8*(GT_CHUNKS-1)+4);
  _PointAddXYZZT<false>(X,Y,ZZ,ZZZ, cx,cy, y0);
  Load256(out+0,X); Load256(out+4,Y); Load256(out+8,ZZ); Load256(out+12,ZZZ);
}

/* prepare's square-free denominator, pinning.cu:1977-1983 */
void h_set_xR(const uint64_t *xR){ for(int i=0;i<4;i++) pin_u2rx_words[i]=xR[i]; }
void h_denom(uint64_t *qx,uint64_t *qzz,uint64_t *qzzz,uint64_t *prod){
  QSB_SLICE_DENOM
}
/* prepare's packed checkpoint, pinning.cu:1995-1999 */
void h_pack(uint64_t *qzz,uint64_t *qzzz,uint64_t *qy,uint64_t *prod){
  QSB_SLICE_PACK
}
/* finish's t,v recovery from tbar,vbar and 1/T, pinning.cu:2103-2105 */
void h_unpack(uint64_t *qzzz,uint64_t *qy,uint64_t *prod){
  QSB_SLICE_UNPACK
}
uint32_t h_tail(uint64_t *t,uint64_t *v,uint64_t *xR,uint64_t *yR,uint64_t *c,
                uint64_t *x1,uint64_t *x2){
  return qsb_xyzz_finish_symmetric_tv(t,v,xR,yR,c,x1,x2);
}
void h_mul(uint64_t *r,uint64_t *a,uint64_t *b){ _ModMult(r,a,b); }
/* unit-level reducer pairs, spelled exactly as _PointAddXYZZ spells them */
void h_mulsub_unfused(uint64_t*r,uint64_t*a,uint64_t*b,uint64_t*c){
  uint64_t S2[4]={a[0],a[1],a[2],a[3]}; _ModMult(S2,b); _ModSub256(r,S2,c); }
void h_sqraddsub2_unfused(uint64_t*r,uint64_t*a,uint64_t*b,uint64_t*c){
  uint64_t T[4]; _ModSqr(T,a); _ModX3Fused(r,T,b,c); }
#if QSB_FUSE_MULSUB
void h_mulsub(uint64_t*r,const uint64_t*a,const uint64_t*b,const uint64_t*c){_ModMulSubCore(r,a,b,c);}
#else
void h_mulsub(uint64_t*r,const uint64_t*a,const uint64_t*b,const uint64_t*c){
  h_mulsub_unfused(r,(uint64_t*)a,(uint64_t*)b,(uint64_t*)c);}
#endif
#if QSB_FUSE_SQRADDSUB2
void h_sqraddsub2(uint64_t*r,const uint64_t*a,const uint64_t*b,const uint64_t*c){_ModSqrAddSub2(r,a,b,c);}
#else
void h_sqraddsub2(uint64_t*r,const uint64_t*a,const uint64_t*b,const uint64_t*c){
  h_sqraddsub2_unfused(r,(uint64_t*)a,(uint64_t*)b,(uint64_t*)c);}
#endif
void h_norm(uint64_t *r){ qsb_field_normalize(r); }
int  h_fuse_mulsub(void){ return QSB_FUSE_MULSUB; }
int  h_fuse_sqraddsub2(void){ return QSB_FUSE_SQRADDSUB2; }
}
"""

def slice_braces(src, decl_rx, occurrence=0, comment_lines=0):
    hits = list(re.finditer(decl_rx, src))
    assert len(hits) > occurrence, (decl_rx, len(hits))
    m = hits[occurrence]
    start = src.index("{", m.end() - 1)
    depth, i = 1, start + 1
    while depth:
        depth += (src[i] == "{") - (src[i] == "}")
        i += 1
    a = src[:m.start()].count("\n") + 1 - comment_lines
    b = src[:i].count("\n") + 1
    return "\n".join(src.split("\n")[a-1:b]), a, b

def slice_lines(src, a, b):
    return "\n".join(src.split("\n")[a-1:b]), a, b

def main(outpath):
    gm = GPUMATH.read_text()
    pn = PINNING.read_text()
    pieces, manifest = [], []

    def take(tag, text, a, b, where):
        pieces.append(text); manifest.append((tag, where, a, b))

    # ---- GPUMath.h -------------------------------------------------------
    take("_IsPositive/_IsNegative", *slice_lines(gm, 84, 85), where="GPUMath.h")
    take("SubP macro",   *slice_lines(gm, 116, 121), where="GPUMath.h")
    take("_ModNeg256(2)", *slice_braces(gm, r"__device__ void _ModNeg256\(uint64_t \*r, uint64_t \*a\)"), where="GPUMath.h")
    take("_ModNeg256(1)", *slice_braces(gm, r"__device__ void _ModNeg256\(uint64_t \*r\)"), where="GPUMath.h")
    take("_ModAdd256",           *slice_braces(gm, r"__device__ void _ModAdd256\("), where="GPUMath.h")
    # whole #if QSB_LAZY .. #endif: _ModSub256 x2, _ModAddLazy, _ModX3Fused, and
    # the non-lazy _ModSub256 x2 under #else
    lz0 = gm.split("\n").index("#if QSB_LAZY") + 1
    lz1 = lz0
    depth = 0
    for i, ln in enumerate(gm.split("\n")[lz0-1:], start=lz0):
        if ln.startswith("#if"): depth += 1
        elif ln.startswith("#endif"):
            depth -= 1
            if depth == 0: lz1 = i; break
    take("#if QSB_LAZY block", *slice_lines(gm, lz0, lz1), where="GPUMath.h")
    take("_ModMultCore",   *slice_braces(gm, r"__device__ __forceinline__ void _ModMultCore\("), where="GPUMath.h")
    take("QSB_FUSE_MULSUB block", *slice_lines(gm, 933, 1108), where="GPUMath.h")
    take("_ModMult(3)",    *slice_braces(gm, r"__device__ void _ModMult\(uint64_t \*r, uint64_t \*a, uint64_t \*b\)"), where="GPUMath.h")
    take("_ModMult(2)",    *slice_braces(gm, r"__device__ void _ModMult\(uint64_t \*r, uint64_t \*a\)"), where="GPUMath.h")
    take("_ModSqr",        *slice_braces(gm, r"__device__ __forceinline__ void _ModSqr\("), where="GPUMath.h")
    take("QSB_FUSE_SQRADDSUB2 block", *slice_lines(gm, 1350, 1644), where="GPUMath.h")
    take("_PointAddXYZZT decl", *slice_lines(gm, 1747, 1751), where="GPUMath.h")
    take("_PointAddXYZZ",  *slice_braces(gm, r"__device__ void _PointAddXYZZ\(uint64_t \*X1"), where="GPUMath.h")
    take("_PointAddXYZZT", *slice_braces(gm, r"__device__ __forceinline__ void _PointAddXYZZT\(", 1, comment_lines=1), where="GPUMath.h")
    take("_PointAddXYZZ_mm", *slice_braces(gm, r"__device__ void _PointAddXYZZ_mm\("), where="GPUMath.h")
    # ---- pinning.cu ------------------------------------------------------
    take("qsb_field_normalize", *slice_braces(pn, r"__device__ __forceinline__ void qsb_field_normalize\("), where="pinning.cu")
    take("qsb_xyzz_finish_symmetric_tv",
         *slice_braces(pn, r"__device__ __forceinline__ uint32_t qsb_xyzz_finish_symmetric_tv\("), where="pinning.cu")

    denom,  da, db = slice_lines(pn, 1976, 1983)
    pack,   pa, pb = slice_lines(pn, 1994, 1999)
    unpack, ua, ub = slice_lines(pn, 2103, 2105)
    manifest += [("denominator stmts", "pinning.cu", da, db),
                 ("pack stmts",        "pinning.cu", pa, pb),
                 ("unpack stmts",      "pinning.cu", ua, ub)]

    body = "\n".join(pieces)
    # the extracted text must be the shipped device source, not something we wrote
    assert "_ModMulSubCore(R, S2, ZZZ1, Y1);" in body
    assert "_ModSqrAddSub2(T, R, PPP, Q);" in body
    assert "#if QSB_FUSE_MULSUB" in body and "#if QSB_FUSE_SQRADDSUB2" in body
    assert "_ModX3Fused" in body and "qsb_field_normalize" in body
    assert "_ModMult(prod,prep_xR,qzz)" in denom and "_ModMult(prod,qzzz)" in denom
    assert "_ModMult(hc,qzz,prod)" in pack
    assert "_ModMult(qzzz,prod)" in unpack

    w = (WRAPPERS.replace("QSB_SLICE_DENOM",  denom)
                 .replace("QSB_SLICE_PACK",   pack)
                 .replace("QSB_SLICE_UNPACK", unpack))
    Path(outpath).write_text(HOST_PRELUDE + body + w)
    for tag, where, a, b in manifest:
        print(f"  {tag:34s} {where}:{a}-{b}")
    print(f"wrote {outpath} ({len(Path(outpath).read_text().splitlines())} lines)")

main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/pin_fused_host.cpp")
