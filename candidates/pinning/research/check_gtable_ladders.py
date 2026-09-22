#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_gtable_ladders.py — bitwise oracle for the chunked gtable ladder build.

v8 parallelized gt_build_ladders: each of the 15 chunks now runs in its own
std::thread with private OpenSSL objects, deriving base_ch directly as
(2^gt_shift(ch) * bscal0) * G instead of the original serial
base *= 2^shift chain.  This oracle extracts the CURRENT implementations of
gt_batch_ladder / gt_ladder_chunk / gt_build_ladders verbatim from
pinning.cu, appends the ORIGINAL serial build as a reference, and compares
both outputs byte-for-byte on a fixed (deterministic) neg_r_inv.

Run: python check_gtable_ladders.py   (needs g++ + OpenSSL headers/libs)
"""
import os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CU = os.path.join(HERE, "..", "pinning.cu")


def extract(src, name):
    """Return the full text of `static ... name(...){ ... }` from src."""
    m = re.search(r"static [^\n]*\b" + re.escape(name) + r"\s*\(", src)
    if not m:
        sys.exit(f"FAIL: {name} not found in pinning.cu")
    start = src.rfind("\n", 0, m.start()) + 1
    i = src.index("{", m.end() - 1)
    depth = 0
    j = i
    while True:
        c = src[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[start : j + 1]
        j += 1


def main():
    src = open(CU, "r", encoding="utf-8").read()
    body = "\n\n".join(
        extract(src, n)
        for n in ("gt_point_to_limbs", "gt_batch_ladder",
                  "gt_ladder_chunk", "gt_build_ladders")
    )

    test = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <system_error>
#include <thread>
extern "C" {
#include <openssl/sha.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
}

#define GT_CHUNKS 15
#define GT_LO 256
#define GT_HI 1024
static int gt_shift(int c) { return c == 0 ? 0 : 17*c+1; }

__BODY__

/* --- ORIGINAL serial reference (pre-v8 code, verbatim logic) --- */
static void gt_build_ladders_ref(uint64_t *hL, uint64_t *hH, const uint8_t neg_r_inv[32]) {
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *x = BN_new(), *y = BN_new(), *shift = BN_new(), *inv2 = BN_new(),
           *order = BN_new(), *nri = BN_new(), *bscal = BN_new();
    EC_POINT *base = EC_POINT_new(grp), *step = EC_POINT_new(grp);
    EC_GROUP_get_order(grp, order, ctx);
    BN_set_word(shift, 2); BN_mod_inverse(inv2, shift, order, ctx);
    BN_lebin2bn(neg_r_inv, 32, nri);
    BN_mod_mul(bscal, inv2, nri, order, ctx);
    EC_POINT_mul(grp, base, bscal, NULL, NULL, ctx);
    memset(hL, 0, (size_t)GT_CHUNKS * GT_LO * 8 * sizeof(uint64_t));
    memset(hH, 0, (size_t)GT_CHUNKS * GT_HI * 8 * sizeof(uint64_t));
    for (int ch = 0; ch < GT_CHUNKS; ch++) {
        if (ch > 0) { BN_set_word(shift, ch==1 ? (1u<<18) : (1u<<17)); EC_POINT_mul(grp, base, NULL, base, shift, ctx); }
        gt_batch_ladder(grp,base,GT_LO-1,
            hL+(size_t)ch*GT_LO*8,x,y,ctx);
        BN_set_word(shift, 256);
        EC_POINT_mul(grp, step, NULL, base, shift, ctx);
        gt_batch_ladder(grp,step,(ch==0?1024:512)-1,
            hH+(size_t)ch*GT_HI*8,x,y,ctx);
    }
    BN_free(x); BN_free(y); BN_free(shift); BN_free(inv2); BN_free(order);
    BN_free(nri); BN_free(bscal);
    EC_POINT_free(base); EC_POINT_free(step);
    EC_GROUP_free(grp); BN_CTX_free(ctx);
}

int main() {
    /* Fixed deterministic neg_r_inv (any 32 bytes exercise the math). */
    uint8_t nri[32];
    for (int i = 0; i < 32; i++) nri[i] = (uint8_t)(0x9Eu + 37*i);

    size_t lb = (size_t)GT_CHUNKS*GT_LO*8*sizeof(uint64_t);
    size_t hb = (size_t)GT_CHUNKS*GT_HI*8*sizeof(uint64_t);
    uint64_t *La=(uint64_t*)malloc(lb), *Ha=(uint64_t*)malloc(hb);
    uint64_t *Lb=(uint64_t*)malloc(lb), *Hb=(uint64_t*)malloc(hb);
    if(!La||!Ha||!Lb||!Hb){fprintf(stderr,"oom\n");return 2;}

    gt_build_ladders(La, Ha, nri);      /* current (possibly threaded) impl  */
    gt_build_ladders_ref(Lb, Hb, nri);  /* serial reference                  */

    size_t bad = 0;
    for (size_t i = 0; i < lb/sizeof(uint64_t); i++) if (La[i]!=Lb[i]) bad++;
    for (size_t i = 0; i < hb/sizeof(uint64_t); i++) if (Ha[i]!=Hb[i]) bad++;
    if (bad) { printf("FAIL: %zu limb words differ\n", bad); return 1; }
    printf("gtable ladder oracle: PASS (%zu + %zu words identical)\n",
           lb/8, hb/8);
    return 0;
}
'''.replace("__BODY__", body)

    with tempfile.TemporaryDirectory() as td:
        cpp = os.path.join(td, "t.cpp")
        exe = os.path.join(td, "t")
        open(cpp, "w", encoding="utf-8").write(test)
        for cxx in ("g++", "cl"):
            pass
        cmd = ["g++", "-O2", "-o", exe, cpp, "-lcrypto"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("compile failed:\n" + r.stderr[-4000:])
        r = subprocess.run([exe], capture_output=True, text=True)
        sys.stdout.write(r.stdout + r.stderr)
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
