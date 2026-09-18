/* End-to-end pinning correctness gate for QSB_FUSE_MULSUB / QSB_FUSE_SQRADDSUB2.
 *
 * Runs the SHIPPED device source text (sliced into libph<mm><ss>.so by
 * audit/extract.py) over whole 128-lane candidate blocks:
 *
 *   15-point deferred XYZZ chain  ->  D = V*(xR*ZZ - X)  ->  block cofactor
 *   collective (C_i/T = 1/D_i)    ->  tbar,vbar across the kernel boundary
 *   ->  t = tbar/T, v = vbar/T    ->  qsb_xyzz_finish_symmetric_tv
 *
 * and compares both x coordinates and both y parities against an independent
 * secp256k1 reference built from OpenSSL EC_POINT_add on exact integers.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <dlfcn.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>

#define N_LANES 128
#define N_PTS   15

typedef void (*fn_chain)(const uint64_t*, uint64_t*);
typedef void (*fn_setxr)(const uint64_t*);
typedef void (*fn_denom)(uint64_t*, uint64_t*, uint64_t*, uint64_t*);
typedef void (*fn_pack )(uint64_t*, uint64_t*, uint64_t*, uint64_t*);
typedef void (*fn_unpk )(uint64_t*, uint64_t*, uint64_t*);
typedef uint32_t (*fn_tail)(uint64_t*,uint64_t*,uint64_t*,uint64_t*,uint64_t*,uint64_t*,uint64_t*);
typedef void (*fn_mul)(uint64_t*, uint64_t*, uint64_t*);
typedef void (*fn_norm)(uint64_t*);
typedef int  (*fn_flag)(void);

struct Arm {
    void *h; const char *name;
    fn_chain chain; fn_setxr setxr; fn_denom denom; fn_pack pack;
    fn_unpk unpk; fn_tail tail; fn_mul mul; fn_norm norm;
    int fm, fs;
};

static void load(Arm &a, const char *path, const char *name) {
    a.h = dlopen(path, RTLD_NOW);
    if (!a.h) { fprintf(stderr, "dlopen %s: %s\n", path, dlerror()); exit(2); }
    a.name = name;
    a.chain=(fn_chain)dlsym(a.h,"h_chain");  a.setxr=(fn_setxr)dlsym(a.h,"h_set_xR");
    a.denom=(fn_denom)dlsym(a.h,"h_denom");  a.pack =(fn_pack )dlsym(a.h,"h_pack");
    a.unpk =(fn_unpk )dlsym(a.h,"h_unpack"); a.tail =(fn_tail )dlsym(a.h,"h_tail");
    a.mul  =(fn_mul  )dlsym(a.h,"h_mul");    a.norm =(fn_norm )dlsym(a.h,"h_norm");
    a.fm = ((fn_flag)dlsym(a.h,"h_fuse_mulsub"))();
    a.fs = ((fn_flag)dlsym(a.h,"h_fuse_sqraddsub2"))();
    if (!a.chain||!a.denom||!a.pack||!a.unpk||!a.tail) { fprintf(stderr,"dlsym\n"); exit(2); }
}

static void bn2l(const BIGNUM *b, uint64_t o[4]) {
    unsigned char buf[32]; BN_bn2binpad(b, buf, 32);
    for (int i=0;i<4;i++){ uint64_t v=0; for(int j=0;j<8;j++) v=(v<<8)|buf[31-(i*8+7-j)]; o[i]=v; }
}
static void l2bn(const uint64_t in[4], BIGNUM *b) {
    unsigned char buf[32];
    for (int i=0;i<4;i++) for(int j=0;j<8;j++) buf[31-(i*8+j)]=(unsigned char)(in[i]>>(8*j));
    BN_bin2bn(buf,32,b);
}
static bool iszero(const uint64_t v[4]) { return (v[0]|v[1]|v[2]|v[3])==0; }

int main(int argc, char **argv) {
    long blocks   = argc > 1 ? atol(argv[1]) : 400;         /* x128 candidates */
    const char *pA = argc > 2 ? argv[2] : "./libph11.so";
    const char *pB = argc > 3 ? argv[3] : "./libph00.so";
    unsigned seed  = argc > 4 ? (unsigned)atoi(argv[4]) : 20260918u;

    Arm A, B; load(A, pA, pA); load(B, pB, pB);
    printf("arm A=%s (MULSUB=%d SQRADDSUB2=%d)   arm B=%s (MULSUB=%d SQRADDSUB2=%d)\n",
           A.name, A.fm, A.fs, B.name, B.fm, B.fs);

    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *P = BN_new(), *ord = BN_new();
    EC_GROUP_get_curve(grp, P, NULL, NULL, ctx);
    EC_GROUP_get_order(grp, ord, ctx);

    /* R = a random fixed problem point; c = 3*xR^2/(2*yR) exactly as the host
     * uploads it into pin_u2rc_words. */
    BIGNUM *k = BN_new(), *xR = BN_new(), *yR = BN_new(), *cbn = BN_new();
    BIGNUM *t1 = BN_new(), *t2 = BN_new(), *inv = BN_new();
    EC_POINT *R = EC_POINT_new(grp), *Rneg = EC_POINT_new(grp);
    BN_set_word(k, 0xC0FFEEu); EC_POINT_mul(grp, R, k, NULL, NULL, ctx);
    EC_POINT_get_affine_coordinates(grp, R, xR, yR, ctx);
    EC_POINT_copy(Rneg, R); EC_POINT_invert(grp, Rneg, ctx);
    BN_mod_sqr(t1, xR, P, ctx); BN_mul_word(t1, 3); BN_mod(t1, t1, P, ctx);
    BN_mod_lshift1(t2, yR, P, ctx); BN_mod_inverse(inv, t2, P, ctx);
    BN_mod_mul(cbn, t1, inv, P, ctx);
    uint64_t lxR[4], lyR[4], lc[4];
    bn2l(xR,lxR); bn2l(yR,lyR); bn2l(cbn,lc);
    A.setxr(lxR); B.setxr(lxR);

    EC_POINT *pt = EC_POINT_new(grp), *sum = EC_POINT_new(grp), *acc = EC_POINT_new(grp);
    BIGNUM *bx = BN_new(), *by = BN_new(), *tt = BN_new();

    long cand=0, cmp_ok=0, skipped=0;
    long mismatchA=0, mismatchB=0, ab_diff=0, repr_diff=0, usable_diff=0;
    long noncanon_zero=0, deg_d0=0, deg_d0_ok=0, ylane_zero=0;
    long partial_lanes=0;

    srandom(seed);
    for (long blk=0; blk<blocks; blk++) {   /* blocks=0 -> unit gate only */
        /* every 17th block is a partial final batch */
        int live = (blk % 17 == 16) ? (1 + (int)(random() % (N_LANES-1))) : N_LANES;
        if (live != N_LANES) partial_lanes += N_LANES - live;
        /* every 23rd block plants a d == 0 candidate (P == R) in lane 3 */
        int plant = (blk % 23 == 22) ? 3 : -1;

        static uint64_t pts[N_LANES][N_PTS*8];
        static uint64_t outA[N_LANES][16], outB[N_LANES][16];
        static uint64_t dA[N_LANES][5], dB[N_LANES][5];
        static uint64_t tbA[N_LANES][4], vbA[N_LANES][4], tbB[N_LANES][4], vbB[N_LANES][4];
        static int okA[N_LANES], okB[N_LANES];
        static BIGNUM *refx1[N_LANES], *refx2[N_LANES];
        static int refpar[N_LANES], refvalid[N_LANES], first=1;
        if (first) { for (int i=0;i<N_LANES;i++){refx1[i]=BN_new();refx2[i]=BN_new();} first=0; }

        for (int L=0; L<live; L++) {
            EC_POINT_set_to_infinity(grp, sum);
            for (int i=0;i<N_PTS;i++) {
                if (plant==L && i==N_PTS-1) {
                    /* last point = R - (sum so far), so the chain lands on R */
                    EC_POINT_copy(acc, sum); EC_POINT_invert(grp, acc, ctx);
                    EC_POINT_add(grp, pt, R, acc, ctx);
                } else {
                    BN_rand_range(k, ord); if (BN_is_zero(k)) BN_one(k);
                    EC_POINT_mul(grp, pt, k, NULL, NULL, ctx);
                }
                if (EC_POINT_is_at_infinity(grp, pt)) { BN_one(k); EC_POINT_mul(grp,pt,k,NULL,NULL,ctx); }
                EC_POINT_get_affine_coordinates(grp, pt, bx, by, ctx);
                bn2l(bx, pts[L]+8*i); bn2l(by, pts[L]+8*i+4);
                EC_POINT_add(grp, sum, sum, pt, ctx);
            }
            refvalid[L] = !EC_POINT_is_at_infinity(grp, sum);
            if (refvalid[L]) {
                /* independent reference: x(P+R), x(P-R) and both y parities */
                EC_POINT_add(grp, acc, sum, R, ctx);
                if (EC_POINT_is_at_infinity(grp, acc)) refvalid[L]=0;
                else { EC_POINT_get_affine_coordinates(grp, acc, refx1[L], tt, ctx);
                       refpar[L] = BN_is_odd(tt) ? 1 : 0; }
            }
            if (refvalid[L]) {
                EC_POINT_add(grp, acc, sum, Rneg, ctx);
                if (EC_POINT_is_at_infinity(grp, acc)) refvalid[L]=0;
                else { EC_POINT_get_affine_coordinates(grp, acc, refx2[L], tt, ctx);
                       refpar[L] |= (BN_is_odd(tt) ? 1 : 0) << 1; }
            }
            A.chain(pts[L], outA[L]);
            B.chain(pts[L], outB[L]);
            if (memcmp(outA[L], outB[L], sizeof outA[L]) != 0) repr_diff++;
        }

        /* ---- per-arm block pipeline ------------------------------------- */
        for (int arm=0; arm<2; arm++) {
            Arm &Z = arm ? B : A;
            uint64_t (*out)[16] = arm ? outB : outA;
            uint64_t (*dd)[5]   = arm ? dB   : dA;
            uint64_t (*tb)[4]   = arm ? tbB  : tbA;
            uint64_t (*vb)[4]   = arm ? vbB  : vbA;
            int *ok             = arm ? okB  : okA;
            static uint64_t a_i[N_LANES][4];
            for (int L=0; L<N_LANES; L++) {
                if (L < live) {
                    uint64_t X[4],ZZ[4],ZZZ[4];
                    memcpy(X,out[L]+0,32); memcpy(ZZ,out[L]+8,32); memcpy(ZZZ,out[L]+12,32);
                    Z.denom(X, ZZ, ZZZ, dd[L]);
                } else { memset(dd[L],0,40); }
                ok[L] = (L < live) && !iszero(dd[L]);
                if (ok[L]) { /* shipped test is on words; flag a non-canonical zero */
                    l2bn(dd[L], tt);
                    BN_mod(tt, tt, P, ctx);
                    if (BN_is_zero(tt)) noncanon_zero++;
                }
                if (ok[L]) memcpy(a_i[L], dd[L], 32);
                else { a_i[L][0]=1; a_i[L][1]=a_i[L][2]=a_i[L][3]=0; }
            }
            /* block collective: T = prod a_i, C_i = prod_{j!=i} a_j */
            uint64_t T[4]={1,0,0,0};
            for (int L=0;L<N_LANES;L++) Z.mul(T,T,a_i[L]);
            static uint64_t C[N_LANES][4];
            for (int L=0;L<N_LANES;L++) {
                uint64_t c0[4]={1,0,0,0};
                for (int j=0;j<N_LANES;j++) if (j!=L) Z.mul(c0,c0,a_i[j]);
                memcpy(C[L],c0,32);
            }
            /* 1/T: exact integer inverse, standing in for _ModInv + root pipeline */
            uint64_t Tinv[5];
            l2bn(T, tt); BN_mod(tt, tt, P, ctx);
            if (BN_is_zero(tt)) { fprintf(stderr,"block root is zero\n"); return 3; }
            BN_mod_inverse(inv, tt, P, ctx); bn2l(inv, Tinv); Tinv[4]=0;
            for (int L=0;L<N_LANES;L++) {
                uint64_t ZZ[4],ZZZ[4],Y[4],Cc[5];
                if (L<live) { memcpy(ZZ,out[L]+8,32); memcpy(ZZZ,out[L]+12,32); memcpy(Y,out[L]+4,32); }
                else { memset(ZZ,0,32); memset(ZZZ,0,32); memset(Y,0,32); }
                memcpy(Cc,C[L],32); Cc[4]=0;
                Z.pack(ZZ, ZZZ, Y, Cc);          /* ZZZ->tbar, Y->vbar */
                if (!ok[L]) { memset(ZZZ,0,32); memset(Y,0,32); }
                memcpy(tb[L],ZZZ,32); memcpy(vb[L],Y,32);
                if (ok[L] && iszero(Y)) ylane_zero++;
                if (ok[L]) {
                    uint64_t tv[5], vv[5], pr[5];
                    memcpy(tv,ZZZ,32); tv[4]=0; memcpy(vv,Y,32); vv[4]=0;
                    memcpy(pr,Tinv,40);
                    Z.unpk(tv, vv, pr);          /* t = tbar/T, v = vbar/T */
                    memcpy(tb[L],tv,32); memcpy(vb[L],vv,32);
                }
            }
        }

        /* ---- compare ----------------------------------------------------- */
        for (int L=0; L<live; L++) {
            if (okA[L] != okB[L]) usable_diff++;
            if (plant==L) {
                deg_d0++;
                if (!okA[L] && !okB[L]) deg_d0_ok++;
                continue;                        /* d==0 is correctly unusable */
            }
            if (!okA[L] || !okB[L]) { skipped++; continue; }
            if (!refvalid[L]) { skipped++; continue; }
            uint64_t x1A[4],x2A[4],x1B[4],x2B[4],tA[5],vA[5],tB[5],vB[5];
            memcpy(tA,tbA[L],32); tA[4]=0; memcpy(vA,vbA[L],32); vA[4]=0;
            memcpy(tB,tbB[L],32); tB[4]=0; memcpy(vB,vbB[L],32); vB[4]=0;
            uint32_t parA = A.tail(tA,vA,lxR,lyR,lc,x1A,x2A);
            uint32_t parB = B.tail(tB,vB,lxR,lyR,lc,x1B,x2B);
            cand++;
            l2bn(x1A,bx); l2bn(x2A,by);
            int badA = (BN_cmp(bx,refx1[L])!=0) || (BN_cmp(by,refx2[L])!=0)
                     || ((int)parA != refpar[L]);
            l2bn(x1B,bx); l2bn(x2B,by);
            int badB = (BN_cmp(bx,refx1[L])!=0) || (BN_cmp(by,refx2[L])!=0)
                     || ((int)parB != refpar[L]);
            if (badA) mismatchA++;
            if (badB) mismatchB++;
            if (memcmp(x1A,x1B,32)||memcmp(x2A,x2B,32)||parA!=parB) ab_diff++;
            if (!badA && !badB) cmp_ok++;
        }
    }

    /* ---- injected degenerates ------------------------------------------- */
    long inj_ok = 0, inj_tot = 0, noncanon_emit = 0;
    {
        uint64_t zero[4]={0,0,0,0}, one[4]={1,0,0,0};
        for (int arm=0; arm<2; arm++) {
            Arm &Z = arm ? B : A;
            uint64_t X[4],ZZ[4],ZZZ[4],d[5];
            /* XYZZ point at infinity: ZZ = ZZZ = 0 */
            memcpy(X,one,32); memcpy(ZZ,zero,32); memcpy(ZZZ,zero,32);
            Z.denom(X,ZZ,ZZZ,d); inj_tot++; if (iszero(d)) inj_ok++;
            /* V == 0 alone */
            memcpy(X,one,32); memcpy(ZZ,one,32); memcpy(ZZZ,zero,32);
            Z.denom(X,ZZ,ZZZ,d); inj_tot++; if (iszero(d)) inj_ok++;
            /* d == 0 exactly: X = xR, ZZ = 1 */
            memcpy(ZZ,one,32); memcpy(ZZZ,one,32); memcpy(X,lxR,32);
            Z.denom(X,ZZ,ZZZ,d); inj_tot++; if (iszero(d)) inj_ok++;
            /* d == 0 through the ONLY non-canonical representative this modulus
             * admits: 2^256 < 2p, so a class has a second representative below
             * 2^256 only if it is below 2^256-p = 2^32+977.  Drive the
             * subtraction to land on exactly p and check what D comes out as. */
            { BIGNUM *s1=BN_new(), *s2=BN_new();
              uint64_t xr1[5]; memcpy(xr1,lxR,32); xr1[4]=0;
              uint64_t prodx[4]; { uint64_t o[4]; Z.mul(o,(uint64_t*)lxR,one); memcpy(prodx,o,32); }
              l2bn(prodx,s1); BN_sub(s2,s1,P);
              if (BN_is_negative(s2)) BN_add(s2,s2,P);   /* fall back to the canonical one */
              bn2l(s2,X); BN_free(s1); BN_free(s2); (void)xr1; }
            memcpy(ZZ,one,32); memcpy(ZZZ,one,32);
            Z.denom(X,ZZ,ZZZ,d); inj_tot++; if (iszero(d)) inj_ok++;
            /* Y == 0 must stay usable: tbar nonzero, vbar zero */
            uint64_t zz[4],vv[4],yy[4],c0[5]={1,0,0,0};
            memcpy(zz,one,32); memcpy(vv,one,32); memcpy(yy,zero,32);
            Z.pack(zz,vv,yy,c0); inj_tot++; if (!iszero(vv) && iszero(yy)) inj_ok++;
        }
    }

    /* ---- unit gate on the two reducers ---------------------------------- */
    long unit=0, unitbadA=0, unitbadB=0, unit_repr=0;
    long unitR=0, unitRbadA=0, unitRbadB=0, unitR_repr=0, noncanonR=0;
    long msBadA=0,msBadB=0,sqBadA=0,sqBadB=0;
    {
        BIGNUM *ba=BN_new(),*bb=BN_new(),*bc=BN_new(),*br=BN_new(),*be=BN_new();
        uint64_t a[4],b[4],c[4],rA[4],rB[4];
        const uint64_t edge[] = {0ULL,1ULL,2ULL,0xFFFFFFFEFFFFFC2FULL,
                                 0xFFFFFFFFFFFFFFFFULL,0x1000003D1ULL};
        for (long i=0;i<200000;i++) {
            for (int j=0;j<4;j++) {
                a[j]=((uint64_t)random()<<40)^((uint64_t)random()<<20)^random();
                b[j]=((uint64_t)random()<<40)^((uint64_t)random()<<20)^random();
                c[j]=((uint64_t)random()<<40)^((uint64_t)random()<<20)^random();
            }
            if (i < 3000) {            /* dense near-modulus / edge coverage */
                BN_copy(ba,P); BN_sub_word(ba,(BN_ULONG)(i%4096)); bn2l(ba,a);
                BN_copy(bb,P); BN_sub_word(bb,(BN_ULONG)((i/7)%4096)); bn2l(bb,b);
                BN_copy(bc,P); BN_sub_word(bc,(BN_ULONG)((i/13)%4096)); bn2l(bc,c);
                if (i%5==0) { a[0]=edge[i%6]; }
                if (i%7==0) { memset(c,0,32); }
            }
            /* a*b - c */
            typedef void (*f4)(uint64_t*,const uint64_t*,const uint64_t*,const uint64_t*);
            ((f4)dlsym(A.h,"h_mulsub"))(rA,a,b,c);
            ((f4)dlsym(B.h,"h_mulsub"))(rB,a,b,c);
            l2bn(a,ba); l2bn(b,bb); l2bn(c,bc);
            BN_mod_mul(be,ba,bb,P,ctx); BN_mod_sub(be,be,bc,P,ctx);
            { int rnd = (i>=3000);
            l2bn(rA,br); BN_mod(br,br,P,ctx); if (BN_cmp(br,be)!=0) { unitbadA++; msBadA++; if(rnd) unitRbadA++; }
            l2bn(rB,br); BN_mod(br,br,P,ctx); if (BN_cmp(br,be)!=0) { unitbadB++; msBadB++; if(rnd) unitRbadB++; }
            if (memcmp(rA,rB,32)!=0) { unit_repr++; if(rnd) unitR_repr++; }
            l2bn(rA,br); if (BN_cmp(br,P)>=0) { noncanon_emit++; if(rnd) noncanonR++; }
            l2bn(rB,br); if (BN_cmp(br,P)>=0) { noncanon_emit++; if(rnd) noncanonR++; }
            if(rnd) unitR++; }
            /* a*a + b - 2c */
            ((f4)dlsym(A.h,"h_sqraddsub2"))(rA,a,b,c);
            ((f4)dlsym(B.h,"h_sqraddsub2"))(rB,a,b,c);
            l2bn(a,ba); l2bn(b,bb); l2bn(c,bc);
            BN_mod_sqr(be,ba,P,ctx); BN_mod_add(be,be,bb,P,ctx);
            BN_mod_lshift1(br,bc,P,ctx); BN_mod_sub(be,be,br,P,ctx);
            { int rnd = (i>=3000);
            l2bn(rA,br); BN_mod(br,br,P,ctx); if (BN_cmp(br,be)!=0) { unitbadA++; sqBadA++; if(rnd) unitRbadA++; }
            l2bn(rB,br); BN_mod(br,br,P,ctx); if (BN_cmp(br,be)!=0) { unitbadB++; sqBadB++; if(rnd) unitRbadB++; }
            if (memcmp(rA,rB,32)!=0) { unit_repr++; if(rnd) unitR_repr++; }
            l2bn(rA,br); if (BN_cmp(br,P)>=0) { noncanon_emit++; if(rnd) noncanonR++; }
            l2bn(rB,br); if (BN_cmp(br,P)>=0) { noncanon_emit++; if(rnd) noncanonR++; }
            if(rnd) unitR++; }
            unit += 2;
        }
    }

    printf("blocks=%ld lanes=%d   candidates compared=%ld  ok=%ld  skipped=%ld\n",
           blocks, N_LANES, cand, cmp_ok, skipped);
    printf("partial-batch inactive lanes exercised = %ld\n", partial_lanes);
    printf("planted d==0 candidates = %ld, both arms marked unusable = %ld\n", deg_d0, deg_d0_ok);
    printf("injected degenerate checks = %ld passed = %ld\n", inj_tot, inj_ok);
    printf("unit reducer cases = %ld  (random band %ld, dense near-p band %ld)\n",
           unit, unitR, unit-unitR);
    printf("  wrong residue  : fused A = %ld (random band %ld)   unfused B = %ld (random band %ld)\n",
           unitbadA, unitRbadA, unitbadB, unitRbadB);
    printf("    by reducer   : a*b-c  A=%ld B=%ld     r*r+e-2q  A=%ld B=%ld\n",
           msBadA, msBadB, sqBadA, sqBadB);
    printf("  raw-word diffs : %ld total, %ld in the random band\n", unit_repr, unitR_repr);
    printf("  outputs >= p   : %ld total, %ld in the random band\n", noncanon_emit, noncanonR);
    printf("lanes with vbar == 0 (Y == 0) that stayed usable = %ld\n", ylane_zero);
    printf("non-canonical zero denominators seen = %ld\n", noncanon_zero);
    printf("usable-decision divergence A vs B = %ld\n", usable_diff);
    printf("fused/unfused raw XYZZ words differed (congruent reps) = %ld\n", repr_diff);
    printf("MISMATCHES vs independent secp256k1 reference: A=%ld  B=%ld\n", mismatchA, mismatchB);
    printf("A vs B differing (x1,x2,parities)                  : %ld\n", ab_diff);
    int fail = (mismatchA || mismatchB || ab_diff || usable_diff || unitRbadA || (unitbadA > unitbadB)
                || deg_d0 != deg_d0_ok || inj_tot != inj_ok);
    printf("%s\n", fail ? "GATE: FAIL" : "GATE: PASS");
    return fail;
}
