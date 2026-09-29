/* Eight-lane current-V3 field operations with promoted-subset SHA16 key hashing.
 * Current V3 batch traversal, paired-key hashes and publication gate are retained.
 * Eight-lane reductions add one XOR product layer; four-lane zero flags are merged
 * in pairs. Requires AVX512F/VL/IFMA; automatically compared with the original engine.
 * Derived from promoted8d07d3e. No prior V1 cg_ifma8 implementation is incorporated. */
namespace v8i {
#define Q8I_TGT "avx2,avx512f,avx512vl,avx512ifma"
#define Q8I_INL __attribute__((target(Q8I_TGT), always_inline)) inline
#define Q8I_FN  __attribute__((target(Q8I_TGT), noinline))
typedef uint64_t V __attribute__((vector_size(64)));
struct vfe { V n[5]; };

static Q8I_INL V vs1(uint64_t x) { return (V){x, x, x, x, x, x, x, x}; }
#define Q8I_LO(acc, a, b) ((V)_mm512_madd52lo_epu64((__m512i)(acc), (__m512i)(a), (__m512i)(b)))
#define Q8I_HI(acc, a, b) ((V)_mm512_madd52hi_epu64((__m512i)(acc), (__m512i)(a), (__m512i)(b)))
#define Q8I_M52 0xFFFFFFFFFFFFFULL
#define Q8I_M48 0xFFFFFFFFFFFFULL
#define Q8I_C   0x1000003D1ULL                 /* 2^256 mod p */
#define Q8I_R   0x1000003D10ULL                /* 2^260 mod p */
/* 2p in radix 2^52 */
#define Q8I_2P0 (2 * 0xFFFFEFFFFFC2FULL)
#define Q8I_2P1 (2 * 0xFFFFFFFFFFFFFULL)
#define Q8I_2P4 (2 * 0x0FFFFFFFFFFFFULL)

/* r = a b mod p (W form in, W form out, n4 <= 2^48); r may alias a or b */
static Q8I_INL void fmul(vfe *r, const vfe *A, const vfe *B) {
    const V a0 = A->n[0], a1 = A->n[1], a2 = A->n[2], a3 = A->n[3], a4 = A->n[4];
    const V b0 = B->n[0], b1 = B->n[1], b2 = B->n[2], b3 = B->n[3], b4 = B->n[4];
    const V z = vs1(0), M = vs1(Q8I_M52), R = vs1(Q8I_R), C = vs1(Q8I_C);
    /* columns: c_k = sum lo(a_i b_j) over i+j = k, plus sum hi(a_i b_j) over i+j = k-1; each < 10 * 2^52 */
    V c0 = Q8I_LO(z, a0, b0);
    V c1 = Q8I_LO(Q8I_LO(Q8I_HI(z, a0, b0), a0, b1), a1, b0);
    V c2 = Q8I_LO(Q8I_LO(Q8I_LO(Q8I_HI(Q8I_HI(z, a0, b1), a1, b0), a0, b2), a1, b1), a2, b0);
    V c3 = Q8I_LO(Q8I_LO(Q8I_LO(Q8I_LO(Q8I_HI(Q8I_HI(Q8I_HI(z, a0, b2), a1, b1), a2, b0), a0, b3), a1, b2), a2, b1), a3, b0);
    V c4 = Q8I_HI(Q8I_HI(Q8I_HI(Q8I_HI(z, a0, b3), a1, b2), a2, b1), a3, b0);
    c4 = Q8I_LO(Q8I_LO(Q8I_LO(Q8I_LO(Q8I_LO(c4, a0, b4), a1, b3), a2, b2), a3, b1), a4, b0);
    V c5 = Q8I_HI(Q8I_HI(Q8I_HI(Q8I_HI(Q8I_HI(z, a0, b4), a1, b3), a2, b2), a3, b1), a4, b0);
    c5 = Q8I_LO(Q8I_LO(Q8I_LO(Q8I_LO(c5, a1, b4), a2, b3), a3, b2), a4, b1);
    V c6 = Q8I_HI(Q8I_HI(Q8I_HI(Q8I_HI(z, a1, b4), a2, b3), a3, b2), a4, b1);
    c6 = Q8I_LO(Q8I_LO(Q8I_LO(c6, a2, b4), a3, b3), a4, b2);
    V c7 = Q8I_HI(Q8I_HI(Q8I_HI(z, a2, b4), a3, b3), a4, b2);
    c7 = Q8I_LO(Q8I_LO(c7, a3, b4), a4, b3);
    V c8 = Q8I_LO(Q8I_HI(Q8I_HI(z, a3, b4), a4, b3), a4, b4);
    V c9 = Q8I_HI(z, a4, b4);                                   /* < 2^46 (a4, b4 < 2^49) */
    /* high columns to 52-bit limbs (c9 stays < 2^52) */
    c6 += c5 >> 52; c5 &= M;
    c7 += c6 >> 52; c6 &= M;
    c8 += c7 >> 52; c7 &= M;
    c9 += c8 >> 52; c8 &= M;
    /* fold 2^(52k) = 2^(52(k-5)) 2^260, 2^260 = R mod p: lo(c_k R) -> column k-5, hi -> k-4 */
    V d0 = Q8I_LO(c0, c5, R);
    V d1 = Q8I_LO(Q8I_HI(c1, c5, R), c6, R);
    V d2 = Q8I_LO(Q8I_HI(c2, c6, R), c7, R);
    V d3 = Q8I_LO(Q8I_HI(c3, c7, R), c8, R);
    V d4 = Q8I_LO(Q8I_HI(c4, c8, R), c9, R);
    const V e5 = Q8I_HI(z, c9, R);                              /* weight 2^260, < 2^32 */
    d4 += d3 >> 52; d3 &= M;
    const V top = (d4 >> 48) + (e5 << 4);                      /* weight 2^256, < 2^37 */
    d4 &= vs1(Q8I_M48);
    d0 = Q8I_LO(d0, top, C);
    d1 = Q8I_HI(d1, top, C);
    d1 += d0 >> 52; d0 &= M;
    d2 += d1 >> 52; d1 &= M;
    d3 += d2 >> 52; d2 &= M;
    d4 += d3 >> 52; d3 &= M;
    r->n[0] = d0; r->n[1] = d1; r->n[2] = d2; r->n[3] = d3; r->n[4] = d4;
}
static Q8I_INL void fsqr(vfe *r, const vfe *a) { fmul(r, a, a); }

/* any limbs < 2^62 -> W form (libsecp256k1 fe_normalize_weak, radix 2^52) */
static Q8I_INL void fwk(vfe *r) {
    const V M = vs1(Q8I_M52);
    V t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4];
    const V x = t4 >> 48; t4 &= vs1(Q8I_M48);
    t0 = Q8I_LO(t0, x, vs1(Q8I_C));                               /* x < 2^14: x C < 2^47 */
    t1 += t0 >> 52; t0 &= M;
    t2 += t1 >> 52; t1 &= M;
    t3 += t2 >> 52; t2 &= M;
    t4 += t3 >> 52; t3 &= M;
    r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4;
}
/* canonical value in [0, p) from limbs < 2^62 (libsecp256k1 fe_normalize, radix 2^52) */
static Q8I_INL void fnorm(vfe *r) {
    const V M = vs1(Q8I_M52);
    V t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4], m;
    V x = t4 >> 48; t4 &= vs1(Q8I_M48);
    t0 = Q8I_LO(t0, x, vs1(Q8I_C));
    t1 += t0 >> 52; t0 &= M;
    t2 += t1 >> 52; t1 &= M; m = t1;
    t3 += t2 >> 52; t2 &= M; m &= t2;
    t4 += t3 >> 52; t3 &= M; m &= t3;
    /* value < 2p here; subtract p once if value >= p (limbs < 2^53: signed compares are exact) */
    const V ge = (V)_mm512_maskz_set1_epi64(_mm512_cmpgt_epi64_mask((__m512i)t0, (__m512i)vs1(0xFFFFEFFFFFC2FULL - 1)), -1);
    x = (t4 >> 48) | ((V)_mm512_maskz_set1_epi64(_mm512_cmpeq_epi64_mask((__m512i)t4, (__m512i)vs1(Q8I_M48)), -1) & (V)_mm512_maskz_set1_epi64(_mm512_cmpeq_epi64_mask((__m512i)m, (__m512i)M), -1) & ge & vs1(1));
    t0 += (vs1(0) - x) & vs1(Q8I_C);                             /* x in {0, 1} */
    t1 += t0 >> 52; t0 &= M;
    t2 += t1 >> 52; t1 &= M;
    t3 += t2 >> 52; t2 &= M;
    t4 += t3 >> 52; t3 &= M;
    t4 &= vs1(Q8I_M48);
    r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4;
}
/* r = a + 2p - b (b in W form); limbs of r < a's + 2^54 */
static Q8I_INL void fsub(vfe *r, const vfe *a, const vfe *b) {
    r->n[0] = a->n[0] + (vs1(Q8I_2P0) - b->n[0]);
    r->n[1] = a->n[1] + (vs1(Q8I_2P1) - b->n[1]);
    r->n[2] = a->n[2] + (vs1(Q8I_2P1) - b->n[2]);
    r->n[3] = a->n[3] + (vs1(Q8I_2P1) - b->n[3]);
    r->n[4] = a->n[4] + (vs1(Q8I_2P4) - b->n[4]);
}
/* r = 2p - a (a in W form) */
static Q8I_INL void fneg(vfe *r, const vfe *a) {
    r->n[0] = vs1(Q8I_2P0) - a->n[0];
    r->n[1] = vs1(Q8I_2P1) - a->n[1];
    r->n[2] = vs1(Q8I_2P1) - a->n[2];
    r->n[3] = vs1(Q8I_2P1) - a->n[3];
    r->n[4] = vs1(Q8I_2P4) - a->n[4];
}
/* r = mask ? a : b (per lane, mask all-ones or zero) */
static Q8I_INL void fsel(vfe *r, V mask, const vfe *a, const vfe *b) {
    for (int k = 0; k < 5; k++) r->n[k] = (V)_mm512_mask_blend_epi64(_mm512_cmpneq_epi64_mask((__m512i)mask, _mm512_setzero_si512()), (__m512i)b->n[k], (__m512i)a->n[k]);
}
static Q8I_INL void fone(vfe *r) { r->n[0] = vs1(1); for (int k = 1; k < 5; k++) r->n[k] = vs1(0); }
/* 4 little-endian 64-bit words per lane (< 2^256) -> radix 2^52 (limbs < 2^52, n4 < 2^48) */
static Q8I_INL void from_w(vfe *r, V w0, V w1, V w2, V w3) {
    const V M = vs1(Q8I_M52);
    r->n[0] = w0 & M;
    r->n[1] = ((w0 >> 52) | (w1 << 12)) & M;
    r->n[2] = ((w1 >> 40) | (w2 << 24)) & M;
    r->n[3] = ((w2 >> 28) | (w3 << 36)) & M;
    r->n[4] = w3 >> 16;
}
/* canonical radix 2^52 -> 4 words per lane */
static Q8I_INL void to_w(V w[4], const vfe *a) {
    const V *n = a->n;
    w[0] = n[0] | (n[1] << 52);
    w[1] = (n[1] >> 12) | (n[2] << 40);
    w[2] = (n[2] >> 24) | (n[3] << 28);
    w[3] = (n[3] >> 36) | (n[4] << 16);
}
static Q8I_INL void set_w(vfe *r, const uint64_t *w) { from_w(r, vs1(w[0]), vs1(w[1]), vs1(w[2]), vs1(w[3])); }
/* r lane l = a lane (l ^ k), k in {1,2,4}. */
static Q8I_INL void fpermx(vfe *r, const vfe *a, int k) {
    const V index = {0u^(unsigned)k,1u^(unsigned)k,2u^(unsigned)k,3u^(unsigned)k,
                     4u^(unsigned)k,5u^(unsigned)k,6u^(unsigned)k,7u^(unsigned)k};
    for (int j=0;j<5;j++)
        r->n[j]=(V)_mm512_permutexvar_epi64((__m512i)index,(__m512i)a->n[j]);
}
/* Keep the existing four-entry transpose twice; combine its two halves. */
static Q8I_INL void tr8(V w[4], const tentry *const *e, int half) {
    v4i::V low[4],high[4];
    v4i::tr4(low,e[0],e[1],e[2],e[3],half);
    v4i::tr4(high,e[4],e[5],e[6],e[7],half);
    for (int k=0;k<4;k++) w[k]=(V)_mm512_inserti64x4(_mm512_castsi256_si512((__m256i)low[k]),(__m256i)high[k],1);
}
static Q8I_INL void gather_x(vfe *x,const tentry *const *e) {V w[4];tr8(w,e,0);from_w(x,w[0],w[1],w[2],w[3]);}
static Q8I_INL void gather_y(vfe *y,const tentry *const *e) {V w[4];tr8(w,e,1);from_w(y,w[0],w[1],w[2],w[3]);}

/* inv lane l = 1 / m lane l (m in W form) with one scalar inversion */
static Q8I_INL void vinv(vfe *inv, const vfe *m) {
    vfe s,q1,q2,q3,oth,t;
    fpermx(&s,m,1);oth=s;fmul(&q1,m,&s);
    fpermx(&s,&q1,2);fmul(&t,&oth,&s);oth=t;fmul(&q2,&q1,&s);
    fpermx(&s,&q2,4);fmul(&t,&oth,&s);oth=t;fmul(&q3,&q2,&s);
    fnorm(&q3);
    V w[4];to_w(w,&q3);
    uint64_t tw[4]={w[0][0],w[1][0],w[2][0],w[3][0]},iw[4];
    scalar_inv(iw,tw);set_w(&t,iw);fmul(inv,&t,&oth);
}

/* SHA16 schedule and four SHA-NI round chains derive from the promoted
 * subset CpuGrindSubset.h at 7813ffe (KH16, originally a33e04c3).
 * Runtime entry is guarded by the existing SHA-NI calibration selection.
 * The complete 16-key mask keeps Q+ in bits0..7 and Q- in bits8..15. */
static __attribute__((target("sha,sse4.1,ssse3,avx,avx2,avx512f,avx512vl"),noinline))
unsigned keyhash16(const uint32_t *m) {
    alignas(64) uint32_t wk[1024];
#define R16(x, n) _mm512_ror_epi32((x), (n))
#define S0_16(x) _mm512_ternarylogic_epi32(R16(x, 7), R16(x, 18), _mm512_srli_epi32(x, 3), 0x96)
#define S1_16(x) _mm512_ternarylogic_epi32(R16(x, 17), R16(x, 19), _mm512_srli_epi32(x, 10), 0x96)
    __m512i W[16];
#pragma GCC unroll 9
    for (int i = 0; i < 9; i++) W[i] = _mm512_load_si512((const void *)(m + 16 * i));
    for (int i = 9; i < 15; i++) W[i] = _mm512_setzero_si512();
    W[15] = _mm512_set1_epi32(264);
    __m512i prev = _mm512_setzero_si512();
#pragma GCC unroll 64
    for (int t = 0; t < 64; t++) {
        __m512i wt;
        if (t < 16) wt = W[t];
        else {
            /* W[t] = s1(W[t-2]) + W[t-7] + s0(W[t-15]) + W[t-16]; the zero words W9..W14 drop out at compile time */
            const int a2 = (t - 2) & 15, a7 = (t - 7) & 15, a15 = (t - 15) & 15, a16 = t & 15;
            const bool z2 = (t - 2) >= 9 && (t - 2) <= 14, z7 = (t - 7) >= 9 && (t - 7) <= 14, z15 = (t - 15) >= 9 && (t - 15) <= 14,
                       z16 = (t - 16) >= 9 && (t - 16) <= 14;
            wt = z16 ? _mm512_setzero_si512() : W[a16];
            if (!z15) wt = _mm512_add_epi32(wt, S0_16(W[a15]));
            if (!z7) wt = _mm512_add_epi32(wt, W[a7]);
            if (!z2) wt = _mm512_add_epi32(wt, S1_16(W[a2]));
            W[a16] = wt;
        }
        const __m512i wkt = _mm512_add_epi32(wt, _mm512_set1_epi32((int)qcg_sha::K256[t]));
        if (t & 1) {                                     /* rounds t - 1, t as dword pairs, key-major within 128-bit lanes */
            _mm512_store_si512((void *)(wk + 32 * (t >> 1)), _mm512_unpacklo_epi32(prev, wkt));
            _mm512_store_si512((void *)(wk + 32 * (t >> 1) + 16), _mm512_unpackhi_epi32(prev, wkt));
        } else prev = wkt;
    }
#undef S0_16
#undef S1_16
#undef R16
    /* key 4L + e: pair p at wk + 32 p + (e >= 2 ? 16 : 0) + 4 L + 2 (e & 1) */
    const __m128i IV0 = _mm_set_epi32((int)0x6a09e667, (int)0xbb67ae85, (int)0x510e527f, (int)0x9b05688c);
    const __m128i IV1 = _mm_set_epi32((int)0x3c6ef372, (int)0xa54ff53a, (int)0x1f83d9ab, (int)0x5be0cd19);
    alignas(64) uint32_t h0[16];
#pragma GCC unroll 1
    for (int L = 0; L < 4; L++) {
        __m128i S0[4], S1[4];
        const uint32_t *base[4] = {wk + 4 * L, wk + 4 * L + 2, wk + 16 + 4 * L, wk + 16 + 4 * L + 2};
#pragma GCC unroll 4
        for (int e = 0; e < 4; e++) { S0[e] = IV0; S1[e] = IV1; }
#pragma GCC unroll 16
        for (int r = 0; r < 16; r++) {
#pragma GCC unroll 4
            for (int e = 0; e < 4; e++) S1[e] = _mm_sha256rnds2_epu32(S1[e], S0[e], _mm_loadl_epi64((const __m128i *)(base[e] + 64 * r)));
#pragma GCC unroll 4
            for (int e = 0; e < 4; e++) S0[e] = _mm_sha256rnds2_epu32(S0[e], S1[e], _mm_loadl_epi64((const __m128i *)(base[e] + 64 * r + 32)));
        }
#pragma GCC unroll 4
        for (int e = 0; e < 4; e++) h0[4 * L + e] = (uint32_t)_mm_extract_epi32(_mm_add_epi32(S0[e], IV0), 3);
    }
#ifdef QCG_HASH_HOOK
    QCG_HASH_HOOK(h0);
#endif
    const __m512i hv = _mm512_load_si512((const void *)h0);   /* pk_prefilter of the 16 keys: bit k = key k passes */
    return (unsigned)_mm512_cmpeq_epi32_mask(_mm512_srli_epi32(hv, 32 - (QSB_ZEROS_N < 32 ? QSB_ZEROS_N : 32)), _mm512_setzero_si512());
}

/* Sixteen pubkeys, hashed with two unchanged eight-way SHA batches.
 * Result bits0..7 are Q+ and bits8..15 are Q-. */
static Q8I_INL unsigned hash_block(const vfe *xp,const vfe *yp,const vfe *xm,const vfe *ym,int use_ni) {
    if (use_ni) {
        V wp[4],wm[4];to_w(wp,xp);to_w(wm,xm);
        const __m512i lo=_mm512_setr_epi32(0,2,4,6,8,10,12,14,0,2,4,6,8,10,12,14);
        const __m512i hi=_mm512_setr_epi32(1,3,5,7,9,11,13,15,1,3,5,7,9,11,13,15);
        __m512i X[8];
        for(int k=0;k<4;k++) {
            X[2*k]=_mm512_mask_blend_epi32(0xff00,_mm512_permutexvar_epi32(lo,(__m512i)wp[k]),_mm512_permutexvar_epi32(lo,(__m512i)wm[k]));
            X[2*k+1]=_mm512_mask_blend_epi32(0xff00,_mm512_permutexvar_epi32(hi,(__m512i)wp[k]),_mm512_permutexvar_epi32(hi,(__m512i)wm[k]));
        }
        __m512i par=_mm512_mask_blend_epi32(0xff00,_mm512_permutexvar_epi32(lo,(__m512i)yp->n[0]),_mm512_permutexvar_epi32(lo,(__m512i)ym->n[0]));
        par=_mm512_and_si512(par,_mm512_set1_epi32(1));
        alignas(64) uint32_t words[9][16];
        _mm512_store_si512(words[0],_mm512_or_si512(_mm512_slli_epi32(_mm512_or_si512(par,_mm512_set1_epi32(2)),24),_mm512_srli_epi32(X[7],8)));
        for(int j=1;j<8;j++)_mm512_store_si512(words[j],_mm512_or_si512(_mm512_slli_epi32(X[8-j],24),_mm512_srli_epi32(X[7-j],8)));
        _mm512_store_si512(words[8],_mm512_or_si512(_mm512_slli_epi32(X[0],24),_mm512_set1_epi32(0x00800000)));
        return keyhash16(&words[0][0]);
    }
    v4i::vfe p[2],y[2],m[2],n[2];
    for (int k=0;k<5;k++) {
        p[0].n[k]=(v4i::V)_mm512_castsi512_si256((__m512i)xp->n[k]);
        y[0].n[k]=(v4i::V)_mm512_castsi512_si256((__m512i)yp->n[k]);
        m[0].n[k]=(v4i::V)_mm512_castsi512_si256((__m512i)xm->n[k]);
        n[0].n[k]=(v4i::V)_mm512_castsi512_si256((__m512i)ym->n[k]);
        p[1].n[k]=(v4i::V)_mm512_extracti64x4_epi64((__m512i)xp->n[k],1);
        y[1].n[k]=(v4i::V)_mm512_extracti64x4_epi64((__m512i)yp->n[k],1);
        m[1].n[k]=(v4i::V)_mm512_extracti64x4_epi64((__m512i)xm->n[k],1);
        n[1].n[k]=(v4i::V)_mm512_extracti64x4_epi64((__m512i)ym->n[k],1);
    }
    const unsigned a=v4i::hash_block(&p[0],&y[0],&m[0],&n[0],use_ni);
    const unsigned b=v4i::hash_block(&p[1],&y[1],&m[1],&n[1],use_ni);
    return (a&15u)|((b&15u)<<4)|((a&240u)<<4)|((b&240u)<<8);
}
static Q8I_INL V zmask8(const uint32_t *d8) {V m;for(int l=0;l<8;l++)m[l]=QCG_ZERO(d8[l])?~0ULL:0;return m;}
static Q8I_INL V negmask8(const uint32_t *d8) {
    return (V)_mm512_cvtepi32_epi64(_mm256_srai_epi32(_mm256_loadu_si256((const __m256i*)d8),31));
}
static Q8I_INL void entries8(const tentry **e,const tentry *T,const uint32_t *d8) {
    for(int l=0;l<8;l++)e[l]=T+(d8[l]&QCG_IDXM);
}
static Q8I_INL void prefetch8(const tentry *T,const uint32_t *d8) {
    for(int l=0;l<8;l++)_mm_prefetch((const char*)(T+(d8[l]&QCG_IDXM)),_MM_HINT_T0);
}

struct bst { vfe xT, dx, dy, ik, lam, l2, t, y3, dxn; };
/* gather T_s; dx = xT - px (1 on zero-digit lanes); dy = +-yT - py; both W form */
static Q8I_INL void bk_pre(bst &c, const tentry *Tj, const uint32_t *d4, uint8_t zf, const vfe &px, const vfe &py, const vfe &one) {
    const tentry *e[8]; entries8(e, Tj, d4);
    V wx[4], wy[4];
    tr8(wx,e,0); tr8(wy,e,1);
    from_w(&c.xT, wx[0], wx[1], wx[2], wx[3]);
    fsub(&c.dx, &c.xT, &px); fwk(&c.dx);
    if (__builtin_expect(zf, 0)) fsel(&c.dx, zmask8(d4), &one, &c.dx);
    vfe yT, ny; from_w(&yT, wy[0], wy[1], wy[2], wy[3]);
    const V sm = negmask8(d4);
    fneg(&ny, &yT);
    fsel(&c.dy, sm, &ny, &yT);
    fsub(&c.dy, &c.dy, &py); fwk(&c.dy);
}
/* x3 = l2 - px - xT -> px (old point kept on zero-digit lanes); t = px_old - x3 */
static Q8I_INL void bk_x3(bst &c, vfe &px, const uint32_t *d4, uint8_t zf) {
    vfe x3;
    fsub(&x3, &c.l2, &px); fsub(&x3, &x3, &c.xT); fwk(&x3);
    fsub(&c.t, &px, &x3); fwk(&c.t);
    if (__builtin_expect(zf, 0)) fsel(&x3, zmask8(d4), &px, &x3);
    px = x3;
}
/* y3 = lam t - py -> py, keeping the old y on zero-digit lanes */
static Q8I_INL void bk_y3(bst &c, vfe &py, const uint32_t *d4, uint8_t zf) {
    vfe y3;
    fsub(&y3, &c.y3, &py); fwk(&y3);
    if (__builtin_expect(zf, 0)) fsel(&y3, zmask8(d4), &py, &y3);
    py = y3;
}
/* forward part of the next step: dxn = xN - x (1 on zero-digit lanes) */
static Q8I_INL void fw_next(bst &c, const tentry *Tn, const uint32_t *dn4, uint8_t zfn, const vfe &x, const vfe &one) {
    const tentry *e[8]; entries8(e, Tn, dn4);
    vfe xN; gather_x(&xN, e);
    fsub(&c.dxn, &xN, &x); fwk(&c.dxn);
    if (__builtin_expect(zfn, 0)) { const V m = zmask8(dn4); fsel(&c.dxn, m, &one, &c.dxn); }
}

struct vstate {
    vfe px[QSB_CG_BMAX / 8 + 4], py[QSB_CG_BMAX / 8 + 4], c[QSB_CG_BMAX / 8 + 4];
    bst tmp;
};

/* One batch: Q+ = z B + A, Q- = Q+ - 2A for the w->n candidates, hash, publish. */
static Q8I_FN void ec_batch(worker_t *w, vstate *vs) {
    shared_t *S = g_cg;
    const int use_ni = S->sha_mode.load(std::memory_order_relaxed) == 2;
    const layout_t &L = S->lay;
    const int n = w->n;
    const int nb = (n + 7) >> 3;
    const tentry *T = S->table;
    vfe *px = vs->px, *py = vs->py, *cc = vs->c;
    vfe one; fone(&one);
    const int nw = L.nwin;
    {
        const uint32_t *dg = w->dig[0];
        const tentry *T0 = T + L.off[0];
        for (int b = 0; b < nb; b++) {
            const tentry *e[8]; for(int l=0;l<8;l++)e[l]=T0+dg[8*b+l];
            if (b + QSB_CG_PF < nb) for (int l = 0; l < 8; l++) _mm_prefetch((const char *)(T0 + dg[8 * (b + QSB_CG_PF) + l]), _MM_HINT_T0);
            gather_x(&px[b], e); gather_y(&py[b], e);
        }
    }
    vfe acc, inv;
    {
        const uint32_t *dg = w->dig[1];
        const uint8_t *zf = w->zf[1];
        const tentry *Tj = T + L.off[1];
        for (int b = 0; b < nb; b++) {
            if (b + QSB_CG_PF < nb) prefetch8(Tj, dg + 8 * (b + QSB_CG_PF));
            const tentry *e[8]; entries8(e, Tj, dg + 8 * b);
            vfe xT, dx; gather_x(&xT, e);
            fsub(&dx, &xT, &px[b]); fwk(&dx);
            if ((zf[2*b]|zf[2*b+1])) fsel(&dx, zmask8(dg + 8 * b), &one, &dx);
            fmul(&cc[b], b ? &cc[b - 1] : &one, &dx);
        }
        acc = cc[nb - 1];
    }
    vfe xD, yD; set_w(&xD, S->dx_w); set_w(&yD, S->dy_w);
    int asc = 1;
    for (int s = 1; s < nw; s++) {
        vinv(&inv, &acc);
        vfe ubuf[2]; ubuf[0] = inv; int ui = 0;
        const vfe *accp = &one;
        const uint32_t *dg = w->dig[s];
        const uint8_t *zf = w->zf[s];
        const tentry *Tj = T + L.off[s];
        const int last = (s + 1 == nw);
        const uint32_t *dgn = last ? NULL : w->dig[s + 1];
        const uint8_t *zfn = last ? NULL : w->zf[s + 1];
        const tentry *Tn = last ? NULL : T + L.off[s + 1];
        const int step = asc ? -1 : 1;
        int b = asc ? nb - 1 : 0;
        for (int it = 0; it < nb; it++, b += step) {
            const int bp = b + step;
            const int bf = b + step * QSB_CG_PF;
            if (bf >= 0 && bf < nb) { prefetch8(Tj, dg + 8 * bf); if (!last) prefetch8(Tn, dgn + 8 * bf); }
            bst &c = vs->tmp;
            bk_pre(c, Tj, dg + 8 * b, (zf[2*b]|zf[2*b+1]), px[b], py[b], one);
            const vfe *ikp;
            if (it + 1 < nb) { fmul(&c.ik, &ubuf[ui], &cc[bp]); fmul(&ubuf[ui ^ 1], &ubuf[ui], &c.dx); ui ^= 1; ikp = &c.ik; }
            else ikp = &ubuf[ui];
            fmul(&c.lam, &c.dy, ikp);
            fsqr(&c.l2, &c.lam);
            bk_x3(c, px[b], dg + 8 * b, (zf[2*b]|zf[2*b+1]));
            fmul(&c.y3, &c.lam, &c.t);
            bk_y3(c, py[b], dg + 8 * b, (zf[2*b]|zf[2*b+1]));
            if (!last) fw_next(c, Tn, dgn + 8 * b, (zfn[2*b]|zfn[2*b+1]), px[b], one);
            else { fsub(&c.dxn, &xD, &px[b]); fwk(&c.dxn); }
            fmul(&cc[b], accp, &c.dxn);
            accp = &cc[b];
        }
        acc = *accp;
        asc = !asc;
    }
    vinv(&inv, &acc);
    {
        vfe ubuf[2]; ubuf[0] = inv; int ui = 0;
        const int step = asc ? -1 : 1;
        int b = asc ? nb - 1 : 0;
        for (int it = 0; it < nb; it++, b += step) {
            const int bp = b + step;
            vfe dx, dy, ik, lam, l2, xm, t, ym;
            fsub(&dx, &xD, &px[b]); fwk(&dx);
            const vfe *ikp;
            if (it + 1 < nb) { fmul(&ik, &ubuf[ui], &cc[bp]); fmul(&ubuf[ui ^ 1], &ubuf[ui], &dx); ui ^= 1; ikp = &ik; }
            else ikp = &ubuf[ui];
            fsub(&dy, &yD, &py[b]); fwk(&dy);
            fmul(&lam, &dy, ikp);
            fsqr(&l2, &lam);
            fsub(&xm, &l2, &px[b]); fsub(&xm, &xm, &xD); fwk(&xm);
            fsub(&t, &px[b], &xm); fwk(&t);
            fmul(&ym, &lam, &t);
            fsub(&ym, &ym, &py[b]);
            vfe xp = px[b], yp = py[b];
            fnorm(&xp); fnorm(&yp); fnorm(&xm); fnorm(&ym);
#ifdef QCG_EC_HOOK
            { V a4[4], b4[4], c4[4], d4[4]; to_w(a4, &xp); to_w(b4, &yp); to_w(c4, &xm); to_w(d4, &ym);
              for (int l = 0; l < 8; l++) { uint64_t X0[4], Y0[4], X1[4], Y1[4];
                  for (int q = 0; q < 4; q++) { X0[q] = a4[q][l]; Y0[q] = b4[q][l]; X1[q] = c4[q][l]; Y1[q] = d4[q][l]; }
                  if (8 * b + l < n) QCG_EC_HOOK(w, 8 * b + l, X0, Y0, X1, Y1); } }
#endif
            const unsigned hm = hash_block(&xp, &yp, &xm, &ym, use_ni);
            if (hm) {
                for (int l = 0; l < 16; l++) if (hm >> l & 1) {
                    const int ci = 8 * b + (l & 7);
                    if (ci < n) publish(w, ci, l >> 3);
                }
            }
        }
    }
}
#undef Q8I_LO
#undef Q8I_HI
#undef Q8I_INL
#undef Q8I_FN
#undef Q8I_TGT
} /* namespace v8i */
