#pragma once
/* Host fixed-key SHA: include inside qcpu after sha16 helpers and
 * gate_publish_exact. Requires S16T, ROR16, SHA16_ROUND, qsha_k, qsha_iv.
 * Independently derived streamed schedule and sparse output on the inherited
 * SHA16 round macro. Startup checks all H0 words against OpenSSL; native
 * compilation and throughput are left to official evaluation.
 *
 * Keep this phase after vec_batch returns: no live fe8 inversion roots here.
 * kw is existing aligned [B/8][9][16] recid-major staged message words.
 * This fixed single-block compressor returns only H0, in the original
 * recid-major vector order. It does not materialize a 64-vector W+K table.
 */
#if QCPU_VEC && QCPU_SHANI
S16T static inline __attribute__((always_inline))
__m512i cpu_keyhash16_fused_h0(const uint32_t *src) {
    __m512i w0 = _mm512_load_si512((const void *)(src + 0 * 16));
    __m512i w1 = _mm512_load_si512((const void *)(src + 1 * 16));
    __m512i w2 = _mm512_load_si512((const void *)(src + 2 * 16));
    __m512i w3 = _mm512_load_si512((const void *)(src + 3 * 16));
    __m512i w4 = _mm512_load_si512((const void *)(src + 4 * 16));
    __m512i w5 = _mm512_load_si512((const void *)(src + 5 * 16));
    __m512i w6 = _mm512_load_si512((const void *)(src + 6 * 16));
    __m512i w7 = _mm512_load_si512((const void *)(src + 7 * 16));
    __m512i w8 = _mm512_load_si512((const void *)(src + 8 * 16));
    __m512i w9 = _mm512_setzero_si512();
    __m512i w10 = _mm512_setzero_si512();
    __m512i w11 = _mm512_setzero_si512();
    __m512i w12 = _mm512_setzero_si512();
    __m512i w13 = _mm512_setzero_si512();
    __m512i w14 = _mm512_setzero_si512();
    __m512i w15 = _mm512_set1_epi32(264);
    __m512i a = _mm512_set1_epi32((int)qsha_iv[0]);
    __m512i b = _mm512_set1_epi32((int)qsha_iv[1]);
    __m512i c = _mm512_set1_epi32((int)qsha_iv[2]);
    __m512i d = _mm512_set1_epi32((int)qsha_iv[3]);
    __m512i e = _mm512_set1_epi32((int)qsha_iv[4]);
    __m512i f = _mm512_set1_epi32((int)qsha_iv[5]);
    __m512i g = _mm512_set1_epi32((int)qsha_iv[6]);
    __m512i h = _mm512_set1_epi32((int)qsha_iv[7]);
    SHA16_ROUND(a,b,c,d,e,f,g,h, _mm512_add_epi32(w0, _mm512_set1_epi32((int)qsha_k[0])));
    SHA16_ROUND(h,a,b,c,d,e,f,g, _mm512_add_epi32(w1, _mm512_set1_epi32((int)qsha_k[1])));
    SHA16_ROUND(g,h,a,b,c,d,e,f, _mm512_add_epi32(w2, _mm512_set1_epi32((int)qsha_k[2])));
    SHA16_ROUND(f,g,h,a,b,c,d,e, _mm512_add_epi32(w3, _mm512_set1_epi32((int)qsha_k[3])));
    SHA16_ROUND(e,f,g,h,a,b,c,d, _mm512_add_epi32(w4, _mm512_set1_epi32((int)qsha_k[4])));
    SHA16_ROUND(d,e,f,g,h,a,b,c, _mm512_add_epi32(w5, _mm512_set1_epi32((int)qsha_k[5])));
    SHA16_ROUND(c,d,e,f,g,h,a,b, _mm512_add_epi32(w6, _mm512_set1_epi32((int)qsha_k[6])));
    SHA16_ROUND(b,c,d,e,f,g,h,a, _mm512_add_epi32(w7, _mm512_set1_epi32((int)qsha_k[7])));
    SHA16_ROUND(a,b,c,d,e,f,g,h, _mm512_add_epi32(w8, _mm512_set1_epi32((int)qsha_k[8])));
    SHA16_ROUND(h,a,b,c,d,e,f,g, _mm512_add_epi32(w9, _mm512_set1_epi32((int)qsha_k[9])));
    SHA16_ROUND(g,h,a,b,c,d,e,f, _mm512_add_epi32(w10, _mm512_set1_epi32((int)qsha_k[10])));
    SHA16_ROUND(f,g,h,a,b,c,d,e, _mm512_add_epi32(w11, _mm512_set1_epi32((int)qsha_k[11])));
    SHA16_ROUND(e,f,g,h,a,b,c,d, _mm512_add_epi32(w12, _mm512_set1_epi32((int)qsha_k[12])));
    SHA16_ROUND(d,e,f,g,h,a,b,c, _mm512_add_epi32(w13, _mm512_set1_epi32((int)qsha_k[13])));
    SHA16_ROUND(c,d,e,f,g,h,a,b, _mm512_add_epi32(w14, _mm512_set1_epi32((int)qsha_k[14])));
    SHA16_ROUND(b,c,d,e,f,g,h,a, _mm512_add_epi32(w15, _mm512_set1_epi32((int)qsha_k[15])));
#define CPU_KF_STEP(I, W, W15, W2, W7, A,B,C,D,E,F,G,H) do { \
    const __m512i s0 = _mm512_ternarylogic_epi32(ROR16(W15, 7), ROR16(W15, 18), _mm512_srli_epi32(W15, 3), 0x96); \
    const __m512i s1 = _mm512_ternarylogic_epi32(ROR16(W2, 17), ROR16(W2, 19), _mm512_srli_epi32(W2, 10), 0x96); \
    W = _mm512_add_epi32(_mm512_add_epi32(W, s0), _mm512_add_epi32(W7, s1)); \
    SHA16_ROUND(A,B,C,D,E,F,G,H, _mm512_add_epi32(W, _mm512_set1_epi32((int)qsha_k[t + (I)]))); \
} while (0)
    /* Ring indices are literal names. Retain a 16-round body rather than
     * forcing 48 separate schedule+round bodies into the instruction cache. */
#pragma GCC unroll 1
    for (int t = 16; t < 64; t += 16) {
        CPU_KF_STEP(0, w0, w1, w14, w9, a,b,c,d,e,f,g,h);
        CPU_KF_STEP(1, w1, w2, w15, w10, h,a,b,c,d,e,f,g);
        CPU_KF_STEP(2, w2, w3, w0, w11, g,h,a,b,c,d,e,f);
        CPU_KF_STEP(3, w3, w4, w1, w12, f,g,h,a,b,c,d,e);
        CPU_KF_STEP(4, w4, w5, w2, w13, e,f,g,h,a,b,c,d);
        CPU_KF_STEP(5, w5, w6, w3, w14, d,e,f,g,h,a,b,c);
        CPU_KF_STEP(6, w6, w7, w4, w15, c,d,e,f,g,h,a,b);
        CPU_KF_STEP(7, w7, w8, w5, w0, b,c,d,e,f,g,h,a);
        CPU_KF_STEP(8, w8, w9, w6, w1, a,b,c,d,e,f,g,h);
        CPU_KF_STEP(9, w9, w10, w7, w2, h,a,b,c,d,e,f,g);
        CPU_KF_STEP(10, w10, w11, w8, w3, g,h,a,b,c,d,e,f);
        CPU_KF_STEP(11, w11, w12, w9, w4, f,g,h,a,b,c,d,e);
        CPU_KF_STEP(12, w12, w13, w10, w5, e,f,g,h,a,b,c,d);
        CPU_KF_STEP(13, w13, w14, w11, w6, d,e,f,g,h,a,b,c);
        CPU_KF_STEP(14, w14, w15, w12, w7, c,d,e,f,g,h,a,b);
        CPU_KF_STEP(15, w15, w0, w13, w8, b,c,d,e,f,g,h,a);
    }
#undef CPU_KF_STEP
    return _mm512_add_epi32(a, _mm512_set1_epi32((int)qsha_iv[0]));
}

/* Startup/reference entry point: export every lane's complete 32-bit H0 in
 * legacy [candidate][recid] order. Compare these 2B words individually against
 * the existing actual-EC/OpenSSL oracle; testing only rare masks is inadequate.
 */
S16T static __attribute__((noinline)) void keyhash16_fused_all_h0(
        const uint32_t *kw, int B, uint32_t *h0) {
    const __m512i interleave = _mm512_setr_epi32(
        0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
    for (int h = 0; h < B / 8; ++h) {
        const __m512i result = cpu_keyhash16_fused_h0(kw + (size_t)h * 9 * 16);
        _mm512_storeu_si512((void *)(h0 + (size_t)h * 16),
                           _mm512_permutexvar_epi32(interleave, result));
    }
}

S16T static inline unsigned cpu_keyhash16_hitmask(__m512i h0) {
    const __m512i upper = _mm512_srli_epi32(h0, 32 - (QSB_ZEROS_N < 32 ? QSB_ZEROS_N : 32));
    return (unsigned)_mm512_cmpeq_epi32_mask(upper, _mm512_setzero_si512());
}

/* Exercise every bit position, empty/full and alternating masks without
 * needing a rare real hash match. Called only after AVX-512 dispatch checks;
 * it does not publish, advance enumeration, or change counters. */
S16T static bool keyhash_fused_mask_selfcheck() {
    alignas(64) uint32_t words[16];
    for (unsigned t = 0; t < 36; ++t) {
        const unsigned m = t < 16 ? (1u << t) : t < 32 ? (65535u ^ (1u << (t - 16))) :
                           t == 32 ? 0u : t == 33 ? 65535u : t == 34 ? 0x5555u : 0xAAAAu;
        for (unsigned j = 0; j < 16; ++j) words[j] = (m & (1u << j)) ? 0u : 0x80000000u;
        const unsigned expected = QSB_ZEROS_N == 0 ? 65535u : m;
        if (cpu_keyhash16_hitmask(_mm512_load_si512((const void *)words)) != expected) return false;
    }
    return true;
}

/* out_pairs holds only nonzero masks: [candidate_base, recid-major mask].
 * Capacity B/4 uint32_t words suffices, even at N=0. The existing h016 buffer
 * has 2B words, so it may be reused without a new allocation or type punning.
 * A hit mask is not an exact authorization to publish.
 */
S16T static __attribute__((noinline)) int keyhash16_fused_sparse(const uint32_t *kw, int B, uint32_t *out_pairs) {
    int n = 0;
    for (int h = 0; h < B / 8; ++h) {
        const __m512i h0 = cpu_keyhash16_fused_h0(kw + (size_t)h * 9 * 16);
        const unsigned m = cpu_keyhash16_hitmask(h0);
        if (m) {
            out_pairs[2 * n] = (uint32_t)(8 * h);
            out_pairs[2 * n + 1] = m;
            ++n;
        }
    }
    return n;
}

/* Call only after keyhash16_fused_sparse returns, with QB and skips from the
 * same completed EC batch. No SHA or EC vector registers need cross OpenSSL.
 * Preserve candidate order, and suppress ri1 only when the ri0 exact gate
 * succeeds. A failed ri0 prefilter verification must still allow ri1.
 */
template <class SkipAccessor>
static void keyhash_sparse_publish(Ctx *c, const uint32_t *pairs, int n,
        const uint8_t *bad, const SkipAccessor &access) {
    for (int i = 0; i < n; ++i) {
        const unsigned base = pairs[2 * i], m = pairs[2 * i + 1];
        unsigned pending = (m | (m >> 8)) & 255u;
        while (pending) {
            const unsigned j = (unsigned)__builtin_ctz(pending);
            pending &= pending - 1;
            const unsigned q = base + j;
            if (bad[q]) continue;
            uint8_t local[9]; const uint8_t *sk = access.get(q, local);
            if ((m & (1u << j)) && gate_publish_exact(c, sk, 0)) continue;
            if (m & (1u << (j + 8))) gate_publish_exact(c, sk, 1);
        }
    }
}
#endif
