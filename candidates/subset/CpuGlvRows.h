#pragma once
/* Included after VecBuf/vec_batch. Signed rowcodes store an offset in units
 * of eight bytes in bits 0..30 and the Y sign in bit 31. Five shared high
 * tables are used twice; planes 10/11 already select the joint +/-C table.
 * No point representation, affine formula or inversion schedule changes.
 */
Q8T static inline __attribute__((always_inline)) __mmask8 cpu_glv_rows8_decode(
        const pt *base, const uint32_t *codes, const pt **rows) {
    static_assert(sizeof(pt) == 64 && sizeof(void *) == 8, "rowcode address units");
    const __m512i words = _mm512_cvtepu32_epi64(_mm256_loadu_si256((const __m256i *)codes));
    const __m512i mask = _mm512_set1_epi64(0x7FFFFFFF);
    const __mmask8 sign = _mm512_cmp_epi64_mask(words, mask, _MM_CMPINT_GT);
    const __m512i address = _mm512_add_epi64(_mm512_set1_epi64((long long)(uintptr_t)base),
        _mm512_slli_epi64(_mm512_and_si512(words, mask), 3));
    _mm512_storeu_si512((void *)rows, address);
    return sign;
}

struct CpuDirectRows8 {
    const pt *base;
    const uint32_t *codes;
    int B;
    bool folded;
    Q8T inline __mmask8 decode(int e, const pt **rows) const {
        const uint32_t *p = folded ? codes + (size_t)(e & 1) * B + (size_t)(e >> 1) * 8
                                   : codes + (size_t)e * 8;
        return cpu_glv_rows8_decode(base, p, rows);
    }
    Q8T inline __mmask8 operator()(int e, const pt **rows) const { return decode(e, rows); }
    Q8T inline void prefetch(int e) const {
        const pt *rows[8]; decode(e, rows);
        for (int j = 0; j < 8; ++j) _mm_prefetch((const char *)rows[j], _MM_HINT_T0);
    }
    Q8T inline __mmask8 load(int e, fe8 &x, fe8 &y) const {
        const pt *rows[8]; const __mmask8 sign = decode(e, rows);
        pt8_load(x, y, rows); return sign;
    }
};

/* Explicit target-qualified functor: C++ lambda call operators do not need
 * to inherit the enclosing function's ISA target for this callback. */
struct CpuNextRows8 {
    CpuDirectRows8 rows;
    const pt **scratch;
    Q8T inline int operator()(int hn, const pt *const **out) const {
        if (rows.folded) {
            rows.decode(2 * hn, scratch); rows.decode(2 * hn + 1, scratch + 8);
            *out = scratch; return 16;
        }
        rows.decode(hn, scratch); *out = scratch; return 8;
    }
};

/* row is the existing digit allocation, now [12][B]. Only the batch selected
 * for GLV+SHA16+actual IFMA EC enters here. Legacy/hybrid scalar batches keep
 * [B][NWMAX]. Local pointer scratch lives across each nxt callback's caller;
 * it is never a pointer to a returned lambda stack frame.
 */
Q8T static void vec_batch_rows(const Ctx *c, const uint32_t *row, const uint8_t *zdig,
        int B, VecBuf &v, bool stage_keys = false) {
    uint32_t *keymsg = nullptr;
#if QCPU_SHANI && QSB_CPU_DIRECT_KEYHASH
    if (stage_keys) keymsg = (uint32_t *)(void *)v.qx;
#endif
    const int G = B / 8, pf = c->pf.load(std::memory_order_relaxed);
    memcpy(v.bad, zdig, (size_t)B);
    const pt *next_rows[16];
    {
        const CpuDirectRows8 first = {c->table + c->woff[0], row, B, false};
        CpuNextRows8 nxt = {{c->table + c->woff[1], row + B, B, false}, next_rows};
        ec8_first(v.X, v.Y, G, pf, first, nxt);
    }
    for (int i = 1; i < 10; ++i) {
        if (i == 5) {
            fe beta; memcpy(beta.v, CPU_GLV_BETA_LE64, sizeof beta.v);
            fe8 beta8; fe8_bcast(beta8, beta);
            for (int h = 0; h < G; ++h) fe8_mul(v.X[h], v.X[h], beta8);
        }
        const CpuDirectRows8 rows = {c->table + c->woff[i], row + (size_t)i * B, B, false};
        const bool final_next = i == 9;
        CpuNextRows8 nxt = {{final_next ? c->cfold : c->table + c->woff[i + 1],
                            row + (size_t)(i + 1) * B, B, final_next}, next_rows};
        ec8_window_rows(v.X, v.Y, v.D, v.P, G, pf, rows, nxt);
    }
    const CpuDirectRows8 rows = {c->cfold, row + (size_t)10 * B, B, true};
    ec8_final_fold_rows(v.X, v.Y, v.D, v.P, G, pf, rows, v.qx, v.qp, keymsg);
}
