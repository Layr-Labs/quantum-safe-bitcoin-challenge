#pragma once
/* Include inside qcpu after epoch16 and CpuGlvVector.h. Startup only: the
 * caller has already established IFMA, SHA16, fast tables and GLV eligibility.
 * This check owns all scratch, never publishes, and never changes Ctx state.
 * Failure disables only the new vector recoder at the call site. */
#if QCPU_VEC && QCPU_SHANI && QSB_CPU_GLV_VECTOR

static const uint64_t CPU_GLV8_CHECK_PROBES[32][4] = {
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x000000000000007FULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000080ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000081ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x00000000000000FFULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000100ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000101ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x000FFFFFFFFFFFFFULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0010000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
    {0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
    {0xFFFFFFFFFFFFFFFFULL, 0x000000FFFFFFFFFFULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000010000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000001ULL, 0x0000000000000000ULL},
    {0xBFD25E8CD0364140ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL},
    {0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL},
    {0xBFD25E8CD0364142ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL},
    {0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL},
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL},
    {0xACAFB2D52683C736ULL, 0x4A3C81BC2D093500ULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL},
    {0xACAFB2D52683C737ULL, 0x4A3C81BC2D093500ULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL},
    {0xACAFB2D52683C738ULL, 0x4A3C81BC2D093500ULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL},
    {0x30D9376D9CB9848CULL, 0x09AFC03D504C8F66ULL, 0x415C6D329C9B8CCFULL, 0xDEA979BBAAA1C174ULL},
    {0x30D9376D9CB9848DULL, 0x09AFC03D504C8F66ULL, 0x415C6D329C9B8CCFULL, 0xDEA979BBAAA1C174ULL},
    {0x30D9376D9CB9848EULL, 0x09AFC03D504C8F66ULL, 0x415C6D329C9B8CCFULL, 0xDEA979BBAAA1C174ULL},
    {0xC02E87FA86ABCFDEULL, 0x8D100AA8CEBB30F7ULL, 0x5F0673AE0AD5BD01ULL, 0x9C30AB654D0458ADULL},
    {0xC02E87FA86ABCFDFULL, 0x8D100AA8CEBB30F7ULL, 0x5F0673AE0AD5BD01ULL, 0x9C30AB654D0458ADULL},
    {0xC02E87FA86ABCFE0ULL, 0x8D100AA8CEBB30F7ULL, 0x5F0673AE0AD5BD01ULL, 0x9C30AB654D0458ADULL},
    {0x8B04358F9FFCF192ULL, 0x8641546BF721B36AULL, 0x45365CC308FAC404ULL, 0xD9A55415B295EE32ULL},
    {0x95F4E9B49F9DB193ULL, 0xE12661ABFE98CABEULL, 0x8EB261669BFA91C8ULL, 0xF4D1D7EDCD4CA8FBULL}
};

/* Deliberately bypass the optimized scalar coefficient/residual helpers. */
static unsigned cpu_glv8_check_reference(const uint64_t input[4], uint32_t out[11]) {
    uint64_t r1[2], r2[2]; unsigned n1, n2;
    cpu_glv_split_reference(input, r1, r2, &n1, &n2);
    const int low2 = cpu_glv_recode(out, r2, n2);
    const int low1 = cpu_glv_recode(out + 5, r1, n1);
    out[10] = cpu_glv_joint_code(low1, low2);
    unsigned bad = 0;
    for (int i = 0; i < 10; ++i) bad |= (unsigned)((out[i] & 0x7FFFFFFFu) == 0);
    return bad;
}

static bool cpu_glv8_check_group(const uint64_t *raw, unsigned active,
                                unsigned off, unsigned stride) {
    if (off > 7 || stride < 11 || stride > 19 || (active & ~255u) ||
        (active & ((1u << off) - 1u))) return false;
    /* A real, valid destination base. Negative indices belong only to masked
     * lanes; canaries cover the prefix, unused rows/columns, and suffix. */
    const unsigned PAD = 16, COUNT = 2 * PAD + 8 * 19;
    uint32_t got[COUNT], want[COUNT];
    uint64_t input_copy[32];
    for (unsigned i = 0; i < 32; ++i) input_copy[i] = raw[i];
    for (unsigned i = 0; i < COUNT; ++i) got[i] = want[i] = 0xA5C39E71u;
    unsigned expected_bad = 0;
    for (unsigned lane = 0; lane < 8; ++lane) if (active & (1u << lane)) {
        uint64_t k[4];
        for (int w = 0; w < 4; ++w) k[w] = raw[w * 8 + lane];
        expected_bad |= cpu_glv8_check_reference(k, want + PAD + (lane - off) * stride) << lane;
    }
    const unsigned bad = cpu_glv_digits8_raw(raw, active, off, got + PAD, stride);
    if (bad != expected_bad) return false; /* includes inactive bad-mask bits */
    for (unsigned i = 0; i < COUNT; ++i) if (got[i] != want[i]) return false;
    for (unsigned i = 0; i < 32; ++i) if (raw[i] != input_copy[i]) return false;
    return true;
}

static uint64_t cpu_glv8_check_random(uint64_t &s) {
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return s * 0x2545F4914F6CDD1DULL;
}

struct CpuGlv8CheckScratch {
    void *raw = nullptr, *dg = nullptr, *vs = nullptr;
    ~CpuGlv8CheckScratch() { free(raw); free(dg); free(vs); }
};

static bool cpu_glv_vector_selfcheck(const Ctx *c) {
    if (!c || !c->vec || !c->glv || !c->fast || !c->s16 || c->nw != 11 ||
        c->ncwin < 1 || c->ncwin > 286 || c->nvar < 1 || c->nvar > c->ncwin ||
        c->npc != (c->ncwin + 15) / 16 || c->nvc != (c->nvar + 15) / 16 ||
        c->nblk < 1 || c->nblk > 15 || c->remlen < 0 || c->remlen > 63 ||
        !c->v16 || !c->p16 || c->pvar.size() < (size_t)c->ncwin ||
        c->blk_const.size() < (size_t)c->nblk ||
        c->pwk.size() < (size_t)c->ncwin * c->nblk) return false;
    for (int p = 0; p < c->ncwin; ++p) if (c->pvar[p] >= c->nvar) return false;
    for (int b = 0; b < c->nblk; ++b) if (!c->pwk[b]) return false;
    try {
        alignas(64) uint64_t raw[4][8];
        for (unsigned base = 0; base < 32; base += 8) {
            for (unsigned lane = 0; lane < 8; ++lane)
                for (int w = 0; w < 4; ++w) raw[w][lane] = CPU_GLV8_CHECK_PROBES[base + lane][w];
            if (!cpu_glv8_check_group(&raw[0][0], 255, 0, 16)) return false;
        }
        uint64_t rng = 0xCBBB9D5DC1059ED8ULL;
        for (int group = 0; group < 8; ++group) {
            for (int lane = 0; lane < 8; ++lane)
                for (int w = 0; w < 4; ++w) raw[w][lane] = cpu_glv8_check_random(rng);
            if (!cpu_glv8_check_group(&raw[0][0], 255, 0, 16)) return false;
        }
        const unsigned mixed[8] = {0, 22, 24, 27, 17, 20, 30, 31};
        for (unsigned rot = 0; rot < 8; ++rot) {
            for (unsigned lane = 0; lane < 8; ++lane)
                for (int w = 0; w < 4; ++w) raw[w][lane] = CPU_GLV8_CHECK_PROBES[mixed[(lane + rot) & 7u]][w];
            if (!cpu_glv8_check_group(&raw[0][0], 255, 0, 16)) return false;
        }
        /* All masks, including every contiguous interval and sparse holes.
         * The full-mask rotations above place both guards and k* in each lane. */
        for (unsigned mask = 0; mask < 256; ++mask) {
            for (unsigned lane = 0; lane < 8; ++lane)
                for (int w = 0; w < 4; ++w) raw[w][lane] = CPU_GLV8_CHECK_PROBES[(mask + lane) & 31u][w];
            unsigned off = 0;
            if (mask) while (!(mask & (1u << off))) ++off;
            const unsigned strides[3] = {11, 16, 19};
            if (!cpu_glv8_check_group(&raw[0][0], mask, off, strides[mask % 3])) return false;
        }
        for (unsigned off = 0; off < 8; ++off) {
            /* Empty masks must return zero without any store. Keep pointers
             * valid so a failed check can safely disable the feature. */
            if (!cpu_glv8_check_group(&raw[0][0], 0, off, 16) ||
                !cpu_glv8_check_group(&raw[0][0], 128, off, 19)) return false;
        }

        CpuGlv8CheckScratch s;
        const size_t raw_words = (size_t)c->npc * 64, dg_words = (size_t)c->npc * 128;
        const size_t vs_words = (size_t)c->nvc * 128;
        /* Each working pointer has 64-byte alignment and 64-byte canaries on
         * both sides. The same vs scratch is reused by the two epoch modes. */
        if (posix_memalign(&s.raw, 64, (raw_words + 16) * sizeof(uint64_t)) ||
            posix_memalign(&s.dg, 64, (dg_words + 32) * sizeof(uint32_t)) ||
            posix_memalign(&s.vs, 64, (vs_words + 32) * sizeof(uint32_t))) return false;
        uint64_t *rp = (uint64_t *)s.raw, *keys = rp + 8;
        uint32_t *dp = (uint32_t *)s.dg, *dg = dp + 16;
        uint32_t *vp = (uint32_t *)s.vs, *vs = vp + 16;
        const uint64_t RPOISON = 0xD4A317C96B02E85FULL;
        const uint32_t DPOISON = 0x6B02E85Fu;
        for (int trial = 0; trial < 3; ++trial) {
            uint32_t est[8]; uint8_t erem[64];
            for (int i = 0; i < 8; ++i)
                est[i] = trial == 0 ? qsha_iv[i] : (uint32_t)cpu_glv8_check_random(rng);
            for (int i = 0; i < 64; ++i) erem[i] = (uint8_t)(i * (trial * 2 + 1) + trial * 83);
            for (size_t i = 0; i < raw_words + 16; ++i) rp[i] = RPOISON;
            for (size_t i = 0; i < dg_words + 32; ++i) dp[i] = DPOISON;
            for (size_t i = 0; i < vs_words + 32; ++i) vp[i] = DPOISON;
            epoch16<false>(c, est, erem, (uint32_t (*)[8])dg, vs, keys);
            for (size_t i = 0; i < raw_words + 16; ++i) if (rp[i] != RPOISON) return false;
            for (size_t i = 0; i < 16; ++i) if (dp[i] != DPOISON || dp[16 + dg_words + i] != DPOISON ||
                vp[i] != DPOISON || vp[16 + vs_words + i] != DPOISON) return false;
            for (size_t i = (size_t)c->ncwin * 8; i < dg_words; ++i) if (dg[i] != DPOISON) return false;
            const std::vector<uint32_t> old_digest(dp, dp + dg_words + 32);
            epoch16<true>(c, est, erem, (uint32_t (*)[8])dg, vs, keys);
            for (size_t i = 0; i < dg_words + 32; ++i) if (dp[i] != old_digest[i]) return false;
            for (size_t i = 0; i < 8; ++i) if (rp[i] != RPOISON || rp[8 + raw_words + i] != RPOISON) return false;
            for (size_t i = 0; i < 16; ++i) if (vp[i] != DPOISON || vp[16 + vs_words + i] != DPOISON) return false;
            for (int p = 0; p < c->npc * 16; ++p) {
                const int source = p < c->ncwin ? p : c->ncwin - 1;
                for (int w = 0; w < 4; ++w) {
                    const uint64_t want = ((uint64_t)dg[(size_t)source * 8 + 6 - 2 * w] << 32) |
                        dg[(size_t)source * 8 + 7 - 2 * w];
                    if (keys[(size_t)(p >> 3) * 32 + w * 8 + (p & 7)] != want) return false;
                }
            }
            for (int p = 0; p < c->ncwin; p += 8) {
                const unsigned take = c->ncwin - p < 8 ? (unsigned)(c->ncwin - p) : 8u;
                if (!cpu_glv8_check_group(keys + (size_t)(p >> 3) * 32, (1u << take) - 1u, 0, 16)) return false;
            }
        }
        return true;
    } catch (...) {
        return false; /* vector allocation failure releases all aligned scratch */
    }
}
#if QSB_CPU_GLV_ROWS
/* This check is independent of the old digits check above. A failure must
 * disable only the row-output dispatch, keeping the old vector recoder. */
static bool cpu_glv8_check_rows(const uint64_t *raw, unsigned active,
    unsigned off, unsigned stride, unsigned base, uint32_t *got, uint32_t *want) {
    if (off > 7 || stride < 8 || stride > QSB_CPU_BATCH || base > stride - 8 ||
        (active & ~255u) || (active & ((1u << off) - 1u))) return false;
    const size_t PAD = 16, count = 2 * PAD + (size_t)12 * stride;
    uint64_t input_copy[32];
    for (unsigned i = 0; i < 32; ++i) input_copy[i] = raw[i];
    for (size_t i = 0; i < count; ++i) got[i] = want[i] = 0xA5C39E71u;
    unsigned expected_bad = 0;
    for (unsigned lane = 0; lane < 8; ++lane) if (active & (1u << lane)) {
        uint64_t k[4]; uint32_t di[11];
        for (unsigned w = 0; w < 4; ++w) k[w] = raw[w * 8 + lane];
        expected_bad |= cpu_glv8_check_reference(k, di) << lane;
        const size_t dest = PAD + base + lane - off;
        for (unsigned i = 0; i < 12; ++i) {
            const uint32_t d = di[i < 10 ? i : 10], mag = d & 0x7FFFFFFFu;
            const uint32_t sign = d >> 31;
            uint64_t row = (uint64_t)(mag ? mag - 1u : 0u) * 8u;
            if (i >= 10) row += (uint64_t)(sign ^ (i - 10)) * CPU_GLV_JOINT_ENTRIES * 8u;
            if (row >= 0x80000000ULL) return false;
            want[dest + (size_t)i * stride] = (uint32_t)row | (sign << 31);
        }
    }
    const unsigned bad = cpu_glv_rowcodes8_raw(raw, active, off, got + PAD + base, stride);
    if (bad != expected_bad) return false;
    for (size_t i = 0; i < count; ++i) if (got[i] != want[i]) return false;
    for (unsigned i = 0; i < 32; ++i) if (raw[i] != input_copy[i]) return false;
    return true;
}

static bool cpu_glv_rowcodes_selfcheck(const Ctx *c) {
    if (!c || !c->vec || !c->glv || !c->fast || !c->s16 || c->nw != 11 || QSB_CPU_BATCH < 19 ||
        c->ncwin < 1 || c->ncwin > 286 || c->nvar < 1 || c->nvar > c->ncwin ||
        c->npc != (c->ncwin + 15) / 16 || c->nvc != (c->nvar + 15) / 16 ||
        c->nblk < 1 || c->nblk > 15 || c->remlen < 0 || c->remlen > 63 ||
        !c->v16 || !c->p16 || c->pvar.size() < (size_t)c->ncwin ||
        c->blk_const.size() < (size_t)c->nblk ||
        c->pwk.size() < (size_t)c->ncwin * c->nblk) return false;
    for (int p = 0; p < c->ncwin; ++p) if (c->pvar[p] >= c->nvar) return false;
    for (int b = 0; b < c->nblk; ++b) if (!c->pwk[b]) return false;
    /* Row units occupy only low31. Check the actual shape, not a fixed
     * historical top-window bound; joint (0,0) still has a valid row zero. */
    for (int i = 0; i < 10; ++i)
        if (!c->went[i] || ((uint64_t)c->went[i] - 1) * 8 >= 0x80000000ULL) return false;
    if (c->went[10] != CPU_GLV_JOINT_ENTRIES ||
        ((uint64_t)2 * CPU_GLV_JOINT_ENTRIES - 1) * 8 >= 0x80000000ULL) return false;
    try {
        std::vector<uint32_t> got((size_t)12 * QSB_CPU_BATCH + 32), want(got.size());
        alignas(64) uint64_t raw[4][8];
        for (unsigned base = 0; base < 32; base += 8) {
            for (unsigned lane = 0; lane < 8; ++lane)
                for (unsigned w = 0; w < 4; ++w) raw[w][lane] = CPU_GLV8_CHECK_PROBES[base + lane][w];
            if (!cpu_glv8_check_rows(&raw[0][0], 255, 0, QSB_CPU_BATCH, base & 7u, got.data(), want.data())) return false;
        }
        uint64_t rng = 0xCBBB9D5DC1059ED8ULL;
        for (unsigned group = 0; group < 8; ++group) {
            for (unsigned lane = 0; lane < 8; ++lane)
                for (unsigned w = 0; w < 4; ++w) raw[w][lane] = cpu_glv8_check_random(rng);
            if (!cpu_glv8_check_rows(&raw[0][0], 255, 0, QSB_CPU_BATCH, group, got.data(), want.data())) return false;
        }
        const unsigned mixed[8] = {0, 22, 24, 27, 17, 20, 30, 31};
        for (unsigned rot = 0; rot < 8; ++rot) {
            for (unsigned lane = 0; lane < 8; ++lane)
                for (unsigned w = 0; w < 4; ++w) raw[w][lane] = CPU_GLV8_CHECK_PROBES[mixed[(lane + rot) & 7u]][w];
            if (!cpu_glv8_check_rows(&raw[0][0], 255, 0, QSB_CPU_BATCH, QSB_CPU_BATCH - 8, got.data(), want.data())) return false;
        }
        for (unsigned mask = 0; mask < 256; ++mask) {
            for (unsigned lane = 0; lane < 8; ++lane)
                for (unsigned w = 0; w < 4; ++w) raw[w][lane] = CPU_GLV8_CHECK_PROBES[(mask + lane) & 31u][w];
            unsigned off = 0;
            if (mask) while (!(mask & (1u << off))) ++off;
            if (!cpu_glv8_check_rows(&raw[0][0], mask, off, 19, mask % 12, got.data(), want.data())) return false;
        }
        for (unsigned off = 0; off < 8; ++off)
            if (!cpu_glv8_check_rows(&raw[0][0], 0, off, QSB_CPU_BATCH, QSB_CPU_BATCH - 8, got.data(), want.data()) ||
                !cpu_glv8_check_rows(&raw[0][0], 128, off, QSB_CPU_BATCH, QSB_CPU_BATCH - 8, got.data(), want.data())) return false;

        CpuGlv8CheckScratch s;
        const size_t raw_words = (size_t)c->npc * 64, vs_words = (size_t)c->nvc * 128;
        const size_t dg_words = (size_t)c->npc * 128;
        if (posix_memalign(&s.raw, 64, (raw_words + 16) * sizeof(uint64_t)) ||
            posix_memalign(&s.dg, 64, (dg_words + 32) * sizeof(uint32_t)) ||
            posix_memalign(&s.vs, 64, (vs_words + 32) * sizeof(uint32_t))) return false;
        uint64_t *rp = (uint64_t *)s.raw, *keys = rp + 8;
        uint32_t *dp = (uint32_t *)s.dg, *dg = dp + 16;
        uint32_t *vp = (uint32_t *)s.vs, *vs = vp + 16;
        const uint64_t RPOISON = 0xD4A317C96B02E85FULL;
        const uint32_t DPOISON = 0x6B02E85Fu;
        for (size_t i = 0; i < raw_words + 16; ++i) rp[i] = RPOISON;
        for (size_t i = 0; i < dg_words + 32; ++i) dp[i] = DPOISON;
        for (size_t i = 0; i < vs_words + 32; ++i) vp[i] = DPOISON;
        uint32_t est[8]; uint8_t erem[64];
        for (unsigned i = 0; i < 8; ++i) est[i] = qsha_iv[i];
        for (unsigned i = 0; i < 64; ++i) erem[i] = (uint8_t)(3 * i + 83);
        epoch16<true>(c, est, erem, (uint32_t (*)[8])dg, vs, keys);
        for (size_t i = 0; i < dg_words + 32; ++i) if (dp[i] != DPOISON) return false;
        for (size_t i = 0; i < 8; ++i) if (rp[i] != RPOISON || rp[8 + raw_words + i] != RPOISON) return false;
        for (size_t i = 0; i < 16; ++i) if (vp[i] != DPOISON || vp[16 + vs_words + i] != DPOISON) return false;
        for (int p = 0; p < c->ncwin; p += 8) {
            const unsigned take = c->ncwin - p < 8 ? (unsigned)(c->ncwin - p) : 8u;
            if (!cpu_glv8_check_rows(keys + (size_t)(p >> 3) * 32, (1u << take) - 1u, 0,
                    QSB_CPU_BATCH, ((unsigned)p * 7) % (QSB_CPU_BATCH - 7), got.data(), want.data())) return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}
#endif /* QSB_CPU_GLV_ROWS */
#endif
