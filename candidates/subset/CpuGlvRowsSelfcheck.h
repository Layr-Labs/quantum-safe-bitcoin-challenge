#pragma once
/* Run only after the original GLV/OpenSSL and raw recoder checks. Compare
 * both address decoding and actual EC output with the retained digit path.
 * Scratch is owned and freed; no candidate/gate/counter state is touched. */
Q8T static __attribute__((noinline)) bool cpu_glv_rows_ec_selfcheck(const Ctx *c) {
    if (!c || !c->glv || !c->vec || !c->cfold || c->nw != 11 ||
        c->went[10] != CPU_GLV_JOINT_ENTRIES) return false;
    try {
        const int B = 32;
        uint32_t dig[B * NWMAX] = {0}, row[12 * B] = {0};
        uint8_t bad[B]; alignas(64) uint64_t raw[4][8];
        uint64_t rng = 0x6A09E667F3BCC909ULL;
        for (int k = 0; k < B; ++k) {
            uint32_t words[8];
            for (int j = 0; j < 4; ++j) {
                rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
                const uint64_t w = rng * 0x2545F4914F6CDD1DULL;
                words[2 * j] = (uint32_t)(w >> 32); words[2 * j + 1] = (uint32_t)w;
                raw[3 - j][k & 7] = w;
            }
            bad[k] = (uint8_t)cpu_glv_digits(dig + (size_t)k * NWMAX, words);
            if ((k & 7) == 7) {
                const unsigned mask = cpu_glv_rowcodes8_raw(&raw[0][0], 255, 0, row + k - 7, B);
                for (int j = 0; j < 8; ++j) if (((mask >> j) & 1u) != bad[k - 7 + j]) return false;
            }
        }
        for (int i = 0; i < 12; ++i) for (int h = 0; h < B / 8; ++h) {
            const pt *base = i < 10 ? c->table + c->woff[i] : c->cfold;
            const pt *rows[8];
            const unsigned sign = cpu_glv_rows8_decode(base, row + (size_t)i * B + h * 8, rows);
            for (int j = 0; j < 8; ++j) {
                const uint32_t d = dig[(size_t)(h * 8 + j) * NWMAX + (i < 10 ? i : 10)];
                const uint32_t a = d & 0x7FFFFFFFu, sg = d >> 31;
                size_t index = (a ? a : 1) - 1;
                if (i >= 10) index += (sg ^ (unsigned)(i - 10)) * c->went[10];
                if (rows[j] != base + index || ((sign >> j) & 1u) != sg) return false;
            }
        }
        CpuGlvSelfcheckState old, next;
        if (!vecbuf_alloc(old.vec, B) || !vecbuf_alloc(next.vec, B)) return false;
        vec_batch(c, dig, bad, B, old.vec);
        vec_batch_rows(c, row, bad, B, next.vec);
        if (memcmp(old.vec.bad, next.vec.bad, B) ||
            memcmp(old.vec.qx, next.vec.qx, (size_t)2 * B * sizeof(fe)) ||
            memcmp(old.vec.qp, next.vec.qp, (size_t)2 * B)) return false;
#if QSB_CPU_DIRECT_KEYHASH
        vec_batch(c, dig, bad, B, old.vec, true);
        vec_batch_rows(c, row, bad, B, next.vec, true);
        if (memcmp(old.vec.bad, next.vec.bad, B) ||
            memcmp(old.vec.qx, next.vec.qx, (size_t)2 * B * 9 * sizeof(uint32_t))) return false;
#endif
        return true;
    } catch (...) { return false; }
}
