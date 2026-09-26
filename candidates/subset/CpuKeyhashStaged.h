#pragma once
/* Include inside namespace qcpu, after the final EC kernels and sha16 helpers.
 * The worker consumes this staging only on the selected IFMA + SHA16 route.
 *
 * kw is 64-byte aligned and contains [B/8][9][16] uint32 message words.
 * Within each vector: lanes 0..7 are recid 0 for candidates h*8..h*8+7;
 * lanes 8..15 are recid 1 for those same candidates. B is a multiple of 32.
 * Only W0..W8 are staged; the fixed 33-byte SHA-256 padding is broadcast.
 * Output h0 retains the existing [candidate][recid] interleaved ordering.
 *
 * Call after the EC batch returns. Calling this in ec8_final_fold's reverse
 * loop would overlap/spill its four live 5-limb inversion chains.
 */
#if QCPU_VEC && QCPU_SHANI
S16T static void keyhash16_staged(const uint32_t *kw, int B, uint32_t *h0) {
    alignas(64) __m512i wk[64];
    const __m512i interleave = _mm512_setr_epi32(
        0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
    for (int h = 0; h < B / 8; h++) {
        const uint32_t *src = kw + (size_t)h * 9 * 16;
        __m512i w[16];
        for (int i = 0; i < 9; i++)
            w[i] = _mm512_load_si512((const void *)(src + (size_t)i * 16));
        for (int i = 9; i < 15; i++) w[i] = _mm512_setzero_si512();
        w[15] = _mm512_set1_epi32(264);
        sha16_sched(wk, w);
        __m512i st[8]; sha16_iv(st); sha16_soa(st, wk);
        _mm512_storeu_si512((void *)(h0 + (size_t)h * 16),
                           _mm512_permutexvar_epi32(interleave, st[0]));
    }
}
/* Run once after table construction, before workers. The two format modes
 * use the actual final EC kernel, then compare all 64 key hashes with OpenSSL.
 * No counters, candidate enumeration, or publication functions are touched. */
static bool keyhash_staged_selfcheck(Ctx *c) {
    if (!c || !c->vec || !c->s16 || c->nw < 1 || c->nw > NWMAX || !c->table) return false;
    struct Owner {
        VecBuf v;
        ~Owner() {
            free(v.X); free(v.Y); free(v.D); free(v.P); free(v.qx); free(v.qp); free(v.bad);
            free(v.rb[0]); free(v.rb[1]); free(v.nb[0]); free(v.nb[1]);
        }
    } own;
    try {
        const int B = 32;
        if (!vecbuf_alloc(own.v, B)) return false;
        uint32_t dig[B * NWMAX] = {0}; uint8_t bad[B] = {0};
        uint32_t expected[2 * B], legacy[2 * B], staged[2 * B];
        uint64_t rng = 0x1F83D9ABFB41BD6BULL;
        for (int k = 0; k < B; ++k) {
            for (int i = 0; i < c->nw; ++i) {
                if (!c->went[i] || c->went[i] > 0x7FFFFFFFu) return false;
                rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
                const uint64_t r = rng * 0x2545F4914F6CDD1DULL;
                dig[(size_t)k * NWMAX + i] = (uint32_t)(r % c->went[i] + 1) | (uint32_t)(r >> 63) << 31;
            }
            if (c->glv && (k & 7) == 0) dig[(size_t)k * NWMAX + c->nw - 1] = 1; /* joint zero pair is finite +/-C */
        }
        vec_batch(c, dig, bad, B, own.v, false);
        for (int e = 0; e < 2 * B; ++e) {
            uint8_t pk[33], digest[32];
            pk[0] = (uint8_t)(2u | (own.v.qp[e] & 1u));
            for (int j = 0; j < 32; ++j)
                pk[j + 1] = (uint8_t)(own.v.qx[e].v[3 - (j >> 3)] >> (56 - 8 * (j & 7)));
            if (!SHA256(pk, sizeof pk, digest)) return false;
            expected[e] = (uint32_t)digest[0] << 24 | (uint32_t)digest[1] << 16 |
                          (uint32_t)digest[2] << 8 | (uint32_t)digest[3];
        }
        keyhash16(own.v.qx, own.v.qp, 2 * B, legacy);
        for (int e = 0; e < 2 * B; ++e) if (legacy[e] != expected[e]) return false;
        vec_batch(c, dig, bad, B, own.v, true); /* reuses qx backing as message words */
        keyhash16_staged((const uint32_t *)(const void *)own.v.qx, B, staged);
        for (int e = 0; e < 2 * B; ++e) if (staged[e] != expected[e]) return false;
#if QSB_CPU_FUSED_KEYHASH
        if (c->keyfused) {
            uint32_t fused[2 * B], pairs[2 * B];
            bool ok = keyhash_fused_mask_selfcheck();
            keyhash16_fused_all_h0((const uint32_t *)(const void *)own.v.qx, B, fused);
            for (int e = 0; e < 2 * B; ++e) if (fused[e] != expected[e]) ok = false;
            const int nm = keyhash16_fused_sparse((const uint32_t *)(const void *)own.v.qx, B, pairs);
            int seen = 0;
            for (int base = 0; base < B; base += 8) {
                unsigned mask = 0;
                for (int ri = 0; ri < 2; ++ri) for (int j = 0; j < 8; ++j)
                    if (pk_prefilter(expected[2 * (base + j) + ri])) mask |= 1u << (8 * ri + j);
                if (mask) {
                    if (seen >= nm || pairs[2 * seen] != (unsigned)base || pairs[2 * seen + 1] != mask) ok = false;
                    ++seen;
                }
            }
            if (seen != nm) ok = false;
            if (!ok) {
                c->keyfused = false;
                printf("  CPU co-grind: streaming key-hash self-check failed; using staged SHA\n");
            }
        }
#endif
        return true;
    } catch (...) { return false; }
}
#endif
