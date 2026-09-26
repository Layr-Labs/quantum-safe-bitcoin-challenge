#pragma once
/* Include inside namespace qcpu before ec8_final/ec8_final_fold.
 * xw[0..3] and ylow MUST come from the existing fe8_canon_words:
 *   fe8_canon_words(xw, x3); fe8_canon_words(yw, y3);
 *   cpu_pkwords8_store(dst, ri, xw, yw[0]);
 * Each __m512i contains eight uint64 lanes for eight consecutive candidates.
 * dst is one group's 9x16 numeric SHA-word slab. Each row is recid-major:
 *   [recid0 candidate0..7, recid1 candidate0..7].
 * The current reverse loop may emit recid1 before recid0. Both write disjoint halves.
 * No field reduction, parity shortcut, scalar coordinate store, or byte swap is used.
 */
static inline __attribute__((always_inline, target("avx512f")))
void cpu_pkwords8_store(uint32_t *dst, int ri, const __m512i xw[4], const __m512i ylow) {
    const __m512i v0 = xw[0], v1 = xw[1], v2 = xw[2], v3 = xw[3];
    const __m512i parity = _mm512_and_si512(ylow, _mm512_set1_epi64(1));
    const __m512i prefix = _mm512_slli_epi64(_mm512_or_si512(parity, _mm512_set1_epi64(2)), 24);
    uint32_t *half = dst + (size_t)ri * 8;
#define CPU_PK_STORE(I, V) _mm256_storeu_si256((__m256i *)(void *)(half + (size_t)(I) * 16), _mm512_cvtepi64_epi32(V))
    /* Narrowing truncates every uint64 lane to low32 in the same candidate order. */
    CPU_PK_STORE(0, _mm512_or_si512(prefix, _mm512_srli_epi64(v3, 40)));
    CPU_PK_STORE(1, _mm512_srli_epi64(v3, 8));
    CPU_PK_STORE(2, _mm512_or_si512(_mm512_slli_epi64(v3, 24), _mm512_srli_epi64(v2, 40)));
    CPU_PK_STORE(3, _mm512_srli_epi64(v2, 8));
    CPU_PK_STORE(4, _mm512_or_si512(_mm512_slli_epi64(v2, 24), _mm512_srli_epi64(v1, 40)));
    CPU_PK_STORE(5, _mm512_srli_epi64(v1, 8));
    CPU_PK_STORE(6, _mm512_or_si512(_mm512_slli_epi64(v1, 24), _mm512_srli_epi64(v0, 40)));
    CPU_PK_STORE(7, _mm512_srli_epi64(v0, 8));
    CPU_PK_STORE(8, _mm512_or_si512(_mm512_slli_epi64(v0, 24), _mm512_set1_epi64(0x00800000)));
#undef CPU_PK_STORE
}
