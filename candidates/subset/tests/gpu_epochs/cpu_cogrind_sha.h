/* cpu_cogrind_sha.h -- SHA-256 compression for the host co-grinder, two independent messages
 * interleaved on the SHA-NI unit.
 *
 * One SHA256RNDS2 depends on the previous one, so a single message runs at the instruction's
 * latency; two independent messages issue back to back and run at its throughput. The co-grinder
 * always has pairs of independent messages (neighbouring window triples of an epoch, the two
 * recids of a candidate), so every hot compression goes through qcg_sha2x(). Round structure,
 * constants and the ABEF/CDGH state packing are the standard SHA-NI formulation (FIPS 180-4
 * rounds; register layout as in Intel's SHA extensions reference code). Compiled with a function
 * target attribute and called only when CPUID reports SHA; otherwise both entry points fall
 * back to OpenSSL's SHA256_Transform, the path e5b67ed2 used for every block.
 */
#ifndef QSB_CPU_COGRIND_SHA_H
#define QSB_CPU_COGRIND_SHA_H
#include <cpuid.h>
#include <immintrin.h>

namespace qcg {

static const uint32_t SHA_K[64] __attribute__((aligned(16))) = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};

static int g_sha_ni = -1;   /* -1 unknown, 0 no, 1 yes; set once in start() */
static inline int sha_ni_available() {
    unsigned a, b, c, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return 0;
    if (!(b & (1u << 29))) return 0;                                   /* SHA */
    if (!__get_cpuid(1, &a, &b, &c, &d)) return 0;
    return (c & (1u << 19)) && (c & (1u << 9));                        /* SSE4.1, SSSE3 */
}

#define QCG_SHA_TGT __attribute__((target("sha,sse4.1,ssse3")))

/* n blocks of two independent messages: sa <- compress*(sa, pa[0..64n)), sb likewise. */
static QCG_SHA_TGT __attribute__((noinline))
void sha2x_ni(uint32_t sa[8], const uint8_t *pa, uint32_t sb[8], const uint8_t *pb, size_t n) {
    const __m128i MASK = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
    __m128i t, a0, a1, b0, b1;
    t = _mm_shuffle_epi32(_mm_loadu_si128((const __m128i *)&sa[0]), 0xB1);      /* CDAB */
    a1 = _mm_shuffle_epi32(_mm_loadu_si128((const __m128i *)&sa[4]), 0x1B);     /* EFGH */
    a0 = _mm_alignr_epi8(t, a1, 8);                                             /* ABEF */
    a1 = _mm_blend_epi16(a1, t, 0xF0);                                          /* CDGH */
    t = _mm_shuffle_epi32(_mm_loadu_si128((const __m128i *)&sb[0]), 0xB1);
    b1 = _mm_shuffle_epi32(_mm_loadu_si128((const __m128i *)&sb[4]), 0x1B);
    b0 = _mm_alignr_epi8(t, b1, 8);
    b1 = _mm_blend_epi16(b1, t, 0xF0);
    for (size_t blk = 0; blk < n; blk++, pa += 64, pb += 64) {
        const __m128i sa0 = a0, sa1 = a1, sb0 = b0, sb1 = b1;
        __m128i wa[4], wb[4];
        for (int i = 0; i < 4; i++) {
            wa[i] = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(pa + 16 * i)), MASK);
            wb[i] = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(pb + 16 * i)), MASK);
        }
#pragma GCC unroll 16
        for (int r = 0; r < 16; r++) {
            const __m128i k = _mm_load_si128((const __m128i *)&SHA_K[4 * r]);
            __m128i ma = _mm_add_epi32(wa[r & 3], k), mb = _mm_add_epi32(wb[r & 3], k);
            a1 = _mm_sha256rnds2_epu32(a1, a0, ma);
            b1 = _mm_sha256rnds2_epu32(b1, b0, mb);
            ma = _mm_shuffle_epi32(ma, 0x0E);
            mb = _mm_shuffle_epi32(mb, 0x0E);
            a0 = _mm_sha256rnds2_epu32(a0, a1, ma);
            b0 = _mm_sha256rnds2_epu32(b0, b1, mb);
            if (r < 12) {   /* schedule words 4(r+4) .. 4(r+4)+3 into the slot of group r */
                wa[r & 3] = _mm_sha256msg2_epu32(_mm_add_epi32(_mm_sha256msg1_epu32(wa[r & 3], wa[(r + 1) & 3]),
                                                               _mm_alignr_epi8(wa[(r + 3) & 3], wa[(r + 2) & 3], 4)),
                                                 wa[(r + 3) & 3]);
                wb[r & 3] = _mm_sha256msg2_epu32(_mm_add_epi32(_mm_sha256msg1_epu32(wb[r & 3], wb[(r + 1) & 3]),
                                                               _mm_alignr_epi8(wb[(r + 3) & 3], wb[(r + 2) & 3], 4)),
                                                 wb[(r + 3) & 3]);
            }
        }
        a0 = _mm_add_epi32(a0, sa0); a1 = _mm_add_epi32(a1, sa1);
        b0 = _mm_add_epi32(b0, sb0); b1 = _mm_add_epi32(b1, sb1);
    }
    t = _mm_shuffle_epi32(a0, 0x1B);                                            /* FEBA */
    a1 = _mm_shuffle_epi32(a1, 0xB1);                                           /* DCHG */
    _mm_storeu_si128((__m128i *)&sa[0], _mm_blend_epi16(t, a1, 0xF0));          /* DCBA */
    _mm_storeu_si128((__m128i *)&sa[4], _mm_alignr_epi8(a1, t, 8));             /* HGFE */
    t = _mm_shuffle_epi32(b0, 0x1B);
    b1 = _mm_shuffle_epi32(b1, 0xB1);
    _mm_storeu_si128((__m128i *)&sb[0], _mm_blend_epi16(t, b1, 0xF0));
    _mm_storeu_si128((__m128i *)&sb[4], _mm_alignr_epi8(b1, t, 8));
}

static inline void sha1x_ossl(uint32_t st[8], const uint8_t *p, size_t n) {
    SHA256_CTX c; memcpy(c.h, st, 32);
    for (size_t i = 0; i < n; i++) SHA256_Transform(&c, p + 64 * i);
    memcpy(st, c.h, 32);
}

/* Two independent messages of n blocks each. */
static inline void qcg_sha2x(uint32_t sa[8], const uint8_t *pa, uint32_t sb[8], const uint8_t *pb, size_t n) {
    if (g_sha_ni > 0) sha2x_ni(sa, pa, sb, pb, n);
    else { sha1x_ossl(sa, pa, n); sha1x_ossl(sb, pb, n); }
}

} /* namespace qcg */
#endif /* QSB_CPU_COGRIND_SHA_H */
