// SPDX-License-Identifier: GPL-3.0-only
#pragma once
/* Fixed-IV SHA256 H0 of TWO independent 33-byte compressed public keys.
 * Derived from the literal qcg_sha::shani_compress2 schedule in DMA/source.
 * Inputs are host-order big-endian SHA words, W0..8 vary, W9..14=0,W15=264.
 * This is one SHA256, not SHA256d. All64 rounds and every schedule update remain.
 * Call only after sha + sse4.1 + ssse3 runtime ISA checks. No global ISA flags.
 */
#include "cg_sha.h"
namespace qsb_pksha_ni_h0 {
using qcg_sha::IV256;
using qcg_sha::K256;
static QSB_SHA_NI void pubkey_h0_pair(const uint32_t *wA, const uint32_t *wB, uint32_t *hA, uint32_t *hB) {
    // Low-to-high lanes are F,E,B,A and H,G,D,C, exactly shani_load_state(IV).
    __m128i A0 = _mm_set_epi32((int)IV256[0], (int)IV256[1], (int)IV256[4], (int)IV256[5]);
    __m128i A1 = _mm_set_epi32((int)IV256[2], (int)IV256[3], (int)IV256[6], (int)IV256[7]);
    __m128i B0 = A0, B1 = A1;
    __m128i MA0, MA1, MA2, MA3, MB0, MB1, MB2, MB3, mA, mB, K;
    MA0 = _mm_loadu_si128((const __m128i *)(wA + 0));  MB0 = _mm_loadu_si128((const __m128i *)(wB + 0));
    MA1 = _mm_loadu_si128((const __m128i *)(wA + 4));  MB1 = _mm_loadu_si128((const __m128i *)(wB + 4));
    // W8 retains its input value; W9..W14=0 and W15=33*8=264.
    MA2 = _mm_cvtsi32_si128((int)wA[8]); MB2 = _mm_cvtsi32_si128((int)wB[8]);
    MA3 = MB3 = _mm_set_epi32(264, 0, 0, 0);
    /* group g: rounds 4g..4g+3 on message vector Mc; Mn = next (msg2 target), Mp = previous
       (alignr source), Mq = the vector msg1 updates */
#define SHANI2_ROUNDS(g, McA, McB)                                                     \
    K = _mm_loadu_si128((const __m128i *)(K256 + 4 * (g)));                            \
    mA = _mm_add_epi32(McA, K); mB = _mm_add_epi32(McB, K);                            \
    A1 = _mm_sha256rnds2_epu32(A1, A0, mA); B1 = _mm_sha256rnds2_epu32(B1, B0, mB);
#define SHANI2_TAIL()                                                                  \
    mA = _mm_shuffle_epi32(mA, 0x0E); mB = _mm_shuffle_epi32(mB, 0x0E);                \
    A0 = _mm_sha256rnds2_epu32(A0, A1, mA); B0 = _mm_sha256rnds2_epu32(B0, B1, mB);
#define SHANI2_MSG2(McA, McB, MpA, MpB, MnA, MnB)                                      \
    MnA = _mm_sha256msg2_epu32(_mm_add_epi32(MnA, _mm_alignr_epi8(McA, MpA, 4)), McA);  \
    MnB = _mm_sha256msg2_epu32(_mm_add_epi32(MnB, _mm_alignr_epi8(McB, MpB, 4)), McB);
#define SHANI2_MSG1(MqA, MqB, McA, McB)                                                \
    MqA = _mm_sha256msg1_epu32(MqA, McA); MqB = _mm_sha256msg1_epu32(MqB, McB);
    /* g = 0 */  SHANI2_ROUNDS(0, MA0, MB0) SHANI2_TAIL()
    /* g = 1 */  SHANI2_ROUNDS(1, MA1, MB1) SHANI2_TAIL() SHANI2_MSG1(MA0, MB0, MA1, MB1)
    /* g = 2 */  SHANI2_ROUNDS(2, MA2, MB2) SHANI2_TAIL() SHANI2_MSG1(MA1, MB1, MA2, MB2)
    /* g = 3: W12..15 are identical constants for both keys. */
    mA = mB = _mm_set_epi32((int)(K256[15] + 264u), (int)K256[14], (int)K256[13], (int)K256[12]);
    A1 = _mm_sha256rnds2_epu32(A1, A0, mA); B1 = _mm_sha256rnds2_epu32(B1, B0, mB);
    /* preserve the original group3 message update / tail ordering */ SHANI2_MSG2(MA3, MB3, MA2, MB2, MA0, MB0) SHANI2_TAIL() SHANI2_MSG1(MA2, MB2, MA3, MB3)
    /* g = 4 */  SHANI2_ROUNDS(4, MA0, MB0) SHANI2_MSG2(MA0, MB0, MA3, MB3, MA1, MB1) SHANI2_TAIL() SHANI2_MSG1(MA3, MB3, MA0, MB0)
    /* g = 5 */  SHANI2_ROUNDS(5, MA1, MB1) SHANI2_MSG2(MA1, MB1, MA0, MB0, MA2, MB2) SHANI2_TAIL() SHANI2_MSG1(MA0, MB0, MA1, MB1)
    /* g = 6 */  SHANI2_ROUNDS(6, MA2, MB2) SHANI2_MSG2(MA2, MB2, MA1, MB1, MA3, MB3) SHANI2_TAIL() SHANI2_MSG1(MA1, MB1, MA2, MB2)
    /* g = 7 */  SHANI2_ROUNDS(7, MA3, MB3) SHANI2_MSG2(MA3, MB3, MA2, MB2, MA0, MB0) SHANI2_TAIL() SHANI2_MSG1(MA2, MB2, MA3, MB3)
    /* g = 8 */  SHANI2_ROUNDS(8, MA0, MB0) SHANI2_MSG2(MA0, MB0, MA3, MB3, MA1, MB1) SHANI2_TAIL() SHANI2_MSG1(MA3, MB3, MA0, MB0)
    /* g = 9 */  SHANI2_ROUNDS(9, MA1, MB1) SHANI2_MSG2(MA1, MB1, MA0, MB0, MA2, MB2) SHANI2_TAIL() SHANI2_MSG1(MA0, MB0, MA1, MB1)
    /* g = 10 */ SHANI2_ROUNDS(10, MA2, MB2) SHANI2_MSG2(MA2, MB2, MA1, MB1, MA3, MB3) SHANI2_TAIL() SHANI2_MSG1(MA1, MB1, MA2, MB2)
    /* g = 11 */ SHANI2_ROUNDS(11, MA3, MB3) SHANI2_MSG2(MA3, MB3, MA2, MB2, MA0, MB0) SHANI2_TAIL() SHANI2_MSG1(MA2, MB2, MA3, MB3)
    /* g = 12 */ SHANI2_ROUNDS(12, MA0, MB0) SHANI2_MSG2(MA0, MB0, MA3, MB3, MA1, MB1) SHANI2_TAIL() SHANI2_MSG1(MA3, MB3, MA0, MB0)
    /* g = 13 */ SHANI2_ROUNDS(13, MA1, MB1) SHANI2_MSG2(MA1, MB1, MA0, MB0, MA2, MB2) SHANI2_TAIL()
    /* g = 14 */ SHANI2_ROUNDS(14, MA2, MB2) SHANI2_MSG2(MA2, MB2, MA1, MB1, MA3, MB3) SHANI2_TAIL()
    /* g = 15 */ SHANI2_ROUNDS(15, MA3, MB3) SHANI2_TAIL()
#undef SHANI2_ROUNDS
#undef SHANI2_TAIL
#undef SHANI2_MSG2
#undef SHANI2_MSG1
    // After all64 rounds, high lane3 of A0/B0 is A64. Only H0 is consumed.
    *hA = (uint32_t)_mm_extract_epi32(A0, 3) + IV256[0];
    *hB = (uint32_t)_mm_extract_epi32(B0, 3) + IV256[0];
}
} // namespace qsb_pksha_ni_h0
