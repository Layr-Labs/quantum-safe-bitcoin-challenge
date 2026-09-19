/* SPDX-License-Identifier: GPL-3.0-only
 * Rolling SHA-256 schedule/compression interleave for the pinning candidate.
 * Uses the existing VanitySearch S2Round/s0/s1 macros; their notices and
 * GPLv3 terms remain in GPUHash.h and COPYING.
 * All indices are compile-time constants. Update each schedule word directly
 * before consuming it, instead of computing sixteen words before any round.
 */
#ifndef QSB_SHA_SCHEDULE_INTERLEAVED_CUH
#define QSB_SHA_SCHEDULE_INTERLEAVED_CUH

#define QSB_SHA_SCHEDULE_STEP(j, a,b,c,d,e,f,g,h, base) do { \
    w[j] += s1(w[((j)+14)&15]) + w[((j)+9)&15] + s0(w[((j)+1)&15]); \
    S2Round(a,b,c,d,e,f,g,h,K[(base)+(j)],w[j]); \
} while (0)

#define QSB_SHA_INTERLEAVED_16(base) do { \
    QSB_SHA_SCHEDULE_STEP(0,a,b,c,d,e,f,g,h,base); \
    QSB_SHA_SCHEDULE_STEP(1,h,a,b,c,d,e,f,g,base); \
    QSB_SHA_SCHEDULE_STEP(2,g,h,a,b,c,d,e,f,base); \
    QSB_SHA_SCHEDULE_STEP(3,f,g,h,a,b,c,d,e,base); \
    QSB_SHA_SCHEDULE_STEP(4,e,f,g,h,a,b,c,d,base); \
    QSB_SHA_SCHEDULE_STEP(5,d,e,f,g,h,a,b,c,base); \
    QSB_SHA_SCHEDULE_STEP(6,c,d,e,f,g,h,a,b,base); \
    QSB_SHA_SCHEDULE_STEP(7,b,c,d,e,f,g,h,a,base); \
    QSB_SHA_SCHEDULE_STEP(8,a,b,c,d,e,f,g,h,base); \
    QSB_SHA_SCHEDULE_STEP(9,h,a,b,c,d,e,f,g,base); \
    QSB_SHA_SCHEDULE_STEP(10,g,h,a,b,c,d,e,f,base); \
    QSB_SHA_SCHEDULE_STEP(11,f,g,h,a,b,c,d,e,base); \
    QSB_SHA_SCHEDULE_STEP(12,e,f,g,h,a,b,c,d,base); \
    QSB_SHA_SCHEDULE_STEP(13,d,e,f,g,h,a,b,c,base); \
    QSB_SHA_SCHEDULE_STEP(14,c,d,e,f,g,h,a,b,base); \
    QSB_SHA_SCHEDULE_STEP(15,b,c,d,e,f,g,h,a,base); \
} while (0)

/* Tip ff275e40 interleaved WMIX+RND for bases 32 and 48 in pubkey33, and
 * explicitly left the sparse first expansion + SHA256_RND(16) as a burst.
 * This macro applies the same schedule/round interleave to that deferred
 * burst, preserving the sparse zero-elided expansion formulas. */
#define QSB_SHA_PUBKEY33_INTERLEAVE_RND16() do { \
    w[0] += s0(w[1]); \
    S2Round(a,b,c,d,e,f,g,h,K[16],w[0]); \
    w[1] += s1(0x108u) + s0(w[2]); \
    S2Round(h,a,b,c,d,e,f,g,K[17],w[1]); \
    w[2] += s1(w[0]) + s0(w[3]); \
    S2Round(g,h,a,b,c,d,e,f,K[18],w[2]); \
    w[3] += s1(w[1]) + s0(w[4]); \
    S2Round(f,g,h,a,b,c,d,e,K[19],w[3]); \
    w[4] += s1(w[2]) + s0(w[5]); \
    S2Round(e,f,g,h,a,b,c,d,K[20],w[4]); \
    w[5] += s1(w[3]) + s0(w[6]); \
    S2Round(d,e,f,g,h,a,b,c,K[21],w[5]); \
    w[6] += s1(w[4]) + 0x108u + s0(w[7]); \
    S2Round(c,d,e,f,g,h,a,b,K[22],w[6]); \
    w[7] += s1(w[5]) + w[0] + s0(w[8]); \
    S2Round(b,c,d,e,f,g,h,a,K[23],w[7]); \
    w[8] += s1(w[6]) + w[1]; \
    S2Round(a,b,c,d,e,f,g,h,K[24],w[8]); \
    w[9]  = s1(w[7]) + w[2]; \
    S2Round(h,a,b,c,d,e,f,g,K[25],w[9]); \
    w[10] = s1(w[8]) + w[3]; \
    S2Round(g,h,a,b,c,d,e,f,K[26],w[10]); \
    w[11] = s1(w[9]) + w[4]; \
    S2Round(f,g,h,a,b,c,d,e,K[27],w[11]); \
    w[12] = s1(w[10]) + w[5]; \
    S2Round(e,f,g,h,a,b,c,d,K[28],w[12]); \
    w[13] = s1(w[11]) + w[6]; \
    S2Round(d,e,f,g,h,a,b,c,K[29],w[13]); \
    w[14] = s1(w[12]) + w[7] + s0(0x108u); \
    S2Round(c,d,e,f,g,h,a,b,K[30],w[14]); \
    w[15] = 0x108u + s1(w[13]) + w[8] + s0(w[0]); \
    S2Round(b,c,d,e,f,g,h,a,K[31],w[15]); \
} while (0)
#endif
