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
#endif
