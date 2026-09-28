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
    QSB_SHA_EXPAND(0); QSB_SHA_EXPAND(1); \
    QSB_SHA_CONSUME(0,a,b,c,d,e,f,g,h,base); QSB_SHA_EXPAND(2); \
    QSB_SHA_CONSUME(1,h,a,b,c,d,e,f,g,base); QSB_SHA_EXPAND(3); \
    QSB_SHA_CONSUME(2,g,h,a,b,c,d,e,f,base); QSB_SHA_EXPAND(4); \
    QSB_SHA_CONSUME(3,f,g,h,a,b,c,d,e,base); QSB_SHA_EXPAND(5); \
    QSB_SHA_CONSUME(4,e,f,g,h,a,b,c,d,base); QSB_SHA_EXPAND(6); \
    QSB_SHA_CONSUME(5,d,e,f,g,h,a,b,c,base); QSB_SHA_EXPAND(7); \
    QSB_SHA_CONSUME(6,c,d,e,f,g,h,a,b,base); QSB_SHA_EXPAND(8); \
    QSB_SHA_CONSUME(7,b,c,d,e,f,g,h,a,base); QSB_SHA_EXPAND(9); \
    QSB_SHA_CONSUME(8,a,b,c,d,e,f,g,h,base); QSB_SHA_EXPAND(10); \
    QSB_SHA_CONSUME(9,h,a,b,c,d,e,f,g,base); QSB_SHA_EXPAND(11); \
    QSB_SHA_CONSUME(10,g,h,a,b,c,d,e,f,base); QSB_SHA_EXPAND(12); \
    QSB_SHA_CONSUME(11,f,g,h,a,b,c,d,e,base); QSB_SHA_EXPAND(13); \
    QSB_SHA_CONSUME(12,e,f,g,h,a,b,c,d,base); QSB_SHA_EXPAND(14); \
    QSB_SHA_CONSUME(13,d,e,f,g,h,a,b,c,base); QSB_SHA_EXPAND(15); \
    QSB_SHA_CONSUME(14,c,d,e,f,g,h,a,b,base); \
    QSB_SHA_CONSUME(15,b,c,d,e,f,g,h,a,base); \
} while (0)

/* Expand two words before starting compression, then keep one future word
 * available. Expansion order is unchanged and no word is overwritten until
 * its previous compression consumer has executed. This exposes independent
 * schedule work without adding a second schedule array or more live words. */
#define QSB_SHA_EXPAND(j) do { \
    w[j] += s1(w[((j)+14)&15]) + w[((j)+9)&15] + s0(w[((j)+1)&15]); \
} while (0)
#define QSB_SHA_CONSUME(j,a,b,c,d,e,f,g,h,base) \
    S2Round(a,b,c,d,e,f,g,h,K[(base)+(j)],w[j])
#endif
