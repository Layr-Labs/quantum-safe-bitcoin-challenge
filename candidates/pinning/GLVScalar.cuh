#pragma once
#include <stdint.h>

#ifndef QSB_BIGTBL
#define QSB_BIGTBL 1
#endif
#if QSB_BIGTBL != 0 && QSB_BIGTBL != 1
#error QSB_BIGTBL must be 0 or 1
#endif

// Four-bank cache geometry from 0xCramJam c13f3832 / 90f89008.
// Keep the promoted GLV12 geometry as an independently compilable control.
#ifndef QSB_FOUR_HOT
#define QSB_FOUR_HOT 1
#endif
#ifndef QSB_DIGIT_LEAN
#define QSB_DIGIT_LEAN 1   /* 1: one-ALU-op-per-step GLV code decode (q9_bigtbl_code_lean) */
#endif
#if QSB_DIGIT_LEAN != 0 && QSB_DIGIT_LEAN != 1
#error QSB_DIGIT_LEAN must be 0 or 1
#endif
#if QSB_DIGIT_LEAN && !QSB_BIGTBL
#error QSB_DIGIT_LEAN is written for the QSB_BIGTBL code layout
#endif
/* QSB_GLV_GLUE (bit mask, default 0): fewer non-multiply instructions in the prepare kernel's GLV
 * split, signed-digit decode and seed. Every value the kernel computes (coefficients, residual
 * magnitudes and signs, the 14 GLV codes, the seed points, the chain state) is the same as the
 * base for every input.
 *  bit 1: q9_product129 (x*d mod 2^129 for the three residual products) from column-pair
 *         accumulators E0/O0/E1/O1 instead of row-wise 32-bit carries, with the word-4 parity
 *         terms (the low bits of the column-4 products, and the sum2 / c1 fix-ups) folded into
 *         the one add that forms word 4.
 *  bit 2: the decode reads the signed residual w = z - s (s = bit 128 of z) instead of |z|:
 *         |z| = w ^ -s, and the -s cancels in every radix chunk's index and sign (q9_bigtbl_code_z).
 *         q9_abs129 and the sign shift are gone; radix chunks use a top-aligned window.
 *  bit 4: coefficient rounding takes bit 31 of w11 as the carry out of w11 + 2^31.
 *  bit 8: (pinning.cu, needs bit 2) Q's two register seed codes are handed to their gathers
 *         as (record, sign mask): the pack into a code and the gather's unpack are gone.
 * 0 leaves the source and PTX unchanged. */
#ifndef QSB_GLV_GLUE
#define QSB_GLV_GLUE 15
#endif
#if QSB_GLV_GLUE < 0 || QSB_GLV_GLUE > 15
#error "QSB_GLV_GLUE is a mask of bits 1, 2, 4 and 8"
#endif
#if (QSB_GLV_GLUE & 8) && !(QSB_GLV_GLUE & 2)
#error "QSB_GLV_GLUE bit 8 (unpacked seed codes) needs bit 2 (signed-residual decode)"
#endif
#define QSB_GLV_EO    ((QSB_GLV_GLUE & 1) != 0)
#define QSB_GLV_ZDEC  ((QSB_GLV_GLUE & 2) != 0)
#define QSB_GLV_RND   ((QSB_GLV_GLUE & 4) != 0)
#define QSB_SEED_GLUE ((QSB_GLV_GLUE & 8) != 0)
#if QSB_FOUR_HOT != 0 && QSB_FOUR_HOT != 1
#error QSB_FOUR_HOT must be 0 or 1
#endif
/* QSB_GLV11=1: P (the phi component) uses five terms instead of six -- segment 0,
 * two appended streaming segments 6 (shift 18, 27 bits, 2^26 records) and 7
 * (shift 45, 28 bits, 2^27 records), then segments 4 and 5 -- so 11 gathers and 10
 * additions per candidate instead of 12 and 11. Q keeps its six GLV12 terms.
 * Layout after i34-9 14675ab0 ("P18"). 0 restores the GLV12 table and chain. */
#ifndef QSB_GLV11
#define QSB_GLV11 0
#endif
#if QSB_GLV11 != 0 && QSB_GLV11 != 1
#error QSB_GLV11 must be 0 or 1
#endif
#if QSB_GLV11 && !(QSB_BIGTBL && QSB_FOUR_HOT)
#error QSB_GLV11 extends the four-hot GLV12 table (QSB_BIGTBL=1, QSB_FOUR_HOT=1)
#endif
#ifndef QSB_QGLV5
#define QSB_QGLV5 0
#endif
#if QSB_QGLV5 != 0 && QSB_QGLV5 != 1
#error QSB_QGLV5 must be 0 or 1
#endif
#if QSB_QGLV5 && !(QSB_GLV11 && QSB_GLV_ZDEC && QSB_SEED_GLUE && QSB_DIGIT_LEAN)
#error QSB_QGLV5 is written for the GLV11 table with the ZDEC decode and register seed glue
#endif
#if QSB_BIGTBL && QSB_FOUR_HOT
#if QSB_GLV11
#define QSB_GT_TOTAL 354501773u
#define QSB_GT_SEGMENTS 8
#else
#define QSB_GT_TOTAL 153175181u
#define QSB_GT_SEGMENTS 6
#endif
#define QSB_GT_RADIX_BITS 14
#define QSB_GT_TOP_CENTER 170559769u
#define QSB_GT_TOP_SHIFT 100u
#else
#define QSB_GT_TOTAL 22893641u
#define QSB_GT_SEGMENTS 6
#define QSB_GT_RADIX_BITS 12
#define QSB_GT_TOP_CENTER 10659985u
#define QSB_GT_TOP_SHIFT 104u
#endif

#if QSB_BIGTBL
// BEGIN QSB_BIGTBL_HOST_EXACT
/* Six terms per signed GLV component. The split below is unchanged. Its
 * rounded reciprocal error gives |r_i| <=
 * 0xa2a8918ca85bafe22016d0b917e4dd77. At shift 104 the largest top
 * field is 10659985, an odd number. Thus d_top=2*f-10659985 is odd,
 * nonzero and in [-10659985,10659985]. No residual truncation is used.
 * QSB_FOUR_HOT=0: physical order 0,1,2,5,3,4, three cached banks.
 * QSB_FOUR_HOT=1: physical order 0..5, first four banks total 48 MiB;
 * widths [18,19,18,18,27], top shift 100, centered at 170559769.
 * Both exactly reconstruct the same bounded signed GLV component.
 * These portable helpers are also compiled verbatim by check_bigtable.py. */
__host__ __device__ __forceinline__ unsigned q9_bigtbl_entries(int c) {
#if QSB_GLV11
    if(c>=6) return c==6?67108864u:134217728u;
#endif
#if QSB_FOUR_HOT
    return c<2?262144u:c<4?131072u:c==4?67108864u:85279885u;
#else
    return c<3 ? 262144u : (c<5 ? 8388608u : 5329993u);
#endif
}
__host__ __device__ __forceinline__ unsigned q9_bigtbl_offset(int c) {
#if QSB_GLV11
    if(c>=6) return c==6?153175181u:220284045u;
#endif
#if QSB_FOUR_HOT
    return c==0?0u:c==1?262144u:c==2?524288u:
           c==3?655360u:c==4?786432u:67895296u;
#else
    return c==0?0u:c==1?262144u:c==2?524288u:c==3?6116425u:
           c==4?14505033u:786432u;
#endif
}
__host__ __device__ __forceinline__ unsigned q9_bigtbl_shift(int c) {
#if QSB_GLV11
    if(c>=6) return c==6?18u:45u;
#endif
#if QSB_FOUR_HOT
    return c==0?0u:c==1?18u:c==2?37u:c==3?55u:c==4?73u:100u;
#else
    return c==0?0u:c==1?18u:c==2?37u:c==3?56u:c==4?80u:104u;
#endif
}
__host__ __device__ __forceinline__ uint32_t q9_bigtbl_code(
    const uint64_t mag[2],unsigned sign,int c) {
    const unsigned shift=q9_bigtbl_shift(c);
    uint64_t wide;
    if(shift<64u) {
        wide=mag[0]>>shift;
        if(shift) wide|=mag[1]<<(64u-shift);
    } else wide=mag[1]>>(shift-64u);
    uint32_t f=(uint32_t)wide,idx,neg_digit;
    if(c==0) {
        idx=f&((1u<<18)-1u);neg_digit=0;
    } else if(c==5) {
        const int32_t d=(int32_t)(2u*f)-(int32_t)QSB_GT_TOP_CENTER;
        neg_digit=(uint32_t)d>>31;
        const uint32_t ad=((uint32_t)d^(0u-neg_digit))+neg_digit;
        idx=(ad-1u)>>1;
    } else {
#if QSB_FOUR_HOT
        const unsigned bits=c==1?19u:c<4?18u:27u;
#else
        const unsigned bits=c<3?19u:24u;
#endif
        f&=(1u<<bits)-1u;
        neg_digit=1u-(f>>(bits-1u));
        idx=(f^(0u-neg_digit))&((1u<<(bits-1u))-1u);
    }
    return (q9_bigtbl_offset(c)+idx)|((neg_digit^sign)<<31);
}
#if QSB_GLV11
/* P's five terms t=0..4 read segments 0,6,7,4,5. Shifts 0,18,45,73,100 keep the
 * signed chain 18->45->73->100 contiguous, so the digit biases telescope to the
 * same segment-0 bias K as GLV12 and the five digits sum exactly to the
 * magnitude. Segments 6 and 7 are plain signed fields of 27 and 28 bits. */
__host__ __device__ __forceinline__ uint32_t q11_bigtbl_code(
    const uint64_t mag[2],unsigned sign,int t) {
    if(t==0) return q9_bigtbl_code(mag,sign,0);
    if(t>=3) return q9_bigtbl_code(mag,sign,t+1);
    const int c=t+5;
    const unsigned shift=q9_bigtbl_shift(c),bits=t==1?27u:28u;
    const uint64_t wide=(mag[0]>>shift)|(mag[1]<<(64u-shift));
    const uint32_t f=(uint32_t)wide&((1u<<bits)-1u);
    const uint32_t neg_digit=1u-(f>>(bits-1u));
    const uint32_t idx=(f^(0u-neg_digit))&((1u<<(bits-1u))-1u);
    return (q9_bigtbl_offset(c)+idx)|((neg_digit^sign)<<31);
}
#endif
// END QSB_BIGTBL_HOST_EXACT

#if QSB_DIGIT_LEAN
/* QSB_DIGIT_LEAN: the same 32-bit code as q9_bigtbl_code(mag,sign,c), written so that
 * every step is one ALU instruction and no multiply-pipe shift or negation is needed.
 * s31 = sign<<31. For every chunk offset(c)+idx < GT_TOTAL < 2^31, so OR-ing the sign
 * bit equals adding it, and each code is idx + offset + sign bit.
 *  - c==0: code = (f & (2^18-1)) | s31 (neg_digit = 0, offset 0).
 *  - radix chunks, width w: with t = bit w-1 of f, neg_digit = 1-t. xs = -t is the
 *    arithmetic shift of f<<(32-w); -neg_digit = ~xs, so idx = (f ^ ~xs) & (2^(w-1)-1),
 *    and bit 31 of ~xs is neg_digit, so ((~xs ^ s31) & 2^31) = (neg_digit^sign)<<31.
 *  - top chunk: C = QSB_GT_TOP_CENTER is odd, d = 2f - C. With h = f - (C+1)/2 we have
 *    d = 2h + 1, so d < 0 exactly when h < 0 (neg_digit = bit 31 of h), and
 *    (|d|-1)/2 = h for h >= 0, -h-1 = ~h for h < 0; that is idx = h ^ (h>>31).
 *    f < 2^28 (mag < 2^128), so h cannot wrap.
 * Values are identical for every mag < 2^128, sign and c. The host build of this
 * function was checked against q9_bigtbl_code on the CPU
 * (every field value of every chunk, both signs, random high bits). */
static_assert((QSB_GT_TOP_CENTER & 1u) == 1u, "top center must be odd");
/* Low 32 bits of (hi:lo) >> r, 0 < r < 32: CUDA __funnelshift_r; host mirror for tests. */
__host__ __device__ __forceinline__ uint32_t q9_funnel_r(uint32_t lo,uint32_t hi,unsigned r) {
#ifdef __CUDA_ARCH__
    return __funnelshift_r(lo,hi,r);
#else
    return (uint32_t)((((uint64_t)hi<<32)|lo)>>(r&31u));
#endif
}
__host__ __device__ __forceinline__ uint32_t q9_bigtbl_code_lean(
    const uint64_t mag[2],uint32_t s31,int c) {
    /* f = bits shift..shift+31 of mag (zero above bit 127): one funnel shift of two
     * 32-bit words, the low 32 bits of (uint32_t)(mag >> shift). */
    const unsigned shift=q9_bigtbl_shift(c);
    const uint32_t ws[5]={(uint32_t)mag[0],(uint32_t)(mag[0]>>32),
                          (uint32_t)mag[1],(uint32_t)(mag[1]>>32),0u};
    const unsigned q=shift>>5,r=shift&31u;
    const uint32_t f=r?q9_funnel_r(ws[q],ws[q+1],r):ws[q];
    if(c==0) {
#ifdef __CUDA_ARCH__
        uint32_t code0;   /* one LOP3: (f & (2^18-1)) | s31 */
        asm("lop3.b32 %0,%1,%2,%3,0xEA;" : "=r"(code0) : "r"(f), "r"((1u<<18)-1u), "r"(s31));
        return code0;
#else
        return (f&((1u<<18)-1u))|s31;
#endif
    }
    if(c==5) {
        const uint32_t h=f-((uint32_t)QSB_GT_TOP_CENTER+1u)/2u;
        const uint32_t s=(uint32_t)((int32_t)h>>31);
        return q9_bigtbl_offset(c)+(h^s)+((h^s31)&0x80000000u);
    }
#if QSB_FOUR_HOT
    const unsigned bits=c==1?19u:c<4?18u:27u;
#else
    const unsigned bits=c<3?19u:24u;
#endif
    const uint32_t xs=(uint32_t)((int32_t)(f<<(32u-bits))>>31);
    const uint32_t idx=(f^~xs)&((1u<<(bits-1u))-1u);
    return q9_bigtbl_offset(c)+idx+((~xs^s31)&0x80000000u);
}
#endif /* QSB_DIGIT_LEAN */
#if QSB_GLV_ZDEC && QSB_DIGIT_LEAN && QSB_FOUR_HOT
/* q9_bigtbl_code_z(w,top,m32,c) == q9_bigtbl_code(mag,sign,c) with mag = w ^ M (M = -s on every
 * word), sign = s = top & 1 and m32 = -s. Field F of mag = F' ^ M (F' the field of w), top bit
 * t = t' ^ s. Radix chunk: idx = t ? F & mask : ~F & mask = t' ? F' & mask : ~F' & mask, and
 * the code sign neg_digit ^ sign = (1 - t) ^ s = 1 - t'; neither depends on s. The window u is
 * top-aligned (t' at bit 31), xs = -t' and v = u ^ ~xs has bit 31 set and idx in bits 32-w..30,
 * so (v >> (32-w)) + offset - 2^(w-1) = offset + idx, and the sign bit ~t' is XORed in (adding
 * 2^31 is flipping bit 31). Chunk 0 and the top chunk use mag's words explicitly. Only bit 0 of
 * top is used. */
__host__ __device__ __forceinline__ uint32_t q9_bigtbl_code_z(
    const uint64_t w[2],uint32_t top,uint32_t m32,int c) {
    const uint32_t ws[4]={(uint32_t)w[0],(uint32_t)(w[0]>>32),(uint32_t)w[1],(uint32_t)(w[1]>>32)};
    if(c==0) {
#ifdef __CUDA_ARCH__
        uint32_t code;
        asm("{\n\t.reg .u32 t;\n\t"
            "lop3.b32 t,%1,%2,%3,0x28;\n\t"      /* (w0 ^ m) & (2^18-1) */
            "lop3.b32 %0,t,%2,0x80000000,0xF8;\n\t}" /* t | (m & 2^31) */
            : "=r"(code) : "r"(ws[0]),"r"(m32),"r"((1u<<18)-1u));
        (void)top;
        return code;
#else
        (void)top;
        return ((ws[0]^m32)&((1u<<18)-1u))|(m32&0x80000000u);
#endif
    }
    if(c==5) {
        const uint32_t cen=((uint32_t)QSB_GT_TOP_CENTER+1u)/2u;
#ifdef __CUDA_ARCH__
        uint32_t code;
        asm("{\n\t.reg .u32 g,h,x,i,t;\n\t"
            "xor.b32 g,%1,%2;\n\t"
            "shr.u32 g,g,4;\n\t"
            "sub.u32 h,g,%3;\n\t"
            "shr.s32 x,h,31;\n\t"
            "xor.b32 i,h,x;\n\t"
            "lop3.b32 t,h,%2,0x80000000,0x28;\n\t" /* (h ^ m) & 2^31 */
            "add.u32 i,i,t;\n\t"
            "add.u32 %0,i,%4;\n\t}"
            : "=r"(code) : "r"(ws[3]),"r"(m32),"r"(cen),"r"(q9_bigtbl_offset(c)));
        return code;
#else
        const uint32_t h=((ws[3]^m32)>>4)-cen;
        const uint32_t s=(uint32_t)((int32_t)h>>31);
        return q9_bigtbl_offset(c)+(h^s)+((h^m32)&0x80000000u);
#endif
    }
    const unsigned bits=c==1?19u:c<4?18u:27u;
    const unsigned r=q9_bigtbl_shift(c)+bits-32u,q=r>>5,rs=r&31u;
    const uint32_t off=q9_bigtbl_offset(c)-(1u<<(bits-1u));
#ifdef __CUDA_ARCH__
    uint32_t code;
    asm("{\n\t.reg .u32 u,x,v;\n\t"
        "shf.r.clamp.b32 u,%1,%2,%3;\n\t"       /* top-aligned window */
        "shr.s32 x,u,31;\n\t"
        "lop3.b32 v,u,x,0,0xC3;\n\t"            /* u ^ ~x */
        "shr.u32 v,v,%4;\n\t"
        "add.u32 v,v,%5;\n\t"                   /* offset + idx */
        "lop3.b32 %0,v,u,0x80000000,0xD2;\n\t}" /* v ^ (~u & 2^31) */
        : "=r"(code) : "r"(ws[q]),"r"(ws[q+1]),"r"(rs),"r"(32u-bits),"r"(off));
    return code;
#else
    const uint32_t u=q9_funnel_r(ws[q],ws[q+1],rs);
    const uint32_t xs=(uint32_t)((int32_t)u>>31);
    const uint32_t v=((u^~xs)>>(32u-bits))+off;
    return v^(~u&0x80000000u);
#endif
}
#endif
#endif
#if QSB_GLV11 && QSB_GLV_ZDEC && QSB_DIGIT_LEAN && QSB_FOUR_HOT
/* q11_bigtbl_code_z(w,top,m32,t) == q11_bigtbl_code(mag,sign,t) with mag = w ^ -s
 * and sign = s = m32 & 1. Terms 0, 3, 4 are GLV12 chunks 0, 4, 5, so they are
 * q9_bigtbl_code_z. Terms 1 and 2 are segments 6 and 7: the radix step of
 * q9_bigtbl_code_z at 27 bits / shift 18 and 28 bits / shift 45. That step's
 * index and code sign are functions of the field bits of w alone, for any
 * signed radix width, so they do not need mag. */
__host__ __device__ __forceinline__ uint32_t q11_radix_code_z(
    const uint32_t ws[4], unsigned shift, unsigned bits, uint32_t offset) {
    const unsigned r=shift+bits-32u,q=r>>5,rs=r&31u;
    const uint32_t off=offset-(1u<<(bits-1u));
#ifdef __CUDA_ARCH__
    uint32_t code;
    asm("{\n\t.reg .u32 u,x,v;\n\t"
        "shf.r.clamp.b32 u,%1,%2,%3;\n\t"
        "shr.s32 x,u,31;\n\t"
        "lop3.b32 v,u,x,0,0xC3;\n\t"
        "shr.u32 v,v,%4;\n\t"
        "add.u32 v,v,%5;\n\t"
        "lop3.b32 %0,v,u,0x80000000,0xD2;\n\t}"
        : "=r"(code) : "r"(ws[q]),"r"(ws[q+1]),"r"(rs),"r"(32u-bits),"r"(off));
    return code;
#else
    const uint32_t u=q9_funnel_r(ws[q],ws[q+1],rs);
    const uint32_t xs=(uint32_t)((int32_t)u>>31);
    const uint32_t v=((u^~xs)>>(32u-bits))+off;
    return v^(~u&0x80000000u);
#endif
}
__host__ __device__ __forceinline__ uint32_t q11_bigtbl_code_z(
    const uint64_t w[2],uint32_t top,uint32_t m32,int t) {
    if(t==0) return q9_bigtbl_code_z(w,top,m32,0);
    if(t>=3) return q9_bigtbl_code_z(w,top,m32,t+1);
    const uint32_t ws[4]={(uint32_t)w[0],(uint32_t)(w[0]>>32),(uint32_t)w[1],(uint32_t)(w[1]>>32)};
    const int c=t+5;
    const unsigned bits=t==1?27u:28u;
    (void)top;
    return q11_radix_code_z(ws,q9_bigtbl_shift(c),bits,q9_bigtbl_offset(c));
}
#endif
#if QSB_GLV_ZDEC && QSB_DIGIT_LEAN && QSB_FOUR_HOT
/* QSB_GLV_GLUE bit 8: the two register seed codes (Q's chunks 0 and 1) as the gather consumes them,
 * rec = code & 0x7fffffff and msk = -(code >> 31) (the Y sign mask), without packing:
 * chunk 0: rec = (w0 ^ m) & (2^18-1), msk = m (its sign bit is s); chunk 1: rec = offset + idx
 * (the radix form before the sign XOR) and, inverted, ~msk = xs = -t' (msk = -(1 - t')). */
__host__ __device__ __forceinline__ void q9_bigtbl_seed_z(const uint64_t w[2],uint32_t m32,int c,
                                                          uint32_t *rec,uint32_t *msk) {
    const uint32_t ws[4]={(uint32_t)w[0],(uint32_t)(w[0]>>32),(uint32_t)w[1],(uint32_t)(w[1]>>32)};
    if(c==0) {
#ifdef __CUDA_ARCH__
        asm("lop3.b32 %0,%1,%2,%3,0x28;" : "=r"(*rec) : "r"(ws[0]),"r"(m32),"r"((1u<<18)-1u));
#else
        *rec=(ws[0]^m32)&((1u<<18)-1u);
#endif
        *msk=m32;
        return;
    }
    /* chunk 1: Q's segment 1 (19 bits at shift 18), or under QSB_QGLV5 segment 6 (27 bits at
     * the same shift 18). The window algebra above holds for any signed radix field. */
#if QSB_QGLV5
    const unsigned bits=27u,seg=6;
#else
    const unsigned bits=19u,seg=1;
#endif
    const unsigned r=q9_bigtbl_shift(seg)+bits-32u,q=r>>5,rs=r&31u;
    const uint32_t off=q9_bigtbl_offset(seg)-(1u<<(bits-1u));
#ifdef __CUDA_ARCH__
    uint32_t xs;
    asm("{\n\t.reg .u32 u,v;\n\t"
        "shf.r.clamp.b32 u,%2,%3,%4;\n\t"
        "shr.s32 %1,u,31;\n\t"
        "lop3.b32 v,u,%1,0,0xC3;\n\t"
        "shr.u32 v,v,%5;\n\t"
        "add.u32 %0,v,%6;\n\t}"
        : "=r"(*rec),"=r"(xs) : "r"(ws[q]),"r"(ws[q+1]),"r"(rs),"r"(32u-bits),"r"(off));
    *msk=xs;   /* chunk 1 returns the complement of the mask */
#else
    const uint32_t u=q9_funnel_r(ws[q],ws[q+1],rs);
    const uint32_t xs=(uint32_t)((int32_t)u>>31);
    *rec=((u^~xs)>>(32u-bits))+off;
    *msk=xs;
#endif
    (void)c;
}
#endif

// QSB/VanitySearch GPLv3 exact wide-product schedule, without field reduction.
__device__ __forceinline__ void q9_wide(uint64_t out[8],const uint64_t a[4],const uint64_t b[4]){
    uint64_t r0,r1,r2,r3,r4,r5,r6,r7;
    asm(
        "{\n"
        "\t.reg .u32 a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7;\n"
        "\t.reg .u64 e0,e1,e2,e3,e4,e5,e6,e7,o0,o1,o2,o3,o4,o5,o6,t,lc;\n"
        "\t.reg .u32 cy,o15;\n"
        "\t.reg .u32 x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;\n"
        "\t.reg .u32 y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14;\n"
        "\tmov.b64 {a0,a1}, %8;\n"
        "\tmov.b64 {a2,a3}, %9;\n"
        "\tmov.b64 {a4,a5}, %10;\n"
        "\tmov.b64 {a6,a7}, %11;\n"
        "\tmov.b64 {b0,b1}, %12;\n"
        "\tmov.b64 {b2,b3}, %13;\n"
        "\tmov.b64 {b4,b5}, %14;\n"
        "\tmov.b64 {b6,b7}, %15;\n"
        "\t.reg .u64 odd_t,odd_lc; .reg .u32 odd_cy;\n"
        "mul.wide.u32 e0, a0, b0;\n"
        "mul.wide.u32 o0, a0, b1;\n"
        "mul.wide.u32 e1, a0, b2;\n"
        "mul.wide.u32 o1, a0, b3;\n"
        "mul.wide.u32 e2, a0, b4;\n"
        "mul.wide.u32 o2, a0, b5;\n"
        "mul.wide.u32 e3, a0, b6;\n"
        "mul.wide.u32 o3, a0, b7;\n"
        "mul.wide.u32 t, a1, b1;\n"
        "mul.wide.u32 odd_t, a1, b0;\n"
        "add.cc.u64 e1, e1, t;\n"
        "mul.wide.u32 t, a1, b3;\n"
        "addc.cc.u64 e2, e2, t;\n"
        "mul.wide.u32 t, a1, b5;\n"
        "addc.cc.u64 e3, e3, t;\n"
        "mul.wide.u32 t, a1, b7;\n"
        "addc.u64 e4, t, 0;\n"
        "add.cc.u64 o0, o0, odd_t;\n"
        "mul.wide.u32 odd_t, a1, b2;\n"
        "addc.cc.u64 o1, o1, odd_t;\n"
        "mul.wide.u32 odd_t, a1, b4;\n"
        "addc.cc.u64 o2, o2, odd_t;\n"
        "mul.wide.u32 odd_t, a1, b6;\n"
        "addc.cc.u64 o3, o3, odd_t;\n"
        "addc.u32 odd_cy, 0, 0;\n"
        "mul.wide.u32 t, a2, b0;\n"
        "cvt.u64.u32 odd_lc, odd_cy;\n"
        "add.cc.u64 e1, e1, t;\n"
        "mul.wide.u32 t, a2, b2;\n"
        "addc.cc.u64 e2, e2, t;\n"
        "mul.wide.u32 t, a2, b4;\n"
        "addc.cc.u64 e3, e3, t;\n"
        "mul.wide.u32 t, a2, b6;\n"
        "addc.cc.u64 e4, e4, t;\n"
        "addc.u32 cy, 0, 0;\n"
        "mul.wide.u32 odd_t, a2, b1;\n"
        "cvt.u64.u32 lc, cy;\n"
        "add.cc.u64 o1, o1, odd_t;\n"
        "mul.wide.u32 odd_t, a2, b3;\n"
        "addc.cc.u64 o2, o2, odd_t;\n"
        "mul.wide.u32 odd_t, a2, b5;\n"
        "addc.cc.u64 o3, o3, odd_t;\n"
        "mul.wide.u32 odd_t, a2, b7;\n"
        "addc.u64 o4, odd_t, odd_lc;\n"
        "mul.wide.u32 t, a3, b1;\n"
        "mul.wide.u32 odd_t, a3, b0;\n"
        "add.cc.u64 e2, e2, t;\n"
        "mul.wide.u32 t, a3, b3;\n"
        "addc.cc.u64 e3, e3, t;\n"
        "mul.wide.u32 t, a3, b5;\n"
        "addc.cc.u64 e4, e4, t;\n"
        "mul.wide.u32 t, a3, b7;\n"
        "addc.u64 e5, t, lc;\n"
        "add.cc.u64 o1, o1, odd_t;\n"
        "mul.wide.u32 odd_t, a3, b2;\n"
        "addc.cc.u64 o2, o2, odd_t;\n"
        "mul.wide.u32 odd_t, a3, b4;\n"
        "addc.cc.u64 o3, o3, odd_t;\n"
        "mul.wide.u32 odd_t, a3, b6;\n"
        "addc.cc.u64 o4, o4, odd_t;\n"
        "addc.u32 odd_cy, 0, 0;\n"
        "mul.wide.u32 t, a4, b0;\n"
        "cvt.u64.u32 odd_lc, odd_cy;\n"
        "add.cc.u64 e2, e2, t;\n"
        "mul.wide.u32 t, a4, b2;\n"
        "addc.cc.u64 e3, e3, t;\n"
        "mul.wide.u32 t, a4, b4;\n"
        "addc.cc.u64 e4, e4, t;\n"
        "mul.wide.u32 t, a4, b6;\n"
        "addc.cc.u64 e5, e5, t;\n"
        "addc.u32 cy, 0, 0;\n"
        "mul.wide.u32 odd_t, a4, b1;\n"
        "cvt.u64.u32 lc, cy;\n"
        "add.cc.u64 o2, o2, odd_t;\n"
        "mul.wide.u32 odd_t, a4, b3;\n"
        "addc.cc.u64 o3, o3, odd_t;\n"
        "mul.wide.u32 odd_t, a4, b5;\n"
        "addc.cc.u64 o4, o4, odd_t;\n"
        "mul.wide.u32 odd_t, a4, b7;\n"
        "addc.u64 o5, odd_t, odd_lc;\n"
        "mul.wide.u32 t, a5, b1;\n"
        "mul.wide.u32 odd_t, a5, b0;\n"
        "add.cc.u64 e3, e3, t;\n"
        "mul.wide.u32 t, a5, b3;\n"
        "addc.cc.u64 e4, e4, t;\n"
        "mul.wide.u32 t, a5, b5;\n"
        "addc.cc.u64 e5, e5, t;\n"
        "mul.wide.u32 t, a5, b7;\n"
        "addc.u64 e6, t, lc;\n"
        "add.cc.u64 o2, o2, odd_t;\n"
        "mul.wide.u32 odd_t, a5, b2;\n"
        "addc.cc.u64 o3, o3, odd_t;\n"
        "mul.wide.u32 odd_t, a5, b4;\n"
        "addc.cc.u64 o4, o4, odd_t;\n"
        "mul.wide.u32 odd_t, a5, b6;\n"
        "addc.cc.u64 o5, o5, odd_t;\n"
        "addc.u32 odd_cy, 0, 0;\n"
        "mul.wide.u32 t, a6, b0;\n"
        "cvt.u64.u32 odd_lc, odd_cy;\n"
        "add.cc.u64 e3, e3, t;\n"
        "mul.wide.u32 t, a6, b2;\n"
        "addc.cc.u64 e4, e4, t;\n"
        "mul.wide.u32 t, a6, b4;\n"
        "addc.cc.u64 e5, e5, t;\n"
        "mul.wide.u32 t, a6, b6;\n"
        "addc.cc.u64 e6, e6, t;\n"
        "addc.u32 cy, 0, 0;\n"
        "mul.wide.u32 odd_t, a6, b1;\n"
        "cvt.u64.u32 lc, cy;\n"
        "add.cc.u64 o3, o3, odd_t;\n"
        "mul.wide.u32 odd_t, a6, b3;\n"
        "addc.cc.u64 o4, o4, odd_t;\n"
        "mul.wide.u32 odd_t, a6, b5;\n"
        "addc.cc.u64 o5, o5, odd_t;\n"
        "mul.wide.u32 odd_t, a6, b7;\n"
        "addc.u64 o6, odd_t, odd_lc;\n"
        "mul.wide.u32 t, a7, b1;\n"
        "mul.wide.u32 odd_t, a7, b0;\n"
        "add.cc.u64 e4, e4, t;\n"
        "mul.wide.u32 t, a7, b3;\n"
        "addc.cc.u64 e5, e5, t;\n"
        "mul.wide.u32 t, a7, b5;\n"
        "addc.cc.u64 e6, e6, t;\n"
        "mul.wide.u32 t, a7, b7;\n"
        "addc.u64 e7, t, lc;\n"
        "add.cc.u64 o3, o3, odd_t;\n"
        "mul.wide.u32 odd_t, a7, b2;\n"
        "addc.cc.u64 o4, o4, odd_t;\n"
        "mul.wide.u32 odd_t, a7, b4;\n"
        "addc.cc.u64 o5, o5, odd_t;\n"
        "mul.wide.u32 odd_t, a7, b6;\n"
        "addc.cc.u64 o6, o6, odd_t;\n"
        "addc.u32 o15, 0, 0;\n"
        "mov.b64 {x0,x1}, e0;\n"
        "\tmov.b64 {x2,x3}, e1;\n"
        "\tmov.b64 {x4,x5}, e2;\n"
        "\tmov.b64 {x6,x7}, e3;\n"
        "\tmov.b64 {x8,x9}, e4;\n"
        "\tmov.b64 {x10,x11}, e5;\n"
        "\tmov.b64 {x12,x13}, e6;\n"
        "\tmov.b64 {x14,x15}, e7;\n"
        "\tmov.b64 {y1,y2}, o0;\n"
        "\tmov.b64 {y3,y4}, o1;\n"
        "\tmov.b64 {y5,y6}, o2;\n"
        "\tmov.b64 {y7,y8}, o3;\n"
        "\tmov.b64 {y9,y10}, o4;\n"
        "\tmov.b64 {y11,y12}, o5;\n"
        "\tmov.b64 {y13,y14}, o6;\n"
        "\tadd.cc.u32 x1, x1, y1;\n"
        "\taddc.cc.u32 x2, x2, y2;\n"
        "\taddc.cc.u32 x3, x3, y3;\n"
        "\taddc.cc.u32 x4, x4, y4;\n"
        "\taddc.cc.u32 x5, x5, y5;\n"
        "\taddc.cc.u32 x6, x6, y6;\n"
        "\taddc.cc.u32 x7, x7, y7;\n"
        "\taddc.cc.u32 x8, x8, y8;\n"
        "\taddc.cc.u32 x9, x9, y9;\n"
        "\taddc.cc.u32 x10, x10, y10;\n"
        "\taddc.cc.u32 x11, x11, y11;\n"
        "\taddc.cc.u32 x12, x12, y12;\n"
        "\taddc.cc.u32 x13, x13, y13;\n"
        "\taddc.cc.u32 x14, x14, y14;\n"
        "\taddc.u32 x15, x15, o15;\n"
        "\t\n"
        "mov.b64 %0, {x0,x1};\n"
        "mov.b64 %1, {x2,x3};\n"
        "mov.b64 %2, {x4,x5};\n"
        "mov.b64 %3, {x6,x7};\n"
        "mov.b64 %4, {x8,x9};\n"
        "mov.b64 %5, {x10,x11};\n"
        "mov.b64 %6, {x12,x13};\n"
        "mov.b64 %7, {x14,x15};\n"
        "}\n"

        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3),"=l"(r4),"=l"(r5),"=l"(r6),"=l"(r7)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    out[0]=r0;out[1]=r1;out[2]=r2;out[3]=r3;out[4]=r4;out[5]=r5;out[6]=r6;out[7]=r7;
}

/* QSB_GLV_LEAN (e2e 2): the split's 32x32 products as explicit PTX mul.wide.u32 /
 * mad.wide.u32, and the high15 overflow count from the add's carry flag.  The C form
 * (uint64_t)a*b is the same exact product, but ptxas lowers it through a 64x64
 * multiply whose zero high halves it keeps in a uniform register (one wasted IADD3
 * per product), and counts overflow with two compares.  Every value is identical;
 * with the switch off the original lines below compile unchanged. */
#ifndef QSB_GLV_LEAN
#define QSB_GLV_LEAN 1
#endif
#if QSB_GLV_LEAN != 0 && QSB_GLV_LEAN != 1
#error QSB_GLV_LEAN must be 0 or 1
#endif
#if QSB_GLV_LEAN
__device__ __forceinline__ uint64_t q9_mulw(uint32_t a,uint32_t b){
    uint64_t r;asm("mul.wide.u32 %0,%1,%2;":"=l"(r):"r"(a),"r"(b));return r;
}
/* a*b+c modulo 2^64, the value of the C form (uint64_t)a*b+c. */
__device__ __forceinline__ uint64_t q9_madw(uint32_t a,uint32_t b,uint64_t c){
    uint64_t r;asm("mad.wide.u32 %0,%1,%2,%3;":"=l"(r):"r"(a),"r"(b),"l"(c));return r;
}
#endif
// GLV lattice and rounded-reciprocal constants from bitcoin-core/secp256k1
// v0.6.0 scalar_impl.h, Copyright (c) 2014 Pieter Wuille, MIT.
// The original MIT license is supplied as COPYING-secp256k1.
#ifndef QSB_GLV_HIGH15
#define QSB_GLV_HIGH15 1
#endif
#if QSB_GLV_HIGH15 != 0 && QSB_GLV_HIGH15 != 1
#error QSB_GLV_HIGH15 must be 0 or 1
#endif

/* Exact original reference and rare out-of-line wrapper. q9_coeff_high15 computes only product
 * diagonals 10..14. The fixed omitted low part is too small to change the
 * bit-383 rounding decision except in FALLBACK_WORD..0x7fffffff. Keeping this
 * path out of line prevents its full 64-product register set from becoming
 * live in the ordinary path. */
__device__ __forceinline__ void q9_coeff_reference(uint64_t out[2],const uint64_t k[4],const uint64_t g[4]){
    uint64_t p[8];q9_wide(p,k,g);
    __uint128_t t=(__uint128_t)p[6]+(p[5]>>63);out[0]=(uint64_t)t;out[1]=p[7]+(uint64_t)(t>>64);
}
template<int WHICH>
__device__ __noinline__ ulonglong2 q9_coeff_fallback(uint64_t k0,uint64_t k1,
                                                      uint64_t k2,uint64_t k3){
    const uint64_t k[4]={k0,k1,k2,k3};
    const uint64_t g1[4]={0xE893209A45DBB031ULL,0x3DAA8A1471E8CA7FULL,
                          0xE86C90E49284EB15ULL,0x3086D221A7D46BCDULL};
    const uint64_t g2[4]={0x1571B4AE8AC47F71ULL,0x221208AC9DF506C6ULL,
                          0x6F547FA90ABFE4C4ULL,0xE4437ED6010E8828ULL};
    uint64_t out[2];q9_coeff_reference(out,k,WHICH==1?g1:g2);
    ulonglong2 r;r.x=out[0];r.y=out[1];return r;
}

__device__ __forceinline__ void q9_high15_add(uint64_t *acc,uint32_t *overflow,uint64_t product){
#if QSB_GLV_LEAN
    /* The same sum and lost-2^64 count, taken from the add's carry flag. */
    asm("{add.cc.u64 %0,%0,%2; addc.u32 %1,%1,0;}":"+l"(*acc),"+r"(*overflow):"l"(product));
#else
    uint64_t before=*acc;*acc=before+product;*overflow+=(uint32_t)(*acc<before);
#endif
}

#ifndef QSB_GLV_COEFF_BOUNDS
#define QSB_GLV_COEFF_BOUNDS 1
#endif
#if QSB_GLV_COEFF_BOUNDS != 0 && QSB_GLV_COEFF_BOUNDS != 1
#error QSB_GLV_COEFF_BOUNDS must be 0 or 1
#endif

/* For the two fixed reciprocals, b7+b6 is respectively0xd85b3dee and
 * 0xe55206fe. Every incoming high15 carry is below3*2^32. Thus the first
 * two products of each diagonal10..13, plus carry, fit64bits:
 * (2^32-1)*(b7+b6)+(3*2^32-1) < 2^64.
 * Later products retain their full overflow accounting. */
__device__ __forceinline__ void q9_high15_begin(uint64_t *acc,uint32_t *overflow,
        uint64_t carry,uint64_t first,uint64_t second){
#if QSB_GLV_COEFF_BOUNDS
    *acc=carry+first+second;*overflow=0;
#else
    *acc=carry;*overflow=0;
    q9_high15_add(acc,overflow,first);
    q9_high15_add(acc,overflow,second);
#endif
}

/* New exact coefficient screen: diagonal 10 contributes only its five high
 * product words. All omitted terms are nonnegative and below 9*2^352 for g1,
 * 8*2^352 for g2. Widening the bit-383 rounding guard preserves exactness.
 * test_glv_coeff.py computes the bound and exercises the actual function. */
#ifndef QSB_GLV_HIGH10_HI
#define QSB_GLV_HIGH10_HI 1
#endif
#if QSB_GLV_HIGH10_HI != 0 && QSB_GLV_HIGH10_HI != 1
#error "QSB_GLV_HIGH10_HI must be 0 or 1"
#endif
__device__ __forceinline__ uint32_t q9_mulhi32(uint32_t a,uint32_t b) {
#ifdef __CUDA_ARCH__
    return __umulhi(a,b);
#else
    return (uint32_t)(((uint64_t)a*b)>>32);
#endif
}

#ifndef QSB_GLV_ROUND_CC
#define QSB_GLV_ROUND_CC 1
#endif
#if QSB_GLV_ROUND_CC != 0 && QSB_GLV_ROUND_CC != 1
#error "QSB_GLV_ROUND_CC must be 0 or 1"
#endif
__device__ __forceinline__ void q9_round_coeff(uint64_t out[2],uint64_t lo,uint64_t hi,uint64_t round) {
#if QSB_GLV_ROUND_CC && defined(__CUDA_ARCH__)
    asm("{add.cc.u64 %0,%2,%4; addc.u64 %1,%3,0;}"
        : "=l"(out[0]),"=l"(out[1]) : "l"(lo),"l"(hi),"l"(round));
#else
    const uint64_t rounded=lo+round;
    out[0]=rounded;out[1]=hi+(uint64_t)(rounded<lo);
#endif
}

template<int WHICH,uint32_t FALLBACK_WORD>
__device__ __forceinline__ void q9_coeff_high15(uint64_t out[2],const uint64_t k[4],const uint64_t g[4]){
    const uint32_t a3=(uint32_t)(k[1]>>32);
    const uint32_t a4=(uint32_t)k[2],a5=(uint32_t)(k[2]>>32);
    const uint32_t a6=(uint32_t)k[3],a7=(uint32_t)(k[3]>>32);
    const uint32_t b3=(uint32_t)(g[1]>>32),b4=(uint32_t)g[2];
    const uint32_t b5=(uint32_t)(g[2]>>32),b6=(uint32_t)g[3],b7=(uint32_t)(g[3]>>32);
    uint64_t carry=0,acc;uint32_t overflow,w10,w11,w12,w13,w14,w15;

    /* Diagonal 10: (3,7)..(7,3). A 64-bit sum is insufficient for five
     * products, so overflow counts its lost 2^64 units explicitly. */
#if QSB_GLV_LEAN
#if QSB_GLV_HIGH10_HI
    // b7+b6 < 2^32 for both production reciprocals: the first sum fits u32.
    const uint32_t first=q9_mulhi32(a3,b7)+q9_mulhi32(a4,b6);
    carry=(uint64_t)first+q9_mulhi32(a5,b5)+q9_mulhi32(a6,b4)+q9_mulhi32(a7,b3);
    w10=0;
#else
    q9_high15_begin(&acc,&overflow,carry,q9_mulw(a3,b7),q9_mulw(a4,b6));
    q9_high15_add(&acc,&overflow,q9_mulw(a5,b5));
    q9_high15_add(&acc,&overflow,q9_mulw(a6,b4));
    q9_high15_add(&acc,&overflow,q9_mulw(a7,b3));
    w10=(uint32_t)acc;carry=(acc>>32)|((uint64_t)overflow<<32);
#endif

    q9_high15_begin(&acc,&overflow,carry,q9_mulw(a4,b7),q9_mulw(a5,b6));
    q9_high15_add(&acc,&overflow,q9_mulw(a6,b5));
    q9_high15_add(&acc,&overflow,q9_mulw(a7,b4));
    w11=(uint32_t)acc;carry=(acc>>32)|((uint64_t)overflow<<32);

    q9_high15_begin(&acc,&overflow,carry,q9_mulw(a5,b7),q9_mulw(a6,b6));
    q9_high15_add(&acc,&overflow,q9_mulw(a7,b5));
    w12=(uint32_t)acc;carry=(acc>>32)|((uint64_t)overflow<<32);

    q9_high15_begin(&acc,&overflow,carry,q9_mulw(a6,b7),q9_mulw(a7,b6));
    w13=(uint32_t)acc;carry=(acc>>32)|((uint64_t)overflow<<32);

    acc=q9_madw(a7,b7,carry);
#else
    q9_high15_begin(&acc,&overflow,carry,(uint64_t)a3*b7,(uint64_t)a4*b6);
    q9_high15_add(&acc,&overflow,(uint64_t)a5*b5);
    q9_high15_add(&acc,&overflow,(uint64_t)a6*b4);
    q9_high15_add(&acc,&overflow,(uint64_t)a7*b3);
    w10=(uint32_t)acc;carry=(acc>>32)|((uint64_t)overflow<<32);

    q9_high15_begin(&acc,&overflow,carry,(uint64_t)a4*b7,(uint64_t)a5*b6);
    q9_high15_add(&acc,&overflow,(uint64_t)a6*b5);
    q9_high15_add(&acc,&overflow,(uint64_t)a7*b4);
    w11=(uint32_t)acc;carry=(acc>>32)|((uint64_t)overflow<<32);

    q9_high15_begin(&acc,&overflow,carry,(uint64_t)a5*b7,(uint64_t)a6*b6);
    q9_high15_add(&acc,&overflow,(uint64_t)a7*b5);
    w12=(uint32_t)acc;carry=(acc>>32)|((uint64_t)overflow<<32);

    q9_high15_begin(&acc,&overflow,carry,(uint64_t)a6*b7,(uint64_t)a7*b6);
    w13=(uint32_t)acc;carry=(acc>>32)|((uint64_t)overflow<<32);

    acc=carry+(uint64_t)a7*b7;
#endif
    w14=(uint32_t)acc;w15=(uint32_t)(acc>>32);
    (void)w10;

    constexpr uint32_t guard=(QSB_GLV_HIGH10_HI && QSB_GLV_LEAN)
        ? (WHICH==1 ? 0x7ffffff7U : 0x7ffffff8U) : FALLBACK_WORD;
    if(w11<guard || w11>=0x80000000U){
        uint64_t lo=(uint64_t)w12|((uint64_t)w13<<32);
        uint64_t hi=(uint64_t)w14|((uint64_t)w15<<32);
#if QSB_GLV_RND
        /* (hi:lo) + (w11 >> 31) mod 2^128: the carry out of w11 + 2^31 is bit 31 of w11 */
        asm("{\n\t.reg .u32 t;\n\t"
            "add.cc.u32 t,%4,0x80000000;\n\t"
            "addc.cc.u64 %0,%2,0;\n\t"
            "addc.u64 %1,%3,0;\n\t}"
            : "=l"(out[0]),"=l"(out[1]) : "l"(lo),"l"(hi),"r"(w11));
#else
        const uint64_t round=(uint64_t)(w11>>31);
        q9_round_coeff(out,lo,hi,round);
#endif
    }else{
        ulonglong2 r=q9_coeff_fallback<WHICH>(k[0],k[1],k[2],k[3]);
        out[0]=r.x;out[1]=r.y;
    }
}

__device__ __forceinline__ void q9_coeff_g1(uint64_t out[2],const uint64_t k[4],const uint64_t g[4]){
#if QSB_GLV_HIGH15
    q9_coeff_high15<1,0x7ffffffcU>(out,k,g);
#else
    q9_coeff_reference(out,k,g);
#endif
}
__device__ __forceinline__ void q9_coeff_g2(uint64_t out[2],const uint64_t k[4],const uint64_t g[4]){
#if QSB_GLV_HIGH15
    q9_coeff_high15<2,0x7ffffffdU>(out,k,g);
#else
    q9_coeff_reference(out,k,g);
#endif
}
__device__ __forceinline__ void q9_sub4(uint64_t out[4],const uint64_t a[4],const uint64_t b[4]){
    uint64_t r0,r1,r2,r3;
    asm("{sub.cc.u64 %0,%4,%8;subc.cc.u64 %1,%5,%9;subc.cc.u64 %2,%6,%10;subc.u64 %3,%7,%11;}"
        :"=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        :"l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    out[0]=r0;out[1]=r1;out[2]=r2;out[3]=r3;
}
template<int W> __device__ __forceinline__ void q9_small_product(uint64_t out[4],const uint64_t a[2],const uint32_t b[W]){
    uint32_t aa[4]={(uint32_t)a[0],(uint32_t)(a[0]>>32),(uint32_t)a[1],(uint32_t)(a[1]>>32)};
    uint32_t rr[9]={0,0,0,0,0,0,0,0,0};
    #pragma unroll
    for(int i=0;i<4;i++){
        uint64_t carry=0;
        #pragma unroll
        for(int j=0;j<W;j++){
            uint64_t t=(uint64_t)aa[i]*b[j]+rr[i+j]+carry;
            rr[i+j]=(uint32_t)t;carry=t>>32;
        }
        rr[i+W]=(uint32_t)carry;
    }
    #pragma unroll
    for(int j=0;j<4;j++)out[j]=(uint64_t)rr[2*j]|((uint64_t)rr[2*j+1]<<32);
}
__device__ __forceinline__ void q9_abs128(uint64_t out[2],unsigned *negative,const uint64_t in[4]){
    unsigned s=(unsigned)(in[3]>>63);uint64_t m=0ULL-s;
    __uint128_t t=(__uint128_t)(in[0]^m)+s;out[0]=(uint64_t)t;out[1]=(in[1]^m)+(uint64_t)(t>>64);*negative=s;
}

#ifndef QSB_GLV_RESIDUAL3
#define QSB_GLV_RESIDUAL3 1
#endif
#if QSB_GLV_RESIDUAL3 != 0 && QSB_GLV_RESIDUAL3 != 1
#error QSB_GLV_RESIDUAL3 must be 0 or 1
#endif
#ifndef QSB_GLV_RESIDUAL129
#define QSB_GLV_RESIDUAL129 1
#endif
#if QSB_GLV_RESIDUAL129 != 0 && QSB_GLV_RESIDUAL129 != 1
#error QSB_GLV_RESIDUAL129 must be 0 or 1
#endif

/* Original four-product residual schedule, retained as the exact OFF path. */
__device__ __forceinline__ void q9_glv_residual_reference(
    const uint64_t k[4],const uint64_t c1[2],const uint64_t c2[2],
    const uint32_t a1[4],const uint32_t a2[5],const uint32_t b1[4],
    uint64_t r1[2],uint64_t r2[2],unsigned *s1,unsigned *s2) {
    uint64_t p[4],q[4],z[4];
    q9_small_product<4>(p,c1,a1);q9_small_product<5>(q,c2,a2);
    q9_sub4(z,k,p);q9_sub4(z,z,q);q9_abs128(r1,s1,z);
    q9_small_product<4>(p,c1,b1);q9_small_product<4>(q,c2,a1);
    q9_sub4(z,p,q);q9_abs128(r2,s2,z);
}

__device__ __forceinline__ void q9_add_upper128(uint64_t x[4],uint64_t lo,uint64_t hi) {
    uint64_t r2,r3;
    asm("{add.cc.u64 %0,%2,%4;addc.u64 %1,%3,%5;}"
        : "=l"(r2),"=l"(r3) : "l"(x[2]),"l"(x[3]),"l"(lo),"l"(hi));
    x[2]=r2;x[3]=r3;
}

/* Three-product residual identity. With c=a+b:
 *   P=a*(c1+c2), Q=b*c2, R=c*c1;
 *   z1=k-P-Q, z2=R-P.
 * c1+c2 is explicitly 129 bits. The carry contributes (a<<128) modulo
 * 2^256; c's implicit top word one contributes (c1<<128). */
__device__ __forceinline__ void q9_glv_residual3(
    const uint64_t k[4],const uint64_t c1[2],const uint64_t c2[2],
    const uint32_t a1[4],const uint32_t a2[5],const uint32_t b1[4],
    uint64_t r1[2],uint64_t r2[2],unsigned *s1,unsigned *s2) {
    uint64_t sum0,sum1;uint32_t sum2;
    asm("{add.cc.u64 %0,%3,%5;addc.cc.u64 %1,%4,%6;addc.u32 %2,0,0;}"
        : "=l"(sum0),"=l"(sum1),"=r"(sum2)
        : "l"(c1[0]),"l"(c1[1]),"l"(c2[0]),"l"(c2[1]));
    const uint64_t sum[2]={sum0,sum1};
    const uint64_t a_lo=(uint64_t)a1[0]|((uint64_t)a1[1]<<32);
    const uint64_t a_hi=(uint64_t)a1[2]|((uint64_t)a1[3]<<32);
    const uint64_t carry_mask=0ULL-(uint64_t)sum2;
    uint64_t p[4],q[4],rr[4],z[4];
    q9_small_product<4>(p,sum,a1);
    q9_add_upper128(p,a_lo&carry_mask,a_hi&carry_mask);
    q9_small_product<4>(q,c2,b1);
    q9_small_product<4>(rr,c1,a2); /* a2[0..3] is c mod 2^128. */
    q9_add_upper128(rr,c1[0],c1[1]);
    q9_sub4(z,k,p);q9_sub4(z,z,q);q9_abs128(r1,s1,z);
    q9_sub4(z,rr,p);q9_abs128(r2,s2,z);
}

/* Exact modulo-2^129 arithmetic is sufficient here: the rounded GLV
 * coefficients guarantee both signed residuals have magnitude below 2^128.
 * The product helper retains words 0..3 and bit 128. Each truncated row's
 * carry and the parity of diagonal four both contribute to that top bit. */
struct q9_u129 { uint64_t lo,hi;uint32_t top; };

#if !QSB_GLV_EO
__device__ __forceinline__ q9_u129 q9_product129(const uint64_t x[2],const uint32_t d[4]) {
    const uint32_t x0=(uint32_t)x[0],x1=(uint32_t)(x[0]>>32);
    const uint32_t x2=(uint32_t)x[1],x3=(uint32_t)(x[1]>>32);
#if QSB_GLV_LEAN
    uint64_t t=q9_mulw(x0,d[0]);const uint32_t w0=(uint32_t)t;uint64_t carry=t>>32;
    t=q9_madw(x0,d[1],carry);uint32_t w1=(uint32_t)t;carry=t>>32;
    t=q9_madw(x0,d[2],carry);uint32_t w2=(uint32_t)t;carry=t>>32;
    t=q9_madw(x0,d[3],carry);uint32_t w3=(uint32_t)t;uint32_t top=(uint32_t)(t>>32);
    t=q9_madw(x1,d[0],w1);w1=(uint32_t)t;carry=t>>32;
    t=q9_madw(x1,d[1],(uint64_t)w2+carry);w2=(uint32_t)t;carry=t>>32;
    t=q9_madw(x1,d[2],(uint64_t)w3+carry);w3=(uint32_t)t;top^=(uint32_t)(t>>32);
    t=q9_madw(x2,d[0],w2);w2=(uint32_t)t;carry=t>>32;
    t=q9_madw(x2,d[1],(uint64_t)w3+carry);w3=(uint32_t)t;top^=(uint32_t)(t>>32);
    t=q9_madw(x3,d[0],w3);w3=(uint32_t)t;top^=(uint32_t)(t>>32);
#else
    uint64_t t=(uint64_t)x0*d[0];const uint32_t w0=(uint32_t)t;uint64_t carry=t>>32;
    t=(uint64_t)x0*d[1]+carry;uint32_t w1=(uint32_t)t;carry=t>>32;
    t=(uint64_t)x0*d[2]+carry;uint32_t w2=(uint32_t)t;carry=t>>32;
    t=(uint64_t)x0*d[3]+carry;uint32_t w3=(uint32_t)t;uint32_t top=(uint32_t)(t>>32);
    t=(uint64_t)x1*d[0]+w1;w1=(uint32_t)t;carry=t>>32;
    t=(uint64_t)x1*d[1]+w2+carry;w2=(uint32_t)t;carry=t>>32;
    t=(uint64_t)x1*d[2]+w3+carry;w3=(uint32_t)t;top^=(uint32_t)(t>>32);
    t=(uint64_t)x2*d[0]+w2;w2=(uint32_t)t;carry=t>>32;
    t=(uint64_t)x2*d[1]+w3+carry;w3=(uint32_t)t;top^=(uint32_t)(t>>32);
    t=(uint64_t)x3*d[0]+w3;w3=(uint32_t)t;top^=(uint32_t)(t>>32);
#endif
    top^=(x1&d[3])^(x2&d[2])^(x3&d[1]);
    q9_u129 r={(uint64_t)w0|((uint64_t)w1<<32),
                 (uint64_t)w2|((uint64_t)w3<<32),top&1U};
    return r;
}
#endif

#if QSB_GLV_EO
/* x*d mod 2^129 from column-pair accumulators: E0 = x0d0 (words 0-1), O0 = x0d1+x1d0
 * (words 1-2, carry cc3 into word 3), E1 = x0d2+x1d1+x2d0 (words 2-3, carries cc4 into
 * word 4), O1 = x0d3+x1d2+x2d1+x3d0 mod 2^64 (words 3-4). Word 4 only matters for its
 * parity (bit 0 of top is bit 128); X is added into it: the caller's parity terms, which
 * include the low bits of x1*d3, x2*d2 and x3*d1 (bit 0 of each is x_i & d_j & 1, so the words
 * x_i with odd d_j are added whole; their bits above bit 0 only reach unused bits of word 4). */
__device__ __forceinline__ q9_u129 q9_product129_rx(const uint64_t x[2],const uint32_t d[4],uint32_t X) {
    uint32_t w0,w1,w2,w3,top;
    asm("{\n\t"
        ".reg .u32 x0,x1,x2,x3,a,b,c,e,t,cc3,cc4;\n\t"
        ".reg .u64 E0,O0,E1,O1,m;\n\t"
        "mov.b64 {x0,x1},%5;\n\t"
        "mov.b64 {x2,x3},%6;\n\t"
        "mul.wide.u32 E0,x0,%7;\n\t"
        "mul.wide.u32 O0,x0,%8;\n\t"
        "mul.wide.u32 m,x1,%7;\n\t"
        "add.cc.u64 O0,O0,m;\n\t"
        "addc.u32 cc3,0,0;\n\t"
        "mul.wide.u32 E1,x0,%9;\n\t"
        "mul.wide.u32 m,x1,%8;\n\t"
        "add.cc.u64 E1,E1,m;\n\t"
        "addc.u32 cc4,0,0;\n\t"
        "mul.wide.u32 m,x2,%7;\n\t"
        "add.cc.u64 E1,E1,m;\n\t"
        "addc.u32 cc4,cc4,0;\n\t"
        "mul.wide.u32 O1,x0,%10;\n\t"
        "mad.wide.u32 O1,x1,%9,O1;\n\t"
        "mad.wide.u32 O1,x2,%8,O1;\n\t"
        "mad.wide.u32 O1,x3,%7,O1;\n\t"
        "mov.b64 {%0,a},E0;\n\t"
        "mov.b64 {b,c},O0;\n\t"
        "add.cc.u32 %1,a,b;\n\t"
        "mov.b64 {a,b},E1;\n\t"
        "addc.cc.u32 %2,a,c;\n\t"
        "mov.b64 {c,e},O1;\n\t"
        "addc.cc.u32 %3,b,c;\n\t"
        "addc.u32 t,e,cc4;\n\t"
        "add.cc.u32 %3,%3,cc3;\n\t"
        "addc.u32 %4,t,%11;\n\t"
        "}"
        : "=r"(w0),"=r"(w1),"=r"(w2),"=r"(w3),"=r"(top)
        : "l"(x[0]),"l"(x[1]),"r"(d[0]),"r"(d[1]),"r"(d[2]),"r"(d[3]),"r"(X));
    q9_u129 r={(uint64_t)w0|((uint64_t)w1<<32),
               (uint64_t)w2|((uint64_t)w3<<32),top};   /* bit 0 of top is bit 128 */
    return r;
}
/* X = extra + the word-4 parity addend of x*d (see above) */
__device__ __forceinline__ q9_u129 q9_product129_r(const uint64_t x[2],const uint32_t d[4],uint32_t extra) {
    const uint32_t x1=(uint32_t)(x[0]>>32),x2=(uint32_t)x[1],x3=(uint32_t)(x[1]>>32);
    return q9_product129_rx(x,d,extra+((d[3]&1U)?x1:0U)+((d[2]&1U)?x2:0U)+((d[1]&1U)?x3:0U));
}
__device__ __forceinline__ q9_u129 q9_product129(const uint64_t x[2],const uint32_t d[4]) {
    q9_u129 r=q9_product129_r(x,d,0U);r.top&=1U;return r;
}
#endif
__device__ __forceinline__ q9_u129 q9_sub129(q9_u129 a,q9_u129 b) {
    q9_u129 r;uint32_t top;
    asm("{sub.cc.u64 %0,%3,%6;subc.cc.u64 %1,%4,%7;subc.u32 %2,%5,%8;}"
        : "=l"(r.lo),"=l"(r.hi),"=r"(top)
        : "l"(a.lo),"l"(a.hi),"r"(a.top),"l"(b.lo),"l"(b.hi),"r"(b.top));
    r.top=top&1U;return r;
}

__device__ __forceinline__ void q9_abs129(uint64_t out[2],unsigned *negative,q9_u129 in) {
    const unsigned s=in.top&1U;const uint64_t m=0ULL-(uint64_t)s;
    const __uint128_t t=(__uint128_t)(in.lo^m)+s;
    out[0]=(uint64_t)t;out[1]=(in.hi^m)+(uint64_t)(t>>64);*negative=s;
}

__device__ __forceinline__ void q9_glv_residual129(
    const uint64_t k[4],const uint64_t c1[2],const uint64_t c2[2],
    const uint32_t a1[4],const uint32_t a2[5],const uint32_t b1[4],
    uint64_t r1[2],uint64_t r2[2],unsigned *s1,unsigned *s2) {
    uint64_t sum0,sum1;uint32_t sum2;
    asm("{add.cc.u64 %0,%3,%5;addc.cc.u64 %1,%4,%6;addc.u32 %2,0,0;}"
        : "=l"(sum0),"=l"(sum1),"=r"(sum2)
        : "l"(c1[0]),"l"(c1[1]),"l"(c2[0]),"l"(c2[1]));
    const uint64_t sum[2]={sum0,sum1};
    q9_u129 p=q9_product129(sum,a1);
    p.top^=(sum2&(a1[0]&1U));
    const q9_u129 q=q9_product129(c2,b1);
    q9_u129 rr=q9_product129(c1,a2); /* a2[0..3] is c modulo 2^128. */
    rr.top^=(uint32_t)(c1[0]&1ULL);   /* c has one implicit bit at 128. */
    const q9_u129 kk={k[0],k[1],(uint32_t)(k[2]&1ULL)};
    const q9_u129 z1=q9_sub129(q9_sub129(kk,p),q);
    const q9_u129 z2=q9_sub129(rr,p);
    q9_abs129(r1,s1,z1);q9_abs129(r2,s2,z2);
}

__device__ __forceinline__ void q9_glv_split(const uint64_t input[4],uint64_t r1[2],uint64_t r2[2],unsigned *s1,unsigned *s2){
    const uint64_t n[4]={0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
    uint64_t k[4]={input[0],input[1],input[2],input[3]};
    if(k[3]==n[3]&&(k[2]>n[2]||(k[2]==n[2]&&(k[1]>n[1]||(k[1]==n[1]&&k[0]>=n[0])))))q9_sub4(k,k,n);
    const uint64_t g1[4]={0xE893209A45DBB031ULL,0x3DAA8A1471E8CA7FULL,0xE86C90E49284EB15ULL,0x3086D221A7D46BCDULL};
    const uint64_t g2[4]={0x1571B4AE8AC47F71ULL,0x221208AC9DF506C6ULL,0x6F547FA90ABFE4C4ULL,0xE4437ED6010E8828ULL};
    const uint32_t a1[4]={0x9284eb15,0xe86c90e4,0xa7d46bcd,0x3086d221};
    const uint32_t a2[5]={0x9d44cfd8,0x57c1108d,0xa8e2f3f6,0x14ca50f7,1};
    const uint32_t b1[4]={0x0abfe4c3,0x6f547fa9,0x010e8828,0xe4437ed6};
    uint64_t c1[2],c2[2];q9_coeff_g1(c1,k,g1);q9_coeff_g2(c2,k,g2);
#if QSB_GLV_RESIDUAL129
    q9_glv_residual129(k,c1,c2,a1,a2,b1,r1,r2,s1,s2);
#elif QSB_GLV_RESIDUAL3
    q9_glv_residual3(k,c1,c2,a1,a2,b1,r1,r2,s1,s2);
#else
    q9_glv_residual_reference(k,c1,c2,a1,a2,b1,r1,r2,s1,s2);
#endif
}

/* QSB_GLV_ZDEC: the decode reads the signed residual directly. With s = bit 128 of z and
 * W = z - s mod 2^128, the magnitude is W for s = 0 and ~W for s = 1 (for s = 1,
 * |z| = (2^128 - z) mod 2^128 = ~(z - 1)), so mag = W ^ M with M = -s. q9_abs129 (mask, four
 * XOR, four-word add) is replaced by the four-word subtraction; q9_bigtbl_code_z undoes the
 * XOR per chunk, where it cancels for the radix chunks. */
#if QSB_GLV_ZDEC
#if !QSB_GLV_RESIDUAL129 || !QSB_DIGIT_LEAN || !QSB_FOUR_HOT
#error "QSB_GLV_ZDEC is written for the RESIDUAL129 / DIGIT_LEAN / FOUR_HOT path"
#endif
/* Only bit 0 of every top word is meaningful on this path (bit 128); no masks. */
__device__ __forceinline__ q9_u129 q9_sub129_z(q9_u129 a,q9_u129 b) {
    q9_u129 r;
    asm("{sub.cc.u64 %0,%3,%6;subc.cc.u64 %1,%4,%7;subc.u32 %2,%5,%8;}"
        : "=l"(r.lo),"=l"(r.hi),"=r"(r.top)
        : "l"(a.lo),"l"(a.hi),"r"(a.top),"l"(b.lo),"l"(b.hi),"r"(b.top));
    return r;
}
/* w = z + M = z - s (M = -s on all 128 bits), m32 = -s, top keeps bit 0 = s. */
__device__ __forceinline__ void q9_zdec(uint64_t w[2],uint32_t *top,uint32_t *m32,q9_u129 z) {
    uint32_t m;
    asm("{\n\t.reg .u64 M;\n\t"
        "bfe.s32 %2,%5,0,1;\n\t"
        "mov.b64 M,{%2,%2};\n\t"
        "add.cc.u64 %0,%3,M;\n\t"
        "addc.u64 %1,%4,M;\n\t}"
        : "=l"(w[0]),"=l"(w[1]),"=r"(m) : "l"(z.lo),"l"(z.hi),"r"(z.top));
    *top=z.top;*m32=m;
}
/* QSB_ZSPLIT_NOPRE (kill switch, default 1): q9_glv_split_z skips the k >= n pre-reduction.
 * k in [n, 2^256) has probability (2^256-n)/2^256 < 2^-127 for the hashed scalar; such a
 * k only changes that one candidate's recovered key, which the host exact gate rejects
 * (a lost candidate, never a false hit). 0 keeps the conditional subtraction. */
#ifndef QSB_ZSPLIT_NOPRE
#define QSB_ZSPLIT_NOPRE 1
#endif
/* Returns q_nonzero | p_nonzero << 1 with p_nonzero = |z1| mod 2^128 != 0, exactly the base's
 * mag != 0: |z| mod 2^128 is 0 exactly when z mod 2^128 is 0. */
__device__ __forceinline__ unsigned q9_glv_split_z(const uint64_t input[4],uint64_t w1[2],uint64_t w2[2],
                                                   uint32_t *t1,uint32_t *t2,uint32_t *m1,uint32_t *m2){
    const uint64_t n[4]={0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
    uint64_t k[4]={input[0],input[1],input[2],input[3]};
#if QSB_ZSPLIT_NOPRE
    (void)n;   /* k >= n needs k[3] == 2^64-1 and k[2] >= 2^64-2: probability < 2^-127 */
#else
#ifndef QSB_GLV_NOREDUCE
#define QSB_GLV_NOREDUCE 1   /* 1: no k >= n pre-reduction (z >= n has probability ~2^-128; the host gate re-checks every hit) -- i34-9 28f6b89d */
#endif
#if !QSB_GLV_NOREDUCE
    if(k[3]==n[3]&&(k[2]>n[2]||(k[2]==n[2]&&(k[1]>n[1]||(k[1]==n[1]&&k[0]>=n[0])))))q9_sub4(k,k,n);
#else
    (void)n;
#endif
#endif
    const uint64_t g1[4]={0xE893209A45DBB031ULL,0x3DAA8A1471E8CA7FULL,0xE86C90E49284EB15ULL,0x3086D221A7D46BCDULL};
    const uint64_t g2[4]={0x1571B4AE8AC47F71ULL,0x221208AC9DF506C6ULL,0x6F547FA90ABFE4C4ULL,0xE4437ED6010E8828ULL};
    const uint32_t a1[4]={0x9284eb15,0xe86c90e4,0xa7d46bcd,0x3086d221};
    const uint32_t a2[5]={0x9d44cfd8,0x57c1108d,0xa8e2f3f6,0x14ca50f7,1};
    const uint32_t b1[4]={0x0abfe4c3,0x6f547fa9,0x010e8828,0xe4437ed6};
    uint64_t c1[2],c2[2];q9_coeff_g1(c1,k,g1);q9_coeff_g2(c2,k,g2);
#if QSB_GLV_EO
    /* sum = c1 + c2 (129 bits) and, for P = sum*a1, the word-4 parity addend of q9_product129_r:
     * sum2 (a1[0] odd: sum2*a1*2^128 adds sum2 at bit 128) plus sum words 1 and 2 (a1[3], a1[2]
     * odd, a1[1] even), folded into the carry capture. */
    uint64_t sum0,sum1;uint32_t xp;
    asm("{\n\t.reg .u32 a,b,c,d,t;\n\t"
        "add.cc.u64 %0,%3,%5;\n\t"
        "addc.cc.u64 %1,%4,%6;\n\t"
        "mov.b64 {a,b},%0;\n\t"
        "mov.b64 {c,d},%1;\n\t"
        "add.u32 t,b,c;\n\t"
        "addc.u32 %2,t,0;\n\t}"
        : "=l"(sum0),"=l"(sum1),"=r"(xp)
        : "l"(c1[0]),"l"(c1[1]),"l"(c2[0]),"l"(c2[1]));
    const uint64_t sum[2]={sum0,sum1};
    static_assert((0x9284eb15u&1u)==1u && (0xa7d46bcdu&1u)==1u && (0x3086d221u&1u)==1u && (0xe86c90e4u&1u)==0u,
                  "parity addend of P assumes a1 words 0, 2, 3 odd and word 1 even");
    const q9_u129 p=q9_product129_rx(sum,a1,xp);
    const q9_u129 q=q9_product129_r(c2,b1,0U);
    const q9_u129 rr=q9_product129_r(c1,a2,(uint32_t)c1[0]); /* a2[0..3] is c mod 2^128; c's bit 128 adds c1 */
#else
    uint64_t sum0,sum1;uint32_t sum2;
    asm("{add.cc.u64 %0,%3,%5;addc.cc.u64 %1,%4,%6;addc.u32 %2,0,0;}"
        : "=l"(sum0),"=l"(sum1),"=r"(sum2)
        : "l"(c1[0]),"l"(c1[1]),"l"(c2[0]),"l"(c2[1]));
    const uint64_t sum[2]={sum0,sum1};
    q9_u129 p=q9_product129(sum,a1);
    p.top^=sum2;                      /* a1[0] is odd */
    const q9_u129 q=q9_product129(c2,b1);
    q9_u129 rr=q9_product129(c1,a2); /* a2[0..3] is c modulo 2^128. */
    rr.top^=(uint32_t)c1[0];          /* c has one implicit bit at 128: bit 0 of c1 */
#endif
    const q9_u129 kk={k[0],k[1],(uint32_t)k[2]};
    const q9_u129 z1=q9_sub129_z(q9_sub129_z(kk,p),q);
    const q9_u129 z2=q9_sub129_z(rr,p);
    q9_zdec(w1,t1,m1,z1);q9_zdec(w2,t2,m2,z2);
    const unsigned p_nonzero=(z1.lo|z1.hi)!=0;
    const unsigned q_nonzero=(z2.lo|z2.hi)!=0;
    return q_nonzero|(p_nonzero<<1);
}
#endif
