#pragma once
/* Host-only exact GLV split and joint-low signed recoding for the Subset CPU
 * co-grinder. Include inside namespace qcpu, after <stdint.h>. C++11 plus the
 * unsigned __int128 extension already used by CpuGrindSubset.h is required.
 *
 * Lattice/reciprocal constants match the local GPL-3 QSB GLVScalar.cuh splitter.
 * No CUDA helper, approximate split, lookup truncation, or signed overflow is
 * used here. Certified high products plus an exact fallback preserve bit-383
 * rounding; the strict GLV bound makes the shared modulo-2^129 residual exact.
 */

#ifndef QSB_CPU_GLV_COEFF6
#define QSB_CPU_GLV_COEFF6 1
#endif
#ifndef QSB_CPU_GLV_RESIDUAL129
#define QSB_CPU_GLV_RESIDUAL129 1
#endif
#if (QSB_CPU_GLV_COEFF6 != 0 && QSB_CPU_GLV_COEFF6 != 1) || (QSB_CPU_GLV_RESIDUAL129 != 0 && QSB_CPU_GLV_RESIDUAL129 != 1)
#error CPU GLV optimization switches must be 0 or 1
#endif

static const unsigned CPU_GLV_LOW_BITS = 8;
static const unsigned CPU_GLV_LOW_HALF = 128;
static const unsigned CPU_GLV_HIGH_WINDOWS = 5;
static const unsigned CPU_GLV_LOGICAL_WINDOWS = 11;
static const uint32_t CPU_GLV_TOP_ENTRIES = 10659986u;
static const uint32_t CPU_GLV_JOINT_ENTRIES = 33025u; /* per +C/-C copy, including (0,0) */
static const unsigned CPU_GLV_HIGH_SHIFTS[5] = {8, 32, 56, 80, 104};
static const uint32_t CPU_GLV_HIGH_ENTRIES[5] = {
    8388608u, 8388608u, 8388608u, 8388608u, CPU_GLV_TOP_ENTRIES
};
static const uint64_t CPU_GLV_BETA_LE64[4] = {
    0xC1396C28719501EEULL, 0x9CF0497512F58995ULL,
    0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL
};
static const uint64_t CPU_GLV_ORDER_LE64[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};
static const uint64_t CPU_GLV_G1_LE64[4] = {
    0xE893209A45DBB031ULL, 0x3DAA8A1471E8CA7FULL,
    0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL
};
static const uint64_t CPU_GLV_G2_LE64[4] = {
    0x1571B4AE8AC47F71ULL, 0x221208AC9DF506C6ULL,
    0x6F547FA90ABFE4C4ULL, 0xE4437ED6010E8828ULL
};
static const uint64_t CPU_GLV_A1_LE64[2] = {
    0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL
};
static const uint64_t CPU_GLV_A2_LE64[3] = {
    0x57C1108D9D44CFD8ULL, 0x14CA50F7A8E2F3F6ULL, 1ULL
};
static const uint64_t CPU_GLV_B1_LE64[2] = {
    0x6F547FA90ABFE4C3ULL, 0xE4437ED6010E8828ULL
};

typedef unsigned __int128 cpu_glv_u128;

/* Schoolbook full product, little-endian base 2^64. Each accumulator is at
 * most (2^64-1)^2 + 2*(2^64-1) = 2^128-1, so no carry is discarded. The final
 * destination of each row has not been touched by earlier rows. */
template <int NA, int NB>
static inline void cpu_glv_mul(uint64_t *out, const uint64_t *a, const uint64_t *b) {
    for (int i = 0; i < NA + NB; ++i) out[i] = 0;
    for (int i = 0; i < NA; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < NB; ++j) {
            const cpu_glv_u128 t = (cpu_glv_u128)a[i] * b[j] + out[i + j] + carry;
            out[i + j] = (uint64_t)t;
            carry = (uint64_t)(t >> 64);
        }
        out[i + NB] = carry;
    }
}

/* Supports out == a or out == b. Arithmetic is unsigned; subtraction is
 * intentionally modulo 2^(64*N), with the comparison retaining the borrow. */
template <int NL>
static inline void cpu_glv_sub(uint64_t *out, const uint64_t *a, const uint64_t *b) {
    unsigned borrow = 0;
    for (int i = 0; i < NL; ++i) {
        const cpu_glv_u128 sub = (cpu_glv_u128)b[i] + borrow;
        const cpu_glv_u128 av = (cpu_glv_u128)a[i];
        out[i] = (uint64_t)(av - sub);
        borrow = (unsigned)(av < sub);
    }
}

static inline void cpu_glv_coeff_reference(uint64_t out[2], const uint64_t k[4], const uint64_t g[4]) {
    uint64_t product[8];
    cpu_glv_mul<4, 4>(product, k, g);
    /* floor((k*g + 2^383) / 2^384). The low 383 bits cannot affect this
     * rounding carry. Both fixed reciprocals keep the rounded result <2^128. */
    const cpu_glv_u128 lo = (cpu_glv_u128)product[6] + (product[5] >> 63);
    out[0] = (uint64_t)lo;
    out[1] = product[7] + (uint64_t)(lo >> 64);
}

/* The slow branch is kept out of the six-product function's register lifetime.
 * It is also the exact answer for any coefficient not covered by a fixed bound. */
static __attribute__((noinline)) void cpu_glv_coeff_fallback(uint64_t out[2], const uint64_t k[4], const uint64_t g[4]) {
    cpu_glv_coeff_reference(out, k, g);
}

template <unsigned DELTA>
static inline void cpu_glv_coeff6(uint64_t out[2], const uint64_t k[4], const uint64_t g[4]) {
    /* With B=2^64, omitted columns contribute delta in [0,DELTA] to word 5.
     * DELTA=floor((3*(B-1)+sum(g_i)-1)/B): 5 for g1, 4 for g2.
     * Diagonal 4 contributes three high halves. */
    const cpu_glv_u128 c0 = (((cpu_glv_u128)k[1] * g[3]) >> 64)
                          + (((cpu_glv_u128)k[2] * g[2]) >> 64)
                          + (((cpu_glv_u128)k[3] * g[1]) >> 64);
    uint64_t w5 = (uint64_t)c0;
    cpu_glv_u128 carry6 = c0 >> 64;
    /* Diagonal 5 can be 129 bits; its carry can be 65 bits. Never sum its
     * two full products in one u128 or narrow carry6 to uint64_t. */
    cpu_glv_u128 product = (cpu_glv_u128)k[2] * g[3];
    cpu_glv_u128 t = (cpu_glv_u128)w5 + (uint64_t)product;
    w5 = (uint64_t)t; carry6 += (product >> 64) + (t >> 64);
    product = (cpu_glv_u128)k[3] * g[2];
    t = (cpu_glv_u128)w5 + (uint64_t)product;
    w5 = (uint64_t)t; carry6 += (product >> 64) + (t >> 64);
    const uint64_t half = 0x8000000000000000ULL;
    if (w5 >= half - DELTA && w5 < half) {
        cpu_glv_coeff_fallback(out, k, g);
        return;
    }
    /* Outside the guarded interval every possible omitted carry gives the
     * same rounded result. Diagonal 6 and its carry fit exactly in u128. */
    const cpu_glv_u128 rounded = (cpu_glv_u128)k[3] * g[3] + carry6 + (w5 >> 63);
    out[0] = (uint64_t)rounded; out[1] = (uint64_t)(rounded >> 64);
}

static inline void cpu_glv_coeff(uint64_t out[2], const uint64_t k[4], const uint64_t g[4]) {
#if QSB_CPU_GLV_COEFF6
    /* Known constant pointers specialize away these tests in the split. A
     * caller passing any other array retains the full reference semantics. */
    if (g == CPU_GLV_G1_LE64) { cpu_glv_coeff6<5>(out, k, g); return; }
    if (g == CPU_GLV_G2_LE64) { cpu_glv_coeff6<4>(out, k, g); return; }
#endif
    cpu_glv_coeff_reference(out, k, g);
}

/* Decode the low 128 bits of a signed 320-bit two's-complement residual.
 * The exact lattice bound |r| < 0xa2a8918ca85bafe22016d0b917e4dd77 < 2^128
 * guarantees that the magnitude fits. No signed integer conversion is used. */
static inline void cpu_glv_abs_residual(uint64_t mag[2], unsigned *negative, const uint64_t r[5]) {
    const unsigned sg = (unsigned)(r[4] >> 63);
    const uint64_t mask = 0ULL - (uint64_t)sg;
    const cpu_glv_u128 lo = (cpu_glv_u128)(r[0] ^ mask) + sg;
    mag[0] = (uint64_t)lo;
    mag[1] = (r[1] ^ mask) + (uint64_t)(lo >> 64);
    *negative = sg;
}

static inline void cpu_glv_normalize(uint64_t k[4], const uint64_t input[4]) {
    for (int i = 0; i < 4; ++i) k[i] = input[i];
    /* input <2^256<2*n, so at most one subtraction is needed. */
    const bool ge = k[3] == CPU_GLV_ORDER_LE64[3] &&
        (k[2] > CPU_GLV_ORDER_LE64[2] ||
         (k[2] == CPU_GLV_ORDER_LE64[2] &&
          (k[1] > CPU_GLV_ORDER_LE64[1] ||
           (k[1] == CPU_GLV_ORDER_LE64[1] && k[0] >= CPU_GLV_ORDER_LE64[0]))));
    if (ge) cpu_glv_sub<4>(k, k, CPU_GLV_ORDER_LE64);
}

static inline void cpu_glv_residual_reference(const uint64_t k[4], const uint64_t c1[2], const uint64_t c2[2],
                                             uint64_t r1[2], uint64_t r2[2], unsigned *negative1, unsigned *negative2) {
    /* r1 = k-c1*a1-c2*a2; r2 = c1*b1-c2*a1. The widest positive product
     * is 128 by 129 bits. Five limbs preserve its sign through both subtractions
     * and avoid relying on a signed 129-bit C++ type. */
    uint64_t p[5] = {0, 0, 0, 0, 0}, q[5], s[5] = {k[0], k[1], k[2], k[3], 0};
    cpu_glv_mul<2, 2>(p, c1, CPU_GLV_A1_LE64);
    cpu_glv_mul<2, 3>(q, c2, CPU_GLV_A2_LE64);
    cpu_glv_sub<5>(s, s, p);
    cpu_glv_sub<5>(s, s, q);
    cpu_glv_abs_residual(r1, negative1, s);

    /* Reset the fifth limb because the 2x2 helper writes only four limbs. */
    p[4] = q[4] = 0;
    cpu_glv_mul<2, 2>(p, c1, CPU_GLV_B1_LE64);
    cpu_glv_mul<2, 2>(q, c2, CPU_GLV_A1_LE64);
    cpu_glv_sub<5>(s, p, q);
    cpu_glv_abs_residual(r2, negative2, s);
}

/* The unoptimized full-product/320-bit route remains callable for startup
 * comparisons, independently of either optimization switch. */
static inline void cpu_glv_split_reference(const uint64_t input[4], uint64_t r1[2], uint64_t r2[2],
                                           unsigned *negative1, unsigned *negative2) {
    uint64_t k[4], c1[2], c2[2];
    cpu_glv_normalize(k, input);
    cpu_glv_coeff_reference(c1, k, CPU_GLV_G1_LE64);
    cpu_glv_coeff_reference(c2, k, CPU_GLV_G2_LE64);
    cpu_glv_residual_reference(k, c1, c2, r1, r2, negative1, negative2);
}

struct CpuGlv129 { uint64_t lo, hi; unsigned top; }; /* top is bit 128, always 0 or 1 */

static inline CpuGlv129 cpu_glv_sum129(const uint64_t a[2], const uint64_t b[2]) {
    cpu_glv_u128 t = (cpu_glv_u128)a[0] + b[0];
    const uint64_t lo = (uint64_t)t;
    t = (cpu_glv_u128)a[1] + b[1] + (t >> 64);
    const CpuGlv129 out = {lo, (uint64_t)t, (unsigned)(t >> 64)};
    return out;
}

/* Three full products determine a product modulo 2^129. Every column-2 term
 * contributes only parity; the two optional input top bits must be retained. */
static inline CpuGlv129 cpu_glv_mul129(const CpuGlv129 &a, const CpuGlv129 &b) {
    cpu_glv_u128 t = (cpu_glv_u128)a.lo * b.lo;
    const uint64_t lo = (uint64_t)t;
    t = (t >> 64) + (cpu_glv_u128)a.lo * b.hi;
    const uint64_t upper = (uint64_t)(t >> 64);
    t = (cpu_glv_u128)a.hi * b.lo + (uint64_t)t;
    const unsigned top = (unsigned)(upper ^ (uint64_t)(t >> 64) ^ (a.hi & b.hi)
                         ^ (a.lo & (uint64_t)b.top) ^ ((uint64_t)a.top & b.lo)) & 1u;
    const CpuGlv129 out = {lo, (uint64_t)t, top};
    return out;
}

static inline CpuGlv129 cpu_glv_sub129(const CpuGlv129 &a, const CpuGlv129 &b) {
    const cpu_glv_u128 lo = (cpu_glv_u128)a.lo - b.lo;
    const cpu_glv_u128 sub = (cpu_glv_u128)b.hi + (unsigned)(a.lo < b.lo);
    const cpu_glv_u128 hi = (cpu_glv_u128)a.hi - sub;
    const unsigned borrow = (unsigned)((cpu_glv_u128)a.hi < sub);
    const CpuGlv129 out = {(uint64_t)lo, (uint64_t)hi, (a.top - b.top - borrow) & 1u};
    return out;
}

static inline void cpu_glv_abs129(uint64_t mag[2], unsigned *negative, const CpuGlv129 &r) {
    /* Strict |r|<2^128 makes bit128 the sign; bit127 would be wrong. */
    const unsigned sg = r.top;
    const uint64_t mask = 0ULL - (uint64_t)sg;
    const cpu_glv_u128 lo = (cpu_glv_u128)(r.lo ^ mask) + sg;
    mag[0] = (uint64_t)lo;
    mag[1] = (r.hi ^ mask) + (uint64_t)(lo >> 64);
    *negative = sg;
}

static inline void cpu_glv_residual129(const uint64_t k[4], const uint64_t c1[2], const uint64_t c2[2],
                                      uint64_t r1[2], uint64_t r2[2], unsigned *negative1, unsigned *negative2) {
    const CpuGlv129 a1 = {CPU_GLV_A1_LE64[0], CPU_GLV_A1_LE64[1], 0};
    const CpuGlv129 a2 = {CPU_GLV_A2_LE64[0], CPU_GLV_A2_LE64[1], (unsigned)CPU_GLV_A2_LE64[2]};
    const CpuGlv129 b1 = {CPU_GLV_B1_LE64[0], CPU_GLV_B1_LE64[1], 0};
    const CpuGlv129 cc1 = {c1[0], c1[1], 0}, cc2 = {c2[0], c2[1], 0};
    const CpuGlv129 sum = cpu_glv_sum129(c1, c2); /* its carry bit can be 1 */
    const CpuGlv129 p = cpu_glv_mul129(sum, a1);
    const CpuGlv129 q = cpu_glv_mul129(cc2, b1);
    const CpuGlv129 r = cpu_glv_mul129(cc1, a2); /* a2 has a nonzero bit128 */
    const CpuGlv129 kk = {k[0], k[1], (unsigned)(k[2] & 1ULL)};
    /* a2=a1+b1: r1=k-(c1+c2)*a1-c2*b1, r2=c1*a2-(c1+c2)*a1. */
    cpu_glv_abs129(r1, negative1, cpu_glv_sub129(cpu_glv_sub129(kk, p), q));
    cpu_glv_abs129(r2, negative2, cpu_glv_sub129(r, p));
}

static inline void cpu_glv_split(const uint64_t input[4], uint64_t r1[2], uint64_t r2[2],
                                 unsigned *negative1, unsigned *negative2) {
    uint64_t k[4], c1[2], c2[2];
    cpu_glv_normalize(k, input);
    cpu_glv_coeff(c1, k, CPU_GLV_G1_LE64);
    cpu_glv_coeff(c2, k, CPU_GLV_G2_LE64);
#if QSB_CPU_GLV_RESIDUAL129
    cpu_glv_residual129(k, c1, c2, r1, r2, negative1, negative2);
#else
    cpu_glv_residual_reference(k, c1, c2, r1, r2, negative1, negative2);
#endif
}

/* Return the signed low digit and write five high digits as abs(d)|sign<<31.
 * Input magnitude is less than the exact 128-bit GLV bound. Internal widths
 * are 8,24,24,24,24 followed by the bounded top at shift 104. */
static inline int cpu_glv_recode(uint32_t high[5], const uint64_t mag[2], unsigned negative) {
    const uint32_t low = (uint32_t)mag[0] & 255u;
    uint32_t carry = (uint32_t)(low > 128u);
    int low_signed = (int)low - (int)(carry * 256u);
    if (negative) low_signed = -low_signed;
    const uint32_t fields[4] = {
        (uint32_t)(mag[0] >> 8) & 0xFFFFFFu,
        (uint32_t)(mag[0] >> 32) & 0xFFFFFFu,
        (uint32_t)((mag[0] >> 56) | (mag[1] << 8)) & 0xFFFFFFu,
        (uint32_t)(mag[1] >> 16) & 0xFFFFFFu
    };
    for (int i = 0; i < 4; ++i) {
        const uint32_t d = fields[i] + carry;
        const uint32_t sg = (uint32_t)(d > 0x800000u);
        const uint32_t a = sg ? 0x1000000u - d : d;
        high[i] = a | ((sg ^ (uint32_t)negative) << 31);
        carry = sg;
    }
    const uint32_t top = (uint32_t)(mag[1] >> 40) + carry;
    high[4] = top | ((uint32_t)negative << 31);
    return low_signed;
}

/* Central symmetry halves the joint grid. Magnitude is index+1 so that the
 * valid (0,0) pair is code 1, never a zero-digit/drop marker. */
static inline uint32_t cpu_glv_joint_code(int low1, int low2) {
    const unsigned sg = (unsigned)(low2 < 0 || (low2 == 0 && low1 < 0));
    const int x = sg ? -low1 : low1;
    const int y = sg ? -low2 : low2;
    const uint32_t index = y == 0 ? (uint32_t)x :
        129u + (uint32_t)(y - 1) * 257u + (uint32_t)(x + 128);
    return (index + 1u) | ((uint32_t)sg << 31);
}

/* SHA digest words are numeric big-endian words (h[0] is the most significant
 * uint32), not host-loaded raw digest bytes. Outputs 0..4: r2 high; 5..9: r1
 * high; 10: joint low index+1 and its canonical sign. Return 1 iff a HIGH
 * digit is zero. Zero low digits, including (0,0), remain valid. */
static inline unsigned cpu_glv_digits(uint32_t dig[11], const uint32_t sha_be[8]) {
    uint64_t z[4];
    for (int i = 0; i < 4; ++i)
        z[i] = ((uint64_t)sha_be[6 - 2 * i] << 32) | sha_be[7 - 2 * i];
    uint64_t r1[2], r2[2];
    unsigned negative1, negative2;
    cpu_glv_split(z, r1, r2, &negative1, &negative2);
    const int low2 = cpu_glv_recode(dig, r2, negative2);
    const int low1 = cpu_glv_recode(dig + 5, r1, negative1);
    dig[10] = cpu_glv_joint_code(low1, low2);
    unsigned bad = 0;
    for (int i = 0; i < 10; ++i) bad |= (unsigned)((dig[i] & 0x7FFFFFFFu) == 0);
    return bad;
}
