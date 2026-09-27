/* cg_fe4.h -- secp256k1 base-field arithmetic on 4x64-bit limbs for the pinning co-grinder.
 * Original code (not derived from libsecp256k1).
 *
 * p = 2^256 - C, C = 0x1000003D1. A field element is any 256-bit integer (4 little-endian
 * 64-bit words) congruent to the value mod p ("lazy": may be >= p). fe*_norm() returns the
 * canonical representative in [0, p).
 *
 * Two implementations with identical semantics:
 *   struct FeAsm : MULX/ADCX/ADOX inline assembly (needs BMI2 + ADX; runtime-checked by callers)
 *   struct FeC   : portable C (unsigned __int128)
 * The inversion (Fermat, p - 2) and everything built on these is templated on the struct.
 */
#ifndef QSB_CG_FE4_H
#define QSB_CG_FE4_H
#include <stdint.h>
#include <string.h>

namespace qcg_fe {

typedef unsigned __int128 u128;
static const uint64_t FE_C = 0x1000003D1ULL;
static const uint64_t FE_P[4] = {0xFFFFFFFEFFFFFC2FULL, ~0ULL, ~0ULL, ~0ULL};

#define QCG_FE_INL __attribute__((always_inline)) inline
/* asm scratch outputs are consumed by an empty asm so nvcc's front end does not warn that they
 * are set but never used */
#define QCG_FE_SINK(v) __asm__ volatile("" : : "r"(v))

/* shared helpers (plain C, correct for any input < 2^256) */
static QCG_FE_INL void fe_norm_c(uint64_t *r) {
    uint64_t s0, s1, s2, s3; unsigned char c;
    c = __builtin_add_overflow(r[0], FE_C, &s0);
    s1 = r[1] + c; c = c && s1 == 0;
    s2 = r[2] + c; c = c && s2 == 0;
    s3 = r[3] + c; c = c && s3 == 0;
    if (c) { r[0] = s0; r[1] = s1; r[2] = s2; r[3] = s3; }
}
static QCG_FE_INL int fe_is_zero_c(const uint64_t *a) {       /* a == 0 mod p */
    uint64_t t[4] = {a[0], a[1], a[2], a[3]}; fe_norm_c(t);
    return (t[0] | t[1] | t[2] | t[3]) == 0;
}

/* ------------------------------------------------------------------ portable C */
struct FeC {
    static QCG_FE_INL void mul(uint64_t *r, const uint64_t *a, const uint64_t *b) {
        uint64_t p[8];
        u128 t;
        uint64_t c;
        /* schoolbook */
        t = (u128)a[0] * b[0]; p[0] = (uint64_t)t; c = (uint64_t)(t >> 64);
        t = (u128)a[0] * b[1] + c; p[1] = (uint64_t)t; c = (uint64_t)(t >> 64);
        t = (u128)a[0] * b[2] + c; p[2] = (uint64_t)t; c = (uint64_t)(t >> 64);
        t = (u128)a[0] * b[3] + c; p[3] = (uint64_t)t; p[4] = (uint64_t)(t >> 64);
        for (int i = 1; i < 4; i++) {
            c = 0;
            for (int j = 0; j < 4; j++) {
                t = (u128)a[i] * b[j] + p[i + j] + c;
                p[i + j] = (uint64_t)t; c = (uint64_t)(t >> 64);
            }
            p[i + 4] = c;
        }
        reduce(r, p);
    }
    static QCG_FE_INL void sqr(uint64_t *r, const uint64_t *a) { uint64_t t[4] = {a[0], a[1], a[2], a[3]}; mul(r, t, t); }
    static QCG_FE_INL void reduce(uint64_t *r, const uint64_t *p) {
        u128 t; uint64_t c = 0, r0, r1, r2, r3;
        t = (u128)p[4] * FE_C + p[0];     r0 = (uint64_t)t; c = (uint64_t)(t >> 64);
        t = (u128)p[5] * FE_C + p[1] + c; r1 = (uint64_t)t; c = (uint64_t)(t >> 64);
        t = (u128)p[6] * FE_C + p[2] + c; r2 = (uint64_t)t; c = (uint64_t)(t >> 64);
        t = (u128)p[7] * FE_C + p[3] + c; r3 = (uint64_t)t; c = (uint64_t)(t >> 64);   /* c < 2^34 */
        t = (u128)c * FE_C + r0; r0 = (uint64_t)t; c = (uint64_t)(t >> 64);
        t = (u128)r1 + c; r1 = (uint64_t)t; c = (uint64_t)(t >> 64);
        t = (u128)r2 + c; r2 = (uint64_t)t; c = (uint64_t)(t >> 64);
        t = (u128)r3 + c; r3 = (uint64_t)t; c = (uint64_t)(t >> 64);
        r0 += c * FE_C;                   /* wrapped value < 2^67, cannot carry again */
        r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
    }
    static QCG_FE_INL void add(uint64_t *r, const uint64_t *a, const uint64_t *b) {
        u128 t; uint64_t c, r0, r1, r2, r3;
        t = (u128)a[0] + b[0];     r0 = (uint64_t)t; c = (uint64_t)(t >> 64);
        t = (u128)a[1] + b[1] + c; r1 = (uint64_t)t; c = (uint64_t)(t >> 64);
        t = (u128)a[2] + b[2] + c; r2 = (uint64_t)t; c = (uint64_t)(t >> 64);
        t = (u128)a[3] + b[3] + c; r3 = (uint64_t)t; c = (uint64_t)(t >> 64);
        /* value = r + c*2^256 == r + c*C */
        t = (u128)r0 + c * FE_C; r0 = (uint64_t)t; c = (uint64_t)(t >> 64);
        t = (u128)r1 + c; r1 = (uint64_t)t; c = (uint64_t)(t >> 64);
        t = (u128)r2 + c; r2 = (uint64_t)t; c = (uint64_t)(t >> 64);
        t = (u128)r3 + c; r3 = (uint64_t)t; c = (uint64_t)(t >> 64);
        r0 += c * FE_C;
        r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
    }
    static QCG_FE_INL void sub(uint64_t *r, const uint64_t *a, const uint64_t *b) {
        /* a - b = a + (2^256 - 1 - b) + 1 - 2^256 ; use a + (p - b) with lazy b: p - b may be
           negative when b >= p, so compute a - b and add p on borrow (twice if needed) */
        uint64_t r0, r1, r2, r3, bw;
        unsigned char c;
        c = __builtin_sub_overflow(a[0], b[0], &r0);
        bw = __builtin_sub_overflow(a[1], b[1], &r1); bw |= __builtin_sub_overflow(r1, (uint64_t)c, &r1); c = (unsigned char)bw;
        bw = __builtin_sub_overflow(a[2], b[2], &r2); bw |= __builtin_sub_overflow(r2, (uint64_t)c, &r2); c = (unsigned char)bw;
        bw = __builtin_sub_overflow(a[3], b[3], &r3); bw |= __builtin_sub_overflow(r3, (uint64_t)c, &r3); c = (unsigned char)bw;
        for (int rep = 0; rep < 2 && c; rep++) {
            /* value + 2^256 -> subtract C to get value + p */
            unsigned char d = __builtin_sub_overflow(r0, FE_C, &r0);
            bw = __builtin_sub_overflow(r1, (uint64_t)d, &r1); d = (unsigned char)bw;
            bw = __builtin_sub_overflow(r2, (uint64_t)d, &r2); d = (unsigned char)bw;
            bw = __builtin_sub_overflow(r3, (uint64_t)d, &r3); d = (unsigned char)bw;
            c = d;
        }
        r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
    }
};

/* ------------------------------------------------------------------ MULX/ADCX/ADOX asm */
struct FeAsm {
    /* r = a * b. r may alias a (row i reads a[i] before r[i] is stored) but must not alias b. */
    static QCG_FE_INL void mul_noalias_b(uint64_t *r, const uint64_t *a, const uint64_t *b) {
        uint64_t t4, t5, t6, t7, x, y, z;
                __asm__ __volatile__(
            "movq 0(%[a]), %%rdx\n\t"
            "mulxq 0(%[b]), %[x], %[y]\n\t"
            "movq %[x], 0(%[r])\n\t"
            "mulxq 8(%[b]), %[x], %[t5]\n\t"
            "addq %[y], %[x]\n\t"
            "mulxq 16(%[b]), %[y], %[t6]\n\t"
            "adcq %[t5], %[y]\n\t"
            "mulxq 24(%[b]), %[t5], %[t4]\n\t"
            "adcq %[t6], %[t5]\n\t"
            "adcq $0, %[t4]\n\t"
            "movq 8(%[a]), %%rdx\n\t"
            "xorl %k[z], %k[z]\n\t"
            "mulxq 0(%[b]), %[t6], %[t7]\n\t"
            "adcxq %[t6], %[x]\n\t"
            "adoxq %[t7], %[y]\n\t"
            "movq %[x], 8(%[r])\n\t"
            "mulxq 8(%[b]), %[t6], %[t7]\n\t"
            "adcxq %[t6], %[y]\n\t"
            "adoxq %[t7], %[t5]\n\t"
            "mulxq 16(%[b]), %[t6], %[t7]\n\t"
            "adcxq %[t6], %[t5]\n\t"
            "adoxq %[t7], %[t4]\n\t"
            "mulxq 24(%[b]), %[t6], %[x]\n\t"
            "adcxq %[t6], %[t4]\n\t"
            "adoxq %[z], %[x]\n\t"
            "adcxq %[z], %[x]\n\t"
            "movq 16(%[a]), %%rdx\n\t"
            "xorl %k[z], %k[z]\n\t"
            "mulxq 0(%[b]), %[t6], %[t7]\n\t"
            "adcxq %[t6], %[y]\n\t"
            "adoxq %[t7], %[t5]\n\t"
            "movq %[y], 16(%[r])\n\t"
            "mulxq 8(%[b]), %[t6], %[t7]\n\t"
            "adcxq %[t6], %[t5]\n\t"
            "adoxq %[t7], %[t4]\n\t"
            "mulxq 16(%[b]), %[t6], %[t7]\n\t"
            "adcxq %[t6], %[t4]\n\t"
            "adoxq %[t7], %[x]\n\t"
            "mulxq 24(%[b]), %[t6], %[y]\n\t"
            "adcxq %[t6], %[x]\n\t"
            "adoxq %[z], %[y]\n\t"
            "adcxq %[z], %[y]\n\t"
            "movq 24(%[a]), %%rdx\n\t"
            "xorl %k[z], %k[z]\n\t"
            "mulxq 0(%[b]), %[t6], %[t7]\n\t"
            "adcxq %[t6], %[t5]\n\t"
            "adoxq %[t7], %[t4]\n\t"
            "movq %[t5], 24(%[r])\n\t"
            "mulxq 8(%[b]), %[t6], %[t7]\n\t"
            "adcxq %[t6], %[t4]\n\t"
            "adoxq %[t7], %[x]\n\t"
            "mulxq 16(%[b]), %[t6], %[t7]\n\t"
            "adcxq %[t6], %[x]\n\t"
            "adoxq %[t7], %[y]\n\t"
            "mulxq 24(%[b]), %[t6], %[t7]\n\t"
            "adcxq %[t6], %[y]\n\t"
            "adoxq %[z], %[t7]\n\t"
            "adcxq %[z], %[t7]\n\t"
            "movabsq $0x1000003D1, %%rdx\n\t"
            "xorl %k[z], %k[z]\n\t"
            "mulxq %[t4], %[t4], %[t6]\n\t"
            "mulxq %[x], %[x], %[t5]\n\t"
            "adcxq 0(%[r]), %[t4]\n\t"
            "adoxq %[t6], %[x]\n\t"
            "adcxq 8(%[r]), %[x]\n\t"
            "mulxq %[y], %[y], %[t6]\n\t"
            "adoxq %[t5], %[y]\n\t"
            "adcxq 16(%[r]), %[y]\n\t"
            "mulxq %[t7], %[t7], %[t5]\n\t"
            "adoxq %[t6], %[t7]\n\t"
            "adcxq 24(%[r]), %[t7]\n\t"
            "adoxq %[z], %[t5]\n\t"
            "adcxq %[z], %[t5]\n\t"
            "mulxq %[t5], %[t6], %[t5]\n\t"
            "addq %[t6], %[t4]\n\t"
            "adcq %[t5], %[x]\n\t"
            "adcq $0, %[y]\n\t"
            "adcq $0, %[t7]\n\t"
            "sbbq %[z], %[z]\n\t"
            "andq %%rdx, %[z]\n\t"
            "addq %[z], %[t4]\n\t"
            "adcq $0, %[x]\n\t"
            "movq %[t4], 0(%[r])\n\t"
            "movq %[x], 8(%[r])\n\t"
            "movq %[y], 16(%[r])\n\t"
            "movq %[t7], 24(%[r])\n\t"
            : [t4] "=&r"(t4), [t5] "=&r"(t5), [t6] "=&r"(t6), [t7] "=&r"(t7), [x] "=&r"(x), [y] "=&r"(y), [z] "=&r"(z)
            : [r] "r"(r), [a] "r"(a), [b] "r"(b)
            : "rdx", "cc", "memory");
        QCG_FE_SINK(t4); QCG_FE_SINK(t5); QCG_FE_SINK(t6); QCG_FE_SINK(t7); QCG_FE_SINK(x); QCG_FE_SINK(y); QCG_FE_SINK(z);
    }
    static QCG_FE_INL void mul(uint64_t *r, const uint64_t *a, const uint64_t *b) {
        if (r == b) { uint64_t t[4] = {b[0], b[1], b[2], b[3]}; mul_noalias_b(r, a, t); }
        else mul_noalias_b(r, a, b);
    }
    /* r = a^2. r may alias a. */
    static QCG_FE_INL void sqr(uint64_t *r, const uint64_t *a) {
        uint64_t a0, t1, t2, t3, t4, t5, t6, t7, x, z;
                __asm__ __volatile__(
            "movq 0(%[a]), %%rdx\n\t"
            "mulxq 8(%[a]), %[t1], %[t2]\n\t"
            "mulxq 16(%[a]), %[x], %[t3]\n\t"
            "addq %[x], %[t2]\n\t"
            "mulxq 24(%[a]), %[x], %[t4]\n\t"
            "adcq %[x], %[t3]\n\t"
            "adcq $0, %[t4]\n\t"
            "movq 8(%[a]), %%rdx\n\t"
            "xorl %k[z], %k[z]\n\t"
            "mulxq 16(%[a]), %[x], %[t5]\n\t"
            "adcxq %[x], %[t3]\n\t"
            "adoxq %[t5], %[t4]\n\t"
            "mulxq 24(%[a]), %[x], %[t5]\n\t"
            "adcxq %[x], %[t4]\n\t"
            "adoxq %[z], %[t5]\n\t"
            "adcxq %[z], %[t5]\n\t"
            "movq 16(%[a]), %%rdx\n\t"
            "mulxq 24(%[a]), %[x], %[t6]\n\t"
            "addq %[x], %[t5]\n\t"
            "adcq $0, %[t6]\n\t"
            "xorl %k[t7], %k[t7]\n\t"
            "addq %[t1], %[t1]\n\t"
            "adcq %[t2], %[t2]\n\t"
            "adcq %[t3], %[t3]\n\t"
            "adcq %[t4], %[t4]\n\t"
            "adcq %[t5], %[t5]\n\t"
            "adcq %[t6], %[t6]\n\t"
            "adcq $0, %[t7]\n\t"
            "movq 0(%[a]), %%rdx\n\t"
            "mulxq %%rdx, %[a0], %[x]\n\t"
            "addq %[x], %[t1]\n\t"
            "movq 8(%[a]), %%rdx\n\t"
            "mulxq %%rdx, %[x], %[z]\n\t"
            "adcq %[x], %[t2]\n\t"
            "adcq %[z], %[t3]\n\t"
            "movq 16(%[a]), %%rdx\n\t"
            "mulxq %%rdx, %[x], %[z]\n\t"
            "adcq %[x], %[t4]\n\t"
            "adcq %[z], %[t5]\n\t"
            "movq 24(%[a]), %%rdx\n\t"
            "mulxq %%rdx, %[x], %[z]\n\t"
            "adcq %[x], %[t6]\n\t"
            "adcq %[z], %[t7]\n\t"
            "movabsq $0x1000003D1, %%rdx\n\t"
            "xorl %k[z], %k[z]\n\t"
            "mulxq %[t4], %[t4], %[x]\n\t"
            "adcxq %[t4], %[a0]\n\t"
            "mulxq %[t5], %[t5], %[t4]\n\t"
            "adoxq %[x], %[t5]\n\t"
            "adcxq %[t5], %[t1]\n\t"
            "mulxq %[t6], %[t6], %[x]\n\t"
            "adoxq %[t4], %[t6]\n\t"
            "adcxq %[t6], %[t2]\n\t"
            "mulxq %[t7], %[t7], %[t4]\n\t"
            "adoxq %[x], %[t7]\n\t"
            "adcxq %[t7], %[t3]\n\t"
            "adoxq %[z], %[t4]\n\t"
            "adcxq %[z], %[t4]\n\t"
            "mulxq %[t4], %[x], %[t4]\n\t"
            "addq %[x], %[a0]\n\t"
            "adcq %[t4], %[t1]\n\t"
            "adcq $0, %[t2]\n\t"
            "adcq $0, %[t3]\n\t"
            "sbbq %[z], %[z]\n\t"
            "andq %%rdx, %[z]\n\t"
            "addq %[z], %[a0]\n\t"
            "adcq $0, %[t1]\n\t"
            "movq %[a0], 0(%[r])\n\t"
            "movq %[t1], 8(%[r])\n\t"
            "movq %[t2], 16(%[r])\n\t"
            "movq %[t3], 24(%[r])\n\t"
            : [a0] "=&r"(a0), [t1] "=&r"(t1), [t2] "=&r"(t2), [t3] "=&r"(t3), [t4] "=&r"(t4), [t5] "=&r"(t5),
              [t6] "=&r"(t6), [t7] "=&r"(t7), [x] "=&r"(x), [z] "=&r"(z)
            : [r] "r"(r), [a] "r"(a)
            : "rdx", "cc", "memory");
        QCG_FE_SINK(a0); QCG_FE_SINK(t1); QCG_FE_SINK(t2); QCG_FE_SINK(t3); QCG_FE_SINK(t4); QCG_FE_SINK(t5); QCG_FE_SINK(t6); QCG_FE_SINK(t7); QCG_FE_SINK(x); QCG_FE_SINK(z);
    }
    static QCG_FE_INL void sub(uint64_t *r, const uint64_t *a, const uint64_t *b) {
        uint64_t r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3], m;
        __asm__(
            "subq 0(%[b]), %[r0]\n\t"
            "sbbq 8(%[b]), %[r1]\n\t"
            "sbbq 16(%[b]), %[r2]\n\t"
            "sbbq 24(%[b]), %[r3]\n\t"
            "sbbq %[m], %[m]\n\t"
            "andq %[c], %[m]\n\t"
            "subq %[m], %[r0]\n\t"
            "sbbq $0, %[r1]\n\t"
            "sbbq $0, %[r2]\n\t"
            "sbbq $0, %[r3]\n\t"
            "sbbq %[m], %[m]\n\t"
            "andq %[c], %[m]\n\t"
            "subq %[m], %[r0]\n\t"
            : [r0] "+&r"(r0), [r1] "+&r"(r1), [r2] "+&r"(r2), [r3] "+&r"(r3), [m] "=&r"(m)
            : [b] "r"(b), [c] "r"(FE_C), "m"(*(const uint64_t(*)[4])b)
            : "cc");
        QCG_FE_SINK(m);
        r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
    }
    static QCG_FE_INL void add(uint64_t *r, const uint64_t *a, const uint64_t *b) {
        uint64_t r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3], m;
        __asm__(
            "addq 0(%[b]), %[r0]\n\t"
            "adcq 8(%[b]), %[r1]\n\t"
            "adcq 16(%[b]), %[r2]\n\t"
            "adcq 24(%[b]), %[r3]\n\t"
            "sbbq %[m], %[m]\n\t"
            "andq %[c], %[m]\n\t"
            "addq %[m], %[r0]\n\t"
            "adcq $0, %[r1]\n\t"
            "adcq $0, %[r2]\n\t"
            "adcq $0, %[r3]\n\t"
            "sbbq %[m], %[m]\n\t"
            "andq %[c], %[m]\n\t"
            "addq %[m], %[r0]\n\t"
            : [r0] "+&r"(r0), [r1] "+&r"(r1), [r2] "+&r"(r2), [r3] "+&r"(r3), [m] "=&r"(m)
            : [b] "r"(b), [c] "r"(FE_C), "m"(*(const uint64_t(*)[4])b)
            : "cc");
        QCG_FE_SINK(m);
        r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
    }
};

/* ------------------------------------------------------------------ generic helpers */
template <class F> static inline void fe_inv(uint64_t *r, const uint64_t *a) {
    /* a^(p-2); addition chain as in libsecp256k1's secp256k1_fe_inv (blocks of 1s: 2,22,223) */
    uint64_t x2[4], x3[4], x6[4], x9[4], x11[4], x22[4], x44[4], x88[4], x176[4], x220[4], x223[4], t[4];
    int j;
    F::sqr(x2, a); F::mul(x2, x2, a);
    F::sqr(x3, x2); F::mul(x3, x3, a);
    memcpy(x6, x3, 32); for (j = 0; j < 3; j++) F::sqr(x6, x6); F::mul(x6, x6, x3);
    memcpy(x9, x6, 32); for (j = 0; j < 3; j++) F::sqr(x9, x9); F::mul(x9, x9, x3);
    memcpy(x11, x9, 32); for (j = 0; j < 2; j++) F::sqr(x11, x11); F::mul(x11, x11, x2);
    memcpy(x22, x11, 32); for (j = 0; j < 11; j++) F::sqr(x22, x22); F::mul(x22, x22, x11);
    memcpy(x44, x22, 32); for (j = 0; j < 22; j++) F::sqr(x44, x44); F::mul(x44, x44, x22);
    memcpy(x88, x44, 32); for (j = 0; j < 44; j++) F::sqr(x88, x88); F::mul(x88, x88, x44);
    memcpy(x176, x88, 32); for (j = 0; j < 88; j++) F::sqr(x176, x176); F::mul(x176, x176, x88);
    memcpy(x220, x176, 32); for (j = 0; j < 44; j++) F::sqr(x220, x220); F::mul(x220, x220, x44);
    memcpy(x223, x220, 32); for (j = 0; j < 3; j++) F::sqr(x223, x223); F::mul(x223, x223, x3);
    memcpy(t, x223, 32); for (j = 0; j < 23; j++) F::sqr(t, t); F::mul(t, t, x22);
    for (j = 0; j < 5; j++) F::sqr(t, t); F::mul(t, t, a);
    for (j = 0; j < 3; j++) F::sqr(t, t); F::mul(t, t, x2);
    for (j = 0; j < 2; j++) F::sqr(t, t); F::mul(r, t, a);
}

/* r = -a (lazy in, canonical-or-lazy out): p - norm(a) */
static QCG_FE_INL void fe_neg_c(uint64_t *r, const uint64_t *a) {
    uint64_t t[4] = {a[0], a[1], a[2], a[3]}; fe_norm_c(t);
    if ((t[0] | t[1] | t[2] | t[3]) == 0) { r[0] = r[1] = r[2] = r[3] = 0; return; }
    unsigned char c; uint64_t bw;
    c = __builtin_sub_overflow(FE_P[0], t[0], &r[0]);
    bw = __builtin_sub_overflow(FE_P[1], t[1], &r[1]); bw |= __builtin_sub_overflow(r[1], (uint64_t)c, &r[1]); c = (unsigned char)bw;
    bw = __builtin_sub_overflow(FE_P[2], t[2], &r[2]); bw |= __builtin_sub_overflow(r[2], (uint64_t)c, &r[2]); c = (unsigned char)bw;
    r[3] = FE_P[3] - t[3] - c;
}

} /* namespace qcg_fe */
#endif
