#pragma once
/* Include inside qcpu after CpuGlvScalar.h and Q8T. Only AVX512F+IFMA are
 * required. Eight independent RAW 256-bit scalars use exact guarded reciprocal
 * coefficients and 52/52/25-bit residuals. The cold g2 correction preserves the
 * normalized scalar path's signed components, including high-zero rejection.
 */

static const uint64_t CPU_GLV8_G1_52[5] = {
    0x3209a45dbb031ULL, 0x1471e8ca7fe89ULL, 0x284eb153daa8aULL,
    0x6bcde86c90e49ULL, 0x03086d221a7d4ULL
};
static const uint64_t CPU_GLV8_G2_52[5] = {
    0x1b4ae8ac47f71ULL, 0xac9df506c6157ULL, 0xabfe4c4221208ULL,
    0x88286f547fa90ULL, 0x0e4437ed6010eULL
};
static const uint64_t CPU_GLV8_A1_52[3] = {
    0xc90e49284eb15ULL, 0x21a7d46bcde86ULL, 0x00000003086d2ULL
};
static const uint64_t CPU_GLV8_A2_52[3] = {
    0x1108d9d44cfd8ULL, 0xf7a8e2f3f657cULL, 0x000000114ca50ULL
};
static const uint64_t CPU_GLV8_B1_52[3] = {
    0x47fa90abfe4c3ULL, 0xd6010e88286f5ULL, 0x0000000e4437eULL
};
static const uint64_t CPU_GLV8_RAW_SPECIAL[4] = {
    0xacafb2d52683c737ULL, 0x4a3c81bc2d093500ULL,
    0xffffffffffffffffULL, 0xffffffffffffffffULL
};

struct CpuGlv8Value { __m512i x0, x1, x2; };

/* A separate pointer-only boundary: no vector value must survive this call.
 * snapshot is [coefficient][radix52 limb][lane], containing six vectors.
 * The generic exact coefficient helper retains its original semantics. */
Q8T static __attribute__((noinline)) void cpu_glv8_patch_raw_coeffs(
    const uint64_t *raw64, uint64_t *snapshot, unsigned guard1, unsigned guard2) {
    for (unsigned which = 0; which < 2; ++which) {
        const unsigned guard = which ? guard2 : guard1;
        for (unsigned lane = 0; lane < 8; ++lane) if ((guard >> lane) & 1u) {
            uint64_t k[4], c[2];
            for (unsigned i = 0; i < 4; ++i) k[i] = raw64[8 * i + lane];
            cpu_glv_coeff_reference(c, k, which ? CPU_GLV_G2_LE64 : CPU_GLV_G1_LE64);
            if (which && k[0] == CPU_GLV8_RAW_SPECIAL[0] &&
                k[1] == CPU_GLV8_RAW_SPECIAL[1] && k[2] == CPU_GLV8_RAW_SPECIAL[2] &&
                k[3] == CPU_GLV8_RAW_SPECIAL[3]) {
                ++c[0];
                c[1] += (uint64_t)(c[0] == 0);
            }
            const unsigned base = 24 * which + lane;
            snapshot[base] = c[0] & 0xfffffffffffffULL;
            snapshot[base + 8] = ((c[0] >> 52) | (c[1] << 12)) & 0xfffffffffffffULL;
            snapshot[base + 16] = c[1] >> 40;
        }
    }
}

/* Nine IFMA operations. The unpropagated sums satisfy s7<5*2^52,
 * s8<3*2^52, so every accumulator and following carry fits uint64.
 * DELTA is 3 for g1 and 4 for g2 over the full raw 256-bit domain. */
template <unsigned DELTA>
Q8T static inline __attribute__((always_inline)) CpuGlv8Value cpu_glv8_coeff52(
    __m512i k2, __m512i k3, __m512i k4, const uint64_t *g, unsigned &guard) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i m52 = _mm512_set1_epi64(0xfffffffffffffULL);
    const __m512i m20 = _mm512_set1_epi64(0xfffffULL);
    __m512i b = _mm512_set1_epi64(g[4]);
    __m512i s7 = _mm512_madd52hi_epu64(zero, k2, b);
    __m512i s8 = _mm512_madd52hi_epu64(zero, k3, b);
    __m512i s9 = _mm512_madd52hi_epu64(zero, k4, b);
    s7 = _mm512_madd52lo_epu64(s7, k3, b);
    s8 = _mm512_madd52lo_epu64(s8, k4, b);
    b = _mm512_set1_epi64(g[3]);
    s7 = _mm512_madd52hi_epu64(s7, k3, b);
    s7 = _mm512_madd52lo_epu64(s7, k4, b);
    s8 = _mm512_madd52hi_epu64(s8, k4, b);
    b = _mm512_set1_epi64(g[2]);
    s7 = _mm512_madd52hi_epu64(s7, k4, b);
    const __m512i low20 = _mm512_and_si512(s7, m20);
    guard = (unsigned)(_mm512_cmp_epu64_mask(low20, _mm512_set1_epi64(0x80000 - DELTA), _MM_CMPINT_GE) &
                       _mm512_cmp_epu64_mask(low20, _mm512_set1_epi64(0x80000), _MM_CMPINT_LT));
    s7 = _mm512_add_epi64(s7, _mm512_set1_epi64(0x80000));
    s8 = _mm512_add_epi64(s8, _mm512_srli_epi64(s7, 52));
    s7 = _mm512_and_si512(s7, m52);
    s9 = _mm512_add_epi64(s9, _mm512_srli_epi64(s8, 52));
    s8 = _mm512_and_si512(s8, m52);
    const CpuGlv8Value out = {
        _mm512_or_si512(_mm512_srli_epi64(s7, 20), _mm512_slli_epi64(_mm512_and_si512(s8, m20), 32)),
        _mm512_or_si512(_mm512_srli_epi64(s8, 20), _mm512_slli_epi64(_mm512_and_si512(s9, m20), 32)),
        _mm512_srli_epi64(s9, 20)
    };
    return out;
}

Q8T static inline __attribute__((always_inline)) CpuGlv8Value cpu_glv8_add129(
    const CpuGlv8Value &a, const CpuGlv8Value &b) {
    const __m512i m52 = _mm512_set1_epi64(0xfffffffffffffULL);
    CpuGlv8Value r;
    r.x0 = _mm512_add_epi64(a.x0, b.x0);
    r.x1 = _mm512_add_epi64(_mm512_add_epi64(a.x1, b.x1), _mm512_srli_epi64(r.x0, 52));
    r.x2 = _mm512_add_epi64(_mm512_add_epi64(a.x2, b.x2), _mm512_srli_epi64(r.x1, 52));
    r.x0 = _mm512_and_si512(r.x0, m52);
    r.x1 = _mm512_and_si512(r.x1, m52);
    r.x2 = _mm512_and_si512(r.x2, _mm512_set1_epi64(0x1ffffff));
    return r;
}

/* Nine IFMA operations for multiplication modulo 2^129. Column 0 contains
 * one low52 term and has no carry. Columns 1/2 are <3*2^52 and <5*2^52. */
Q8T static inline __attribute__((always_inline)) CpuGlv8Value cpu_glv8_mul129(
    const CpuGlv8Value &a, const uint64_t *b52) {
    const __m512i zero = _mm512_setzero_si512();
    __m512i b = _mm512_set1_epi64(b52[0]);
    CpuGlv8Value r;
    r.x0 = _mm512_madd52lo_epu64(zero, a.x0, b);
    r.x1 = _mm512_madd52hi_epu64(zero, a.x0, b);
    r.x1 = _mm512_madd52lo_epu64(r.x1, a.x1, b);
    r.x2 = _mm512_madd52hi_epu64(zero, a.x1, b);
    r.x2 = _mm512_madd52lo_epu64(r.x2, a.x2, b);
    b = _mm512_set1_epi64(b52[1]);
    r.x1 = _mm512_madd52lo_epu64(r.x1, a.x0, b);
    r.x2 = _mm512_madd52hi_epu64(r.x2, a.x0, b);
    r.x2 = _mm512_madd52lo_epu64(r.x2, a.x1, b);
    b = _mm512_set1_epi64(b52[2]);
    r.x2 = _mm512_madd52lo_epu64(r.x2, a.x0, b);
    r.x2 = _mm512_add_epi64(r.x2, _mm512_srli_epi64(r.x1, 52));
    r.x1 = _mm512_and_si512(r.x1, _mm512_set1_epi64(0xfffffffffffffULL));
    r.x2 = _mm512_and_si512(r.x2, _mm512_set1_epi64(0x1ffffff));
    return r;
}

Q8T static inline __attribute__((always_inline)) CpuGlv8Value cpu_glv8_sub129(
    const CpuGlv8Value &a, const CpuGlv8Value &b) {
    const __m512i one = _mm512_set1_epi64(1);
    const __m512i m52 = _mm512_set1_epi64(0xfffffffffffffULL);
    __mmask8 borrow = _mm512_cmp_epu64_mask(a.x0, b.x0, _MM_CMPINT_LT);
    CpuGlv8Value r;
    r.x0 = _mm512_and_si512(_mm512_sub_epi64(a.x0, b.x0), m52);
    const __m512i b1 = _mm512_mask_add_epi64(b.x1, borrow, b.x1, one);
    borrow = _mm512_cmp_epu64_mask(a.x1, b1, _MM_CMPINT_LT);
    r.x1 = _mm512_and_si512(_mm512_sub_epi64(a.x1, b1), m52);
    const __m512i b2 = _mm512_mask_add_epi64(b.x2, borrow, b.x2, one);
    r.x2 = _mm512_and_si512(_mm512_sub_epi64(a.x2, b2), _mm512_set1_epi64(0x1ffffff));
    return r;
}

template <bool RowCodes>
Q8T static inline __attribute__((always_inline)) void cpu_glv8_store_high(
    uint32_t *out, unsigned digit, __m512i index, unsigned active8,
    __m512i value, unsigned stride) {
    /* The base pointer is never shifted backwards. Negative inactive indices
     * are harmless: the masked scatter never evaluates their memory access. */
    if (RowCodes) {
        const __m512i sign = _mm512_and_si512(value, _mm512_set1_epi64(0x80000000ULL));
        __m512i mag = _mm512_and_si512(value, _mm512_set1_epi64(0x7fffffff));
        /* The original zero/drop mask was collected before this safe-row
         * clamp. A zero row offset itself is not a rejection marker. */
        mag = _mm512_mask_add_epi64(mag, _mm512_cmpeq_epi64_mask(mag, _mm512_setzero_si512()),
                                    mag, _mm512_set1_epi64(1));
        value = _mm512_or_si512(_mm512_slli_epi64(_mm512_sub_epi64(mag, _mm512_set1_epi64(1)), 3), sign);
    }
    _mm512_mask_i64scatter_epi32(out + (RowCodes ? (size_t)digit * stride : digit), (__mmask8)active8, index,
                               _mm512_cvtepi64_epi32(value), 4);
}

template <bool RowCodes>
Q8T static inline __attribute__((always_inline)) void cpu_glv8_store_joint(
    uint32_t *out, __m512i index, unsigned active8, __m512i joint, unsigned stride) {
    if (RowCodes) {
        const __m512i sign = _mm512_and_si512(joint, _mm512_set1_epi64(0x80000000ULL));
        const __mmask8 negative = _mm512_cmpneq_epi64_mask(sign, _mm512_setzero_si512());
        const __m512i mag = _mm512_and_si512(joint, _mm512_set1_epi64(0x7fffffff));
        const __m512i offset = _mm512_slli_epi64(_mm512_sub_epi64(mag, _mm512_set1_epi64(1)), 3);
        const __m512i half = _mm512_set1_epi64((uint64_t)CPU_GLV_JOINT_ENTRIES * 8);
        const __m512i row0 = _mm512_or_si512(_mm512_mask_add_epi64(offset, negative, offset, half), sign);
        const __m512i row1 = _mm512_or_si512(_mm512_mask_add_epi64(offset, (__mmask8)~negative, offset, half), sign);
        _mm512_mask_i64scatter_epi32(out + (size_t)10 * stride, (__mmask8)active8, index,
                                   _mm512_cvtepi64_epi32(row0), 4);
        _mm512_mask_i64scatter_epi32(out + (size_t)11 * stride, (__mmask8)active8, index,
                                   _mm512_cvtepi64_epi32(row1), 4);
    } else {
        _mm512_mask_i64scatter_epi32(out + 10, (__mmask8)active8, index,
                                   _mm512_cvtepi64_epi32(joint), 4);
    }
}

template <bool RowCodes>
Q8T static inline __attribute__((always_inline)) void cpu_glv8_high_digit(
    __m512i field, unsigned negative, __mmask8 &carry, unsigned &bad,
    uint32_t *out, unsigned digit, __m512i index, unsigned active8, unsigned stride) {
    const __m512i d = _mm512_mask_add_epi64(field, carry, field, _mm512_set1_epi64(1));
    carry = _mm512_cmp_epu64_mask(d, _mm512_set1_epi64(0x800000), _MM_CMPINT_GT);
    const __m512i mag = _mm512_mask_sub_epi64(d, carry, _mm512_set1_epi64(0x1000000), d);
    bad |= (unsigned)_mm512_cmpeq_epi64_mask(mag, _mm512_setzero_si512());
    const __m512i code = _mm512_mask_or_epi64(mag, (__mmask8)((unsigned)carry ^ negative),
                                           mag, _mm512_set1_epi64(0x80000000ULL));
    cpu_glv8_store_high<RowCodes>(out, digit, index, active8, code, stride);
}

/* Recode one signed residue, releasing each high digit after its scatter.
 * Return the signed low8 digit in eight 64-bit lanes. Bit128 is the sign. */
template <bool RowCodes>
Q8T static inline __attribute__((always_inline)) __m512i cpu_glv8_recode(
    CpuGlv8Value v, uint32_t *out, unsigned first, __m512i index,
    unsigned active8, unsigned &bad, unsigned stride) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i m52 = _mm512_set1_epi64(0xfffffffffffffULL);
    const __m512i m25 = _mm512_set1_epi64(0x1ffffff);
    const unsigned negative = (unsigned)_mm512_cmpneq_epi64_mask(
        _mm512_and_si512(v.x2, _mm512_set1_epi64(0x1000000)), zero);
    v.x0 = _mm512_mask_xor_epi64(v.x0, (__mmask8)negative, v.x0, m52);
    v.x1 = _mm512_mask_xor_epi64(v.x1, (__mmask8)negative, v.x1, m52);
    v.x2 = _mm512_mask_xor_epi64(v.x2, (__mmask8)negative, v.x2, m25);
    v.x0 = _mm512_mask_add_epi64(v.x0, (__mmask8)negative, v.x0, _mm512_set1_epi64(1));
    v.x1 = _mm512_add_epi64(v.x1, _mm512_srli_epi64(v.x0, 52));
    v.x0 = _mm512_and_si512(v.x0, m52);
    v.x2 = _mm512_and_si512(_mm512_add_epi64(v.x2, _mm512_srli_epi64(v.x1, 52)), m25);
    v.x1 = _mm512_and_si512(v.x1, m52);
    const __m512i lo = _mm512_or_si512(v.x0, _mm512_slli_epi64(v.x1, 52));
    const __m512i hi = _mm512_or_si512(_mm512_srli_epi64(v.x1, 12), _mm512_slli_epi64(v.x2, 40));
    __m512i low = _mm512_and_si512(lo, _mm512_set1_epi64(255));
    __mmask8 carry = _mm512_cmp_epu64_mask(low, _mm512_set1_epi64(128), _MM_CMPINT_GT);
    low = _mm512_mask_sub_epi64(low, carry, low, _mm512_set1_epi64(256));
    low = _mm512_mask_sub_epi64(low, (__mmask8)negative, zero, low);
    const __m512i m24 = _mm512_set1_epi64(0xffffff);
    cpu_glv8_high_digit<RowCodes>(_mm512_and_si512(_mm512_srli_epi64(lo, 8), m24), negative, carry, bad, out, first, index, active8, stride);
    cpu_glv8_high_digit<RowCodes>(_mm512_and_si512(_mm512_srli_epi64(lo, 32), m24), negative, carry, bad, out, first + 1, index, active8, stride);
    cpu_glv8_high_digit<RowCodes>(_mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(lo, 56), _mm512_slli_epi64(hi, 8)), m24), negative, carry, bad, out, first + 2, index, active8, stride);
    cpu_glv8_high_digit<RowCodes>(_mm512_and_si512(_mm512_srli_epi64(hi, 16), m24), negative, carry, bad, out, first + 3, index, active8, stride);
    __m512i top = _mm512_srli_epi64(hi, 40);
    top = _mm512_mask_add_epi64(top, carry, top, _mm512_set1_epi64(1));
    bad |= (unsigned)_mm512_cmpeq_epi64_mask(top, zero);
    top = _mm512_mask_or_epi64(top, (__mmask8)negative, top, _mm512_set1_epi64(0x80000000ULL));
    cpu_glv8_store_high<RowCodes>(out, first + 4, index, active8, top, stride);
    return low;
}

/* Both output policies share every coefficient, cold correction, residual
 * and signed-digit operation. Only destination indices and final stores vary. */
template <bool RowCodes>
Q8T static inline __attribute__((always_inline)) unsigned cpu_glv8_raw_impl(
    const uint64_t *raw64, unsigned active8, unsigned lane_offset,
    uint32_t *out, unsigned stride) {
    active8 &= 255u;
    if (!active8) return 0;
    CpuGlv8Value k, c1, c2;
    unsigned guard1, guard2, bit128;
    {
        const __m512i m52 = _mm512_set1_epi64(0xfffffffffffffULL);
        __m512i r0 = _mm512_loadu_si512((const void *)(raw64));
        __m512i r1 = _mm512_loadu_si512((const void *)(raw64 + 8));
        __m512i r2 = _mm512_loadu_si512((const void *)(raw64 + 16));
        __m512i r3 = _mm512_loadu_si512((const void *)(raw64 + 24));
        bit128 = (unsigned)_mm512_cmpneq_epi64_mask(_mm512_and_si512(r2, _mm512_set1_epi64(1)), _mm512_setzero_si512());
        k.x0 = _mm512_and_si512(r0, m52);
        r0 = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(r0, 52), _mm512_slli_epi64(r1, 12)), m52);
        r1 = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(r1, 40), _mm512_slli_epi64(r2, 24)), m52);
        r2 = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(r2, 28), _mm512_slli_epi64(r3, 36)), m52);
        r3 = _mm512_srli_epi64(r3, 16);
        c1 = cpu_glv8_coeff52<3>(r1, r2, r3, CPU_GLV8_G1_52, guard1);
        c2 = cpu_glv8_coeff52<4>(r1, r2, r3, CPU_GLV8_G2_52, guard2);
        k.x1 = r0;
        k.x2 = _mm512_and_si512(r1, _mm512_set1_epi64(0x1ffffff));
    }
    guard1 &= active8;
    guard2 &= active8;
    if (guard1 | guard2) {
        alignas(64) uint64_t snapshot[48];
        _mm512_store_si512((void *)(snapshot), c1.x0);
        _mm512_store_si512((void *)(snapshot + 8), c1.x1);
        _mm512_store_si512((void *)(snapshot + 16), c1.x2);
        _mm512_store_si512((void *)(snapshot + 24), c2.x0);
        _mm512_store_si512((void *)(snapshot + 32), c2.x1);
        _mm512_store_si512((void *)(snapshot + 40), c2.x2);
        cpu_glv8_patch_raw_coeffs(raw64, snapshot, guard1, guard2);
        /* Every live vector is replaced from memory after the scalar call. */
        c1.x0 = _mm512_load_si512((const void *)(snapshot));
        c1.x1 = _mm512_load_si512((const void *)(snapshot + 8));
        c1.x2 = _mm512_load_si512((const void *)(snapshot + 16));
        c2.x0 = _mm512_load_si512((const void *)(snapshot + 24));
        c2.x1 = _mm512_load_si512((const void *)(snapshot + 32));
        c2.x2 = _mm512_load_si512((const void *)(snapshot + 40));
        const __m512i r0 = _mm512_loadu_si512((const void *)(raw64));
        const __m512i r1 = _mm512_loadu_si512((const void *)(raw64 + 8));
        const __m512i m52 = _mm512_set1_epi64(0xfffffffffffffULL);
        k.x0 = _mm512_and_si512(r0, m52);
        k.x1 = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(r0, 52), _mm512_slli_epi64(r1, 12)), m52);
        k.x2 = _mm512_mask_or_epi64(_mm512_srli_epi64(r1, 40), (__mmask8)bit128,
                                   _mm512_srli_epi64(r1, 40), _mm512_set1_epi64(0x1000000));
    }
    CpuGlv8Value p;
    {
        const CpuGlv8Value sum = cpu_glv8_add129(c1, c2);
        p = cpu_glv8_mul129(sum, CPU_GLV8_A1_52);
    }
    {
        const CpuGlv8Value q = cpu_glv8_mul129(c2, CPU_GLV8_B1_52);
        k = cpu_glv8_sub129(cpu_glv8_sub129(k, p), q);
    }
    CpuGlv8Value r = cpu_glv8_mul129(c1, CPU_GLV8_A2_52);
    r = cpu_glv8_sub129(r, p);
    /* F's widening 32x32 multiplication handles lane*stride without DQ.
     * Only the index becomes negative for inactive lanes before lane_offset. */
    const unsigned step = RowCodes ? 1u : stride;
    __m512i index = _mm512_mul_epu32(_mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7), _mm512_set1_epi64(step));
    index = _mm512_sub_epi64(index, _mm512_set1_epi64((uint64_t)lane_offset * step));
    unsigned bad = 0;
    const __m512i low2 = cpu_glv8_recode<RowCodes>(r, out, 0, index, active8, bad, stride);
    const __m512i low1 = cpu_glv8_recode<RowCodes>(k, out, 5, index, active8, bad, stride);
    const __m512i zero = _mm512_setzero_si512();
    const __mmask8 joint_negative = _mm512_cmp_epi64_mask(low2, zero, _MM_CMPINT_LT) |
        (_mm512_cmpeq_epi64_mask(low2, zero) & _mm512_cmp_epi64_mask(low1, zero, _MM_CMPINT_LT));
    const __m512i x = _mm512_mask_sub_epi64(low1, joint_negative, zero, low1);
    const __m512i y = _mm512_mask_sub_epi64(low2, joint_negative, zero, low2);
    __m512i joint = _mm512_add_epi64(_mm512_add_epi64(_mm512_slli_epi64(y, 8), y), x);
    joint = _mm512_add_epi64(joint, _mm512_set1_epi64(1));
    joint = _mm512_mask_or_epi64(joint, joint_negative, joint, _mm512_set1_epi64(0x80000000ULL));
    cpu_glv8_store_joint<RowCodes>(out, index, active8, joint, stride);
    return bad & active8;
}

/* raw64 is read-only [4][8], low64 limb first. active8 uses ORIGINAL lane
 * positions. Require lane_offset in [0,7], stride>=11, and every active lane
 * j>=lane_offset. Write its eleven words to out[(j-lane_offset)*stride+i].
 * Return the high-zero bad mask in original lane positions, intersect active8.
 * An empty mask returns without reading either pointer. */
Q8T static __attribute__((noinline)) unsigned cpu_glv_digits8_raw(
    const uint64_t *raw64, unsigned active8, unsigned lane_offset,
    uint32_t *out, unsigned stride) {
    return cpu_glv8_raw_impl<false>(raw64, active8, lane_offset, out, stride);
}

/* Direct row planes: active lane j writes out[i*row_stride+j-lane_offset].
 * The caller supplies its candidate-offset base and row_stride=QSB_CPU_BATCH;
 * all active destinations must fit each of the twelve planes. Bit31 retains
 * the y sign; low31 is the table offset in uint64 words. High planes 0..9
 * use (max(magnitude,1)-1)*8. Joint planes 10/11 include (sign XOR recid)
 * times CPU_GLV_JOINT_ENTRIES*8 relative to cfold, while both retain sign.
 * Require lane_offset in [0,7] and active j>=lane_offset. Mask/bad and empty
 * behavior are identical to the old digits API. No raw normalization occurs. */
Q8T static __attribute__((noinline)) unsigned cpu_glv_rowcodes8_raw(
    const uint64_t *raw64, unsigned active8, unsigned lane_offset,
    uint32_t *out, unsigned row_stride) {
    return cpu_glv8_raw_impl<true>(raw64, active8, lane_offset, out, row_stride);
}
