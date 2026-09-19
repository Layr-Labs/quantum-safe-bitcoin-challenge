#pragma once
#include <stdint.h>

// Independent fixed-width implementation of the pinned secp256k1 GLV lattice.
// Numeric basis/rounding constants: bitcoin-core/secp256k1 46db787112beabdb5e17e0dc35680716f1057e7b.
// Isolated validation probe, not a production candidate or a timing claim.
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define GLV_HD __host__ __device__ __forceinline__
#define GLV_UNROLL _Pragma("unroll")
#else
#define GLV_HD inline
#define GLV_UNROLL
#endif
#ifndef GLV_MUTANT
#define GLV_MUTANT 0
#endif

namespace glv_probe {
using W = uint64_t;
struct U128 { W v[2]; };
struct U192 { W v[3]; };
struct U256 { W v[4]; };
struct U512 { W v[8]; };
struct Signed128 { W lo, hi; unsigned negative; };
struct Split { Signed128 first, second; bool range_ok; };
struct Digits { int32_t v[8]; bool range_ok; };
struct RecodedSplit { Digits first, second; bool range_ok; };

GLV_HD W high_product(W a, W b) {
#ifdef __CUDA_ARCH__
    return __umul64hi(a, b);
#else
    return W((__uint128_t(a) * b) >> 64);
#endif
}

// a*b + old + carry <= (2^64-1)^2 + 2*(2^64-1) = 2^128-1.
GLV_HD void multiply_accumulate(W a, W b, W old, W carry, W &lo, W &hi) {
    W product = a * b;
    hi = high_product(a, b);
    W first = product + old;
#if GLV_MUTANT != 2
    hi += W(first < product);
#endif
    lo = first + carry;
    hi += W(lo < first);
}

GLV_HD U512 multiply_full(U256 a, U256 b) {
    U512 out = {};
    GLV_UNROLL
    for (int i = 0; i < 4; ++i) {
        W carry = 0;
        GLV_UNROLL
        for (int j = 0; j < 4; ++j) {
            W lo, hi;
            multiply_accumulate(a.v[i], b.v[j], out.v[i+j], carry, lo, hi);
            out.v[i+j] = lo;
            carry = hi;
        }
        out.v[i+4] = carry;
    }
    return out;
}

GLV_HD U192 rounded_high128(U256 a, U256 b) {
    U512 p = multiply_full(a, b);
#if GLV_MUTANT == 1
    W round = 0;
#else
    W round = p.v[5] >> 63;
#endif
    U192 result = {{p.v[6] + round, p.v[7], 0}};
    W carry = W(result.v[0] < p.v[6]);
    result.v[1] += carry;
    result.v[2] = W(result.v[1] < p.v[7]);
    return result;
}

GLV_HD U192 multiply_low192(U128 a, U192 b) {
    U192 out = {};
    GLV_UNROLL
    for (int i = 0; i < 2; ++i) {
        W carry = 0;
        GLV_UNROLL
        for (int j = 0; j < 3-i; ++j) {
            W lo, hi;
            multiply_accumulate(a.v[i], b.v[j], out.v[i+j], carry, lo, hi);
            out.v[i+j] = lo;
            carry = hi;
        }
        // All further product bits are discarded modulo 2^192.
    }
    return out;
}

GLV_HD U192 subtract192(U192 a, U192 b) {
    U192 result;
    W borrow = 0;
    GLV_UNROLL
    for (int i = 0; i < 3; ++i) {
        W sub = b.v[i] + borrow;
        W next = W(sub < b.v[i]) | W(a.v[i] < sub);
        result.v[i] = a.v[i] - sub;
#if GLV_MUTANT == 5
        borrow = 0;
#else
        borrow = next;
#endif
        (void)next;
    }
    return result;
}

GLV_HD U256 canonical(U256 k) {
    const U256 n = {{0xbfd25e8cd0364141ULL, 0xbaaedce6af48a03bULL,
                    0xfffffffffffffffeULL, 0xffffffffffffffffULL}};
    bool greater_equal = true;
    GLV_UNROLL
    for (int i = 3; i >= 0; --i) {
        if (k.v[i] != n.v[i]) {
            greater_equal = k.v[i] > n.v[i];
            break;
        }
    }
    if (greater_equal) {
        W borrow = 0;
        GLV_UNROLL
        for (int i = 0; i < 4; ++i) {
            W sub = n.v[i] + borrow;
            W next = W(sub < n.v[i]) | W(k.v[i] < sub);
            k.v[i] -= sub;
            borrow = next;
        }
    }
    return k;
}

GLV_HD Signed128 signed_magnitude(U192 value, bool &in_range) {
#if GLV_MUTANT == 3
    unsigned negative = unsigned(value.v[1] >> 63);
#else
    unsigned negative = unsigned(value.v[2] >> 63);
#endif
    if (negative) {
        const U192 zero = {};
        value = subtract192(zero, value);
    }
    in_range = value.v[2] == 0;
    return {value.v[0], value.v[1], negative};
}

GLV_HD Split split(U256 raw) {
    const U256 g1 = {{0xe893209a45dbb031ULL, 0x3daa8a1471e8ca7fULL,
                     0xe86c90e49284eb15ULL, 0x3086d221a7d46bcdULL}};
    const U256 g2 = {{0x1571b4ae8ac47f71ULL, 0x221208ac9df506c6ULL,
                     0x6f547fa90abfe4c4ULL, 0xe4437ed6010e8828ULL}};
    const U192 a1 = {{0xe86c90e49284eb15ULL, 0x3086d221a7d46bcdULL, 0}};
    const U192 b1 = {{0x6f547fa90abfe4c3ULL, 0xe4437ed6010e8828ULL, 0}};
#if GLV_MUTANT == 4
    const U192 a2 = {{0x57c1108d9d44cfd8ULL, 0x14ca50f7a8e2f3f6ULL, 0}};
#else
    const U192 a2 = {{0x57c1108d9d44cfd8ULL, 0x14ca50f7a8e2f3f6ULL, 1}};
#endif
    U256 k = canonical(raw);
    U192 rounded1 = rounded_high128(k, g1), rounded2 = rounded_high128(k, g2);
    U128 c1 = {{rounded1.v[0], rounded1.v[1]}};
    U128 c2 = {{rounded2.v[0], rounded2.v[1]}};
    U192 low_k = {{k.v[0], k.v[1], k.v[2]}};
    U192 first = subtract192(subtract192(low_k, multiply_low192(c1, a1)),
                            multiply_low192(c2, a2));
    U192 second = subtract192(multiply_low192(c1, b1), multiply_low192(c2, a1));
    bool ok1, ok2;
    Signed128 s1 = signed_magnitude(first, ok1), s2 = signed_magnitude(second, ok2);
    return {s1, s2, ok1 && ok2 && rounded1.v[2] == 0 && rounded2.v[2] == 0};
}

// Seven balanced low digits, followed by the complete signed quotient at 2^112.
// All shifts and multiword arithmetic are unsigned, including negative values.
GLV_HD Digits recode_wide(Signed128 value, unsigned max_top) {
    U192 q = {{value.lo, value.hi, 0}};
    bool valid_sign = value.negative <= 1 &&
                      ((value.lo | value.hi) != 0 || value.negative == 0);
#if GLV_MUTANT != 9
    if (value.negative) q = subtract192(U192{}, q);
#endif
    Digits out = {};
    GLV_UNROLL
    for (int i = 0; i < 7; ++i) {
        int32_t digit = int32_t(q.v[0] & 65535);
#if GLV_MUTANT == 6
        if (digit >= 32768) digit -= 65536;
#else
        if (digit > 32768) digit -= 65536;
#endif
        out.v[i] = digit;
        W extension = digit < 0 ? ~W(0) : 0;
        U192 signed_digit = {{W(int64_t(digit)), extension, extension}};
        q = subtract192(q, signed_digit);
#if GLV_MUTANT == 7
        extension = 0;
#else
        extension = W(0) - (q.v[2] >> 63);
#endif
        q = {{(q.v[0] >> 16) | (q.v[1] << 48),
              (q.v[1] >> 16) | (q.v[2] << 48),
              (q.v[2] >> 16) | (extension << 48)}};
    }
    bool top_in_range;
    Signed128 top = signed_magnitude(q, top_in_range);
    out.range_ok = valid_sign && top_in_range && top.hi == 0 &&
                   top.lo <= max_top && max_top <= 65536;
    if (out.range_ok) {
        int32_t magnitude = int32_t(top.lo);
        out.v[7] = top.negative ? -magnitude : magnitude;
    }
#if GLV_MUTANT == 8
    out.v[7] = int32_t((W(int64_t(out.v[7])) + 32768) & 65535) - 32768;
#endif
#if GLV_MUTANT == 10
    if ((value.lo | value.hi) == 0) out.v[0] = 1;
#endif
    return out;
}

GLV_HD RecodedSplit split_recode(U256 raw) {
    Split values = split(raw);
    Digits first = recode_wide(values.first, 41641);
    Digits second = recode_wide(values.second, 35429);
    return {first, second, values.range_ok && first.range_ok && second.range_ok};
}
} // namespace glv_probe
