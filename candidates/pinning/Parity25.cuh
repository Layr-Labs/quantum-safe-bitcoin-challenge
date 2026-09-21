// Research only: specialized to promoted 66fede0 raw _ModMultCore contract.
// No production include. Caller MUST execute original full raw multiply and parity on false.
#pragma once
__device__ __forceinline__ bool qsb_try_parity25(
    const uint64_t *a, const uint64_t *b, const uint64_t *offset,
    uint32_t neg, uint32_t &out) {
    const uint32_t a0 = uint32_t(a[0]);
    const uint32_t a1 = uint32_t(a[0] >> 32);
    const uint32_t a2 = uint32_t(a[1]);
    const uint32_t a3 = uint32_t(a[1] >> 32);
    const uint32_t a4 = uint32_t(a[2]);
    const uint32_t a5 = uint32_t(a[2] >> 32);
    const uint32_t a6 = uint32_t(a[3]);
    const uint32_t a7 = uint32_t(a[3] >> 32);
    const uint32_t b0 = uint32_t(b[0]);
    const uint32_t b1 = uint32_t(b[0] >> 32);
    const uint32_t b2 = uint32_t(b[1]);
    const uint32_t b3 = uint32_t(b[1] >> 32);
    const uint32_t b4 = uint32_t(b[2]);
    const uint32_t b5 = uint32_t(b[2] >> 32);
    const uint32_t b6 = uint32_t(b[3]);
    const uint32_t b7 = uint32_t(b[3] >> 32);
    uint64_t low = 0;
    low += __umulhi(a0, b6);
    low += __umulhi(a1, b5);
    low += __umulhi(a2, b4);
    low += __umulhi(a3, b3);
    low += __umulhi(a4, b2);
    low += __umulhi(a5, b1);
    low += __umulhi(a6, b0);
    low += uint64_t(a0) * b7;
    low += uint64_t(a1) * b6;
    low += uint64_t(a2) * b5;
    low += uint64_t(a3) * b4;
    low += uint64_t(a4) * b3;
    low += uint64_t(a5) * b2;
    low += uint64_t(a6) * b1;
    low += uint64_t(a7) * b0;
    const uint32_t odd = (a1 & b7) ^ (a2 & b6) ^ (a3 & b5) ^ (a4 & b4) ^ (a5 & b3) ^ (a6 & b2) ^ (a7 & b1);
    low += uint64_t(odd & 1u) << 32;
    const uint32_t lo = uint32_t(low);
    if (lo >= 0xfffffff4u) return false;
    const uint32_t product_bit = uint32_t(low >> 32) & 1u;
    const uint64_t p57 = uint64_t(a5) * b7;
    const uint64_t p66 = uint64_t(a6) * b6;
    const uint64_t p67 = uint64_t(a6) * b7;
    const uint64_t p75 = uint64_t(a7) * b5;
    const uint64_t p76 = uint64_t(a7) * b6;
    const uint64_t p77 = uint64_t(a7) * b7;
    uint64_t carry = 0;
    carry += uint64_t(__umulhi(a4, b7)) + uint64_t(__umulhi(a5, b6)) + uint64_t(__umulhi(a6, b5)) + uint64_t(__umulhi(a7, b4)) + uint64_t(uint32_t(p57)) + uint64_t(uint32_t(p66)) + uint64_t(uint32_t(p75));
    const uint32_t t0 = uint32_t(carry);
    carry >>= 32;
    carry += uint64_t((p57 >> 32)) + uint64_t((p66 >> 32)) + uint64_t(uint32_t(p67)) + uint64_t((p75 >> 32)) + uint64_t(uint32_t(p76));
    const uint32_t t1 = uint32_t(carry);
    carry >>= 32;
    carry += uint64_t((p67 >> 32)) + uint64_t((p76 >> 32)) + uint64_t(uint32_t(p77));
    const uint32_t t2 = uint32_t(carry);
    carry >>= 32;
    carry += uint64_t((p77 >> 32));
    const uint32_t t3 = uint32_t(carry);
    carry >>= 32;
    // K * top, retaining five low words; sixth word cannot affect parity.
    carry = uint64_t(t0) * 977u;
    const uint32_t z0 = uint32_t(carry);
    carry >>= 32;
    carry += uint64_t(t1) * 977u + t0;
    const uint32_t z1 = uint32_t(carry);
    carry >>= 32;
    carry += uint64_t(t2) * 977u + t1;
    const uint32_t z2 = uint32_t(carry);
    carry >>= 32;
    carry += uint64_t(t3) * 977u + t2;
    const uint32_t z3 = uint32_t(carry);
    carry >>= 32;
    const uint32_t z4 = uint32_t(uint64_t(t3) + carry);
    // floor(K*hi/2^224) is certified if adding 9*K cannot carry through bit96.
    carry = uint64_t(z0) + 8793u;
    carry = uint64_t(z1) + 9u + (carry >> 32);
    carry = uint64_t(z2) + (carry >> 32);
    if (carry >> 32) return false;
    const uint64_t f = ((uint64_t(z4) << 32) | z3) + lo;
    const uint32_t r = uint32_t(f);
    if (r >= 0xfffffff3u) return false;
    // Conservative top-word quotient/zero certificate; ambiguous bounds fall back.
    const uint64_t lower = uint64_t(r) + uint32_t(offset[3] >> 32);
    const uint64_t upper = lower + 14u;
    if ((lower >> 32) != (upper >> 32)) return false;
    if (uint32_t(lower) == 0u || uint32_t(upper) == 0xffffffffu) return false;
    out = ((a0 & b0) ^ product_bit ^ uint32_t(f >> 32) ^
           uint32_t(offset[0]) ^ uint32_t(lower >> 32) ^ neg) & 1u;
    return true;
}
