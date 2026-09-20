/* Sparse-schedule SHA-256 for the Fast 11-byte locktime tail block
 * (delta B, scarletbright 7f965b4d). Pad shape: W[0..2] live, W[3..14]=0,
 * W[15]=9995*8=79960. Continues from an existing midstate. The first 16
 * rounds and the first in-place WMIX drop zero addends; later rounds use the
 * generic SHA256_RND / WMIX schedule. Bit-identical to _SHA256Transform on
 * that padded block. */
__device__ __forceinline__ void _SHA256TransformFastTail11(
    uint32_t state[8], uint32_t w0, uint32_t w1, uint32_t w2)
{
    const uint32_t L = 9995u * 8u; /* 79960 */
    uint32_t t1;
    uint32_t t2;

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    uint32_t w[16];
    w[0] = w0;
    w[1] = w1;
    w[2] = w2;
#pragma unroll
    for (int i = 3; i < 15; i++) w[i] = 0;
    w[15] = L;

    S2Round(a, b, c, d, e, f, g, h, K[0], w[0]);
    S2Round(h, a, b, c, d, e, f, g, K[1], w[1]);
    S2Round(g, h, a, b, c, d, e, f, K[2], w[2]);
    S2Round(f, g, h, a, b, c, d, e, K[3], 0u);
    S2Round(e, f, g, h, a, b, c, d, K[4], 0u);
    S2Round(d, e, f, g, h, a, b, c, K[5], 0u);
    S2Round(c, d, e, f, g, h, a, b, K[6], 0u);
    S2Round(b, c, d, e, f, g, h, a, K[7], 0u);
    S2Round(a, b, c, d, e, f, g, h, K[8], 0u);
    S2Round(h, a, b, c, d, e, f, g, K[9], 0u);
    S2Round(g, h, a, b, c, d, e, f, K[10], 0u);
    S2Round(f, g, h, a, b, c, d, e, K[11], 0u);
    S2Round(e, f, g, h, a, b, c, d, K[12], 0u);
    S2Round(d, e, f, g, h, a, b, c, K[13], 0u);
    S2Round(c, d, e, f, g, h, a, b, K[14], 0u);
    S2Round(b, c, d, e, f, g, h, a, K[15], L);

    {
        w[0] += s0(w[1]);
        w[1] += s1(L) + s0(w[2]);
        w[2] += s1(w[0]);
        w[3]  = s1(w[1]);
        w[4]  = s1(w[2]);
        w[5]  = s1(w[3]);
        w[6]  = s1(w[4]) + L;
        w[7]  = s1(w[5]) + w[0];
        w[8]  = s1(w[6]) + w[1];
        w[9]  = s1(w[7]) + w[2];
        w[10] = s1(w[8]) + w[3];
        w[11] = s1(w[9]) + w[4];
        w[12] = s1(w[10]) + w[5];
        w[13] = s1(w[11]) + w[6];
        w[14] = s1(w[12]) + w[7] + s0(L);
        w[15] += s1(w[13]) + w[8] + s0(w[0]);
    }

    SHA256_RND(16);
    WMIX();
    SHA256_RND(32);
    WMIX();
    SHA256_RND(48);

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

/* Sparse-schedule SHA-256 for the SHA256d second compression (delta D,
 * preludebrace bc77eb42): 32-byte message = first digest as eight words,
 * fixed pad W[8]=0x80000000, W[9..14]=0, W[15]=256, from the SHA-256 IV.
 * Bit-identical to _SHA256Initialize + _SHA256Transform on that block. */
__device__ __forceinline__ void _SHA256TransformDigest32(
    uint32_t out[8], const uint32_t m[8])
{
    uint32_t t1;
    uint32_t t2;

    uint32_t a = 0x6a09e667u;
    uint32_t b = 0xbb67ae85u;
    uint32_t c = 0x3c6ef372u;
    uint32_t d = 0xa54ff53au;
    uint32_t e = 0x510e527fu;
    uint32_t f = 0x9b05688cu;
    uint32_t g = 0x1f83d9abu;
    uint32_t h = 0x5be0cd19u;

    uint32_t w[16];
#pragma unroll
    for (int i = 0; i < 8; i++) w[i] = m[i];

    S2Round(a, b, c, d, e, f, g, h, K[0], w[0]);
    S2Round(h, a, b, c, d, e, f, g, K[1], w[1]);
    S2Round(g, h, a, b, c, d, e, f, K[2], w[2]);
    S2Round(f, g, h, a, b, c, d, e, K[3], w[3]);
    S2Round(e, f, g, h, a, b, c, d, K[4], w[4]);
    S2Round(d, e, f, g, h, a, b, c, K[5], w[5]);
    S2Round(c, d, e, f, g, h, a, b, K[6], w[6]);
    S2Round(b, c, d, e, f, g, h, a, K[7], w[7]);
    S2Round(a, b, c, d, e, f, g, h, K[8], 0x80000000u);
    S2Round(h, a, b, c, d, e, f, g, K[9], 0u);
    S2Round(g, h, a, b, c, d, e, f, K[10], 0u);
    S2Round(f, g, h, a, b, c, d, e, K[11], 0u);
    S2Round(e, f, g, h, a, b, c, d, K[12], 0u);
    S2Round(d, e, f, g, h, a, b, c, K[13], 0u);
    S2Round(c, d, e, f, g, h, a, b, K[14], 0u);
    S2Round(b, c, d, e, f, g, h, a, K[15], 256u);

    {
        /* First schedule expansion; w[9..14]=0 and w[8]/w[15] are the fixed
         * pad words. s0(0)=s1(0)=0, so zero terms vanish. */
        w[0] += s0(w[1]);
        w[1] += s1(256u) + s0(w[2]);
        w[2] += s1(w[0]) + s0(w[3]);
        w[3] += s1(w[1]) + s0(w[4]);
        w[4] += s1(w[2]) + s0(w[5]);
        w[5] += s1(w[3]) + s0(w[6]);
        w[6] += s1(w[4]) + 256u + s0(w[7]);
        w[7] += s1(w[5]) + w[0] + s0(0x80000000u);
        w[8]  = 0x80000000u + s1(w[6]) + w[1];
        w[9]  = s1(w[7]) + w[2];
        w[10] = s1(w[8]) + w[3];
        w[11] = s1(w[9]) + w[4];
        w[12] = s1(w[10]) + w[5];
        w[13] = s1(w[11]) + w[6];
        w[14] = s1(w[12]) + w[7] + s0(256u);
        w[15] = 256u + s1(w[13]) + w[8] + s0(w[0]);
    }

    SHA256_RND(16);
    WMIX();
    SHA256_RND(32);
    WMIX();
    SHA256_RND(48);

    out[0] = 0x6a09e667u + a;
    out[1] = 0xbb67ae85u + b;
    out[2] = 0x3c6ef372u + c;
    out[3] = 0xa54ff53au + d;
    out[4] = 0x510e527fu + e;
    out[5] = 0x9b05688cu + f;
    out[6] = 0x1f83d9abu + g;
    out[7] = 0x5be0cd19u + h;
}

/* Sparse-schedule SHA-256 for the 33-byte compressed public key (delta D):
 * live words pb[0..8], W[9..14]=0, W[15]=0x108, from the SHA-256 IV.
 * Bit-identical to _SHA256Initialize + _SHA256Transform on that block. */
__device__ __forceinline__ void _SHA256TransformPubkey33(
    uint32_t out[8], const uint32_t m[9])
{
    uint32_t t1;
    uint32_t t2;

    uint32_t a = 0x6a09e667u;
    uint32_t b = 0xbb67ae85u;
    uint32_t c = 0x3c6ef372u;
    uint32_t d = 0xa54ff53au;
    uint32_t e = 0x510e527fu;
    uint32_t f = 0x9b05688cu;
    uint32_t g = 0x1f83d9abu;
    uint32_t h = 0x5be0cd19u;

    uint32_t w[16];
#pragma unroll
    for (int i = 0; i < 9; i++) w[i] = m[i];

    S2Round(a, b, c, d, e, f, g, h, K[0], w[0]);
    S2Round(h, a, b, c, d, e, f, g, K[1], w[1]);
    S2Round(g, h, a, b, c, d, e, f, K[2], w[2]);
    S2Round(f, g, h, a, b, c, d, e, K[3], w[3]);
    S2Round(e, f, g, h, a, b, c, d, K[4], w[4]);
    S2Round(d, e, f, g, h, a, b, c, K[5], w[5]);
    S2Round(c, d, e, f, g, h, a, b, K[6], w[6]);
    S2Round(b, c, d, e, f, g, h, a, K[7], w[7]);
    S2Round(a, b, c, d, e, f, g, h, K[8], w[8]);
    S2Round(h, a, b, c, d, e, f, g, K[9], 0u);
    S2Round(g, h, a, b, c, d, e, f, K[10], 0u);
    S2Round(f, g, h, a, b, c, d, e, K[11], 0u);
    S2Round(e, f, g, h, a, b, c, d, K[12], 0u);
    S2Round(d, e, f, g, h, a, b, c, K[13], 0u);
    S2Round(c, d, e, f, g, h, a, b, K[14], 0u);
    S2Round(b, c, d, e, f, g, h, a, K[15], 0x108u);

    {
        /* First schedule expansion; w[9..14]=0 and w[15]=0x108 is fixed. */
        w[0] += s0(w[1]);
        w[1] += s1(0x108u) + s0(w[2]);
        w[2] += s1(w[0]) + s0(w[3]);
        w[3] += s1(w[1]) + s0(w[4]);
        w[4] += s1(w[2]) + s0(w[5]);
        w[5] += s1(w[3]) + s0(w[6]);
        w[6] += s1(w[4]) + 0x108u + s0(w[7]);
        w[7] += s1(w[5]) + w[0] + s0(w[8]);
        w[8] += s1(w[6]) + w[1];
        w[9]  = s1(w[7]) + w[2];
        w[10] = s1(w[8]) + w[3];
        w[11] = s1(w[9]) + w[4];
        w[12] = s1(w[10]) + w[5];
        w[13] = s1(w[11]) + w[6];
        w[14] = s1(w[12]) + w[7] + s0(0x108u);
        w[15] = 0x108u + s1(w[13]) + w[8] + s0(w[0]);
    }

    SHA256_RND(16);
    /* Scheduling-only experiment: interleave the two dense schedule expansions
     * with their compression rounds. All 64 SHA-256 rounds remain intact. */
    QSB_SHA_INTERLEAVED_16(32);
    QSB_SHA_INTERLEAVED_16(48);

    out[0] = 0x6a09e667u + a;
    out[1] = 0xbb67ae85u + b;
    out[2] = 0x3c6ef372u + c;
    out[3] = 0xa54ff53au + d;
    out[4] = 0x510e527fu + e;
    out[5] = 0x9b05688cu + f;
    out[6] = 0x1f83d9abu + g;
    out[7] = 0x5be0cd19u + h;
}

