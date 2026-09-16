/* pinning.cu — QSB pinning grinder (benchmark candidate)
 *
 * Per candidate (sequence, locktime):
 *   z  = SHA256d(pin_prefix || suffix[seq,lt])
 *   u1 = -z * r^-1 mod n
 *   Q1 = u1*G + u2*R   (recid 0)      Q2 = u1*G - u2*R   (recid 1)
 *   hit iff leading_zero_bits(SHA256(compress(Qi))) >= QSB_ZEROS_N
 *
 * Design (the field arithmetic comes from the GPLv3 VanitySearch GPUMath.h
 * beside this file, see COPYING; everything else is in this file):
 *
 *   * u1*G is a fixed-base comb with WBITS-bit windows (default 24: 11
 *     windows, 10 point additions) over an (x,y)-interleaved affine table
 *     that is built ON THE GPU at start-up in ~2 s (Montgomery-batched
 *     inversions; no OpenSSL table, no /tmp cache).  Window digits are made
 *     uniformly non-zero by folding a constant -D*G into the chunk-0 table,
 *     so the comb is a branch-free chain: one affine+affine add, then
 *     projective mixed adds (9M+2S each) with no inversion in the loop.
 *   * u2*R and -u2*R share the x coordinate, so the two recids share
 *     v = xA*Z - X, v^2, v^3 and Z3; both come out with ONE common Z.
 *   * That single Z per candidate is inverted with a warp-wide Montgomery
 *     batch (shuffle prefix/suffix products): one _ModInv per 32 candidates.
 *     The seed kernel paid two full _ModInv per candidate.
 *   * The suffix SHA block that does not contain the locktime is hashed once
 *     per sequence on the host; the kernel does 1 preimage block, the second
 *     SHA256, and one pubkey SHA per recid (4 transforms per candidate).
 *   * Host loop: 4M candidates per launch, one 4-byte readback per launch,
 *     hits appended to results/pinning_hit_<gpu>.txt in the bridge's format.
 *
 * Build:  nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm
 * Usage:  ./pinning <pinning.bin> [gpu_index] [total_gpus] [global_offset] [single_hash]
 *                   [selftest] [seconds=<s>] [debug <seq> <lt>]
 * Hits:   sequence=<u32>\nlocktime=<u32>\nrecid=<0|1>\n
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/stat.h>
#include <cuda_runtime.h>

#include "GPUMath.h"

extern "C" {
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
}

#ifndef QSB_ZEROS_N
#define QSB_ZEROS_N 24
#endif

#ifndef QSB_BLK
#define QSB_BLK 256          /* threads per block == batch-inversion group */
#endif
#ifndef QSB_MINBLK
#define QSB_MINBLK 2         /* __launch_bounds__ min blocks per SM */
#endif
#ifndef QSB_BATCH
#define QSB_BATCH (1u << 22) /* candidates per launch (multiple of QSB_BLK) */
#endif

#ifndef WBITS
#define WBITS 24             /* window width; even, 16..24. 24 -> 11 windows, 10.7 GB table */
#endif
#define NWIN ((256 + WBITS - 1) / WBITS)
#define WLAST (256 - WBITS * (NWIN - 1))
#define WSIZE (1ull << WBITS)
#define HB (WBITS / 2)
#define HSIZE (1u << HB)
#define NENTRIES ((size_t)(NWIN - 1) * WSIZE + ((size_t)1 << WLAST))
#ifndef QSB_GENK
#define QSB_GENK 32          /* entries per thread in the batched table generator */
#endif
#define CHK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)

/* ------------------------------------------------------------------ */
/* constant memory: per-sequence SHA template + problem constants      */
/* ------------------------------------------------------------------ */
__constant__ uint32_t c_state[8];      /* SHA state before the in-kernel block(s) */
__constant__ uint32_t c_words[32];     /* up to 2 padded blocks, lt bytes zero */
__constant__ int      c_nblk;          /* 1 or 2 in-kernel blocks */
__constant__ int      c_ltw[4];        /* word index of lt byte b */
__constant__ int      c_lts[4];        /* shift of lt byte b inside its word */
__constant__ uint64_t c_nri[4];        /* -r^-1 mod n */
__constant__ uint64_t c_ax[4], c_ay[4];/* u2*R affine */

/* ------------------------------------------------------------------ */
/* SHA-256                                                             */
/* ------------------------------------------------------------------ */
__constant__ uint32_t c_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) { return __funnelshift_r(x, x, n); }
#define SS0(x) (rotr(x,2)^rotr(x,13)^rotr(x,22))
#define SS1(x) (rotr(x,6)^rotr(x,11)^rotr(x,25))
#define ss0(x) (rotr(x,7)^rotr(x,18)^((x)>>3))
#define ss1(x) (rotr(x,17)^rotr(x,19)^((x)>>10))
#define CH(x,y,z) (((x)&(y))^(~(x)&(z)))
#define MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))

__device__ __forceinline__ void sha_transform(uint32_t s[8], uint32_t w[16]) {
    uint32_t a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
#pragma unroll
    for (int i = 0; i < 64; i++) {
        if (i >= 16) {
            w[i&15] = ss1(w[(i-2)&15]) + w[(i-7)&15] + ss0(w[(i-15)&15]) + w[(i-16)&15];
        }
        uint32_t t1 = h + SS1(e) + CH(e,f,g) + c_K[i] + w[i&15];
        uint32_t t2 = SS0(a) + MAJ(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s[0]+=a; s[1]+=b; s[2]+=c; s[3]+=d; s[4]+=e; s[5]+=f; s[6]+=g; s[7]+=h;
}

/* host SHA-256 transform (same algorithm) for the per-sequence midstate */
static uint32_t h_rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static const uint32_t h_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };
static void host_sha_transform(uint32_t s[8], const uint32_t in[16]) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = in[i];
    for (int i = 16; i < 64; i++) {
        uint32_t x = w[i-15], y = w[i-2];
        w[i] = (h_rotr(y,17)^h_rotr(y,19)^(y>>10)) + w[i-7] + (h_rotr(x,7)^h_rotr(x,18)^(x>>3)) + w[i-16];
    }
    uint32_t a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + (h_rotr(e,6)^h_rotr(e,11)^h_rotr(e,25)) + ((e&f)^(~e&g)) + h_K[i] + w[i];
        uint32_t t2 = (h_rotr(a,2)^h_rotr(a,13)^h_rotr(a,22)) + ((a&b)^(a&c)^(b&c));
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s[0]+=a; s[1]+=b; s[2]+=c; s[3]+=d; s[4]+=e; s[5]+=f; s[6]+=g; s[7]+=h;
}

/* ------------------------------------------------------------------ */
/* scalar multiplication mod n (secp256k1 group order)                 */
/* ------------------------------------------------------------------ */
__device__ void gpu_scalar_mulmod(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t p[8] = {0};
    for (int i = 0; i < 4; i++) {
        __uint128_t carry = 0;
        for (int j = 0; j < 4; j++) {
            __uint128_t v = (__uint128_t)p[i+j] + (__uint128_t)a[i]*b[j] + carry;
            p[i+j] = (uint64_t)v; carry = v >> 64;
        }
        p[i+4] = (uint64_t)carry;
    }
    const uint64_t C0=0x402DA1732FC9BEBFULL, C1=0x4551231950B75FC4ULL;
    uint64_t q[8] = {0};
    for (int i = 0; i < 4; i++) q[i] = p[i];
    {   uint64_t carry = 0;
        for (int i = 0; i < 4; i++) { __uint128_t v = (__uint128_t)q[i] + (__uint128_t)p[4+i]*C0 + carry; q[i]=(uint64_t)v; carry=v>>64; }
        q[4] = carry; }
    {   uint64_t carry = 0;
        for (int i = 0; i < 4; i++) { __uint128_t v = (__uint128_t)q[i+1] + (__uint128_t)p[4+i]*C1 + carry; q[i+1]=(uint64_t)v; carry=v>>64; }
        __uint128_t v = (__uint128_t)q[5] + carry; q[5]=(uint64_t)v;
        if (v>>64) { v = (__uint128_t)q[6] + (v>>64); q[6]=(uint64_t)v; if (v>>64) q[7] += (uint64_t)(v>>64); } }
    {   uint64_t carry = 0;
        for (int i = 0; i < 4; i++) { __uint128_t v = (__uint128_t)q[i+2] + (__uint128_t)p[4+i] + carry; q[i+2]=(uint64_t)v; carry=v>>64; }
        if (carry) { __uint128_t v = (__uint128_t)q[6] + carry; q[6]=(uint64_t)v; if (v>>64) q[7] += (uint64_t)(v>>64); } }
    uint64_t r2[8] = {0};
    for (int i = 0; i < 4; i++) r2[i] = q[i];
    {   uint64_t carry = 0;
        for (int i = 0; i < 4; i++) { __uint128_t v = (__uint128_t)r2[i] + (__uint128_t)q[4+i]*C0 + carry; r2[i]=(uint64_t)v; carry=v>>64; }
        r2[4] = carry; }
    {   uint64_t carry = 0;
        for (int i = 0; i < 4; i++) { __uint128_t v = (__uint128_t)r2[i+1] + (__uint128_t)q[4+i]*C1 + carry; r2[i+1]=(uint64_t)v; carry=v>>64; }
        __uint128_t v = (__uint128_t)r2[5] + carry; r2[5]=(uint64_t)v;
        if (v>>64) r2[6] += (uint64_t)(v>>64); }
    {   uint64_t carry = 0;
        for (int i = 0; i < 4; i++) { __uint128_t v = (__uint128_t)r2[i+2] + (__uint128_t)q[4+i] + carry; r2[i+2]=(uint64_t)v; carry=v>>64; }
        if (carry) r2[6] += carry; }
    uint64_t res[5];
    for (int i = 0; i < 5; i++) res[i] = r2[i];
    const uint64_t N[4]={0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
    for (int rep = 0; rep < 5; rep++) {
        int ge;
        if (res[4] > 0) ge = 1;
        else { ge = 1; for (int i = 3; i >= 0; i--) { if (res[i] > N[i]) { ge = 1; break; } if (res[i] < N[i]) { ge = 0; break; } } }
        if (!ge) break;
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t old = res[i], sub = N[i] + borrow;
            res[i] = old - sub;
            borrow = (old < sub || (borrow && N[i] == 0xFFFFFFFFFFFFFFFFULL)) ? 1 : 0;
        }
        if (res[4] > 0) res[4]--;
    }
    for (int i = 0; i < 4; i++) r[i] = res[i];
}

/* ------------------------------------------------------------------ */
/* small helpers                                                       */
/* ------------------------------------------------------------------ */
__device__ __forceinline__ void set1(uint64_t r[4]) { r[0]=1; r[1]=0; r[2]=0; r[3]=0; }
__device__ __forceinline__ void cpy(uint64_t r[4], const uint64_t a[4]) { r[0]=a[0]; r[1]=a[1]; r[2]=a[2]; r[3]=a[3]; }

__device__ __forceinline__ void shfl_up4(uint64_t r[4], const uint64_t a[4], int d) {
    r[0] = __shfl_up_sync(0xffffffffu, a[0], d);
    r[1] = __shfl_up_sync(0xffffffffu, a[1], d);
    r[2] = __shfl_up_sync(0xffffffffu, a[2], d);
    r[3] = __shfl_up_sync(0xffffffffu, a[3], d);
}
__device__ __forceinline__ void shfl_down4(uint64_t r[4], const uint64_t a[4], int d) {
    r[0] = __shfl_down_sync(0xffffffffu, a[0], d);
    r[1] = __shfl_down_sync(0xffffffffu, a[1], d);
    r[2] = __shfl_down_sync(0xffffffffu, a[2], d);
    r[3] = __shfl_down_sync(0xffffffffu, a[3], d);
}
__device__ __forceinline__ void shfl4(uint64_t r[4], const uint64_t a[4], int src) {
    r[0] = __shfl_sync(0xffffffffu, a[0], src);
    r[1] = __shfl_sync(0xffffffffu, a[1], src);
    r[2] = __shfl_sync(0xffffffffu, a[2], src);
    r[3] = __shfl_sync(0xffffffffu, a[3], src);
}

/* Warp-wide batched inversion: every lane supplies z (nonzero, < p) and
 * receives z^-1.  One _ModInv per warp (uniform across the warp, so it costs
 * 1/32 of a _ModInv per candidate); 12 _ModMult per lane. */
__device__ __forceinline__ void warp_batch_inv(uint64_t z[4], uint64_t out[4]) {
    const int lane = threadIdx.x & 31;
    uint64_t p[4], s[4], t[4];
    cpy(p, z); cpy(s, z);
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        shfl_up4(t, p, d);
        if (lane < d) set1(t);
        _ModMult(p, p, t);
        shfl_down4(t, s, d);
        if (lane + d > 31) set1(t);
        _ModMult(s, s, t);
    }
    uint64_t tot[5]; shfl4(tot, p, 31); tot[4] = 0;
    _ModInv(tot);                       /* uniform across the warp */
    uint64_t pm[4], sp[4];
    shfl_up4(pm, p, 1);   if (lane == 0)  set1(pm);
    shfl_down4(sp, s, 1); if (lane == 31) set1(sp);
    _ModMult(t, tot, pm);
    _ModMult(out, t, sp);
}

/* affine P += T with a supplied inverse of (xT - xP) */
__device__ __forceinline__ void aff_add_inv(uint64_t px[4], uint64_t py[4],
                                            const uint64_t tx[4], const uint64_t ty[4],
                                            uint64_t inv[4]) {
    uint64_t dy[4], lam[4], l2[4], x3[4], t[4];
    _ModSub256(dy, (uint64_t*)ty, py);
    _ModMult(lam, dy, inv);
    _ModSqr(l2, lam);
    _ModSub256(x3, l2, px);
    _ModSub256(x3, (uint64_t*)tx);      /* x3 = lam^2 - px - tx */
    _ModSub256(t, px, x3);              /* px - x3 */
    _ModMult(t, t, lam);
    _ModSub256(py, t, py);              /* y3 = lam*(px-x3) - py */
    cpy(px, x3);
}

/* full affine add with its own inversion (table generation only) */
__device__ void aff_add_slow(uint64_t px[4], uint64_t py[4], const uint64_t tx[4], const uint64_t ty[4]) {
    uint64_t dx[5];
    _ModSub256(dx, (uint64_t*)tx, px); dx[4] = 0;
    _ModInv(dx);
    aff_add_inv(px, py, tx, ty, dx);
}

/* ------------------------------------------------------------------ */
/* G-table generation on the GPU                                       */
/*   entry(c, k) = (k+1) * B_c + O_c,  k in [0, 65536)                 */
/*   B_c = 2^(16c) G,  O_0 = -D*G (D = sum 2^(16c)), O_c = infinity    */
/*   host supplies B_c and S0_c = B_c + O_c for every chunk.           */
/* ------------------------------------------------------------------ */
struct AffPt { uint64_t x[4]; uint64_t y[4]; };

/* one thread per chunk: T[i] = i*B (i=1..HSIZE-1), S[j] = S0 + HSIZE j B (j=0..HSIZE-1) */
__global__ void k_gen_serial(const AffPt *base, const AffPt *base2, const AffPt *s0, AffPt *T, AffPt *S) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= NWIN) return;
    AffPt *Tc = T + (size_t)c * HSIZE, *Sc = S + (size_t)c * HSIZE;
    uint64_t bx[4], by[4]; cpy(bx, base[c].x); cpy(by, base[c].y);
    uint64_t px[4], py[4];
    cpy(Tc[1].x, bx); cpy(Tc[1].y, by);
    /* T[2] = 2B is a doubling, which the affine add cannot do: host supplies it */
    cpy(px, base2[c].x); cpy(py, base2[c].y);
    cpy(Tc[2].x, px); cpy(Tc[2].y, py);
    for (int i = 3; i < (int)HSIZE; i++) { aff_add_slow(px, py, bx, by); cpy(Tc[i].x, px); cpy(Tc[i].y, py); }
    uint64_t rx[4], ry[4]; cpy(rx, px); cpy(ry, py);
    aff_add_slow(rx, ry, bx, by);               /* R = HSIZE * B */
    cpy(px, s0[c].x); cpy(py, s0[c].y);
    cpy(Sc[0].x, px); cpy(Sc[0].y, py);
    for (int j = 1; j < (int)HSIZE; j++) { aff_add_slow(px, py, rx, ry); cpy(Sc[j].x, px); cpy(Sc[j].y, py); }
}

__device__ __forceinline__ void store_entry(uint4 *tab, size_t idx, const uint64_t px[4], const uint64_t py[4]) {
    uint4 *e = tab + idx * 4;
    e[0] = make_uint4((uint32_t)px[0], (uint32_t)(px[0]>>32), (uint32_t)px[1], (uint32_t)(px[1]>>32));
    e[1] = make_uint4((uint32_t)px[2], (uint32_t)(px[2]>>32), (uint32_t)px[3], (uint32_t)(px[3]>>32));
    e[2] = make_uint4((uint32_t)py[0], (uint32_t)(py[0]>>32), (uint32_t)py[1], (uint32_t)(py[1]>>32));
    e[3] = make_uint4((uint32_t)py[2], (uint32_t)(py[2]>>32), (uint32_t)py[3], (uint32_t)(py[3]>>32));
}

/* entry(c, HSIZE j + i) = S[j] + T[i]  (i == 0: S[j]; c>0,k==1: 2B).
 * Each thread produces QSB_GENK consecutive entries (same j) with one
 * Montgomery-batched inversion. grid.x covers nentries/QSB_GENK, grid.y = chunk. */
__global__ void k_gen_table(const AffPt *T, const AffPt *S, uint4 *tab, uint32_t nentries, int c) {
    const int K = QSB_GENK;
    uint32_t k0 = (blockIdx.x * blockDim.x + threadIdx.x) * K;
    if (k0 >= nentries) return;
    const AffPt *Tc = T + (size_t)c * HSIZE, *Sc = S + (size_t)c * HSIZE;
    uint32_t j = k0 >> HB, i0 = k0 & (HSIZE - 1);
    uint64_t sx[4], sy[4]; cpy(sx, Sc[j].x); cpy(sy, Sc[j].y);
    uint64_t dx[K][4], pre[K][4];
    for (int m = 0; m < K; m++) {
        uint32_t i = i0 + m, k = k0 + m;
        int special = (i == 0) || (c > 0 && k == 1);
        if (special) set1(dx[m]); else _ModSub256(dx[m], (uint64_t*)Tc[i].x, sx);
        if (m == 0) cpy(pre[0], dx[0]); else _ModMult(pre[m], pre[m-1], dx[m]);
    }
    uint64_t inv[5]; cpy(inv, pre[K-1]); inv[4] = 0;
    _ModInv(inv);
    for (int m = K - 1; m >= 0; m--) {
        uint64_t im[4];
        if (m > 0) { _ModMult(im, inv, pre[m-1]); _ModMult(inv, inv, dx[m]); } else cpy(im, inv);
        uint32_t i = i0 + m, k = k0 + m;
        uint64_t px[4], py[4]; cpy(px, sx); cpy(py, sy);
        if (c > 0 && k == 1) { cpy(px, Tc[2].x); cpy(py, Tc[2].y); }
        else if (i != 0) aff_add_inv(px, py, Tc[i].x, Tc[i].y, im);
        store_entry(tab, (size_t)c * WSIZE + k, px, py);
    }
}

__device__ __forceinline__ void load_entry(const uint4 *__restrict__ tab, int c, uint32_t d, uint64_t x[4], uint64_t y[4]) {
    const uint4 *e = tab + ((size_t)c * WSIZE + d) * 4;
    uint4 a = __ldg(e), b = __ldg(e + 1), cc = __ldg(e + 2), dd = __ldg(e + 3);
    x[0] = ((uint64_t)a.y << 32) | a.x;  x[1] = ((uint64_t)a.w << 32) | a.z;
    x[2] = ((uint64_t)b.y << 32) | b.x;  x[3] = ((uint64_t)b.w << 32) | b.z;
    y[0] = ((uint64_t)cc.y << 32) | cc.x; y[1] = ((uint64_t)cc.w << 32) | cc.z;
    y[2] = ((uint64_t)dd.y << 32) | dd.x; y[3] = ((uint64_t)dd.w << 32) | dd.z;
}

/* ------------------------------------------------------------------ */
/* the grinder                                                         */
/* ------------------------------------------------------------------ */
__device__ __forceinline__ int gate(const uint32_t h[8]) {
#if QSB_ZEROS_N <= 32
    return (h[0] >> (32 - QSB_ZEROS_N)) == 0u || QSB_ZEROS_N == 0;
#else
    int z = 0;
    for (int i = 0; i < 8; i++) { if (h[i] == 0) { z += 32; continue; } z += __clz(h[i]); break; }
    return z >= QSB_ZEROS_N;
#endif
}

/* SHA256(02/03 || x) -> h */
__device__ __forceinline__ void hash_pubkey(const uint64_t x[4], const uint64_t y[4], uint32_t h[8]) {
    uint32_t B[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        B[j] = (uint32_t)(x[(7 - j) >> 1] >> (((7 - j) & 1) * 32));   /* big-endian word j of x */
    }
    uint32_t prefix = 0x02u + (uint32_t)(y[0] & 1);
    uint32_t w[16];
    w[0] = (prefix << 24) | (B[0] >> 8);
#pragma unroll
    for (int j = 1; j < 8; j++) w[j] = (B[j-1] << 24) | (B[j] >> 8);
    w[8] = (B[7] << 24) | 0x800000u;
#pragma unroll
    for (int j = 9; j < 15; j++) w[j] = 0;
    w[15] = 264;
    h[0]=0x6a09e667; h[1]=0xbb67ae85; h[2]=0x3c6ef372; h[3]=0xa54ff53a;
    h[4]=0x510e527f; h[5]=0x9b05688c; h[6]=0x1f83d9ab; h[7]=0x5be0cd19;
    sha_transform(h, w);
}

__global__ void __launch_bounds__(QSB_BLK, QSB_MINBLK)
k_grind(const uint4 *__restrict__ tab, uint32_t start_lt,
        uint32_t *d_hit_cnt, uint2 *d_hits, int hit_cap)
{
    const uint32_t idx = blockIdx.x * QSB_BLK + threadIdx.x;
    const uint32_t lt = start_lt + idx;

    /* --- z = SHA256d(preimage) --- */
    uint32_t st[8];
#pragma unroll
    for (int i = 0; i < 8; i++) st[i] = c_state[i];
    uint32_t w[16];
    if (c_nblk == 2) {
#pragma unroll
        for (int i = 0; i < 16; i++) w[i] = c_words[i];
#pragma unroll
        for (int b = 0; b < 4; b++) if (c_ltw[b] < 16) w[c_ltw[b]] |= ((lt >> (8*b)) & 0xffu) << c_lts[b];
        sha_transform(st, w);
#pragma unroll
        for (int i = 0; i < 16; i++) w[i] = c_words[16 + i];
#pragma unroll
        for (int b = 0; b < 4; b++) if (c_ltw[b] >= 16) w[c_ltw[b] - 16] |= ((lt >> (8*b)) & 0xffu) << c_lts[b];
        sha_transform(st, w);
    } else {
#pragma unroll
        for (int i = 0; i < 16; i++) w[i] = c_words[i];
#pragma unroll
        for (int b = 0; b < 4; b++) w[c_ltw[b]] |= ((lt >> (8*b)) & 0xffu) << c_lts[b];
        sha_transform(st, w);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) w[i] = st[i];
    w[8] = 0x80000000u;
#pragma unroll
    for (int i = 9; i < 15; i++) w[i] = 0;
    w[15] = 256;
    st[0]=0x6a09e667; st[1]=0xbb67ae85; st[2]=0x3c6ef372; st[3]=0xa54ff53a;
    st[4]=0x510e527f; st[5]=0x9b05688c; st[6]=0x1f83d9ab; st[7]=0x5be0cd19;
    sha_transform(st, w);

    /* --- u1 = -z r^-1 mod n --- */
    uint64_t z[4], u1[4], nri[4];
    z[3] = ((uint64_t)st[0] << 32) | st[1];
    z[2] = ((uint64_t)st[2] << 32) | st[3];
    z[1] = ((uint64_t)st[4] << 32) | st[5];
    z[0] = ((uint64_t)st[6] << 32) | st[7];
    cpy(nri, c_nri);
    gpu_scalar_mulmod(u1, nri, z);

    /* --- P = u1*G by 16-window comb: affine start, then projective mixed adds --- */
    uint64_t px[4], py[4], pz[5], tx[4], ty[4];
    uint32_t uw[9], dig[NWIN];
#pragma unroll
    for (int i = 0; i < 4; i++) { uw[2*i] = (uint32_t)u1[i]; uw[2*i+1] = (uint32_t)(u1[i] >> 32); }
    uw[8] = 0;
#define DIGIT(c) (__funnelshift_r(uw[((c) * WBITS) >> 5], uw[(((c) * WBITS) >> 5) + 1], ((c) * WBITS) & 31) & (uint32_t)(WSIZE - 1))
#pragma unroll
    for (int c = 0; c < NWIN; c++) dig[c] = DIGIT(c);
    load_entry(tab, 0, dig[0], px, py);
    load_entry(tab, 1, dig[1], tx, ty);
    {   /* affine + affine -> projective (Z1 = 1 specialisation) */
        uint64_t u[4], v[4], us2[4], vs2[4], vs3[4], vs2v2[4], a[4], t[4];
        _ModSub256(u, ty, py);
        _ModSub256(v, tx, px);
        _ModSqr(us2, u);
        _ModSqr(vs2, v);
        _ModMult(vs3, vs2, v);
        _ModMult(vs2v2, vs2, px);
        _ModAdd256(t, vs2v2, vs2v2);
        _ModSub256(a, us2, vs3);
        _ModSub256(a, t);                 /* a = u^2 - v^3 - 2 v^2 x1 */
        _ModMult(px, v, a);               /* X3 = v a */
        _ModMult(t, vs3, py);             /* v^3 y1 */
        _ModSub256(py, vs2v2, a);
        _ModMult(py, py, u);
        _ModSub256(py, t);                /* Y3 = u (v^2 x1 - a) - v^3 y1 */
        cpy(pz, vs3); pz[4] = 0;          /* Z3 = v^3 */
    }
#ifdef QSB_PREFETCH
    load_entry(tab, 2, dig[2], tx, ty);
#pragma unroll 1
    for (int c = 2; c < NWIN - 1; c++) {
        uint64_t nx[4], ny[4];
        load_entry(tab, c + 1, dig[c + 1], nx, ny);
        _PointAddSecp256k1(px, py, pz, tx, ty);
        cpy(tx, nx); cpy(ty, ny);
    }
    _PointAddSecp256k1(px, py, pz, tx, ty);
#else
#pragma unroll 1
    for (int c = 2; c < NWIN; c++) {
        load_entry(tab, c, dig[c], tx, ty);
        _PointAddSecp256k1(px, py, pz, tx, ty);
    }
#endif

    /* --- Q1 = P + A, Q2 = P - A sharing v = xA Z - X and Z3 --- */
    uint64_t x1[4], y1[4], x2[4], y2[4], zz[4];
    {
        uint64_t ax[4], ay[4], t[4], u[4], un[4], v[4], us2[4], vs2[4], vs3[4], us2w[4], vs2v2[4], v2x2[4], a[4], vs3y[4];
        cpy(ax, c_ax); cpy(ay, c_ay);
        _ModMult(t, ay, pz);              /* yA Z */
        _ModSub256(u, t, py);             /* u  = yA Z - Y */
        _ModAdd256(un, t, py); _ModNeg256(un);   /* u' = -yA Z - Y */
        _ModMult(v, ax, pz);
        _ModSub256(v, px);                /* v = xA Z - X */
        _ModSqr(vs2, v);
        _ModMult(vs3, vs2, v);
        _ModMult(vs2v2, vs2, px);
        _ModAdd256(v2x2, vs2v2, vs2v2);
        _ModMult(vs3y, vs3, py);          /* v^3 Y */
        _ModMult(zz, vs3, pz);            /* Z3 (shared) */
        /* recid 0 */
        _ModSqr(us2, u);
        _ModMult(us2w, us2, pz);
        _ModSub256(a, us2w, vs3); _ModSub256(a, v2x2);
        _ModMult(x1, v, a);
        _ModSub256(y1, vs2v2, a); _ModMult(y1, y1, u); _ModSub256(y1, vs3y);
        /* recid 1 */
        _ModSqr(us2, un);
        _ModMult(us2w, us2, pz);
        _ModSub256(a, us2w, vs3); _ModSub256(a, v2x2);
        _ModMult(x2, v, a);
        _ModSub256(y2, vs2v2, a); _ModMult(y2, y2, un); _ModSub256(y2, vs3y);
    }
    uint64_t zinv[4];
    warp_batch_inv(zz, zinv);
    _ModMult(x1, x1, zinv); _ModMult(y1, y1, zinv);
    _ModMult(x2, x2, zinv); _ModMult(y2, y2, zinv);

    uint32_t h[8];
    int found = 0, recid = 0;
    hash_pubkey(x1, y1, h);
    if (gate(h)) { found = 1; recid = 0; }
    hash_pubkey(x2, y2, h);
    if (gate(h)) { if (!found) recid = 1; found += 1; }
    if (found) {
        /* report every hit; both recids hitting at once is astronomically rare
         * but recorded correctly (found == 2 -> recid 0 first, then recid 1). */
        uint32_t pos = atomicAdd(d_hit_cnt, 1u);
        if (pos < (uint32_t)hit_cap) d_hits[pos] = make_uint2(idx, (uint32_t)recid);
        if (found == 2) {
            pos = atomicAdd(d_hit_cnt, 1u);
            if (pos < (uint32_t)hit_cap) d_hits[pos] = make_uint2(idx, 1u);
        }
    }
}


/* ---- debug: one candidate, all intermediates (launch <<<1,32>>>) ---- */
__device__ void dump4(const char *n, const uint64_t *v) {
    printf("DBG %s %016llx%016llx%016llx%016llx\n", n, (unsigned long long)v[3], (unsigned long long)v[2], (unsigned long long)v[1], (unsigned long long)v[0]);
}
__global__ void k_debug(const uint4 *__restrict__ tab, uint32_t lt) {
    uint32_t st[8];
    for (int i = 0; i < 8; i++) st[i] = c_state[i];
    uint32_t w[16];
    if (c_nblk == 2) {
        for (int i = 0; i < 16; i++) w[i] = c_words[i];
        for (int b = 0; b < 4; b++) if (c_ltw[b] < 16) w[c_ltw[b]] |= ((lt >> (8*b)) & 0xffu) << c_lts[b];
        sha_transform(st, w);
        for (int i = 0; i < 16; i++) w[i] = c_words[16 + i];
        for (int b = 0; b < 4; b++) if (c_ltw[b] >= 16) w[c_ltw[b] - 16] |= ((lt >> (8*b)) & 0xffu) << c_lts[b];
        sha_transform(st, w);
    } else {
        for (int i = 0; i < 16; i++) w[i] = c_words[i];
        for (int b = 0; b < 4; b++) w[c_ltw[b]] |= ((lt >> (8*b)) & 0xffu) << c_lts[b];
        sha_transform(st, w);
    }
    if (threadIdx.x == 0) { printf("DBG sha1 "); for (int i = 0; i < 8; i++) printf("%08x", st[i]); printf("\n"); }
    for (int i = 0; i < 8; i++) w[i] = st[i];
    w[8] = 0x80000000u; for (int i = 9; i < 15; i++) w[i] = 0; w[15] = 256;
    st[0]=0x6a09e667; st[1]=0xbb67ae85; st[2]=0x3c6ef372; st[3]=0xa54ff53a; st[4]=0x510e527f; st[5]=0x9b05688c; st[6]=0x1f83d9ab; st[7]=0x5be0cd19;
    sha_transform(st, w);
    uint64_t z[4], u1[4], nri[4];
    z[3] = ((uint64_t)st[0] << 32) | st[1]; z[2] = ((uint64_t)st[2] << 32) | st[3];
    z[1] = ((uint64_t)st[4] << 32) | st[5]; z[0] = ((uint64_t)st[6] << 32) | st[7];
    cpy(nri, c_nri);
    gpu_scalar_mulmod(u1, nri, z);
    if (threadIdx.x == 0) { dump4("z", z); dump4("nri", nri); dump4("u1", u1); }
    uint64_t px[4], py[4], pz[5], tx[4], ty[4];
    uint32_t uw[9];
    for (int i = 0; i < 4; i++) { uw[2*i] = (uint32_t)u1[i]; uw[2*i+1] = (uint32_t)(u1[i] >> 32); }
    uw[8] = 0;
    if (threadIdx.x == 0) { printf("DBG digits"); for (int c = 0; c < NWIN; c++) printf(" %u", DIGIT(c)); printf("\n"); }
    load_entry(tab, 0, DIGIT(0), px, py);
    load_entry(tab, 1, DIGIT(1), tx, ty);
    if (threadIdx.x == 0) { dump4("e0x", px); dump4("e0y", py); dump4("e1x", tx); dump4("e1y", ty); }
    {
        uint64_t u[4], v[4], us2[4], vs2[4], vs3[4], vs2v2[4], a[4], t[4];
        _ModSub256(u, ty, py); _ModSub256(v, tx, px);
        _ModSqr(us2, u); _ModSqr(vs2, v); _ModMult(vs3, vs2, v); _ModMult(vs2v2, vs2, px);
        _ModAdd256(t, vs2v2, vs2v2); _ModSub256(a, us2, vs3); _ModSub256(a, t);
        _ModMult(px, v, a); _ModMult(t, vs3, py); _ModSub256(py, vs2v2, a); _ModMult(py, py, u); _ModSub256(py, t);
        cpy(pz, vs3); pz[4] = 0;
    }
    for (int c = 2; c < NWIN; c++) { load_entry(tab, c, DIGIT(c), tx, ty); _PointAddSecp256k1(px, py, pz, tx, ty); }
    {   /* normalize P for printing */
        uint64_t zi[5]; cpy(zi, pz); zi[4] = 0; _ModInv(zi);
        uint64_t ax_[4], ay_[4]; _ModMult(ax_, px, zi); _ModMult(ay_, py, zi);
        if (threadIdx.x == 0) { dump4("Px", ax_); dump4("Py", ay_); }
    }
    uint64_t x1[4], y1[4], x2[4], y2[4], zz[4];
    {
        uint64_t ax[4], ay[4], t[4], u[4], un[4], v[4], us2[4], vs2[4], vs3[4], us2w[4], vs2v2[4], v2x2[4], a[4], vs3y[4];
        cpy(ax, c_ax); cpy(ay, c_ay);
        _ModMult(t, ay, pz); _ModSub256(u, t, py); _ModAdd256(un, t, py); _ModNeg256(un);
        _ModMult(v, ax, pz); _ModSub256(v, px);
        _ModSqr(vs2, v); _ModMult(vs3, vs2, v); _ModMult(vs2v2, vs2, px); _ModAdd256(v2x2, vs2v2, vs2v2);
        _ModMult(vs3y, vs3, py); _ModMult(zz, vs3, pz);
        _ModSqr(us2, u); _ModMult(us2w, us2, pz); _ModSub256(a, us2w, vs3); _ModSub256(a, v2x2);
        _ModMult(x1, v, a); _ModSub256(y1, vs2v2, a); _ModMult(y1, y1, u); _ModSub256(y1, vs3y);
        _ModSqr(us2, un); _ModMult(us2w, us2, pz); _ModSub256(a, us2w, vs3); _ModSub256(a, v2x2);
        _ModMult(x2, v, a); _ModSub256(y2, vs2v2, a); _ModMult(y2, y2, un); _ModSub256(y2, vs3y);
    }
    uint64_t zinv[4];
    warp_batch_inv(zz, zinv);
    _ModMult(x1, x1, zinv); _ModMult(y1, y1, zinv); _ModMult(x2, x2, zinv); _ModMult(y2, y2, zinv);
    uint32_t h[8];
    hash_pubkey(x1, y1, h);
    if (threadIdx.x == 0) { dump4("Q1x", x1); dump4("Q1y", y1); printf("DBG h1 "); for (int i = 0; i < 8; i++) printf("%08x", h[i]); printf("\n"); }
    hash_pubkey(x2, y2, h);
    if (threadIdx.x == 0) { dump4("Q2x", x2); dump4("Q2y", y2); printf("DBG h2 "); for (int i = 0; i < 8; i++) printf("%08x", h[i]); printf("\n"); }
}

/* ------------------------------------------------------------------ */
/* host                                                                */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t midstate[8];
    uint32_t suffix_len;
    uint8_t *suffix;
    uint32_t total_preimage_len;
    uint32_t seq_offset;
    uint32_t lt_offset;
    uint8_t neg_r_inv[32];
    uint8_t u2r_x[32];
    uint8_t u2r_y[32];
} pinning_params_t;

static int load_params(const char *fn, pinning_params_t *p) {
    FILE *f = fopen(fn, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fn); return -1; }
    if (fread(p->midstate, 4, 8, f) != 8) goto err;
    for (int i = 0; i < 8; i++) {
        uint8_t *b = (uint8_t*)&p->midstate[i];
        p->midstate[i] = ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|b[3];
    }
    if (fread(&p->suffix_len, 4, 1, f) != 1) goto err;
    if (p->suffix_len > 120) { fprintf(stderr, "suffix too long\n"); goto err; }
    p->suffix = (uint8_t*)calloc(256, 1);
    if (fread(p->suffix, 1, p->suffix_len, f) != p->suffix_len) goto err;
    if (fread(&p->total_preimage_len, 4, 1, f) != 1) goto err;
    if (fread(&p->seq_offset, 4, 1, f) != 1) goto err;
    if (fread(&p->lt_offset, 4, 1, f) != 1) goto err;
    if (fread(p->neg_r_inv, 1, 32, f) != 32) goto err;
    if (fread(p->u2r_x, 1, 32, f) != 32) goto err;
    if (fread(p->u2r_y, 1, 32, f) != 32) goto err;
    fclose(f);
    printf("  Loaded: preimage=%u, suffix=%u, seq@%u, lt@%u\n",
           p->total_preimage_len, p->suffix_len, p->seq_offset, p->lt_offset);
    return 0;
err:
    fprintf(stderr, "Error reading %s\n", fn); fclose(f); return -1;
}

static void le_bytes_to_limbs(const uint8_t b[32], uint64_t r[4]) {
    for (int i = 0; i < 4; i++) { r[i] = 0; for (int k = 0; k < 8; k++) r[i] |= (uint64_t)b[i*8+k] << (8*k); }
}

/* Prepare the per-sequence SHA template: padded suffix with seq patched,
 * lt bytes zero; hash the block(s) that do not contain lt on the host. */
static void upload_seq_template(const pinning_params_t *pp, uint32_t seq) {
    uint8_t buf[192]; memset(buf, 0, sizeof buf);
    memcpy(buf, pp->suffix, pp->suffix_len);
    for (int b = 0; b < 4; b++) buf[pp->seq_offset + b] = (seq >> (8*b)) & 0xff;
    for (int b = 0; b < 4; b++) buf[pp->lt_offset + b] = 0;
    int nblk = (pp->suffix_len < 56) ? 1 : 2;
    buf[pp->suffix_len] = 0x80;
    uint64_t bit_len = (uint64_t)pp->total_preimage_len * 8;
    int last = nblk*64 - 8;
    for (int b = 0; b < 8; b++) buf[last + b] = (bit_len >> (56 - 8*b)) & 0xff;
    uint32_t words[32];
    for (int i = 0; i < nblk*16; i++)
        words[i] = ((uint32_t)buf[i*4]<<24)|((uint32_t)buf[i*4+1]<<16)|((uint32_t)buf[i*4+2]<<8)|buf[i*4+3];
    uint32_t state[8]; memcpy(state, pp->midstate, 32);
    int skip = (nblk == 2 && pp->lt_offset >= 64) ? 1 : 0;
    if (skip) host_sha_transform(state, words);
    int kn = nblk - skip;
    uint32_t kw[32]; memset(kw, 0, sizeof kw);
    memcpy(kw, words + 16*skip, 64*kn);
    int ltw[4], lts[4];
    for (int b = 0; b < 4; b++) {
        int off = (int)pp->lt_offset + b - 64*skip;
        ltw[b] = off >> 2; lts[b] = 24 - 8*(off & 3);
    }
    CHK(cudaMemcpyToSymbol(c_state, state, 32));
    CHK(cudaMemcpyToSymbol(c_words, kw, sizeof kw));
    CHK(cudaMemcpyToSymbol(c_nblk, &kn, sizeof kn));
    CHK(cudaMemcpyToSymbol(c_ltw, ltw, sizeof ltw));
    CHK(cudaMemcpyToSymbol(c_lts, lts, sizeof lts));
}

/* OpenSSL helpers for the (tiny) host-side EC work */
static void pt_to_aff(EC_GROUP *grp, const EC_POINT *pt, AffPt *out, BN_CTX *ctx) {
    BIGNUM *x = BN_new(), *y = BN_new();
    EC_POINT_get_affine_coordinates(grp, pt, x, y, ctx);
    uint8_t xb[32], yb[32]; memset(xb, 0, 32); memset(yb, 0, 32);
    BN_bn2bin(x, xb + (32 - BN_num_bytes(x)));
    BN_bn2bin(y, yb + (32 - BN_num_bytes(y)));
    for (int i = 0; i < 4; i++) { out->x[i] = 0; out->y[i] = 0;
        for (int k = 0; k < 8; k++) { out->x[i] |= (uint64_t)xb[31 - i*8 - k] << (8*k); out->y[i] |= (uint64_t)yb[31 - i*8 - k] << (8*k); } }
    BN_free(x); BN_free(y);
}

static void limbs_to_bn(const uint64_t a[4], BIGNUM *bn) {
    uint8_t be[32];
    for (int i = 0; i < 4; i++) for (int k = 0; k < 8; k++) be[31 - i*8 - k] = (a[i] >> (8*k)) & 0xff;
    BN_bin2bn(be, 32, bn);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <pinning.bin> [gpu_index] [total_gpus] [global_offset] [single_hash] [selftest]\n", argv[0]);
        return 1;
    }
    int gpu_index = (argc >= 3) ? atoi(argv[2]) : 0;
    int total_gpus_override = (argc >= 4) ? atoi(argv[3]) : 0;
    int global_offset = (argc >= 5) ? atoi(argv[4]) : 0;
    int selftest = 0;
    double max_seconds = 0;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "selftest") == 0) selftest = 1;
        if (strncmp(argv[i], "seconds=", 8) == 0) max_seconds = atof(argv[i] + 8);
    }
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    CHK(cudaSetDevice(gpu_index));
    cudaDeviceProp prop; CHK(cudaGetDeviceProperties(&prop, gpu_index));
    printf("QSB Pinning Grinder (affine comb, batched inversion) [GPU %d]\n", gpu_index);
    printf("  GPU: %s (%d SMs)  N=%d  block=%d batch=%u\n", prop.name, prop.multiProcessorCount, QSB_ZEROS_N, QSB_BLK, QSB_BATCH);

    pinning_params_t pp;
    if (load_params(argv[1], &pp) < 0) return 1;

    /* ---- host EC constants (OpenSSL) ---- */
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx = BN_CTX_new();
    AffPt h_base[NWIN], h_base2[NWIN], h_s0[NWIN];
    {
        /* B_c = 2^(16c) G ; D = sum_c 2^(16c) ; C = -D G ; S0_0 = G + C, S0_c = B_c */
        BIGNUM *k = BN_new(), *D = BN_new(), *one = BN_new();
        BN_one(one); BN_zero(D);
        EC_POINT *pt = EC_POINT_new(grp);
        for (int c = 0; c < NWIN; c++) {
            BN_lshift(k, one, WBITS*c);
            BN_add(D, D, k);
            EC_POINT_mul(grp, pt, k, NULL, NULL, ctx);
            pt_to_aff(grp, pt, &h_base[c], ctx);
            h_s0[c] = h_base[c];
            BN_lshift(k, one, WBITS*c + 1);
            EC_POINT_mul(grp, pt, k, NULL, NULL, ctx);
            pt_to_aff(grp, pt, &h_base2[c], ctx);
        }
        EC_POINT *C = EC_POINT_new(grp);
        EC_POINT_mul(grp, C, D, NULL, NULL, ctx);
        EC_POINT_invert(grp, C, ctx);
        EC_POINT_add(grp, pt, EC_GROUP_get0_generator(grp), C, ctx);
        pt_to_aff(grp, pt, &h_s0[0], ctx);
        EC_POINT_free(pt); EC_POINT_free(C); BN_free(k); BN_free(D); BN_free(one);
    }
    /* u2R affine and -r^-1 to constant memory */
    {
        uint64_t nri[4], ax[4], ay[4];
        le_bytes_to_limbs(pp.neg_r_inv, nri);
        le_bytes_to_limbs(pp.u2r_x, ax);
        le_bytes_to_limbs(pp.u2r_y, ay);
        CHK(cudaMemcpyToSymbol(c_nri, nri, 32));
        CHK(cudaMemcpyToSymbol(c_ax, ax, 32));
        CHK(cudaMemcpyToSymbol(c_ay, ay, 32));
    }

    /* ---- G-table on the GPU ---- */
    size_t tab_bytes = NENTRIES * 64;
    uint4 *d_tab; CHK(cudaMalloc(&d_tab, tab_bytes));
    {
        AffPt *d_base, *d_base2, *d_s0, *d_T, *d_S;
        CHK(cudaMalloc(&d_base, sizeof h_base)); CHK(cudaMalloc(&d_base2, sizeof h_base2)); CHK(cudaMalloc(&d_s0, sizeof h_s0));
        CHK(cudaMemcpy(d_base2, h_base2, sizeof h_base2, cudaMemcpyHostToDevice));
        CHK(cudaMalloc(&d_T, sizeof(AffPt) * NWIN * HSIZE)); CHK(cudaMalloc(&d_S, sizeof(AffPt) * NWIN * HSIZE));
        CHK(cudaMemcpy(d_base, h_base, sizeof h_base, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(d_s0, h_s0, sizeof h_s0, cudaMemcpyHostToDevice));
        k_gen_serial<<<NWIN, 1>>>(d_base, d_base2, d_s0, d_T, d_S);
        CHK(cudaGetLastError());
        for (int c = 0; c < NWIN; c++) {
            uint32_t ne = (c == NWIN - 1) ? (1u << WLAST) : (uint32_t)WSIZE;
            dim3 g((ne / QSB_GENK + 127) / 128, 1);
            k_gen_table<<<g, 128>>>(d_T, d_S, d_tab, ne, c);
        }
        CHK(cudaGetLastError());
        CHK(cudaDeviceSynchronize());
        cudaFree(d_base); cudaFree(d_base2); cudaFree(d_s0); cudaFree(d_T); cudaFree(d_S);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        printf("  GTable built on GPU in %.2fs\n", (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9);
    }
    if (selftest) {
        /* compare random table entries against OpenSSL */
        uint4 *h_tab = (uint4*)malloc(tab_bytes);
        CHK(cudaMemcpy(h_tab, d_tab, tab_bytes, cudaMemcpyDeviceToHost));
        BIGNUM *k = BN_new(), *D = BN_new(), *one = BN_new(), *m = BN_new(), *ordr = BN_new();
        BN_one(one); BN_zero(D);
        EC_GROUP_get_order(grp, ordr, ctx);
        for (int c = 0; c < NWIN; c++) { BN_lshift(k, one, WBITS*c); BN_add(D, D, k); }
        EC_POINT *pt = EC_POINT_new(grp);
        int bad = 0;
        srand(12345);
        for (int c = 0; c < NWIN; c++) for (int n = 0; n < 24; n++) {
            uint32_t ne = (c == NWIN - 1) ? (1u << WLAST) : (uint32_t)WSIZE;
            const uint32_t fixed_k[3] = {0u, 1u, ne - 1};
            uint32_t kk = (n < 3) ? fixed_k[n] : (uint32_t)(((uint64_t)rand() * 65536ull + rand()) % ne);
            /* multiple m = (kk+1) * 2^(W c) - (c==0 ? D : 0) mod order */
            BN_set_word(m, kk + 1); BN_lshift(m, m, WBITS*c);
            if (c == 0) BN_mod_sub(m, m, D, ordr, ctx);
            EC_POINT_mul(grp, pt, m, NULL, NULL, ctx);
            AffPt ref; pt_to_aff(grp, pt, &ref, ctx);
            uint4 *e = h_tab + ((size_t)c * WSIZE + kk) * 4;
            uint64_t ex[4] = { ((uint64_t)e[0].y<<32)|e[0].x, ((uint64_t)e[0].w<<32)|e[0].z, ((uint64_t)e[1].y<<32)|e[1].x, ((uint64_t)e[1].w<<32)|e[1].z };
            uint64_t ey[4] = { ((uint64_t)e[2].y<<32)|e[2].x, ((uint64_t)e[2].w<<32)|e[2].z, ((uint64_t)e[3].y<<32)|e[3].x, ((uint64_t)e[3].w<<32)|e[3].z };
            if (memcmp(ex, ref.x, 32) || memcmp(ey, ref.y, 32)) { bad++; printf("  selftest MISMATCH c=%d k=%u\n", c, kk); }
        }
        printf("  selftest: table %s (%d mismatches)\n", bad ? "FAILED" : "OK", bad);
        free(h_tab); EC_POINT_free(pt); BN_free(k); BN_free(D); BN_free(one); BN_free(m); BN_free(ordr);
        if (bad) return 2;
    }

    /* ---- hit buffers ---- */
    const int HIT_CAP = 4096;
    uint32_t *d_hit_cnt; uint2 *d_hits;
    CHK(cudaMalloc(&d_hit_cnt, 4)); CHK(cudaMalloc(&d_hits, HIT_CAP * sizeof(uint2)));
    CHK(cudaMemset(d_hit_cnt, 0, 4));
    uint2 *h_hits = (uint2*)malloc(HIT_CAP * sizeof(uint2));

    int num_gpus = 0; cudaGetDeviceCount(&num_gpus); if (num_gpus < 1) num_gpus = 1;
    int effective_total = (total_gpus_override > 0) ? total_gpus_override : num_gpus;
    int effective_id = global_offset + gpu_index;
    uint32_t SEQ_MIN = 0x80000000u;

    for (int i = 3; i + 2 < argc; i++) if (strcmp(argv[i], "debug") == 0) {
        uint32_t dseq = (uint32_t)strtoul(argv[i+1], NULL, 0), dlt = (uint32_t)strtoul(argv[i+2], NULL, 0);
        upload_seq_template(&pp, dseq);
        printf("DBG seq %u lt %u\n", dseq, dlt);
        k_debug<<<1, 32>>>(d_tab, dlt);
        CHK(cudaDeviceSynchronize());
        return 0;
    }
    mkdir("results", 0755);
    char fname[256];
    snprintf(fname, sizeof(fname), "results/pinning_hit_%d.txt", gpu_index);

    printf("\n  === Search: full 32-bit locktime per sequence, seq=[0x%08X+], GPU %d (global %d of %d) ===\n",
           SEQ_MIN, gpu_index, effective_id, effective_total);
    fflush(stdout);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    double last_print = 0;
    uint64_t total_searched = 0, total_hits = 0;
    const uint32_t GRID = QSB_BATCH / QSB_BLK;

    for (uint32_t seq = SEQ_MIN + effective_id; ; seq += effective_total) {
        upload_seq_template(&pp, seq);
        for (uint64_t lt0 = 0; lt0 < (1ull << 32); lt0 += QSB_BATCH) {
            k_grind<<<GRID, QSB_BLK>>>(d_tab, (uint32_t)lt0, d_hit_cnt, d_hits, HIT_CAP);
            uint32_t h_cnt = 0;
            CHK(cudaMemcpy(&h_cnt, d_hit_cnt, 4, cudaMemcpyDeviceToHost)); /* syncs */
            total_searched += QSB_BATCH;
            if (h_cnt) {
                int nh = (h_cnt > (uint32_t)HIT_CAP) ? HIT_CAP : (int)h_cnt;
                CHK(cudaMemcpy(h_hits, d_hits, nh * sizeof(uint2), cudaMemcpyDeviceToHost));
                CHK(cudaMemset(d_hit_cnt, 0, 4));
                FILE *f = fopen(fname, "a");
                if (f) {
                    for (int i = 0; i < nh; i++)
                        fprintf(f, "sequence=%u\nlocktime=%u\nrecid=%u\n", seq, (uint32_t)lt0 + h_hits[i].x, h_hits[i].y);
                    fclose(f);
                }
                total_hits += nh;
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double el = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
            if (el - last_print >= 2.0) {
                last_print = el;
                printf("  [GPU %d] seq 0x%08X, %lluM total, %.1fM/s, %llu hits, %.0fs\n",
                       gpu_index, seq, (unsigned long long)(total_searched/1000000), total_searched/el/1e6,
                       (unsigned long long)total_hits, el);
                fflush(stdout);
            }
            if (max_seconds > 0 && el >= max_seconds) {
                printf("\n  Done: %lluM in %.0fs (%.1fM/s), hits=%llu\n",
                       (unsigned long long)(total_searched/1000000), el, total_searched/el/1e6, (unsigned long long)total_hits);
                return 0;
            }
        }
    }
    return 0;
}
