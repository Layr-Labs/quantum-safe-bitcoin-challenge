#pragma once
/* pair_sha.cuh: the interleaved two-state SHA-256 compression, lifted verbatim
 * out of tree.cu so that exactly the same text can also be compiled by a host
 * C++ compiler and differentially tested against OpenSSL. The device build is
 * byte-identical to the in-tree version: the portability shim is macro-only
 * (QSB_PS_K/QSB_PS_I alias the __constant__ tables, qsb_ps_byte_perm aliases
 * __byte_perm), so with the feature macro off the emitted PTX does not move. */
#ifndef ZLAB_PAIRSHA
#define ZLAB_PAIRSHA 0
#endif
#ifndef QSB_PAIR_SHA
#define QSB_PAIR_SHA 0
#endif

#ifdef __CUDACC__
/* Device side: K, I, __byte_perm and the S0/S1/s0/s1/Ch/Maj macros all come
 * from GPUHash.h, already included by the translation unit. */
#define QSB_PS_DEV __device__
#define QSB_PS_INL __forceinline__
#define QSB_PS_K K
#define QSB_PS_I I
#define qsb_ps_byte_perm(a, b, sel) __byte_perm((a), (b), (sel))
#else
/* Host shim: only what the compression body names, nothing else. */
#include <stdint.h>
#define QSB_PS_DEV
#define QSB_PS_INL inline
static const uint32_t QSB_PS_K[64] = {
	0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
	0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
	0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
	0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
	0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
	0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
	0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
	0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
	0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
	0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
	0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
	0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
	0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
	0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
	0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
	0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
};
static const uint32_t QSB_PS_I[8] = {
	0x6a09e667ul, 0xbb67ae85ul, 0x3c6ef372ul, 0xa54ff53aul,
	0x510e527ful, 0x9b05688cul, 0x1f83d9abul, 0x5be0cd19ul,
};
#ifndef ROR
#define ROR(x,n) ((x>>n)|(x<<(32-n)))
#endif
#ifndef S0
#define S0(x) (ROR(x,2) ^ ROR(x,13) ^ ROR(x,22))
#define S1(x) (ROR(x,6) ^ ROR(x,11) ^ ROR(x,25))
#define s0(x) (ROR(x,7) ^ ROR(x,18) ^ (x >> 3))
#define s1(x) (ROR(x,17) ^ ROR(x,19) ^ (x >> 10))
#endif
#ifndef Maj
#define Maj(x,y,z) ((x & y) | (z & (x | y)))
#define Ch(x,y,z) (z ^ (x & (y ^ z)))
#endif
/* __byte_perm: {b,a} viewed as bytes 7..0, one selector nibble per output byte
 * (nibble bit 3 = replicate sign of the picked byte). */
static inline uint32_t qsb_ps_byte_perm(uint32_t a, uint32_t b, uint32_t sel) {
    uint8_t src[8];
    for (int i = 0; i < 4; i++) src[i]     = (uint8_t)(a >> (8 * i));
    for (int i = 0; i < 4; i++) src[4 + i] = (uint8_t)(b >> (8 * i));
    uint32_t r = 0;
    for (int i = 0; i < 4; i++) {
        unsigned n = (sel >> (4 * i)) & 0xFu;
        uint8_t v = src[n & 7u];
        if (n & 8u) v = (v & 0x80u) ? 0xFFu : 0x00u;
        r |= (uint32_t)v << (8 * i);
    }
    return r;
}
#endif

#if ZLAB_PAIRSHA || QSB_PAIR_SHA
/* Two independent SHA-256 compressions from the IV, rounds interleaved. */
#define ZP_RND2(k) { \
  for (int zr = 0; zr < 16; zr++) { \
    S2RoundZ(a0,b0,c0,d0,e0,f0,g0,h0,x0,QSB_PS_K[k+zr],w0[zr]); \
    S2RoundZ(a1,b1,c1,d1,e1,f1,g1,h1,x1,QSB_PS_K[k+zr],w1[zr]); \
  } }
#define S2RoundZ(a,b,c,d,e,f,g,h,x,k,w) { \
    uint32_t zt1 = h + S1(e) + Ch(e,f,g) + (k) + (w); \
    uint32_t zt2 = S0(a) + Maj(a,b,c); \
    d += zt1; x = zt1 + zt2; \
    h=g; g=f; f=e; e=d; d=c; c=b; b=a; a=x; }
#define ZP_WMIX(w) { \
    for (int zi = 0; zi < 16; zi++) w[zi] += s1(w[(zi+14)&15]) + w[(zi+9)&15] + s0(w[(zi+1)&15]); }
QSB_PS_DEV QSB_PS_INL void zlab_sha256_pair_h0(uint32_t *w0, uint32_t *w1, uint32_t *out0, uint32_t *out1) {
    uint32_t a0=QSB_PS_I[0],b0=QSB_PS_I[1],c0=QSB_PS_I[2],d0=QSB_PS_I[3],e0=QSB_PS_I[4],f0=QSB_PS_I[5],g0=QSB_PS_I[6],h0=QSB_PS_I[7],x0;
    uint32_t a1=QSB_PS_I[0],b1=QSB_PS_I[1],c1=QSB_PS_I[2],d1=QSB_PS_I[3],e1=QSB_PS_I[4],f1=QSB_PS_I[5],g1=QSB_PS_I[6],h1=QSB_PS_I[7],x1;
    #pragma unroll 1
    for (int blk = 0; blk < 64; blk += 16) {
        if (blk) { ZP_WMIX(w0); ZP_WMIX(w1); }
        ZP_RND2(blk);
    }
    out0[0]=QSB_PS_I[0]+a0;out0[1]=QSB_PS_I[1]+b0;out0[2]=QSB_PS_I[2]+c0;out0[3]=QSB_PS_I[3]+d0;
    out0[4]=QSB_PS_I[4]+e0;out0[5]=QSB_PS_I[5]+f0;out0[6]=QSB_PS_I[6]+g0;out0[7]=QSB_PS_I[7]+h0;
    out1[0]=QSB_PS_I[0]+a1;out1[1]=QSB_PS_I[1]+b1;out1[2]=QSB_PS_I[2]+c1;out1[3]=QSB_PS_I[3]+d1;
    out1[4]=QSB_PS_I[4]+e1;out1[5]=QSB_PS_I[5]+f1;out1[6]=QSB_PS_I[6]+g1;out1[7]=QSB_PS_I[7]+h1;
}

/* One 33-byte compressed pubkey (0x02|0x03 prefix + big-endian x) padded into
 * a single 16-word SHA-256 block. Extracted verbatim from the per-recid body
 * of qsb_k2s_gate so the gate and the host mirror test share one packing. */
QSB_PS_DEV QSB_PS_INL void qsb_pair_sha_preimage(
    uint64_t sx0, uint64_t sx1, uint64_t sx2, uint64_t sx3, uint32_t y_odd, uint32_t *pb) {
    uint32_t x32[8]={(uint32_t)sx0,(uint32_t)(sx0>>32),(uint32_t)sx1,(uint32_t)(sx1>>32),
                     (uint32_t)sx2,(uint32_t)(sx2>>32),(uint32_t)sx3,(uint32_t)(sx3>>32)};
    uint8_t prefix_byte = 0x2+(uint8_t)(y_odd&1u);
    pb[0]=qsb_ps_byte_perm(x32[7],prefix_byte,0x4321);
    pb[1]=qsb_ps_byte_perm(x32[7],x32[6],0x0765);pb[2]=qsb_ps_byte_perm(x32[6],x32[5],0x0765);
    pb[3]=qsb_ps_byte_perm(x32[5],x32[4],0x0765);pb[4]=qsb_ps_byte_perm(x32[4],x32[3],0x0765);
    pb[5]=qsb_ps_byte_perm(x32[3],x32[2],0x0765);pb[6]=qsb_ps_byte_perm(x32[2],x32[1],0x0765);
    pb[7]=qsb_ps_byte_perm(x32[1],x32[0],0x0765);pb[8]=qsb_ps_byte_perm(x32[0],0x80,0x0456);
    pb[9]=0;pb[10]=0;pb[11]=0;pb[12]=0;pb[13]=0;pb[14]=0;pb[15]=0x108;
}
#endif
