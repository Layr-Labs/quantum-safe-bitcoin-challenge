/* SPDX-License-Identifier: GPL-3.0-only
 * Rectangle probe: derived from the promoted pinning SHA implementation.
 * Included after qsb_tail_pre/qsb_make_tail_pre and SHA round macros.
 */
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#ifndef QSB_RECT
#define QSB_RECT 0
#endif
#if QSB_RECT != 0 && QSB_RECT != 1024 && QSB_RECT != 2048
#error "QSB_RECT must be 0, 1024 or 2048"
#endif
static_assert(sizeof(qsb_tail_pre)==76, "rectangle descriptor ABI");
static_assert(offsetof(qsb_tail_pre,mid)==0 && offsetof(qsb_tail_pre,v)==32 &&
              offsetof(qsb_tail_pre,v2y)==56 && offsetof(qsb_tail_pre,d4)==72,
              "rectangle descriptor offsets");
#ifdef __CUDACC__
#define QSB_RECT_HD __host__ __device__
#define QSB_RECT_LOAD(p) __ldg(p)
#else
#define QSB_RECT_HD
#define QSB_RECT_LOAD(p) (*(p))
#endif
/* Slot base is the first sequence in the batch; batches start at column zero.
 * Subbatches advance the descriptor pointer by off/width, never the slot base. */
QSB_RECT_HD static inline uint32_t qsb_rect_col(uint32_t idx,unsigned width) { return idx&(width-1u); }
QSB_RECT_HD static inline uint32_t qsb_rect_row(uint32_t idx,unsigned width) { return idx/width; }
static inline void qsb_rect_decode(uint32_t seq0,uint32_t lt0,unsigned width,uint32_t idx,
                                   uint32_t *seq,uint32_t *lt) {
    *seq=seq0+(width ? qsb_rect_row(idx,width) : 0u);
    *lt=lt0+(width ? qsb_rect_col(idx,width) : idx);
}
static inline unsigned qsb_rect_arm_width(int arm) { return arm==1||arm==4 ? 2048u : arm==3 ? 1024u : 0u; }
static inline uint32_t qsb_rect_arm_begin(int arm) {
    const uint32_t b[5]={0x80000000u,0x88000000u,0x81000000u,0xa8000000u,0x98000000u};
    return b[arm];
}
static inline uint32_t qsb_rect_arm_end(int arm) {
    const uint32_t e[5]={0x81000000u,0x98000000u,0x82000000u,0xc0000000u,0xa8000000u};
    return e[arm];
}
static inline bool qsb_rect_slice(int arm,uint64_t j,uint32_t *seq) {
    if(arm<0||arm>=5) return false;
    const unsigned w=qsb_rect_arm_width(arm);
    const uint64_t step=w ? (1ull<<30)/w : 1ull;
    if(j>=384 || j>=((uint64_t)qsb_rect_arm_end(arm)-qsb_rect_arm_begin(arm))/step) return false;
    *seq=qsb_rect_arm_begin(arm)+(uint32_t)(j*step);return true;
}
/* Host descriptor construction is shared verbatim with portable tests. */
template<class Compress>
static bool qsb_rect_descriptors(const uint32_t prefix[8],const uint8_t suffix[75],
        uint32_t seq0,unsigned n,uint32_t stop,uint32_t w2,qsb_tail_pre *out,Compress compress) {
    if(seq0<0x80000000u || (uint64_t)seq0+n>stop || stop>0xc0000000u) return false;
    for(unsigned i=0;i<n;i++) {
        uint8_t b[64];memcpy(b,suffix,64);
        uint32_t seq=seq0+i;for(int k=0;k<4;k++) b[31+k]=(uint8_t)(seq>>(8*k));
        uint32_t mid[8];memcpy(mid,prefix,32);compress(mid,b);
        qsb_make_tail_pre(out+i,mid,w2);
    }
    return true;
}
static bool qsb_rect_schedule(const uint8_t suffix[75],unsigned width,uint32_t *kw) {
    if(width!=1024 && width!=2048) return false;
    static const uint32_t K[64]={
        0x428A2F98u,0x71374491u,0xB5C0FBCFu,0xE9B5DBA5u,0x3956C25Bu,0x59F111F1u,0x923F82A4u,0xAB1C5ED5u,
        0xD807AA98u,0x12835B01u,0x243185BEu,0x550C7DC3u,0x72BE5D74u,0x80DEB1FEu,0x9BDC06A7u,0xC19BF174u,
        0xE49B69C1u,0xEFBE4786u,0x0FC19DC6u,0x240CA1CCu,0x2DE92C6Fu,0x4A7484AAu,0x5CB0A9DCu,0x76F988DAu,
        0x983E5152u,0xA831C66Du,0xB00327C8u,0xBF597FC7u,0xC6E00BF3u,0xD5A79147u,0x06CA6351u,0x14292967u,
        0x27B70A85u,0x2E1B2138u,0x4D2C6DFCu,0x53380D13u,0x650A7354u,0x766A0ABBu,0x81C2C92Eu,0x92722C85u,
        0xA2BFE8A1u,0xA81A664Bu,0xC24B8B70u,0xC76C51A3u,0xD192E819u,0xD6990624u,0xF40E3585u,0x106AA070u,
        0x19A4C116u,0x1E376C08u,0x2748774Cu,0x34B0BCB5u,0x391C0CB3u,0x4ED8AA4Au,0x5B9CCA4Fu,0x682E6FF3u,
        0x748F82EEu,0x78A5636Fu,0x84C87814u,0x8CC70208u,0x90BEFFFAu,0xA4506CEBu,0xBEF9A3F7u,0xC67178F2u};
    for(unsigned col=0;col<width;col++) {
        uint8_t b[64]={};memcpy(b,suffix+64,11);
        uint32_t lt=500000000u+col;for(int k=0;k<4;k++) b[3+k]=(uint8_t)(lt>>(8*k));
        b[11]=0x80;const uint64_t bits=9995ull*8;
        for(int k=0;k<8;k++) b[63-k]=(uint8_t)(bits>>(8*k));
        uint32_t w[64];
        for(int k=0;k<16;k++) w[k]=((uint32_t)b[4*k]<<24)|((uint32_t)b[4*k+1]<<16)|((uint32_t)b[4*k+2]<<8)|b[4*k+3];
        for(int k=16;k<64;k++) {
            uint32_t x=w[k-15],y=w[k-2];
            w[k]=w[k-16]+(qsb_h_ror(x,7)^qsb_h_ror(x,18)^(x>>3))+w[k-7]+(qsb_h_ror(y,17)^qsb_h_ror(y,19)^(y>>10));
        }
        for(int k=16;k<64;k++) kw[(k-16)*width+col]=K[k]+w[k];
    }
    return true;
}
#define QSB_RECT_RL(a, b, c, d, e, f, g, h, kw) \
    t1 = h + S1(e) + Ch(e,f,g) + (kw) + QSB_Z; \
    t2 = S0(a) + Maj(a,b,c); \
    d += t1 + QSB_Z; \
    h = t1 + t2;
template<unsigned WIDTH>
__device__ __forceinline__ void qsb_rect_tail(
    uint32_t state[8], uint32_t lane, uint32_t u0, uint32_t w1, uint32_t w2, const qsb_tail_pre &tp, const uint32_t *kw, uint32_t col)
{
    const uint32_t L = 9995u * 8u; /* 79960 */
    uint32_t t1;
    uint32_t t2;

    uint32_t a = tp.mid[0];
    uint32_t b = tp.mid[1];
    uint32_t c = tp.mid[2];
    uint32_t d = tp.mid[3];
    uint32_t e = tp.mid[4];
    uint32_t f = tp.mid[5];
    uint32_t g = tp.mid[6];
    uint32_t h = tp.mid[7];
    (void)c;

    /* round 0: v0 + W0, v1 + W0 */
    h = lane + QSB_UB(tp.v[0] + u0);
    d = lane + QSB_UB(tp.v[1] + u0);
    /* round 1 */
    t1 = S1(d) + ((d & e) | (~d & f)) + QSB_UB(tp.v2y + w1);   /* (d&e) + (~d&f): disjoint bits */
    t2 = S0(h) + (h & tp.mx);
    c = tp.c2y + t1;
    g = t1 + t2;
    /* round 2 */
    t1 = tp.v[3] + S1(c) + Ch(c,d,e);      t2 = S0(g) + Maj(g,h,a); b += t1; f = t1 + t2;
    /* round 3 */
    t1 = tp.v[4] + S1(b) + Ch(b,c,d);      t2 = S0(f) + Maj(f,g,h); a += t1; e = t1 + t2;
    QSB_RL(e, f, g, h, a, b, c, d, qsb_klit(4));
    QSB_RL(d, e, f, g, h, a, b, c, qsb_klit(5));
    QSB_RL(c, d, e, f, g, h, a, b, qsb_klit(6));
    QSB_RL(b, c, d, e, f, g, h, a, qsb_klit(7));
    QSB_RL(a, b, c, d, e, f, g, h, qsb_klit(8));
    QSB_RL(h, a, b, c, d, e, f, g, qsb_klit(9));
    QSB_RL(g, h, a, b, c, d, e, f, qsb_klit(10));
    QSB_RL(f, g, h, a, b, c, d, e, qsb_klit(11));
    QSB_RL(e, f, g, h, a, b, c, d, qsb_klit(12));
    QSB_RL(d, e, f, g, h, a, b, c, qsb_klit(13));
    QSB_RL(c, d, e, f, g, h, a, b, qsb_klit(14));
    QSB_RL(b, c, d, e, f, g, h, a, qsb_klit(15) + L);

    QSB_RECT_RL(a, b, c, d, e, f, g, h, QSB_RECT_LOAD(kw + 0u*WIDTH + col));
    QSB_RECT_RL(h, a, b, c, d, e, f, g, QSB_RECT_LOAD(kw + 1u*WIDTH + col));
    QSB_RECT_RL(g, h, a, b, c, d, e, f, QSB_RECT_LOAD(kw + 2u*WIDTH + col));
    QSB_RECT_RL(f, g, h, a, b, c, d, e, QSB_RECT_LOAD(kw + 3u*WIDTH + col));
    QSB_RECT_RL(e, f, g, h, a, b, c, d, QSB_RECT_LOAD(kw + 4u*WIDTH + col));
    QSB_RECT_RL(d, e, f, g, h, a, b, c, QSB_RECT_LOAD(kw + 5u*WIDTH + col));
    QSB_RECT_RL(c, d, e, f, g, h, a, b, QSB_RECT_LOAD(kw + 6u*WIDTH + col));
    QSB_RECT_RL(b, c, d, e, f, g, h, a, QSB_RECT_LOAD(kw + 7u*WIDTH + col));
    QSB_RECT_RL(a, b, c, d, e, f, g, h, QSB_RECT_LOAD(kw + 8u*WIDTH + col));
    QSB_RECT_RL(h, a, b, c, d, e, f, g, QSB_RECT_LOAD(kw + 9u*WIDTH + col));
    QSB_RECT_RL(g, h, a, b, c, d, e, f, QSB_RECT_LOAD(kw + 10u*WIDTH + col));
    QSB_RECT_RL(f, g, h, a, b, c, d, e, QSB_RECT_LOAD(kw + 11u*WIDTH + col));
    QSB_RECT_RL(e, f, g, h, a, b, c, d, QSB_RECT_LOAD(kw + 12u*WIDTH + col));
    QSB_RECT_RL(d, e, f, g, h, a, b, c, QSB_RECT_LOAD(kw + 13u*WIDTH + col));
    QSB_RECT_RL(c, d, e, f, g, h, a, b, QSB_RECT_LOAD(kw + 14u*WIDTH + col));
    QSB_RECT_RL(b, c, d, e, f, g, h, a, QSB_RECT_LOAD(kw + 15u*WIDTH + col));
    QSB_RECT_RL(a, b, c, d, e, f, g, h, QSB_RECT_LOAD(kw + 16u*WIDTH + col));
    QSB_RECT_RL(h, a, b, c, d, e, f, g, QSB_RECT_LOAD(kw + 17u*WIDTH + col));
    QSB_RECT_RL(g, h, a, b, c, d, e, f, QSB_RECT_LOAD(kw + 18u*WIDTH + col));
    QSB_RECT_RL(f, g, h, a, b, c, d, e, QSB_RECT_LOAD(kw + 19u*WIDTH + col));
    QSB_RECT_RL(e, f, g, h, a, b, c, d, QSB_RECT_LOAD(kw + 20u*WIDTH + col));
    QSB_RECT_RL(d, e, f, g, h, a, b, c, QSB_RECT_LOAD(kw + 21u*WIDTH + col));
    QSB_RECT_RL(c, d, e, f, g, h, a, b, QSB_RECT_LOAD(kw + 22u*WIDTH + col));
    QSB_RECT_RL(b, c, d, e, f, g, h, a, QSB_RECT_LOAD(kw + 23u*WIDTH + col));
    QSB_RECT_RL(a, b, c, d, e, f, g, h, QSB_RECT_LOAD(kw + 24u*WIDTH + col));
    QSB_RECT_RL(h, a, b, c, d, e, f, g, QSB_RECT_LOAD(kw + 25u*WIDTH + col));
    QSB_RECT_RL(g, h, a, b, c, d, e, f, QSB_RECT_LOAD(kw + 26u*WIDTH + col));
    QSB_RECT_RL(f, g, h, a, b, c, d, e, QSB_RECT_LOAD(kw + 27u*WIDTH + col));
    QSB_RECT_RL(e, f, g, h, a, b, c, d, QSB_RECT_LOAD(kw + 28u*WIDTH + col));
    QSB_RECT_RL(d, e, f, g, h, a, b, c, QSB_RECT_LOAD(kw + 29u*WIDTH + col));
    QSB_RECT_RL(c, d, e, f, g, h, a, b, QSB_RECT_LOAD(kw + 30u*WIDTH + col));
    QSB_RECT_RL(b, c, d, e, f, g, h, a, QSB_RECT_LOAD(kw + 31u*WIDTH + col));
    QSB_RECT_RL(a, b, c, d, e, f, g, h, QSB_RECT_LOAD(kw + 32u*WIDTH + col));
    QSB_RECT_RL(h, a, b, c, d, e, f, g, QSB_RECT_LOAD(kw + 33u*WIDTH + col));
    QSB_RECT_RL(g, h, a, b, c, d, e, f, QSB_RECT_LOAD(kw + 34u*WIDTH + col));
    QSB_RECT_RL(f, g, h, a, b, c, d, e, QSB_RECT_LOAD(kw + 35u*WIDTH + col));
    QSB_RECT_RL(e, f, g, h, a, b, c, d, QSB_RECT_LOAD(kw + 36u*WIDTH + col));
    QSB_RECT_RL(d, e, f, g, h, a, b, c, QSB_RECT_LOAD(kw + 37u*WIDTH + col));
    QSB_RECT_RL(c, d, e, f, g, h, a, b, QSB_RECT_LOAD(kw + 38u*WIDTH + col));
    QSB_RECT_RL(b, c, d, e, f, g, h, a, QSB_RECT_LOAD(kw + 39u*WIDTH + col));
    QSB_RECT_RL(a, b, c, d, e, f, g, h, QSB_RECT_LOAD(kw + 40u*WIDTH + col));
    QSB_RECT_RL(h, a, b, c, d, e, f, g, QSB_RECT_LOAD(kw + 41u*WIDTH + col));
    QSB_RECT_RL(g, h, a, b, c, d, e, f, QSB_RECT_LOAD(kw + 42u*WIDTH + col));
    QSB_RECT_RL(f, g, h, a, b, c, d, e, QSB_RECT_LOAD(kw + 43u*WIDTH + col));
    QSB_RECT_RL(e, f, g, h, a, b, c, d, QSB_RECT_LOAD(kw + 44u*WIDTH + col));
    QSB_RECT_RL(d, e, f, g, h, a, b, c, QSB_RECT_LOAD(kw + 45u*WIDTH + col));
    QSB_RECT_RL(c, d, e, f, g, h, a, b, QSB_RECT_LOAD(kw + 46u*WIDTH + col));
    QSB_R63_FF04(QSB_RECT_LOAD(kw + 47u*WIDTH + col) + tp.mid[0], tp.d4, state[0], state[4]);
    state[1] = tp.mid[1] + b;
    state[2] = tp.mid[2] + c;
    state[3] = tp.mid[3] + d;
    state[5] = tp.mid[5] + f;
    state[6] = tp.mid[6] + g;
    state[7] = tp.mid[7] + h;
}
#undef QSB_RECT_RL
