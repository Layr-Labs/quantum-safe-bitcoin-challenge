// SPDX-License-Identifier: GPL-3.0-only
// Exact parity-window core ported from public PR885 (EvanYan1024, 3e166ba4).
// Subset adapter: fallback preserves the inherited speculative qsb_fmul/qsb_fadd path.
#pragma once
#ifndef QSB_PW_HI_APPROX
#define QSB_PW_HI_APPROX 0
#endif

#ifndef QSB_K2S_PARITY_WINDOW
#define QSB_K2S_PARITY_WINDOW 1
#endif
#if QSB_K2S_PARITY_WINDOW
/* Public PR965 by @Portablelle narrows PR885's bounded window from 27 to 18
 * products. Keep the original window as the rare guard-reject path so the
 * subset's speculative field fallback is selected identically on all inputs. */
#ifndef QSB_K2S_PARITY_NARROW
#define QSB_K2S_PARITY_NARROW 1
#endif

__device__ __forceinline__ void qsb_parity_window_words(
uint64_t &mid, uint64_t &top, const uint64_t *a, const uint64_t *b) {
asm(
"{\n"
".reg .u32 a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7;\n"
".reg .u32 pcarry,lo,hi,top,mid0,mid1,bit;\n"
".reg .u64 acc,t,mid,high;\n"
"mov.b64 {a0,a1}, %2;\n"
"mov.b64 {a2,a3}, %3;\n"
"mov.b64 {a4,a5}, %4;\n"
"mov.b64 {a6,a7}, %5;\n"
"mov.b64 {b0,b1}, %6;\n"
"mov.b64 {b2,b3}, %7;\n"
"mov.b64 {b4,b5}, %8;\n"
"mov.b64 {b6,b7}, %9;\n"
"mul.wide.u32 acc,a0,b5;\n"
"mov.u32 pcarry,0;\n"
"mul.wide.u32 t,a1,b4;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 pcarry,pcarry,0;\n"
"mul.wide.u32 t,a2,b3;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 pcarry,pcarry,0;\n"
"mul.wide.u32 t,a3,b2;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 pcarry,pcarry,0;\n"
"mul.wide.u32 t,a4,b1;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 pcarry,pcarry,0;\n"
"mul.wide.u32 t,a5,b0;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 pcarry,pcarry,0;\n"
"mov.b64 {lo,hi},acc;\n"
"mov.b64 acc,{hi,pcarry};\n"
"mov.u32 top,0;\n"
"mul.wide.u32 t,a0,b6;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 top,top,0;\n"
"mul.wide.u32 t,a1,b5;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 top,top,0;\n"
"mul.wide.u32 t,a2,b4;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 top,top,0;\n"
"mul.wide.u32 t,a3,b3;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 top,top,0;\n"
"mul.wide.u32 t,a4,b2;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 top,top,0;\n"
"mul.wide.u32 t,a5,b1;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 top,top,0;\n"
"mul.wide.u32 t,a6,b0;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 top,top,0;\n"
"mov.b64 {lo,hi},acc;\n"
"mov.b64 mid,{hi,top};\n"
"mul.wide.u32 t,a0,b7;\n"
"add.u64 mid,mid,t;\n"
"mul.wide.u32 t,a1,b6;\n"
"add.u64 mid,mid,t;\n"
"mul.wide.u32 t,a2,b5;\n"
"add.u64 mid,mid,t;\n"
"mul.wide.u32 t,a3,b4;\n"
"add.u64 mid,mid,t;\n"
"mul.wide.u32 t,a4,b3;\n"
"add.u64 mid,mid,t;\n"
"mul.wide.u32 t,a5,b2;\n"
"add.u64 mid,mid,t;\n"
"mul.wide.u32 t,a6,b1;\n"
"add.u64 mid,mid,t;\n"
"mul.wide.u32 t,a7,b0;\n"
"add.u64 mid,mid,t;\n"
"mov.b64 {mid0,mid1},mid;\n"
"and.b32 bit,a1,b7;\n"
"xor.b32 mid1,mid1,bit;\n"
"and.b32 bit,a2,b6;\n"
"xor.b32 mid1,mid1,bit;\n"
"and.b32 bit,a3,b5;\n"
"xor.b32 mid1,mid1,bit;\n"
"and.b32 bit,a4,b4;\n"
"xor.b32 mid1,mid1,bit;\n"
"and.b32 bit,a5,b3;\n"
"xor.b32 mid1,mid1,bit;\n"
"and.b32 bit,a6,b2;\n"
"xor.b32 mid1,mid1,bit;\n"
"and.b32 bit,a7,b1;\n"
"xor.b32 mid1,mid1,bit;\n"
"and.b32 mid1,mid1,1;\n"
"mov.b64 %0,{mid0,mid1};\n"
"mul.wide.u32 acc,a5,b7;\n"
"mov.u32 pcarry,0;\n"
"mul.wide.u32 t,a6,b6;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 pcarry,pcarry,0;\n"
"mul.wide.u32 t,a7,b5;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 pcarry,pcarry,0;\n"
"mov.b64 {lo,hi},acc;\n"
"mov.b64 acc,{hi,pcarry};\n"
"mov.u32 top,0;\n"
"mul.wide.u32 t,a6,b7;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 top,top,0;\n"
"mul.wide.u32 t,a7,b6;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 top,top,0;\n"
"mov.b64 {lo,hi},acc;\n"
"mov.b64 high,{hi,top};\n"
"mul.wide.u32 t,a7,b7;\n"
"add.u64 high,high,t;\n"
"mov.u64 %1,high;\n"
"}\n"
: "=l"(mid),"=l"(top)
: "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
}

/* Exact dependency-split schedule; neither window nor rejection guards change.
 * The degree6 sum needs its complete carry count; degree7 is modulo2^64.
 * Default off until an interleaved existing-verifier throughput gate wins. */
#ifndef QSB_PW_SPLIT_ACC
#define QSB_PW_SPLIT_ACC 0
#endif
#if QSB_PW_SPLIT_ACC != 0 && QSB_PW_SPLIT_ACC != 1
#error "QSB_PW_SPLIT_ACC must be 0 or 1"
#endif
/* Only bit32 and low32 of qn are consumed. With top=lo+2^32*hi,
 * bit32(top+977*hi+xn+beta_hi) = bit32(lo+977*hi+xn+beta_hi)
 * XOR bit0(hi), including u64 wrap. All other carries remain complete;
 * the original narrow guard and original fallback are unchanged. */
#ifndef QSB_PW_BIT32
#define QSB_PW_BIT32 0
#endif
#if QSB_PW_BIT32 != 0 && QSB_PW_BIT32 != 1
#error "QSB_PW_BIT32 must be 0 or 1"
#endif
#if QSB_PW_BIT32 && (!QSB_K2S_PARITY_NARROW || QSB_PW_QN)
#error "QSB_PW_BIT32 requires the original narrow qn form"
#endif
#if QSB_K2S_PARITY_NARROW
__device__ __forceinline__ void qsb_parity_window_words_narrow(
uint64_t &mid, uint64_t &top, const uint64_t *a, const uint64_t *b) {
asm(
"{\n"
".reg .u32 a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7;\n"
".reg .u32 pcarry,lo,hi,top,mid0,mid1,bit;\n"
".reg .u64 acc,t,mid,high;\n"
"mov.b64 {a0,a1}, %2;\n"
"mov.b64 {a2,a3}, %3;\n"
"mov.b64 {a4,a5}, %4;\n"
"mov.b64 {a6,a7}, %5;\n"
"mov.b64 {b0,b1}, %6;\n"
"mov.b64 {b2,b3}, %7;\n"
"mov.b64 {b4,b5}, %8;\n"
"mov.b64 {b6,b7}, %9;\n"
#if QSB_PW_SPLIT_ACC
".reg .u64 tmp;\n"
"mul.wide.u32 acc,a0,b6;\n"
"mul.wide.u32 t,a1,b5;\n"
"mov.u32 top,0;\n"
"mov.u32 pcarry,0;\n"
"mul.wide.u32 tmp,a2,b4;\n"
"add.cc.u64 acc,acc,tmp;\n"
"addc.u32 top,top,0;\n"
"mul.wide.u32 tmp,a3,b3;\n"
"add.cc.u64 t,t,tmp;\n"
"addc.u32 pcarry,pcarry,0;\n"
"mul.wide.u32 tmp,a4,b2;\n"
"add.cc.u64 acc,acc,tmp;\n"
"addc.u32 top,top,0;\n"
"mul.wide.u32 tmp,a5,b1;\n"
"add.cc.u64 t,t,tmp;\n"
"addc.u32 pcarry,pcarry,0;\n"
"mul.wide.u32 tmp,a6,b0;\n"
"add.cc.u64 acc,acc,tmp;\n"
"addc.u32 top,top,0;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 top,top,pcarry;\n"
#else
"mul.wide.u32 acc,a0,b6;\n"
"mov.u32 top,0;\n"
"mul.wide.u32 t,a1,b5;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 top,top,0;\n"
"mul.wide.u32 t,a2,b4;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 top,top,0;\n"
"mul.wide.u32 t,a3,b3;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 top,top,0;\n"
"mul.wide.u32 t,a4,b2;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 top,top,0;\n"
"mul.wide.u32 t,a5,b1;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 top,top,0;\n"
"mul.wide.u32 t,a6,b0;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 top,top,0;\n"
#endif
"mov.b64 {lo,hi},acc;\n"
"mov.b64 mid,{hi,top};\n"
#if QSB_PW_SPLIT_ACC
"mad.wide.u32 mid,a0,b7,mid;\n"
"mul.wide.u32 acc,a1,b6;\n"
"mad.wide.u32 mid,a2,b5,mid;\n"
"mad.wide.u32 acc,a3,b4,acc;\n"
"mad.wide.u32 mid,a4,b3,mid;\n"
"mad.wide.u32 acc,a5,b2,acc;\n"
"mad.wide.u32 mid,a6,b1,mid;\n"
"mad.wide.u32 acc,a7,b0,acc;\n"
"add.u64 mid,mid,acc;\n"
#else
"mul.wide.u32 t,a0,b7;\n"
"add.u64 mid,mid,t;\n"
"mul.wide.u32 t,a1,b6;\n"
"add.u64 mid,mid,t;\n"
"mul.wide.u32 t,a2,b5;\n"
"add.u64 mid,mid,t;\n"
"mul.wide.u32 t,a3,b4;\n"
"add.u64 mid,mid,t;\n"
"mul.wide.u32 t,a4,b3;\n"
"add.u64 mid,mid,t;\n"
"mul.wide.u32 t,a5,b2;\n"
"add.u64 mid,mid,t;\n"
"mul.wide.u32 t,a6,b1;\n"
"add.u64 mid,mid,t;\n"
"mul.wide.u32 t,a7,b0;\n"
"add.u64 mid,mid,t;\n"
#endif
"mov.b64 {mid0,mid1},mid;\n"
"and.b32 bit,a1,b7;\n"
"xor.b32 mid1,mid1,bit;\n"
"and.b32 bit,a2,b6;\n"
"xor.b32 mid1,mid1,bit;\n"
"and.b32 bit,a3,b5;\n"
"xor.b32 mid1,mid1,bit;\n"
"and.b32 bit,a4,b4;\n"
"xor.b32 mid1,mid1,bit;\n"
"and.b32 bit,a5,b3;\n"
"xor.b32 mid1,mid1,bit;\n"
"and.b32 bit,a6,b2;\n"
"xor.b32 mid1,mid1,bit;\n"
"and.b32 bit,a7,b1;\n"
"xor.b32 mid1,mid1,bit;\n"
"and.b32 mid1,mid1,1;\n"
"mov.b64 %0,{mid0,mid1};\n"
"mul.wide.u32 acc,a6,b7;\n"
"mov.u32 top,0;\n"
"mul.wide.u32 t,a7,b6;\n"
"add.cc.u64 acc,acc,t;\n"
"addc.u32 top,top,0;\n"
"mov.b64 {lo,hi},acc;\n"
"mov.b64 high,{hi,top};\n"
"mul.wide.u32 t,a7,b7;\n"
"add.u64 high,high,t;\n"
"mov.u64 %1,high;\n"
"}\n"
: "=l"(mid),"=l"(top)
: "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
}

#endif

#if QSB_PW_HI_APPROX
#define qsb_parity_product_window qsb_parity_product_window_reference
#endif
__device__ __forceinline__ uint32_t qsb_parity_product_window(
const uint64_t *a, const uint64_t *b, const uint64_t *beta, uint32_t neg) {
uint64_t mid,top;
#if QSB_K2S_PARITY_NARROW
qsb_parity_window_words_narrow(mid,top,a,b);
const uint32_t xn=(uint32_t)mid;
#if QSB_PW_BIT32
const uint64_t qn=(uint64_t)(uint32_t)top+977ULL*(top>>32)+xn+(beta[3]>>32);
#elif QSB_PW_QN == 2
/* QSB_PW_QN 2: the same u64 sum with the two 32-bit addends typed as 32-bit values. */
const uint64_t qn=top+(uint64_t)(977u*0u)+977ULL*(uint32_t)(top>>32)+(uint64_t)xn+(uint64_t)(uint32_t)(beta[3]>>32);
#elif QSB_PW_QN
/* QSB_PW_QN (tree.cu): the same u64 sum, as mad.wide.u32 + one three-input add pair. */
uint64_t qn;
asm("{\n\t.reg .u32 th,lo,hi;\n\t"
"mov.b64 {lo,th}, %1;\n\t"
"mad.wide.u32 %0, th, 977, %1;\n\t"
"mov.b64 {lo,hi}, %0;\n\t"
"add.cc.u32 lo, lo, %2;\n\t"
"addc.u32 hi, hi, 0;\n\t"
"add.cc.u32 lo, lo, %3;\n\t"
"addc.u32 hi, hi, 0;\n\t"
"mov.b64 %0, {lo,hi};\n\t}"
: "=l"(qn) : "l"(top), "r"(xn), "r"((uint32_t)(beta[3]>>32)));
#else
const uint64_t qn=top+977ULL*(top>>32)+xn+(beta[3]>>32);
#endif
/* D5 omission changes the middle accumulator by at most 6; D12
     * omission changes the high accumulator by at most 3. Including a
     * possible high-word carry gives |Q_old-Q_new|<=986. These stronger
     * guards imply the original fast-path guards and preserve its bit 32. */
if(xn<0xfffffff9u && (uint32_t)qn<0xfffff47fu)
#if QSB_PW_BIT32
return (uint32_t)(((a[0]&b[0])^(mid>>32)^beta[0]^(qn>>32)^(top>>32)^neg)&1u);
#else
return (uint32_t)(((a[0]&b[0])^(mid>>32)^beta[0]^(qn>>32)^neg)&1u);
#endif
/* On a narrow rejection, execute the original 27-product decision and
     * original speculative fallback. This preserves the old result even on
     * directed operands where that fallback differs from exact field math. */
#endif
qsb_parity_window_words(mid,top,a,b);
const uint32_t x7=(uint32_t)mid;
// Only bit 32 and bits 0..31 of q are used. u64 overflow is harmless.
const uint64_t q=top+977ULL*(top>>32)+x7+(beta[3]>>32);
// Unknown carries change q by at most 1958. Exclude the final all-one
// limb too, so the baseline sum-parity exceptional correction cannot fire.
if(x7!=0xffffffffu && (uint32_t)q<0xfffff859u) {
return (uint32_t)(((a[0]&b[0])^(mid>>32)^beta[0]^(q>>32)^neg)&1u);
}
uint64_t raw[4];
qsb_fmul(raw,a,b);
qsb_fadd(raw,raw,beta);
return (uint32_t)((raw[0]^neg)&1u);
}
#if QSB_PW_HI_APPROX
#undef qsb_parity_product_window
/* Iter46: replace nine wide products by high-half-only products. D6's low
 * halves contribute a carry c6 in [0,6]; D13's low halves contribute c13 in
 * [0,1]. The new middle M is smaller by c6, the new top T by c13. Require
 * M.lo+6 < the old narrow guard, and Q.lo+984 < its old guard (984=6+978).
 * Then M bit32 and Q bit32 cannot change and the old narrow fast path is
 * guaranteed to accept. Otherwise execute the unchanged original decision,
 * including both rejection fallbacks. Inputs and speculative semantics stay
 * identical; this is not a wider unproven parity window. */
__device__ __forceinline__ void qsb_parity_window_words_hi(
uint64_t &mid, uint64_t &top, const uint64_t *a, const uint64_t *b) {
asm("{\n"
".reg .u32 a<8>,b<8>,v,lo,hi,bit,ml,mh;\n"
".reg .u64 m,t,h;\n"
"mov.b64 {a0,a1},%2; mov.b64 {a2,a3},%3;\n"
"mov.b64 {a4,a5},%4; mov.b64 {a6,a7},%5;\n"
"mov.b64 {b0,b1},%6; mov.b64 {b2,b3},%7;\n"
"mov.b64 {b4,b5},%8; mov.b64 {b6,b7},%9;\n"
"mul.hi.u32 lo,a0,b6; mov.u32 hi,0;\n"
"mul.hi.u32 v,a1,b5; add.cc.u32 lo,lo,v; addc.u32 hi,hi,0;\n"
"mul.hi.u32 v,a2,b4; add.cc.u32 lo,lo,v; addc.u32 hi,hi,0;\n"
"mul.hi.u32 v,a3,b3; add.cc.u32 lo,lo,v; addc.u32 hi,hi,0;\n"
"mul.hi.u32 v,a4,b2; add.cc.u32 lo,lo,v; addc.u32 hi,hi,0;\n"
"mul.hi.u32 v,a5,b1; add.cc.u32 lo,lo,v; addc.u32 hi,hi,0;\n"
"mul.hi.u32 v,a6,b0; add.cc.u32 lo,lo,v; addc.u32 hi,hi,0;\n"
"mov.b64 m,{lo,hi};\n"
"mad.wide.u32 m,a0,b7,m; mad.wide.u32 m,a1,b6,m;\n"
"mad.wide.u32 m,a2,b5,m; mad.wide.u32 m,a3,b4,m;\n"
"mad.wide.u32 m,a4,b3,m; mad.wide.u32 m,a5,b2,m;\n"
"mad.wide.u32 m,a6,b1,m; mad.wide.u32 m,a7,b0,m;\n"
"mov.b64 {ml,mh},m;\n"
"and.b32 bit,a1,b7; xor.b32 mh,mh,bit;\n"
"and.b32 bit,a2,b6; xor.b32 mh,mh,bit;\n"
"and.b32 bit,a3,b5; xor.b32 mh,mh,bit;\n"
"and.b32 bit,a4,b4; xor.b32 mh,mh,bit;\n"
"and.b32 bit,a5,b3; xor.b32 mh,mh,bit;\n"
"and.b32 bit,a6,b2; xor.b32 mh,mh,bit;\n"
"and.b32 bit,a7,b1; xor.b32 mh,mh,bit;\n"
"and.b32 mh,mh,1; mov.b64 %0,{ml,mh};\n"
"mul.hi.u32 lo,a6,b7; mul.hi.u32 v,a7,b6;\n"
"add.cc.u32 lo,lo,v; addc.u32 hi,0,0;\n"
"mov.b64 h,{lo,hi}; mad.wide.u32 h,a7,b7,h;\n"
"mov.u64 %1,h;\n}"
: "=l"(mid),"=l"(top)
: "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
}
__device__ __forceinline__ uint32_t qsb_parity_product_window(
const uint64_t *a, const uint64_t *b, const uint64_t *beta, uint32_t neg) {
uint64_t mid,top;
qsb_parity_window_words_hi(mid,top,a,b);
const uint32_t xn=(uint32_t)mid;
const uint64_t q=top+977ULL*(top>>32)+xn+(beta[3]>>32);
if(xn<0xfffffff3u && (uint32_t)q<0xfffff0a7u)
return (uint32_t)(((a[0]&b[0])^(mid>>32)^beta[0]^(q>>32)^neg)&1u);
return qsb_parity_product_window_reference(a,b,beta,neg);
}
#endif
/* Two COMPLETE serial windows interleaved; not split accumulators. All inputs
 * are snapshotted before any output write, and CC producer/capture groups stay
 * contiguous because PTX has one condition-code carry per thread. */
#ifndef QSB_PW_PAIR
#define QSB_PW_PAIR 0
#endif
#if QSB_PW_PAIR != 0 && QSB_PW_PAIR != 1
#error "QSB_PW_PAIR must be 0 or 1"
#endif
#if QSB_PW_HI_APPROX != 0 && QSB_PW_HI_APPROX != 1
#error "QSB_PW_HI_APPROX must be 0 or 1"
#endif
#if QSB_PW_HI_APPROX && (!QSB_K2S_PARITY_NARROW || QSB_PW_SPLIT_ACC || QSB_PW_PAIR || QSB_PW_BIT32)
#error "QSB_PW_HI_APPROX requires the serial original narrow window"
#endif
#if QSB_PW_PAIR
#if !QSB_K2S_PARITY_NARROW || !QSB_NEGFOLD_PARITY || QSB_K2S_CENTER_SQR != 2 || QSB_K2S_SUM_BASIS || QSB_PW_SPLIT_ACC || QSB_PW_BIT32 || QSB_TAIL_STAGGER || QSB_TAIL_WEAVE
#error "QSB_PW_PAIR requires the unsplit narrow, original-qn, negfold, center2, non-sum-basis post3"
#endif
__device__ __forceinline__ void qsb_parity_window_words_pair(
uint64_t &mid0, uint64_t &top0, uint64_t &mid1, uint64_t &top1,
const uint64_t *a0, const uint64_t *b0,
const uint64_t *a1, const uint64_t *b1) {
asm(
"{\n"
".reg .u32 a00,a10,a20,a30,a40,a50,a60,a70,b00,b10,b20,b30,b40,b50,b60,b70;\n"
".reg .u32 pcarry0,lo0,hi0,top0,mid00,mid10,bit0;\n"
".reg .u64 acc0,t0,mid0,high0;\n"
".reg .u32 a01,a11,a21,a31,a41,a51,a61,a71,b01,b11,b21,b31,b41,b51,b61,b71;\n"
".reg .u32 pcarry1,lo1,hi1,top1,mid01,mid11,bit1;\n"
".reg .u64 acc1,t1,mid1,high1;\n"
"mov.b64 {a00,a10}, %4;\n"
"mov.b64 {a20,a30}, %5;\n"
"mov.b64 {a40,a50}, %6;\n"
"mov.b64 {a60,a70}, %7;\n"
"mov.b64 {b00,b10}, %8;\n"
"mov.b64 {b20,b30}, %9;\n"
"mov.b64 {b40,b50}, %10;\n"
"mov.b64 {b60,b70}, %11;\n"
"mov.b64 {a01,a11}, %12;\n"
"mov.b64 {a21,a31}, %13;\n"
"mov.b64 {a41,a51}, %14;\n"
"mov.b64 {a61,a71}, %15;\n"
"mov.b64 {b01,b11}, %16;\n"
"mov.b64 {b21,b31}, %17;\n"
"mov.b64 {b41,b51}, %18;\n"
"mov.b64 {b61,b71}, %19;\n"
"mul.wide.u32 acc0,a00,b60;\n"
"mul.wide.u32 acc1,a01,b61;\n"
"mov.u32 top0,0;\n"
"mov.u32 top1,0;\n"
"mul.wide.u32 t0,a10,b50;\n"
"add.cc.u64 acc0,acc0,t0;\n"
"addc.u32 top0,top0,0;\n"
"mul.wide.u32 t1,a11,b51;\n"
"add.cc.u64 acc1,acc1,t1;\n"
"addc.u32 top1,top1,0;\n"
"mul.wide.u32 t0,a20,b40;\n"
"add.cc.u64 acc0,acc0,t0;\n"
"addc.u32 top0,top0,0;\n"
"mul.wide.u32 t1,a21,b41;\n"
"add.cc.u64 acc1,acc1,t1;\n"
"addc.u32 top1,top1,0;\n"
"mul.wide.u32 t0,a30,b30;\n"
"add.cc.u64 acc0,acc0,t0;\n"
"addc.u32 top0,top0,0;\n"
"mul.wide.u32 t1,a31,b31;\n"
"add.cc.u64 acc1,acc1,t1;\n"
"addc.u32 top1,top1,0;\n"
"mul.wide.u32 t0,a40,b20;\n"
"add.cc.u64 acc0,acc0,t0;\n"
"addc.u32 top0,top0,0;\n"
"mul.wide.u32 t1,a41,b21;\n"
"add.cc.u64 acc1,acc1,t1;\n"
"addc.u32 top1,top1,0;\n"
"mul.wide.u32 t0,a50,b10;\n"
"add.cc.u64 acc0,acc0,t0;\n"
"addc.u32 top0,top0,0;\n"
"mul.wide.u32 t1,a51,b11;\n"
"add.cc.u64 acc1,acc1,t1;\n"
"addc.u32 top1,top1,0;\n"
"mul.wide.u32 t0,a60,b00;\n"
"add.cc.u64 acc0,acc0,t0;\n"
"addc.u32 top0,top0,0;\n"
"mul.wide.u32 t1,a61,b01;\n"
"add.cc.u64 acc1,acc1,t1;\n"
"addc.u32 top1,top1,0;\n"
"mov.b64 {lo0,hi0},acc0;\n"
"mov.b64 {lo1,hi1},acc1;\n"
"mov.b64 mid0,{hi0,top0};\n"
"mov.b64 mid1,{hi1,top1};\n"
"mul.wide.u32 t0,a00,b70;\n"
"add.u64 mid0,mid0,t0;\n"
"mul.wide.u32 t1,a01,b71;\n"
"add.u64 mid1,mid1,t1;\n"
"mul.wide.u32 t0,a10,b60;\n"
"add.u64 mid0,mid0,t0;\n"
"mul.wide.u32 t1,a11,b61;\n"
"add.u64 mid1,mid1,t1;\n"
"mul.wide.u32 t0,a20,b50;\n"
"add.u64 mid0,mid0,t0;\n"
"mul.wide.u32 t1,a21,b51;\n"
"add.u64 mid1,mid1,t1;\n"
"mul.wide.u32 t0,a30,b40;\n"
"add.u64 mid0,mid0,t0;\n"
"mul.wide.u32 t1,a31,b41;\n"
"add.u64 mid1,mid1,t1;\n"
"mul.wide.u32 t0,a40,b30;\n"
"add.u64 mid0,mid0,t0;\n"
"mul.wide.u32 t1,a41,b31;\n"
"add.u64 mid1,mid1,t1;\n"
"mul.wide.u32 t0,a50,b20;\n"
"add.u64 mid0,mid0,t0;\n"
"mul.wide.u32 t1,a51,b21;\n"
"add.u64 mid1,mid1,t1;\n"
"mul.wide.u32 t0,a60,b10;\n"
"add.u64 mid0,mid0,t0;\n"
"mul.wide.u32 t1,a61,b11;\n"
"add.u64 mid1,mid1,t1;\n"
"mul.wide.u32 t0,a70,b00;\n"
"add.u64 mid0,mid0,t0;\n"
"mul.wide.u32 t1,a71,b01;\n"
"add.u64 mid1,mid1,t1;\n"
"mov.b64 {mid00,mid10},mid0;\n"
"mov.b64 {mid01,mid11},mid1;\n"
"and.b32 bit0,a10,b70;\n"
"and.b32 bit1,a11,b71;\n"
"xor.b32 mid10,mid10,bit0;\n"
"xor.b32 mid11,mid11,bit1;\n"
"and.b32 bit0,a20,b60;\n"
"and.b32 bit1,a21,b61;\n"
"xor.b32 mid10,mid10,bit0;\n"
"xor.b32 mid11,mid11,bit1;\n"
"and.b32 bit0,a30,b50;\n"
"and.b32 bit1,a31,b51;\n"
"xor.b32 mid10,mid10,bit0;\n"
"xor.b32 mid11,mid11,bit1;\n"
"and.b32 bit0,a40,b40;\n"
"and.b32 bit1,a41,b41;\n"
"xor.b32 mid10,mid10,bit0;\n"
"xor.b32 mid11,mid11,bit1;\n"
"and.b32 bit0,a50,b30;\n"
"and.b32 bit1,a51,b31;\n"
"xor.b32 mid10,mid10,bit0;\n"
"xor.b32 mid11,mid11,bit1;\n"
"and.b32 bit0,a60,b20;\n"
"and.b32 bit1,a61,b21;\n"
"xor.b32 mid10,mid10,bit0;\n"
"xor.b32 mid11,mid11,bit1;\n"
"and.b32 bit0,a70,b10;\n"
"and.b32 bit1,a71,b11;\n"
"xor.b32 mid10,mid10,bit0;\n"
"xor.b32 mid11,mid11,bit1;\n"
"and.b32 mid10,mid10,1;\n"
"and.b32 mid11,mid11,1;\n"
"mul.wide.u32 acc0,a60,b70;\n"
"mul.wide.u32 acc1,a61,b71;\n"
"mov.u32 top0,0;\n"
"mov.u32 top1,0;\n"
"mul.wide.u32 t0,a70,b60;\n"
"add.cc.u64 acc0,acc0,t0;\n"
"addc.u32 top0,top0,0;\n"
"mul.wide.u32 t1,a71,b61;\n"
"add.cc.u64 acc1,acc1,t1;\n"
"addc.u32 top1,top1,0;\n"
"mov.b64 {lo0,hi0},acc0;\n"
"mov.b64 {lo1,hi1},acc1;\n"
"mov.b64 high0,{hi0,top0};\n"
"mov.b64 high1,{hi1,top1};\n"
"mul.wide.u32 t0,a70,b70;\n"
"add.u64 high0,high0,t0;\n"
"mul.wide.u32 t1,a71,b71;\n"
"add.u64 high1,high1,t1;\n"
"mov.b64 %0,{mid00,mid10};\n"
"mov.u64 %1,high0;\n"
"mov.b64 %2,{mid01,mid11};\n"
"mov.u64 %3,high1;\n"
"}\n"
: "=l"(mid0),"=l"(top0),"=l"(mid1),"=l"(top1)
: "l"(a0[0]),"l"(a0[1]),"l"(a0[2]),"l"(a0[3]),
"l"(b0[0]),"l"(b0[1]),"l"(b0[2]),"l"(b0[3]),
"l"(a1[0]),"l"(a1[1]),"l"(a1[2]),"l"(a1[3]),
"l"(b1[0]),"l"(b1[1]),"l"(b1[2]),"l"(b1[3]));
}
/* Each stream preserves every qn variant and its original rejection fallback. */
__device__ __forceinline__ uint32_t qsb_parity_window_pair_decide(
uint64_t mid, uint64_t top,
const uint64_t *a, const uint64_t *b, const uint64_t *beta, uint32_t neg) {
const uint32_t xn=(uint32_t)mid;
#if QSB_PW_QN == 2
/* QSB_PW_QN 2: the same u64 sum with the two 32-bit addends typed as 32-bit values. */
const uint64_t qn=top+(uint64_t)(977u*0u)+977ULL*(uint32_t)(top>>32)+(uint64_t)xn+(uint64_t)(uint32_t)(beta[3]>>32);
#elif QSB_PW_QN
/* QSB_PW_QN (tree.cu): the same u64 sum, as mad.wide.u32 + one three-input add pair. */
uint64_t qn;
asm("{\n\t.reg .u32 th,lo,hi;\n\t"
"mov.b64 {lo,th}, %1;\n\t"
"mad.wide.u32 %0, th, 977, %1;\n\t"
"mov.b64 {lo,hi}, %0;\n\t"
"add.cc.u32 lo, lo, %2;\n\t"
"addc.u32 hi, hi, 0;\n\t"
"add.cc.u32 lo, lo, %3;\n\t"
"addc.u32 hi, hi, 0;\n\t"
"mov.b64 %0, {lo,hi};\n\t}"
: "=l"(qn) : "l"(top), "r"(xn), "r"((uint32_t)(beta[3]>>32)));
#else
const uint64_t qn=top+977ULL*(top>>32)+xn+(beta[3]>>32);
#endif
/* D5 omission changes the middle accumulator by at most 6; D12
     * omission changes the high accumulator by at most 3. Including a
     * possible high-word carry gives |Q_old-Q_new|<=986. These stronger
     * guards imply the original fast-path guards and preserve its bit 32. */
if(xn<0xfffffff9u && (uint32_t)qn<0xfffff47fu)
return (uint32_t)(((a[0]&b[0])^(mid>>32)^beta[0]^(qn>>32)^neg)&1u);
return qsb_parity_product_window(a,b,beta,neg);
}
__device__ __forceinline__ uint32_t qsb_parity_product_window_pair(
const uint64_t *a0, const uint64_t *b0,
const uint64_t *a1, const uint64_t *b1, const uint64_t *beta) {
uint64_t mid0,top0,mid1,top1;
qsb_parity_window_words_pair(mid0,top0,mid1,top1,a0,b0,a1,b1);
const uint32_t p0=qsb_parity_window_pair_decide(mid0,top0,a0,b0,beta,1u);
const uint32_t p1=qsb_parity_window_pair_decide(mid1,top1,a1,b1,beta,0u);
return p0|(p1<<1);
}
#endif

#endif
