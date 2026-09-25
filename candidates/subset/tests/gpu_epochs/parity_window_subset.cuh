// SPDX-License-Identifier: GPL-3.0-only
// Exact parity-window core ported from public PR885 (EvanYan1024, 3e166ba4).
// Subset adapter: fallback preserves the inherited speculative qsb_fmul/qsb_fadd path.
#pragma once
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
#ifndef QSB_K2S_PARITY_HIGH
#define QSB_K2S_PARITY_HIGH 1
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

#if QSB_K2S_PARITY_HIGH
__device__ __forceinline__ void qsb_parity_window_words_high(
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
        "mul.hi.u32 hi,a0,b6;\n"
        "cvt.u64.u32 mid,hi;\n"
        "mul.hi.u32 hi,a1,b5;\n"
        "cvt.u64.u32 t,hi;\n"
        "add.u64 mid,mid,t;\n"
        "mul.hi.u32 hi,a2,b4;\n"
        "cvt.u64.u32 t,hi;\n"
        "add.u64 mid,mid,t;\n"
        "mul.hi.u32 hi,a3,b3;\n"
        "cvt.u64.u32 t,hi;\n"
        "add.u64 mid,mid,t;\n"
        "mul.hi.u32 hi,a4,b2;\n"
        "cvt.u64.u32 t,hi;\n"
        "add.u64 mid,mid,t;\n"
        "mul.hi.u32 hi,a5,b1;\n"
        "cvt.u64.u32 t,hi;\n"
        "add.u64 mid,mid,t;\n"
        "mul.hi.u32 hi,a6,b0;\n"
        "cvt.u64.u32 t,hi;\n"
        "add.u64 mid,mid,t;\n"
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

#endif

__device__ __forceinline__ uint32_t qsb_parity_product_window(
    const uint64_t *a, const uint64_t *b, const uint64_t *beta, uint32_t neg) {
    uint64_t mid,top;
#if QSB_K2S_PARITY_NARROW
#if QSB_K2S_PARITY_HIGH
    qsb_parity_window_words_high(mid,top,a,b);
    const uint32_t xh=(uint32_t)mid;
    const uint64_t qh=top+977ULL*(top>>32)+xh+(beta[3]>>32);
    // Omitting the low halves of diagonal 6 removes a carry in [0,6].
    // These guards imply the original narrow guards and preserve both bits 32.
    if(xh<0xfffffff3u && (uint32_t)qh<0xfffff479u)
        return (uint32_t)(((a[0]&b[0])^(mid>>32)^beta[0]^(qh>>32)^neg)&1u);
#endif
    qsb_parity_window_words_narrow(mid,top,a,b);
    const uint32_t xn=(uint32_t)mid;
    const uint64_t qn=top+977ULL*(top>>32)+xn+(beta[3]>>32);
    /* D5 omission changes the middle accumulator by at most 6; D12
     * omission changes the high accumulator by at most 3. Including a
     * possible high-word carry gives |Q_old-Q_new|<=986. These stronger
     * guards imply the original fast-path guards and preserve its bit 32. */
    if(xn<0xfffffff9u && (uint32_t)qn<0xfffff47fu)
        return (uint32_t)(((a[0]&b[0])^(mid>>32)^beta[0]^(qn>>32)^neg)&1u);
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
#endif
