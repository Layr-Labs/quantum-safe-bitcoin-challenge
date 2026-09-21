// SPDX-License-Identifier: GPL-3.0-only
// CUDA/RTX 4090 differential-validated against the full packed product; see SUBMISSION.md.
// Include after qsb_packed_raw_mul and qsb_sum_parity in PackedRecovery.cuh.
#pragma once
#if !QSB_C31 || !QSB_SHORT_CARRY || !QSB_FIELD_SC || !QSB_PARITY_SUM
#error "ParityWindow requires the audited C31/short-carry/field-SC/parity-sum path"
#endif
#if defined(QSB_RP_SQR) && QSB_RP_SQR
#error "RP_SQR changes the multiplication contract; re-audit ParityWindow first"
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

__device__ __forceinline__ uint32_t qsb_parity_product_window(
    const uint64_t *a, const uint64_t *b, const uint64_t *beta, uint32_t neg) {
    uint64_t mid,top;
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
    qsb_packed_raw_mul(raw,a,b);
    return qsb_sum_parity(raw,beta,neg);
}
