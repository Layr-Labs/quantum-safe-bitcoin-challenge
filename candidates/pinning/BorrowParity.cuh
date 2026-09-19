#pragma once
// Parity of the existing four-limb subtraction with an odd-p borrow correction.
// Only the final borrow and the two input low bits are needed.
__device__ __forceinline__ uint32_t qsb_borrow_parity(const uint64_t *a,const uint64_t *b) {
    uint64_t borrow;
    asm("{ .reg .u64 t;\n"
        "sub.cc.u64 t,%1,%5;\n"
        "subc.cc.u64 t,%2,%6;\n"
        "subc.cc.u64 t,%3,%7;\n"
        "subc.cc.u64 t,%4,%8;\n"
        "subc.u64 %0,0,0; }"
        : "=l"(borrow)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
          "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    return (uint32_t)((a[0]^b[0]^borrow)&1u);
}
