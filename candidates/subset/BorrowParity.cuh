#pragma once
// Compare most-significant limbs first, exiting as soon as they differ.
// Odd modulus: adding p after an unsigned borrow flips the low bit.
__device__ __forceinline__ uint32_t qsb_borrow_parity(const uint64_t *a,const uint64_t *b) {
 uint32_t borrow;
#ifndef __CUDA_ARCH__
 /* host reference of the same comparison: borrow = (a < b) as 256-bit unsigned */
 borrow = (a[3]!=b[3]) ? (a[3]<b[3]) : (a[2]!=b[2]) ? (a[2]<b[2]) : (a[1]!=b[1]) ? (a[1]<b[1]) : (a[0]<b[0]);
#else
 asm("{ .reg .pred ne,lt;\n"
     "setp.ne.u64 ne,%4,%8; setp.lt.u64 lt,%4,%8; @ne bra done;\n"
     "setp.ne.u64 ne,%3,%7; setp.lt.u64 lt,%3,%7; @ne bra done;\n"
     "setp.ne.u64 ne,%2,%6; setp.lt.u64 lt,%2,%6; @ne bra done;\n"
     "setp.lt.u64 lt,%1,%5;\n"
     "done: selp.u32 %0,1,0,lt; }"
     : "=r"(borrow)
     : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
       "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
#endif
 return (uint32_t)((a[0]^b[0]^borrow)&1u);
}
