#pragma once
/* Private root64 lowering only. ZiRoot64Pair is defined by root64.cuh.
 * All outputs are complete128-bit bit patterns, not modular field residues. */
__device__ __forceinline__ ZiRoot64Pair zi_root64_shared_uu(uint64_t a,uint64_t b){
    ZiRoot64Pair r;
    asm volatile(
        "{\n\t"
        ".reg .b32 a0,a1,b0,b1,w0,w1,v0,v1,c;\n\t"
        ".reg .b64 p,q,h,z;\n\t"
        "mov.b64 {a0,a1},%2;\n\t"
        "mov.b64 {b0,b1},%3;\n\t"
        "mul.wide.u32 p,a0,b0;\n\t"
        "mov.b64 {w0,w1},p;\n\t"
        "mul.wide.u32 q,a0,b1;\n\t"
        "mov.b64 {v0,v1},q;\n\t"
        "add.cc.u32 w1,w1,v0;\n\t"
        "addc.u32 c,0,0;\n\t"
        "cvt.u64.u32 h,v1;\n\t"
        "mul.wide.u32 q,a1,b0;\n\t"
        "mov.b64 {v0,v1},q;\n\t"
        "add.cc.u32 w1,w1,v0;\n\t"
        "addc.u32 c,c,0;\n\t" // c is0,1,2, not a one-bit carry.
        "cvt.u64.u32 z,v1;\n\t"
        "add.u64 h,h,z;\n\t"
        "mul.wide.u32 q,a1,b1;\n\t"
        "add.u64 h,h,q;\n\t"
        "cvt.u64.u32 z,c;\n\t"
        "add.u64 h,h,z;\n\t"
        "mov.b64 %0,{w0,w1};\n\t"
        "mov.b64 %1,h;\n\t"
        "}"
        : "=l"(r.lo),"=l"(r.hi) : "l"(a),"l"(b));
    return r;
}
/* a is a signed64 bit pattern, |a|<=2^60. t is an exact sign-extended
 * signed32 value; the root producer proves t in[-17,16] for cap<=32.
 * We require the conservative[-18,17] envelope. The RESULT stays signed128.
 * a=al+2^32*ah, where al=signed32(low(a)), ah=high_signed(a)+(low(a)>>31).
 * p0=al*t, p1=ah*t; both fit signed64. Return p0+(p1<<32) exactly. */
__device__ __forceinline__ ZiRoot64Pair zi_root64_bounded_st(uint64_t a,uint64_t t){
    ZiRoot64Pair r;
    asm volatile(
        "{\n\t"
        ".reg .b32 al,ah,top,unused,s,p0l,p0h,p1l,p1h,wh,hh;\n\t"
        ".reg .b64 p0,p1;\n\t"
        "mov.b64 {al,ah},%2;\n\t"
        "mov.b64 {top,unused},%3;\n\t"
        "shr.u32 s,al,31;\n\t"
        "add.u32 ah,ah,s;\n\t"
        "mul.wide.s32 p0,al,top;\n\t"
        "mul.wide.s32 p1,ah,top;\n\t"
        "mov.b64 {p0l,p0h},p0;\n\t"
        "mov.b64 {p1l,p1h},p1;\n\t"
        "shr.s32 hh,p0h,31;\n\t" // sign extension of p0 above bit63.
        "add.cc.u32 wh,p0h,p1l;\n\t"
        "addc.u32 hh,hh,p1h;\n\t" // signed high32 plus binary low carry.
        "mov.b64 %0,{p0l,wh};\n\t"
        "cvt.s64.s32 %1,hh;\n\t"
        "}"
        : "=l"(r.lo),"=l"(r.hi) : "l"(a),"l"(t));
    return r;
}
