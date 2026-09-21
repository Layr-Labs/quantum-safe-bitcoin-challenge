// GLV608. Canonical a,b in [0,p); output is -(a+b) mod p, also canonical.
#pragma once
__device__ __forceinline__ void qsb_dual_x_neg_sum(uint64_t *out,const uint64_t *a,const uint64_t *b) {
 uint64_t r0,r1,r2,r3;
 asm("{\n\t.reg .u64 t0,t1,t2,t3,u0,u1,u2,u3,borrow,k;\n\t"
 "sub.cc.u64 t0,0xFFFFFFFEFFFFFC2F,%4;\n\t"
 "subc.cc.u64 t1,0xFFFFFFFFFFFFFFFF,%5;\n\t"
 "subc.cc.u64 t2,0xFFFFFFFFFFFFFFFF,%6;\n\t"
 "subc.u64 t3,0xFFFFFFFFFFFFFFFF,%7;\n\t"
 "sub.cc.u64 u0,t0,%8;\n\t"
 "subc.cc.u64 u1,t1,%9;\n\t"
 "subc.cc.u64 u2,t2,%10;\n\t"
 "subc.cc.u64 u3,t3,%11;\n\t"
 "subc.u64 borrow,0,0;\n\t"
 "and.b64 k,borrow,0xFFFFFFFEFFFFFC2F;\n\t"
 "add.cc.u64 u0,u0,k;\n\t"
 "addc.cc.u64 u1,u1,borrow;\n\t"
 "addc.cc.u64 u2,u2,borrow;\n\t"
 "addc.u64 u3,u3,borrow;\n\t"
 "mov.u64 %0,u0;mov.u64 %1,u1;mov.u64 %2,u2;mov.u64 %3,u3;\n\t}"
 : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
 : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
   "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
 // p itself occurs only when both inputs are zero; fold it to canonical zero.
 uint64_t keep=-(uint64_t)!((r1&r2&r3)==UINT64_MAX && r0==0xFFFFFFFEFFFFFC2FULL);
 out[0]=r0&keep;out[1]=r1&keep;out[2]=r2&keep;out[3]=r3&keep;
}
