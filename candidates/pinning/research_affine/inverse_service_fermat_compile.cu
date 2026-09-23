// SPDX-License-Identifier: GPL-3.0-only
// Compile-only inverse resource experiment. Exact multiplier copied literally
// from pinning.cu qsb_field_mul. No GPU execution or timing has been performed.
#include "inverse_service.cuh"
#ifndef QSB_SERVICE_BLOCKS
#define QSB_SERVICE_BLOCKS 4
#endif
__device__ __forceinline__ void qsb_field_mul(uint64_t *out,uint64_t *a,uint64_t *b){
    uint64_t r0,r1,r2,r3;
    asm("{\n\t.reg .u32 a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7;\n\t.reg .u64 e0,e1,e2,e3,e4,e5,e6,e7,o0,o1,o2,o3,o4,o5,o6,t,lc;\n\t.reg .u32 cy,o15;\n\t.reg .u32 x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;\n\t.reg .u32 y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14;\n\tmov.b64 {a0,a1}, %4;\n\tmov.b64 {a2,a3}, %5;\n\tmov.b64 {a4,a5}, %6;\n\tmov.b64 {a6,a7}, %7;\n\tmov.b64 {b0,b1}, %8;\n\tmov.b64 {b2,b3}, %9;\n\tmov.b64 {b4,b5}, %10;\n\tmov.b64 {b6,b7}, %11;\n\t.reg .u64 odd_t,odd_lc; .reg .u32 odd_cy;\nmul.wide.u32 e0, a0, b0;\nmul.wide.u32 o0, a0, b1;\nmul.wide.u32 e1, a0, b2;\nmul.wide.u32 o1, a0, b3;\nmul.wide.u32 e2, a0, b4;\nmul.wide.u32 o2, a0, b5;\nmul.wide.u32 e3, a0, b6;\nmul.wide.u32 o3, a0, b7;\nmul.wide.u32 t, a1, b1;\nmul.wide.u32 odd_t, a1, b0;\nadd.cc.u64 e1, e1, t;\nmul.wide.u32 t, a1, b3;\naddc.cc.u64 e2, e2, t;\nmul.wide.u32 t, a1, b5;\naddc.cc.u64 e3, e3, t;\nmul.wide.u32 t, a1, b7;\naddc.u64 e4, t, 0;\nadd.cc.u64 o0, o0, odd_t;\nmul.wide.u32 odd_t, a1, b2;\naddc.cc.u64 o1, o1, odd_t;\nmul.wide.u32 odd_t, a1, b4;\naddc.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a1, b6;\naddc.cc.u64 o3, o3, odd_t;\naddc.u32 odd_cy, 0, 0;\nmul.wide.u32 t, a2, b0;\nadd.cc.u64 e1, e1, t;\nmul.wide.u32 t, a2, b2;\naddc.cc.u64 e2, e2, t;\nmul.wide.u32 t, a2, b4;\naddc.cc.u64 e3, e3, t;\nmul.wide.u32 t, a2, b6;\naddc.cc.u64 e4, e4, t;\naddc.u32 cy, 0, 0;\nmul.wide.u32 odd_t, a2, b1;\nmov.b64 odd_lc, {odd_cy, cy};\nadd.cc.u64 o1, o1, odd_t;\nmul.wide.u32 odd_t, a2, b3;\naddc.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a2, b5;\naddc.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a2, b7;\naddc.u64 o4, odd_t, odd_lc;\nmul.wide.u32 t, a3, b1;\nmul.wide.u32 odd_t, a3, b0;\nadd.cc.u64 e2, e2, t;\nmul.wide.u32 t, a3, b3;\naddc.cc.u64 e3, e3, t;\nmul.wide.u32 t, a3, b5;\naddc.cc.u64 e4, e4, t;\nmul.wide.u32 t, a3, b7;\naddc.u64 e5, t, 0;\nadd.cc.u64 o1, o1, odd_t;\nmul.wide.u32 odd_t, a3, b2;\naddc.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a3, b4;\naddc.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a3, b6;\naddc.cc.u64 o4, o4, odd_t;\naddc.u32 odd_cy, 0, 0;\nmul.wide.u32 t, a4, b0;\nadd.cc.u64 e2, e2, t;\nmul.wide.u32 t, a4, b2;\naddc.cc.u64 e3, e3, t;\nmul.wide.u32 t, a4, b4;\naddc.cc.u64 e4, e4, t;\nmul.wide.u32 t, a4, b6;\naddc.cc.u64 e5, e5, t;\naddc.u32 cy, 0, 0;\nmul.wide.u32 odd_t, a4, b1;\nmov.b64 odd_lc, {odd_cy, cy};\nadd.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a4, b3;\naddc.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a4, b5;\naddc.cc.u64 o4, o4, odd_t;\nmul.wide.u32 odd_t, a4, b7;\naddc.u64 o5, odd_t, odd_lc;\nmul.wide.u32 t, a5, b1;\nmul.wide.u32 odd_t, a5, b0;\nadd.cc.u64 e3, e3, t;\nmul.wide.u32 t, a5, b3;\naddc.cc.u64 e4, e4, t;\nmul.wide.u32 t, a5, b5;\naddc.cc.u64 e5, e5, t;\nmul.wide.u32 t, a5, b7;\naddc.u64 e6, t, 0;\nadd.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a5, b2;\naddc.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a5, b4;\naddc.cc.u64 o4, o4, odd_t;\nmul.wide.u32 odd_t, a5, b6;\naddc.cc.u64 o5, o5, odd_t;\naddc.u32 odd_cy, 0, 0;\nmul.wide.u32 t, a6, b0;\nadd.cc.u64 e3, e3, t;\nmul.wide.u32 t, a6, b2;\naddc.cc.u64 e4, e4, t;\nmul.wide.u32 t, a6, b4;\naddc.cc.u64 e5, e5, t;\nmul.wide.u32 t, a6, b6;\naddc.cc.u64 e6, e6, t;\naddc.u32 cy, 0, 0;\nmul.wide.u32 odd_t, a6, b1;\nmov.b64 odd_lc, {odd_cy, cy};\nadd.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a6, b3;\naddc.cc.u64 o4, o4, odd_t;\nmul.wide.u32 odd_t, a6, b5;\naddc.cc.u64 o5, o5, odd_t;\nmul.wide.u32 odd_t, a6, b7;\naddc.u64 o6, odd_t, odd_lc;\nmul.wide.u32 t, a7, b1;\nmul.wide.u32 odd_t, a7, b0;\nadd.cc.u64 e4, e4, t;\nmul.wide.u32 t, a7, b3;\naddc.cc.u64 e5, e5, t;\nmul.wide.u32 t, a7, b5;\naddc.cc.u64 e6, e6, t;\nmul.wide.u32 t, a7, b7;\naddc.u64 e7, t, 0;\nadd.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a7, b2;\naddc.cc.u64 o4, o4, odd_t;\nmul.wide.u32 odd_t, a7, b4;\naddc.cc.u64 o5, o5, odd_t;\nmul.wide.u32 odd_t, a7, b6;\naddc.cc.u64 o6, o6, odd_t;\naddc.u32 o15, 0, 0;\nmov.b64 {x0,x1}, e0;\n\tmov.b64 {x2,x3}, e1;\n\tmov.b64 {x4,x5}, e2;\n\tmov.b64 {x6,x7}, e3;\n\tmov.b64 {x8,x9}, e4;\n\tmov.b64 {x10,x11}, e5;\n\tmov.b64 {x12,x13}, e6;\n\tmov.b64 {x14,x15}, e7;\n\tmov.b64 {y1,y2}, o0;\n\tmov.b64 {y3,y4}, o1;\n\tmov.b64 {y5,y6}, o2;\n\tmov.b64 {y7,y8}, o3;\n\tmov.b64 {y9,y10}, o4;\n\tmov.b64 {y11,y12}, o5;\n\tmov.b64 {y13,y14}, o6;\n\tadd.cc.u32 x1, x1, y1;\n\taddc.cc.u32 x2, x2, y2;\n\taddc.cc.u32 x3, x3, y3;\n\taddc.cc.u32 x4, x4, y4;\n\taddc.cc.u32 x5, x5, y5;\n\taddc.cc.u32 x6, x6, y6;\n\taddc.cc.u32 x7, x7, y7;\n\taddc.cc.u32 x8, x8, y8;\n\taddc.cc.u32 x9, x9, y9;\n\taddc.cc.u32 x10, x10, y10;\n\taddc.cc.u32 x11, x11, y11;\n\taddc.cc.u32 x12, x12, y12;\n\taddc.cc.u32 x13, x13, y13;\n\taddc.cc.u32 x14, x14, y14;\n\taddc.u32 x15, x15, o15;\n\t.reg .u64 r0,r1,r2,r3,h0,h1,h2,h3,f0,f1,f2,f3,g0,g1,g2,g3;\n\t.reg .u32 f8,g8,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,w0,w1,w2,w3,w4,w5,w6,w7,m0,m1,m2;\n\tmov.b64 r0, {x0,x1}; mov.b64 r1, {x2,x3}; mov.b64 r2, {x4,x5}; mov.b64 r3, {x6,x7};\n\tmov.b64 h0, {x8,x9}; mov.b64 h1, {x10,x11}; mov.b64 h2, {x12,x13}; mov.b64 h3, {x14,x15};\n\tmul.wide.u32 t, x8, 977;  add.cc.u64  f0, r0, t;\n\tmul.wide.u32 t, x10, 977; addc.cc.u64 f1, r1, t;\n\tmul.wide.u32 t, x12, 977; addc.cc.u64 f2, r2, t;\n\tmul.wide.u32 t, x14, 977; addc.cc.u64 f3, r3, t;\n\taddc.u32 f8, 0, 0;\n\tmul.wide.u32 t, x9, 977;  add.cc.u64  g0, h0, t;\n\tmul.wide.u32 t, x11, 977; addc.cc.u64 g1, h1, t;\n\tmul.wide.u32 t, x13, 977; addc.cc.u64 g2, h2, t;\n\tmul.wide.u32 t, x15, 977; addc.cc.u64 g3, h3, t;\n\taddc.u32 g8, 0, 0;\n\tmov.b64 {z0,z1}, f0;\n\tmov.b64 {z2,z3}, f1;\n\tmov.b64 {z4,z5}, f2;\n\tmov.b64 {z6,z7}, f3;\n\tmov.b64 {w0,w1}, g0;\n\tmov.b64 {w2,w3}, g1;\n\tmov.b64 {w4,w5}, g2;\n\tmov.b64 {w6,w7}, g3;\n\tadd.cc.u32  z1, z1, w0;\n\taddc.cc.u32 z2, z2, w1;\n\taddc.cc.u32 z3, z3, w2;\n\taddc.cc.u32 z4, z4, w3;\n\taddc.cc.u32 z5, z5, w4;\n\taddc.cc.u32 z6, z6, w5;\n\taddc.cc.u32 z7, z7, w6;\n\taddc.cc.u32 z8, f8, w7;\n\taddc.u32    z9, g8, 0;\n\tmul.wide.u32 t, z8, 977; mov.b64 {m0,m1}, t;\n\tmad.lo.u32 m1, z9, 977, m1;\n\tadd.cc.u32 m1, m1, z8;\n\taddc.u32 m2, z9, 0;\n\tadd.cc.u32 z0, z0, m0; addc.cc.u32 z1, z1, m1; addc.cc.u32 z2, z2, m2;\n\taddc.cc.u32 z3, z3, 0;\n\taddc.cc.u32 z4, z4, 0;\n\taddc.cc.u32 z5, z5, 0;\n\taddc.cc.u32 z6, z6, 0;\n\taddc.cc.u32 z7, z7, 0;\n    .reg .u32 cf, k0, k1, v0, v1, v2, v3, v4, v5, v6, v7, borrow;\n    .reg .pred take;\n    addc.u32 cf, 0, 0;\n    mul.lo.u32 k0, cf, 977;\n    add.cc.u32 z0, z0, k0;\n    addc.cc.u32 z1, z1, cf;\n    addc.u32 z2, z2, 0;\nmov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.b64 %2, {z4,z5}; mov.b64 %3, {z6,z7};\n\t}\n"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
          "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    out[0]=r0;out[1]=r1;out[2]=r2;out[3]=r3;out[4]=0;
}

__device__ __forceinline__ void norm(uint64_t *r) {
    if((r[1]&r[2]&r[3])==UINT64_MAX && r[0]>=UINT64_C(0xfffffffefffffc2f)) {
        r[0]-=UINT64_C(0xfffffffefffffc2f);r[1]=r[2]=r[3]=0;
    }
}
__device__ __forceinline__ void cp(uint64_t out[5],const uint64_t in[5]) {
    for(int j=0;j<5;++j)out[j]=in[j];
}
__device__ __forceinline__ void sq(uint64_t out[5],int n) {
    #pragma unroll 1
    for(int j=0;j<n;++j)qsb_field_mul(out,out,out);
}
struct FermatInverse {
    __device__ void operator()(uint64_t out[4],const uint64_t in[4]) const {
        uint64_t x[5]={in[0],in[1],in[2],in[3],0};norm(x);
        uint64_t x2[5],x3[5],x22[5],x44[5],t[5],u[5];
        cp(x2,x);sq(x2,1);qsb_field_mul(x2,x2,x); // 3
        cp(x3,x2);sq(x3,1);qsb_field_mul(x3,x3,x); // 7
        cp(t,x3);sq(t,3);qsb_field_mul(t,t,x3); // 63
        sq(t,3);qsb_field_mul(t,t,x3); // 511
        sq(t,2);qsb_field_mul(t,t,x2); // 2047
        cp(u,x3);sq(u,1);qsb_field_mul(u,u,x3); // 21
        sq(u,1);qsb_field_mul(x2,u,x2); // 45, overwrites dead x^3
        cp(x22,t);sq(x22,11);qsb_field_mul(x22,x22,t); // 2^22-1
        cp(x44,x22);sq(x44,22);qsb_field_mul(x44,x44,x22); // 2^44-1
        cp(t,x44);sq(t,44);qsb_field_mul(t,t,x44); // 2^88-1
        cp(u,t);sq(t,88);qsb_field_mul(t,t,u); // 2^176-1
        sq(t,44);qsb_field_mul(t,t,x44); // 2^220-1
        sq(t,3);qsb_field_mul(t,t,x3); // 2^223-1
        sq(t,23);qsb_field_mul(t,t,x22); // 2^246-2^22-1
        sq(t,10);qsb_field_mul(t,t,x2); // 2^256-2^32-979 = p-2
        norm(t);for(int j=0;j<4;++j)out[j]=t[j];
    }
};
__global__ __launch_bounds__(128,QSB_SERVICE_BLOCKS) void qsb_inverse_service_fermat_compile_probe(
    qsb_inverse_service::Slot *slots,unsigned workers,unsigned services) {
    const unsigned worker=blockIdx.x-services;
    if(blockIdx.x<services) {
        qsb_inverse_service::service(slots,workers,blockIdx.x*blockDim.x+threadIdx.x,FermatInverse{});
        return;
    }
    if(worker<workers && threadIdx.x==0) {
        const uint64_t root[4]={2,3,4,5};uint64_t inverse[4];
        qsb_inverse_service::worker_publish(slots[worker],1,root);
        qsb_inverse_service::worker_wait(slots[worker],1,inverse);
        qsb_inverse_service::worker_stop(slots[worker]);
    }
}
