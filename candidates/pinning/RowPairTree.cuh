// Two adjacent lanes split four complete rows each of the promoted multiplier.
// All 64 word products and all carries of the 512-bit product are retained.
#pragma once
__device__ __forceinline__ void qsb_pair_partial(
    uint64_t *p,uint64_t a0,uint64_t a1,const uint64_t *b) {
    asm(
        "{\n"
        "\t.reg .u32 a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7;\n"
        "\t.reg .u64 e0,e1,e2,e3,e4,e5,e6,e7,o0,o1,o2,o3,o4,o5,o6,t,lc;\n"
        "\t.reg .u32 cy,o15;\n"
        "\t.reg .u32 x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;\n"
        "\t.reg .u32 y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14;\n"
        "\tmov.b64 {a0,a1}, %6;\n"
        "\tmov.b64 {a2,a3}, %7;\n"
        "\t\n"
        "\t\n"
        "\tmov.b64 {b0,b1}, %8;\n"
        "\tmov.b64 {b2,b3}, %9;\n"
        "\tmov.b64 {b4,b5}, %10;\n"
        "\tmov.b64 {b6,b7}, %11;\n"
        "\t.reg .u64 odd_t,odd_lc; .reg .u32 odd_cy;\n"
        "mul.wide.u32 e0, a0, b0;\n"
        "mul.wide.u32 o0, a0, b1;\n"
        "mul.wide.u32 e1, a0, b2;\n"
        "mul.wide.u32 o1, a0, b3;\n"
        "mul.wide.u32 e2, a0, b4;\n"
        "mul.wide.u32 o2, a0, b5;\n"
        "mul.wide.u32 e3, a0, b6;\n"
        "mul.wide.u32 o3, a0, b7;\n"
        "mul.wide.u32 t, a1, b1;\n"
        "mul.wide.u32 odd_t, a1, b0;\n"
        "add.cc.u64 e1, e1, t;\n"
        "mul.wide.u32 t, a1, b3;\n"
        "addc.cc.u64 e2, e2, t;\n"
        "mul.wide.u32 t, a1, b5;\n"
        "addc.cc.u64 e3, e3, t;\n"
        "mul.wide.u32 t, a1, b7;\n"
        "addc.u64 e4, t, 0;\n"
        "add.cc.u64 o0, o0, odd_t;\n"
        "mul.wide.u32 odd_t, a1, b2;\n"
        "addc.cc.u64 o1, o1, odd_t;\n"
        "mul.wide.u32 odd_t, a1, b4;\n"
        "addc.cc.u64 o2, o2, odd_t;\n"
        "mul.wide.u32 odd_t, a1, b6;\n"
        "addc.cc.u64 o3, o3, odd_t;\n"
        "addc.u32 odd_cy, 0, 0;\n"
        "mul.wide.u32 t, a2, b0;\n"
        "add.cc.u64 e1, e1, t;\n"
        "mul.wide.u32 t, a2, b2;\n"
        "addc.cc.u64 e2, e2, t;\n"
        "mul.wide.u32 t, a2, b4;\n"
        "addc.cc.u64 e3, e3, t;\n"
        "mul.wide.u32 t, a2, b6;\n"
        "addc.cc.u64 e4, e4, t;\n"
        "addc.u32 cy, 0, 0;\n"
        "mul.wide.u32 odd_t, a2, b1;\n"
        "mov.b64 odd_lc, {odd_cy, cy};\n"
        "add.cc.u64 o1, o1, odd_t;\n"
        "mul.wide.u32 odd_t, a2, b3;\n"
        "addc.cc.u64 o2, o2, odd_t;\n"
        "mul.wide.u32 odd_t, a2, b5;\n"
        "addc.cc.u64 o3, o3, odd_t;\n"
        "mul.wide.u32 odd_t, a2, b7;\n"
        "addc.u64 o4, odd_t, odd_lc;\n"
        "mul.wide.u32 t, a3, b1;\n"
        "mul.wide.u32 odd_t, a3, b0;\n"
        "add.cc.u64 e2, e2, t;\n"
        "mul.wide.u32 t, a3, b3;\n"
        "addc.cc.u64 e3, e3, t;\n"
        "mul.wide.u32 t, a3, b5;\n"
        "addc.cc.u64 e4, e4, t;\n"
        "mul.wide.u32 t, a3, b7;\n"
        "addc.u64 e5, t, 0;\n"
        "add.cc.u64 o1, o1, odd_t;\n"
        "mul.wide.u32 odd_t, a3, b2;\n"
        "addc.cc.u64 o2, o2, odd_t;\n"
        "mul.wide.u32 odd_t, a3, b4;\n"
        "addc.cc.u64 o3, o3, odd_t;\n"
        "mul.wide.u32 odd_t, a3, b6;\n"
        "addc.cc.u64 o4, o4, odd_t;\n"
        "addc.u32 odd_cy, 0, 0;\n"
        "mov.b64 {x0,x1},e0;\n"
        "mov.b64 {x2,x3},e1;\n"
        "mov.b64 {x4,x5},e2;\n"
        "mov.b64 {x6,x7},e3;\n"
        "mov.b64 {x8,x9},e4;\n"
        "mov.b64 {x10,x11},e5;\n"
        "mov.b64 {y1,y2},o0;\n"
        "mov.b64 {y3,y4},o1;\n"
        "mov.b64 {y5,y6},o2;\n"
        "mov.b64 {y7,y8},o3;\n"
        "mov.b64 {y9,y10},o4;\n"
        "add.cc.u32 x1,x1,y1;\n"
        "addc.cc.u32 x2,x2,y2;\n"
        "addc.cc.u32 x3,x3,y3;\n"
        "addc.cc.u32 x4,x4,y4;\n"
        "addc.cc.u32 x5,x5,y5;\n"
        "addc.cc.u32 x6,x6,y6;\n"
        "addc.cc.u32 x7,x7,y7;\n"
        "addc.cc.u32 x8,x8,y8;\n"
        "addc.cc.u32 x9,x9,y9;\n"
        "addc.cc.u32 x10,x10,y10;\n"
        "addc.u32 x11,x11,odd_cy;\n"
        "mov.b64 %0,{x0,x1};\n"
        "mov.b64 %1,{x2,x3};\n"
        "mov.b64 %2,{x4,x5};\n"
        "mov.b64 %3,{x6,x7};\n"
        "mov.b64 %4,{x8,x9};\n"
        "mov.b64 %5,{x10,x11};\n"
        "}\n"
        : "=l"(p[0]),"=l"(p[1]),"=l"(p[2]),"=l"(p[3]),"=l"(p[4]),"=l"(p[5])
        : "l"(a0),"l"(a1),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
}
__device__ __forceinline__ unsigned qsb_pair_add4(
    const unsigned *a,const unsigned *b,unsigned *out) {
    unsigned carry;
    asm(
        "{\n"
        ".reg .u32 x0,x1,x2,x3,c;\n"
        "add.cc.u32 x0,%5,%9;\n"
        "addc.cc.u32 x1,%6,%10;\n"
        "addc.cc.u32 x2,%7,%11;\n"
        "addc.cc.u32 x3,%8,%12;\n"
        "addc.u32 c,0,0;\n"
        "mov.u32 %0,x0; mov.u32 %1,x1; mov.u32 %2,x2; mov.u32 %3,x3; mov.u32 %4,c;\n"
        "}\n"
        : "=r"(out[0]),"=r"(out[1]),"=r"(out[2]),"=r"(out[3]),"=r"(carry)
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(b[2]),"r"(b[3]));
    return carry;
}
__device__ __forceinline__ unsigned qsb_pair_inc4(unsigned *out,unsigned seed) {
    unsigned carry;
    asm(
        "{\n"
        ".reg .u32 x0,x1,x2,x3,c;\n"
        "add.cc.u32 x0,%5,%9;\n"
        "addc.cc.u32 x1,%6,0;\n"
        "addc.cc.u32 x2,%7,0;\n"
        "addc.cc.u32 x3,%8,0;\n"
        "addc.u32 c,0,0;\n"
        "mov.u32 %0,x0; mov.u32 %1,x1; mov.u32 %2,x2; mov.u32 %3,x3; mov.u32 %4,c;\n"
        "}\n"
        : "=r"(out[0]),"=r"(out[1]),"=r"(out[2]),"=r"(out[3]),"=r"(carry)
        : "r"(out[0]),"r"(out[1]),"r"(out[2]),"r"(out[3]),"r"(seed));
    return carry;
}
__device__ __forceinline__ void qsb_pair_finish8(unsigned *out,unsigned seed,unsigned middle) {
    asm(
        "{\n"
        ".reg .u32 x0,x1,x2,x3,x4,x5,x6,x7;\n"
        "add.cc.u32 x0,%8,%16;\n"
        "addc.cc.u32 x1,%9,0;\n"
        "addc.cc.u32 x2,%10,0;\n"
        "addc.cc.u32 x3,%11,0;\n"
        "addc.cc.u32 x4,%12,%17;\n"
        "addc.cc.u32 x5,%13,0;\n"
        "addc.cc.u32 x6,%14,0;\n"
        "addc.u32 x7,%15,0;\n"
        "mov.u32 %0,x0; mov.u32 %1,x1; mov.u32 %2,x2; mov.u32 %3,x3;\n"
        "mov.u32 %4,x4; mov.u32 %5,x5; mov.u32 %6,x6; mov.u32 %7,x7;\n"
        "}\n"
        : "=r"(out[0]),"=r"(out[1]),"=r"(out[2]),"=r"(out[3]),
          "=r"(out[4]),"=r"(out[5]),"=r"(out[6]),"=r"(out[7])
        : "r"(out[0]),"r"(out[1]),"r"(out[2]),"r"(out[3]),
          "r"(out[4]),"r"(out[5]),"r"(out[6]),"r"(out[7]),"r"(seed),"r"(middle));
}
__device__ __forceinline__ void qsb_tree_pair_product(
    const uint64_t *ap,int as,const uint64_t *bp,int bs,uint64_t *op,int os,unsigned mask) {
    const unsigned lane=threadIdx.x&1u;
    unsigned d[8];
    {
        uint64_t b[4],p[6];
        #pragma unroll
        for(int k=0;k<4;k++) b[k]=bp[k*bs];
        qsb_pair_partial(p,ap[(2*lane)*as],ap[(2*lane+1)*as],b);
        unsigned x[12],external[4],sum[4];
        #pragma unroll
        for(int k=0;k<6;k++) {x[2*k]=(unsigned)p[k];x[2*k+1]=(unsigned)(p[k]>>32);}
        #pragma unroll
        for(int k=0;k<4;k++) external[k]=__shfl_xor_sync(mask,lane?x[k]:x[8+k],1,2);
        unsigned carry=qsb_pair_add4(x+4,external,sum);
        unsigned incoming=__shfl_up_sync(mask,carry,1,2);
        #pragma unroll
        for(int k=0;k<4;k++) {d[k]=lane?sum[k]:x[k];d[4+k]=lane?x[8+k]:sum[k];}
        qsb_pair_finish8(d,lane?incoming:0,lane?carry:0);
    }
    unsigned lo[4],hi[4];
    #pragma unroll
    for(int k=0;k<4;k++) {
        unsigned other=__shfl_xor_sync(mask,lane?d[k]:d[4+k],1,2);
        lo[k]=lane?other:d[k];hi[k]=lane?d[4+k]:other;
    }
    // First fold is L + (2^32+977)*H with every carry preserved.
    // Its high word and the low96 second fold match promoted C31 exactly.
    unsigned prev=__shfl_up_sync(mask,hi[3],1,2);
    unsigned fl[4],fh[4],pfh[4],out[4];
    #pragma unroll
    for(int k=0;k<4;k++) {
        uint64_t f=(uint64_t)lo[k]+977ULL*hi[k]+(k?hi[k-1]:(lane?prev:0));
        fl[k]=(unsigned)f;fh[k]=(unsigned)(f>>32);
    }
    prev=__shfl_up_sync(mask,fh[3],1,2);
    #pragma unroll
    for(int k=0;k<4;k++) pfh[k]=k?fh[k-1]:(lane?prev:0);
    unsigned carry=qsb_pair_add4(fl,pfh,out);
    unsigned incoming=__shfl_up_sync(mask,carry,1,2);
    carry+=qsb_pair_inc4(out,lane?incoming:0);
    unsigned h=hi[3]+fh[3]+carry;
    h=__shfl_sync(mask,h,1,2);
    if(lane==0) {
        uint64_t t0=(uint64_t)out[0]+977ULL*h;
        uint64_t t1=(uint64_t)out[1]+h+(t0>>32);
        out[0]=(unsigned)t0;out[1]=(unsigned)t1;out[2]+=(unsigned)(t1>>32);
    }
    #pragma unroll
    for(int k=0;k<2;k++) op[(2*lane+k)*os]=(uint64_t)out[2*k]|((uint64_t)out[2*k+1]<<32);
}
template<int N> __device__ __forceinline__ void qsb_tree_pair_up(
    uint64_t (*products)[2*N],int offset,int count) {
    unsigned tid=threadIdx.x;int half=count>>1;
    if(tid<32) {
        unsigned mask=__ballot_sync(0xffffffffu,tid<(unsigned)(half*2));
        if(tid<(unsigned)(half*2)) {
            int job=tid/2;
            qsb_tree_pair_product(&products[0][offset+job],2*N,
                &products[0][offset+half+job],2*N,&products[0][offset+count+job],2*N,mask);
        }
    }
}
template<int N> __device__ __forceinline__ void qsb_tree_pair_down(
    uint64_t (*products)[2*N],uint64_t (*excluded)[N],int offset,int count) {
    unsigned tid=threadIdx.x;int half=count>>1;
    if(tid<32) {
        unsigned mask=__ballot_sync(0xffffffffu,tid<(unsigned)(count*2));
        if(tid<(unsigned)(count*2)) {
            int job=tid/2;
            qsb_tree_pair_product(&excluded[0][offset+count-N+(job&(half-1))],N,
                &products[0][offset+(job^half)],2*N,&excluded[0][offset-N+job],N,mask);
        }
    }
}
