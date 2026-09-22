// SPDX-License-Identifier: GPL-3.0-only
// Experimental CTA-local affine comb. All lanes participate in every inverse.
#pragma once
#ifndef QSB_PIN_AFFINE
#define QSB_PIN_AFFINE 1
#endif

#if QSB_PIN_AFFINE
#ifndef QSB_AFFINE_WARP_SYNC
#define QSB_AFFINE_WARP_SYNC 1
#endif
#ifndef QSB_AFFINE_QUAD
#define QSB_AFFINE_QUAD 1
#endif
#if QSB_AFFINE_QUAD && !defined(QSB_AFFINE_HOST_ORACLE)
#include "hm41_quad_inverse.cuh"
#endif
__device__ __forceinline__ bool qsb_affine_zero(const uint64_t *a) {
    return (a[0]|a[1]|a[2]|a[3])==0;
}
// Canonical, full-carry subtraction. This path does not use C31 shortcuts.
__device__ __forceinline__ void qsb_affine_sub(uint64_t *out,const uint64_t *a,const uint64_t *b) {
#ifdef __CUDA_ARCH__

    uint64_t r0,r1,r2,r3;
    asm("{\n.reg .u64 t0,t1,t2,t3,t4,s0,s1,s2,s3,d0,d1,d2,d3,d4,k;\n.reg .pred choose;\nsub.cc.u64 t0,%4,%8;\nsubc.cc.u64 t1,%5,%9; subc.cc.u64 t2,%6,%10; subc.cc.u64 t3,%7,%11;\nsubc.u64 k,0,0; and.b64 k,k,0x1000003D1;\nsub.cc.u64 t0,t0,k; subc.cc.u64 t1,t1,0; subc.cc.u64 t2,t2,0;\nsubc.u64 t3,t3,0;\nmov.u64 %0,t0; mov.u64 %1,t1; mov.u64 %2,t2; mov.u64 %3,t3;\n}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    out[0]=r0;out[1]=r1;out[2]=r2;out[3]=r3;

#else

    uint64_t borrow=0;
    #pragma unroll
    for(unsigned k=0;k<4;k++) {
        const __uint128_t d=(__uint128_t)a[k]-b[k]-borrow;
        out[k]=(uint64_t)d;borrow=(uint64_t)(d>>64)&1u;
    }
    const uint64_t correction=borrow*0x1000003d1ULL;
    __uint128_t d=(__uint128_t)out[0]-correction;out[0]=(uint64_t)d;
    borrow=(uint64_t)(d>>64)&1u;
    #pragma unroll
    for(unsigned k=1;k<4;k++) {
        d=(__uint128_t)out[k]-borrow;out[k]=(uint64_t)d;borrow=(uint64_t)(d>>64)&1u;
    }

#endif
}
__device__ __forceinline__ void qsb_affine_add(uint64_t *out,const uint64_t *a,const uint64_t *b) {
#ifdef __CUDA_ARCH__

    uint64_t r0,r1,r2,r3;
    asm("{\n.reg .u64 t0,t1,t2,t3,t4,s0,s1,s2,s3,d0,d1,d2,d3,d4,k;\n.reg .pred choose;\nadd.cc.u64 s0,%4,%8;\naddc.cc.u64 s1,%5,%9; addc.cc.u64 s2,%6,%10; addc.cc.u64 s3,%7,%11;\naddc.u64 t4,0,0;\nsub.cc.u64 t0,s0,0xFFFFFFFEFFFFFC2F;\nsubc.cc.u64 t1,s1,0xFFFFFFFFFFFFFFFF;\nsubc.cc.u64 t2,s2,0xFFFFFFFFFFFFFFFF;\nsubc.cc.u64 t3,s3,0xFFFFFFFFFFFFFFFF;\nsubc.u64 t4,t4,0;\nsetp.ge.s64 choose,t4,0;\nselp.u64 %0,t0,s0,choose; selp.u64 %1,t1,s1,choose;\nselp.u64 %2,t2,s2,choose; selp.u64 %3,t3,s3,choose;\n}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    out[0]=r0;out[1]=r1;out[2]=r2;out[3]=r3;

#else

    uint64_t carry=0;
    #pragma unroll
    for(unsigned k=0;k<4;k++) {
        const __uint128_t s=(__uint128_t)a[k]+b[k]+carry;
        out[k]=(uint64_t)s;carry=(uint64_t)(s>>64);
    }
    __uint128_t s=(__uint128_t)out[0]+carry*0x1000003d1ULL;
    out[0]=(uint64_t)s;carry=(uint64_t)(s>>64);
    #pragma unroll
    for(unsigned k=1;k<4;k++) {
        s=(__uint128_t)out[k]+carry;out[k]=(uint64_t)s;carry=(uint64_t)(s>>64);
    }
    qsb_field_normalize(out);

#endif
}
__device__ __forceinline__ void qsb_affine_mul(uint64_t *out,const uint64_t *a,const uint64_t *b) {
    uint64_t tmp[5];
    qsb_field_mul(tmp,const_cast<uint64_t*>(a),const_cast<uint64_t*>(b));
    qsb_field_normalize(tmp);Load256(out,tmp);
}

// Products and inverse nodes share one packed 2*N array. One parent thread
// loads both children before replacing either, preventing sibling read races.
template<unsigned N>
__device__ __forceinline__ void qsb_affine_inverse(uint64_t *value,uint64_t (*nodes)[2*N]) {
    const unsigned tid=threadIdx.x;
    #pragma unroll
    for(unsigned k=0;k<4;k++)nodes[k][tid]=value[k];
    __syncthreads();
    unsigned offset=0;
    #pragma unroll 1
    for(unsigned count=N;count>1;count>>=1) {
        const unsigned half=count/2;
        if(tid<half) {
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(unsigned k=0;k<4;k++){a[k]=nodes[k][offset+tid];b[k]=nodes[k][offset+half+tid];}
            qsb_field_mul(out,a,b);
            #pragma unroll
            for(unsigned k=0;k<4;k++)nodes[k][offset+count+tid]=out[k];
        }
        offset+=count;
#if QSB_AFFINE_WARP_SYNC
        // Once all writers/readers are in warp zero, other warps may wait
        // at the later full-block handoff without joining each small level.
        if(count>64)__syncthreads();else __syncwarp();
#else
        __syncthreads();
#endif
    }
    if(tid<(QSB_AFFINE_QUAD?4u:1u)) {
        uint64_t root[5];
        #pragma unroll
        for(unsigned k=0;k<4;k++)root[k]=nodes[k][2*N-2];
        root[4]=0;qsb_field_normalize(root);
#if QSB_AFFINE_QUAD
        hm41_quad_inverse(root,tid);
#else
        _ModInv(root);
#endif
        if(tid==0){
            #pragma unroll
            for(unsigned k=0;k<4;k++)nodes[k][2*N-2]=root[k];
        }
    }
#if QSB_AFFINE_WARP_SYNC
    __syncwarp();
#else
    __syncthreads();
#endif
    offset=2*N-4;
    #pragma unroll 1
    for(unsigned count=2;count<=N;count<<=1) {
        const unsigned half=count/2;
        if(tid<half) {
            uint64_t a[5],b[5],inv[5],out[5];
            #pragma unroll
            for(unsigned k=0;k<4;k++){
                a[k]=nodes[k][offset+tid];b[k]=nodes[k][offset+half+tid];
                inv[k]=nodes[k][offset+count+tid];
            }
            qsb_field_mul(out,inv,b);
            #pragma unroll
            for(unsigned k=0;k<4;k++)nodes[k][offset+tid]=out[k];
            qsb_field_mul(out,inv,a);
            #pragma unroll
            for(unsigned k=0;k<4;k++)nodes[k][offset+half+tid]=out[k];
        }
#if QSB_AFFINE_WARP_SYNC
        // The next level expands the reader set to count lanes.
        if(count>=64)__syncthreads();else __syncwarp();
#else
        __syncthreads();
#endif
        if(count<N)offset-=count*2;
    }
    #pragma unroll
    for(unsigned k=0;k<4;k++)value[k]=nodes[k][tid];
    qsb_field_normalize(value);
    // Each lane reads only its own leaf. Its next write is to that same
    // leaf, then the next upward pass synchronizes all lanes before reuse.
    // The recovery handoff has its own full-block barrier.
}

// 0: ordinary/doubling add; 1: infinity + Q; 2: opposite points -> infinity.
__device__ __forceinline__ unsigned qsb_affine_prepare(
    const uint64_t *x,const uint64_t *y,bool infinity,
    const uint64_t *qx,const uint64_t *qy,uint64_t *d,uint64_t *num) {
    if(infinity){d[0]=1;d[1]=d[2]=d[3]=0;return 1;}
    qsb_affine_sub(d,qx,x);
    qsb_affine_sub(num,qy,y);
    if(!qsb_affine_zero(d))return 0;
    if(!qsb_affine_zero(num)||qsb_affine_zero(y)){
        d[0]=1;d[1]=d[2]=d[3]=0;return 2;
    }
    qsb_affine_add(d,y,y);
    uint64_t xx[4];qsb_affine_mul(xx,x,x);
    qsb_affine_add(num,xx,xx);qsb_affine_add(num,num,xx);
    return 0;
}
__device__ __forceinline__ void qsb_affine_finish(
    uint64_t *x,uint64_t *y,bool &infinity,const uint64_t *qx,const uint64_t *qy,
    const uint64_t *inverse,const uint64_t *num,unsigned action) {
    if(action==1){Load256(x,qx);Load256(y,qy);infinity=false;return;}
    if(action==2){infinity=true;return;}
    uint64_t slope[4],nx[4],ny[4];
    qsb_affine_mul(slope,num,inverse);qsb_affine_mul(nx,slope,slope);
    qsb_affine_sub(nx,nx,x);qsb_affine_sub(nx,nx,qx);
    qsb_affine_sub(ny,x,nx);qsb_affine_mul(ny,ny,slope);qsb_affine_sub(ny,ny,y);
    Load256(x,nx);Load256(y,ny);
}

__device__ __forceinline__ uint64_t *qsb_affine_arena() {
    // 7.5 KiB digits + 0.5 KiB padding + 8 KiB destructive product tree.
    // After the comb, the first 12 KiB becomes the existing recovery tree.
    __shared__ uint64_t words[16*QSB_TREE_N];return words;
}
__device__ __forceinline__ void qsb_affine_decode(const uint64_t *k) {
    uint64_t M[4];int negative;qsb_signed_recode_setup(k,M,&negative);
    volatile uint32_t *codes=(volatile uint32_t*)qsb_affine_arena();
    #pragma unroll
    for(int c=0;c<GT_CHUNKS;c++) {
        const unsigned pos=c==0?1u:17u*c+2u;
        const unsigned j=pos/64u,sh=pos%64u;
        uint64_t value=M[j]>>sh;
        if(j<3 && sh>46u)value|=M[j+1]<<(64u-sh);
        const unsigned bits=c==0?18u:17u;
        uint32_t f=(uint32_t)value&((1u<<bits)-1u);
        int32_t tm=c==GT_CHUNKS-1?-negative:(int32_t)(f>>(bits-1u))-1;
        uint32_t idx=(f^(uint32_t)tm)&((1u<<(bits-1u))-1u);
        uint32_t neg=(uint32_t)(tm<0);
        codes[(size_t)c*QSB_TREE_N+threadIdx.x]=idx|(neg<<31);
    }
}
__device__ __forceinline__ void qsb_affine_load(const uint8_t *table,unsigned c,
    unsigned base,uint64_t *x,uint64_t *y) {
    volatile uint32_t *codes=(volatile uint32_t*)qsb_affine_arena();
    uint32_t code=codes[(size_t)c*QSB_TREE_N+threadIdx.x];
    { uint32_t m32=(uint32_t)((int32_t)code>>31); gt_load_signed_flat_m(table,base,code&0x1ffffu,((uint64_t)m32<<32)|m32,x,y); }  /* P6: same mask, one SHF */
}

__device__ void qsb_fixed_affine_block(uint64_t *X,uint64_t *Y,
    uint64_t *U,uint64_t *V,const uint64_t *scalar,const uint8_t *table) {
    auto nodes=(uint64_t (*)[2*QSB_TREE_N])(qsb_affine_arena()+8*QSB_TREE_N);
    qsb_affine_decode(scalar);
    qsb_affine_load(table,0,gt_offset(0),X,Y);
#if QSB_YOFF
    // Offset table ordinates can exceed p; normalize after the exact subtraction.
    const uint64_t c[4]={0x800001e8ULL,0,0,0};
    qsb_field_normalize(Y);qsb_affine_sub(Y,Y,c);
#endif
    bool infinity=false;
    #pragma unroll 1
    for(unsigned chunk=1;chunk<GT_CHUNKS;chunk++) {
        uint64_t qx[4],qy[4],d[5],num[4];
        qsb_affine_load(table,chunk,gt_offset(chunk),qx,qy);
#if QSB_YOFF
        qsb_field_normalize(qy);qsb_affine_sub(qy,qy,c);
#endif
        const unsigned action=qsb_affine_prepare(X,Y,infinity,qx,qy,d,num);
        qsb_affine_inverse<QSB_TREE_N>(d,nodes);
        qsb_affine_finish(X,Y,infinity,qx,qy,d,num,action);
    }
    U[0]=V[0]=infinity?0:1;U[1]=U[2]=U[3]=V[1]=V[2]=V[3]=0;
#if QSB_NEG_Y_MAC
    const uint64_t zero[4]={0,0,0,0};qsb_affine_sub(Y,zero,Y);
#endif
}
#endif
