// SPDX-License-Identifier: GPL-3.0-only
// R16-W5: m=32, 512 denominator owners, one 32 KiB arena.
// Coordinate-major queues avoid a stride-16 shared-memory access.
// Queue epochs and product-tree epochs never overlap: all reads precede
// the full CTA barrier before the first write of the next interpretation.
#pragma once

__device__ __forceinline__ bool w5_zero(const uint64_t *a) {
    return (a[0]|a[1]|a[2]|a[3])==0;
}
__device__ __forceinline__ void w5_one(uint64_t *a) {
    a[0]=1;a[1]=a[2]=a[3]=0;
}
__device__ __forceinline__ void w5_clear(uint64_t *a) {
    a[0]=a[1]=a[2]=a[3]=0;
}
__device__ __forceinline__ void w5_mul(uint64_t *r,uint64_t *a,uint64_t *b) {
    uint64_t t[5];qsb_field_mul(t,a,b);qsb_field_normalize(t);Load256(r,t);
}
#include "W5Square.cuh"
__device__ __forceinline__ void w5_sqr(uint64_t *r,uint64_t *a) {
    w5_square_exact(r,a);qsb_field_normalize(r);
}
__device__ __forceinline__ void w5_get(uint64_t *x,uint64_t *y,
    const uint64_t arena[4][1024],int point) {
    #pragma unroll
    for(int k=0;k<4;k++){x[k]=arena[k][point];y[k]=arena[k][512+point];}
}
__device__ __forceinline__ void w5_put(uint64_t arena[4][1024],int point,
    const uint64_t *x,const uint64_t *y) {
    #pragma unroll
    for(int k=0;k<4;k++){arena[k][point]=x[k];arena[k][512+point]=y[k];}
}
// kind 0: regular slope; 1: copy anchor; 2: infinity. Complete affine law.
__device__ __forceinline__ unsigned w5_add_prepare(uint64_t *x,uint64_t *y,
    uint64_t *sum,uint64_t *num,uint64_t *den,uint64_t *bx,uint64_t *by) {
    if(w5_zero(x)&&w5_zero(y)){Load256(x,bx);Load256(y,by);w5_one(den);return 1;}
    if(w5_zero(bx)&&w5_zero(by)){w5_one(den);return 1;}
    _ModSub256(den,bx,x);
    _ModAdd256(sum,x,bx);
    _ModSub256(num,by,y);
    if(w5_zero(den)) {
        if(!w5_zero(num)||w5_zero(y)){w5_one(den);return 2;}
        w5_sqr(num,x);_ModAdd256(den,num,num);_ModAdd256(num,den,num);
        _ModAdd256(den,y,y);
    }
    return 0;
}
__device__ __forceinline__ void w5_add_finish(unsigned kind,uint64_t *x,
    uint64_t *y,uint64_t *sum,uint64_t *num,uint64_t *inv) {
    if(kind==2){w5_clear(x);w5_clear(y);return;}
    if(kind==1)return;
    uint64_t nx[4],t[4];
    w5_mul(num,num,inv);w5_sqr(nx,num);_ModSub256(nx,sum);
    _ModSub256(t,x,nx);w5_mul(t,num,t);_ModSub256(y,t,y);Load256(x,nx);
}
__device__ __forceinline__ unsigned w5_recover(uint64_t *x,uint64_t *y,
    uint64_t *inv,uint64_t *xp,uint64_t *xm) {
    uint64_t a[4]={pin_u2rx_words[0],pin_u2rx_words[1],pin_u2rx_words[2],pin_u2rx_words[3]};
    uint64_t b[4]={pin_u2ry_words[0],pin_u2ry_words[1],pin_u2ry_words[2],pin_u2ry_words[3]};
    uint64_t c[4]={pin_recovery_c[0],pin_recovery_c[1],pin_recovery_c[2],pin_recovery_c[3]};
    uint64_t l[4],m[4],sum[4],t[4],v[4];
    if(w5_zero(x)&&w5_zero(y)) {
        Load256(xp,a);Load256(xm,a);
        return 12u|(unsigned)(b[0]&1u)|((unsigned)((b[0]&1u)^1u)<<1);
    }
    _ModSub256(t,a,x);
    if(w5_zero(t)) {
        // P=R or P=-R: exactly one recovered point is infinity.
        w5_sqr(xp,c);_ModSub256(xp,a);_ModSub256(xp,a);
        _ModSub256(t,a,xp);w5_mul(v,c,t);_ModSub256(v,b);
        _ModSub256(t,y,b);
        if(w5_zero(t)){w5_clear(xm);return 4u|(unsigned)(v[0]&1u);}
        Load256(xm,xp);w5_clear(xp);return 8u|((unsigned)((v[0]&1u)^1u)<<1);
    }
    _ModSub256(l,b,y);w5_mul(l,l,inv);
    _ModAdd256(m,b,y);w5_mul(m,m,inv);_ModAdd256(sum,l,m);
    _ModSub256(t,l,c);w5_mul(xp,sum,t);_ModAdd256(xp,xp,a);
    _ModSub256(t,m,c);w5_mul(xm,sum,t);_ModAdd256(xm,xm,a);
    _ModSub256(t,a,xp);w5_mul(v,l,t);_ModSub256(v,b);
    unsigned status=12u|(unsigned)(v[0]&1u);
    _ModSub256(t,a,xm);w5_mul(v,m,t);_ModSub256(v,b,v);
    return status|((unsigned)(v[0]&1u)<<1);
}

// Register-first window issue adapts fkiene PR 522 to each parallel pair.
// The second window is decoded after the first lookup has been issued.
__device__ __forceinline__ uint32_t w5_decode(const uint64_t *M,unsigned c,int negative) {
    const unsigned pos=16*c+1,j=pos/64,sh=pos%64;
    uint32_t f=(uint32_t)(M[j]>>sh)&0xffffu;
    if(j<3 && sh==49)f|=(uint32_t)(M[j+1]&1u)<<15;
    int tm=c==15?-negative:(int)(f>>15)-1;
    return ((f^(uint32_t)tm)&0x7fffu)|((uint32_t)(tm<0)<<31);
}

__global__ void __launch_bounds__(512,1) w5_r16(const uint8_t *table,
    ulonglong2 *saved,int batch_size) {
    __shared__ uint64_t arena[4][1024]; // ready points <-> heap tree
    __shared__ uint64_t scalars[4][32];
    __shared__ int negatives[32];
    const unsigned tid=threadIdx.x,lane=tid&31u;
    const int base=(int)blockIdx.x*(32*QSB_W5_COHORTS);
    const int left=batch_size-base;
    const int cohorts=left<32*QSB_W5_COHORTS ? (left+31)/32 : QSB_W5_COHORTS;
    const unsigned stage=tid<256?0u:tid<384?1u:tid<448?2u:tid<480?3u:4u;
    for(int wave=0;wave<cohorts+4;wave++) {
        if(tid<32 && wave<cohorts) {
            const int idx=base+32*wave+(int)lane;
            uint64_t k[4]={0,0,0,0};
            if(idx<batch_size){
                ulonglong2 lo=qsb_ld_v2(saved+idx),hi=qsb_ld_v2(saved+(size_t)batch_size+idx);
                k[0]=lo.x;k[1]=lo.y;k[2]=hi.x;k[3]=hi.y;
            }
            uint64_t M[4];int negative;qsb_signed_recode_setup(k,M,&negative);
            #pragma unroll
            for(int j=0;j<4;j++)scalars[j][lane]=M[j];
            negatives[lane]=negative;
        }
        __syncthreads();
        const int cohort=wave-(int)stage;
        const int idx=base+32*cohort+(int)lane;
        const bool active=cohort>=0 && cohort<cohorts && idx<batch_size;
        uint64_t x[4],y[4],sum[4],num[4],den[5];unsigned kind=2;
        w5_clear(x);w5_clear(y);w5_clear(sum);w5_clear(num);w5_one(den);den[4]=0;
        if(active) {
            if(stage<4) {
                uint64_t bx[4],by[4];
                if(stage==0) {
                    uint64_t M[4];
                    #pragma unroll
                    for(int j=0;j<4;j++)M[j]=scalars[j][lane];
                    const unsigned c=(tid>>5)*2;
                    const int negative=negatives[lane];
                    uint32_t ca=w5_decode(M,c,negative);
                    gt_load_signed_flat_m(table,gt_offset(c),ca&0x7fffu,0ULL-(ca>>31),x,y);
                    uint32_t cb=w5_decode(M,c+1,negative);
                    gt_load_signed_flat_m(table,gt_offset(c+1),cb&0x7fffu,0ULL-(cb>>31),bx,by);
                } else {
                    unsigned first=stage==1?((tid-256)>>5)*64+lane:
                        stage==2?256+((tid-384)>>5)*64+lane:384+lane;
                    w5_get(x,y,arena,first);w5_get(bx,by,arena,first+32);
                }
                kind=w5_add_prepare(x,y,sum,num,den,bx,by);
            } else {
                w5_get(x,y,arena,448+lane);
                uint64_t a[4]={pin_u2rx_words[0],pin_u2rx_words[1],pin_u2rx_words[2],pin_u2rx_words[3]};
                _ModSub256(den,a,x);
                if(w5_zero(den)||(w5_zero(x)&&w5_zero(y)))w5_one(den);
            }
        }
        // All previous queue reads complete before replacing it with leaves.
        __syncthreads();
        #pragma unroll
        for(int k=0;k<4;k++)arena[k][512+tid]=den[k];
        __syncthreads();
        #pragma unroll 1
        for(unsigned width=256;width; width>>=1) {
            if(tid<width) {
                uint64_t a[5],b[5],v[5];
                #pragma unroll
                for(int k=0;k<4;k++){a[k]=arena[k][2*(width+tid)];b[k]=arena[k][2*(width+tid)+1];}
                a[4]=b[4]=0;qsb_field_mul(v,a,b);
                #pragma unroll
                for(int k=0;k<4;k++)arena[k][width+tid]=v[k];
            }
            __syncthreads();
        }
        if(tid==0) {
            uint64_t r[5];
            #pragma unroll
            for(int k=0;k<4;k++)r[k]=arena[k][1];
            r[4]=0;qsb_field_normalize(r);_ModInv(r);
            #pragma unroll
            for(int k=0;k<4;k++)arena[k][1]=r[k];
        }
        __syncthreads();
        // One owner reads BOTH siblings before replacing either with inverses.
        #pragma unroll 1
        for(unsigned width=1;width<256;width<<=1) {
            if(tid<width) {
                uint64_t a[5],b[5],r[5],li[5],ri[5];
                #pragma unroll
                for(int k=0;k<4;k++){
                    r[k]=arena[k][width+tid];a[k]=arena[k][2*(width+tid)];b[k]=arena[k][2*(width+tid)+1];
                }
                a[4]=b[4]=r[4]=0;qsb_field_mul(li,r,b);qsb_field_mul(ri,r,a);
                #pragma unroll
                for(int k=0;k<4;k++){arena[k][2*(width+tid)]=li[k];arena[k][2*(width+tid)+1]=ri[k];}
            }
            __syncthreads();
        }
        // Final leaf expansion is register-only: no 512-element inverse store.
        {
            uint64_t parent[5],sibling[5];
            #pragma unroll
            for(int k=0;k<4;k++){
                parent[k]=arena[k][256+tid/2];sibling[k]=arena[k][512+(tid^1u)];
            }
            parent[4]=sibling[4]=0;qsb_field_mul(den,parent,sibling);
        }
        qsb_field_normalize(den);
        // Every final leaf inverse is in registers before queue reuse.
        __syncthreads();
        if(stage<4) {
            w5_add_finish(kind,x,y,sum,num,den);w5_put(arena,(int)tid,x,y);
        } else if(active) {
            unsigned status=w5_recover(x,y,den,sum,num);
            size_t stride=(size_t)batch_size;
            qsb_st_v2(saved+idx,sum[0],sum[1]);qsb_st_v2(saved+stride+idx,sum[2],sum[3]);
            qsb_st_v2(saved+2*stride+idx,num[0],num[1]);qsb_st_v2(saved+3*stride+idx,num[2],num[3]);
            qsb_st_v2(saved+4*stride+idx,status,0);
        }
        __syncthreads();
    }
}
