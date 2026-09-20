// SPDX-License-Identifier: GPL-3.0-only
// Seven affine pairs -> eight-point deferred XYZZ chain.
// See DESIGN and audits for coefficient bounds, checkpoints, and exceptions.
#pragma once
#include "HybridField.cuh"
#include "HybridCofactor.cuh"

// Canonical addition, retaining every carry. Inputs <p.
__device__ __forceinline__ void hy_add(uint64_t *out,const uint64_t *a,const uint64_t *b) {
    uint64_t v[4],carry=0;
    #pragma unroll
    for(int j=0;j<4;j++){
        __uint128_t t=(__uint128_t)a[j]+b[j]+carry;
        v[j]=(uint64_t)t;carry=(uint64_t)(t>>64);
    }
    uint64_t k=carry*0x1000003D1ULL;carry=0;
    #pragma unroll
    for(int j=0;j<4;j++){
        __uint128_t t=(__uint128_t)v[j]+(j==0?k:0)+carry;
        v[j]=(uint64_t)t;carry=(uint64_t)(t>>64);
    }
    Load256(out,v);qsb_field_normalize(out);
}
// Shared planes0..19 hold five prefixes;20..23 hold the signed recode residue.
__device__ __forceinline__ void hy_put(unsigned slot,const uint64_t *v) {
    volatile uint64_t *s=qsb_digit_arena();unsigned t=threadIdx.x;
    #pragma unroll
    for(int j=0;j<4;j++)s[(4*slot+j)*QSB_TREE_N+t]=v[j];
}
__device__ __forceinline__ void hy_get(unsigned slot,uint64_t *v) {
    volatile uint64_t *s=qsb_digit_arena();unsigned t=threadIdx.x;
    #pragma unroll
    for(int j=0;j<4;j++)v[j]=s[(4*slot+j)*QSB_TREE_N+t];
}
__device__ __forceinline__ uint32_t hy_code(unsigned c,int negative) {
    unsigned bits=c==0?18u:17u,bit=c==0?1u:17u*c+2u;
    unsigned limb=bit>>6,shift=bit&63u;volatile uint64_t *s=qsb_digit_arena();unsigned t=threadIdx.x;
    uint64_t z=s[(20+limb)*QSB_TREE_N+t]>>shift;
    if(shift+bits>64 && limb<3)z|=s[(21+limb)*QSB_TREE_N+t]<<(64-shift);
    uint32_t f=(uint32_t)z&((1u<<bits)-1u);
    int32_t mask=c==14?-negative:(int32_t)(f>>(bits-1))-1;
    return ((f^(uint32_t)mask)&((1u<<(bits-1))-1u))|((uint32_t)(mask<0)<<31);
}
__device__ __forceinline__ void hy_load_x(const uint8_t *table,unsigned c,int neg,uint64_t *x) {
    uint32_t code=hy_code(c,neg);
    const ulonglong2 *p=(const ulonglong2*)(table+((size_t)gt_offset(c)+(code&0x1ffffu))*64);
    ulonglong2 a=__ldg(p),b=__ldg(p+1);
    x[0]=a.x;x[1]=a.y;x[2]=b.x;x[3]=b.y;
}
__device__ __forceinline__ void hy_load(const uint8_t *table,unsigned c,int neg,uint64_t *x,uint64_t *y) {
    uint32_t code=hy_code(c,neg);
    gt_load_signed_flat_m(table,gt_offset(c),code&0x1ffffu,0ULL-(code>>31),x,y);
}
__device__ __forceinline__ void hy_den(const uint8_t *table,unsigned pair,int neg,uint64_t *d) {
    uint64_t a[4],b[4];hy_load_x(table,2*pair,neg,a);hy_load_x(table,2*pair+1,neg,b);
    hy_sub(d,b,a);
}
__device__ __forceinline__ void hy_checkpoint(const uint64_t *k,const uint8_t *table,
    bool active,int n,ulonglong2 *saved,uint64_t *roots) {
    uint64_t M[4],value[5],d[4];int negative;
    qsb_signed_recode_setup(k,M,&negative);hy_put(5,M);
    hy_den(table,0,negative,value);
    #pragma unroll 1
    for(unsigned j=1;j<7;j++){hy_den(table,j,negative,d);hy_mul(value,value,d);}
    if(!active){value[0]=1;value[1]=value[2]=value[3]=0;}
    value[4]=0;
    // All lanes must finish reading M before the cofactor overlay.
    __syncthreads();
    uint64_t (*products)[2*QSB_TREE_N]=(uint64_t (*)[2*QSB_TREE_N])qsb_digit_arena();
    uint64_t (*excluded)[QSB_TREE_N]=(uint64_t (*)[QSB_TREE_N])(qsb_digit_arena()+8*QSB_TREE_N);
    hy_cofactor_prepare<QSB_TREE_N>(value,roots,products,excluded);
    if(active){
        size_t i=(size_t)blockIdx.x*QSB_TREE_N+threadIdx.x,s=(size_t)n;
        qsb_st_v2(saved+i,k[0],k[1]);qsb_st_v2(saved+s+i,k[2],k[3]);
        qsb_st_v2(saved+2*s+i,value[0],value[1]);qsb_st_v2(saved+3*s+i,value[2],value[3]);
    }
}
// Only0 and +/-c modN can be exceptional in the reversed chain.
// Reduce the raw SHA scalar exactly; this helper executes once per candidate.
__device__ __forceinline__ unsigned hy_exception(const uint64_t *input) {
    uint64_t k[4];Load256(k,input);
    if(k[3]==GT_ORDER_N[3] && (k[2]>GT_ORDER_N[2] ||
       (k[2]==GT_ORDER_N[2] && (k[1]>GT_ORDER_N[1] || (k[1]==GT_ORDER_N[1] && k[0]>=GT_ORDER_N[0]))))){
        uint64_t borrow=0;
        #pragma unroll
        for(int j=0;j<4;j++){
            __uint128_t t=(__uint128_t)k[j]-GT_ORDER_N[j]-borrow;
            k[j]=(uint64_t)t;borrow=(uint64_t)(t>>64)&1;
        }
    }
    if((k[1]|k[2]|k[3])==0){
        if(k[0]==0)return 1;
        if(k[0]==0x4D0364141ULL)return 2;
    }
    if(k[3]==0xffffffffffffffffULL && k[2]==0xfffffffffffffffeULL &&
       k[1]==0xbaaedce6af48a03bULL && k[0]==0xbfd25e8800000000ULL)return 2;
    return 0;
}
// Out-of-line exact exceptional double; never used by ordinary scalars.
__device__ __noinline__ void hy_double(uint64_t *x,uint64_t *y) {
    uint64_t a[4],b[4],s[4],inv[5];
    hy_add(inv,y,y);inv[4]=0;_ModInv(inv);
    hy_square(a,x);hy_add(b,a,a);hy_add(a,a,b);hy_mul(s,a,inv);
    hy_square(a,s);hy_sub(a,a,x);hy_sub(a,a,x);
    hy_sub(b,x,a);hy_mul(b,s,b);hy_sub(y,b,y);Load256(x,a);
}
// prefix is d0...d(j-1), absent for j0. r is inverse(d0...dj).
__device__ __forceinline__ void hy_affine_pair(const uint8_t *table,unsigned j,int negative,
    uint64_t *r,const uint64_t *prefix,uint64_t *x,uint64_t *y) {
    uint64_t bx[4],by[4],iv[4],d[4];
    hy_load(table,2*j,negative,x,y);hy_load(table,2*j+1,negative,bx,by);
    if(j){hy_mul(iv,r,prefix);hy_sub(d,bx,x);hy_mul(r,r,d);}else{Load256(iv,r);}
    hy_sub(by,by,y);hy_mul(by,by,iv);  // slope
    hy_add(bx,bx,x);hy_square(iv,by);hy_sub(iv,iv,bx); // pair x
    hy_sub(x,x,iv);hy_mul(x,by,x);hy_sub(y,x,y);Load256(x,iv);
}
__global__ void __launch_bounds__(128,4) hy_pair_prepare(const uint8_t *table,
    ulonglong2 *saved,uint64_t *roots,int n) {
    const size_t i=(size_t)blockIdx.x*QSB_TREE_N+threadIdx.x,s=(size_t)n;
    const bool active=i<s;
    uint64_t k[4]={1,0,0,0},r[4]={1,0,0,0},I[4],prefix[4],M[4];
    if(active){
        ulonglong2 a=qsb_ld_v2(saved+i),b=qsb_ld_v2(saved+s+i);
        ulonglong2 c=qsb_ld_v2(saved+2*s+i),d=qsb_ld_v2(saved+3*s+i);
        k[0]=a.x;k[1]=a.y;k[2]=b.x;k[3]=b.y;
        r[0]=c.x;r[1]=c.y;r[2]=d.x;r[3]=d.y;
    }
    #pragma unroll
    for(int j=0;j<4;j++)I[j]=roots[4ull*blockIdx.x+j];
    hy_mul(r,r,I);
    unsigned exception=hy_exception(k);int negative;
    qsb_signed_recode_setup(k,M,&negative);hy_put(5,M);
    hy_den(table,0,negative,prefix);hy_put(0,prefix);
    #pragma unroll 1
    for(unsigned j=1;j<6;j++){
        uint64_t d[4];hy_den(table,j,negative,d);hy_mul(prefix,prefix,d);
        if(j<5)hy_put(j,prefix);
    }
    uint64_t X[4],Y[4],U[4],V[4],anchor[4],x[4],y[4];
    hy_load(table,14,negative,X,anchor);
    hy_affine_pair(table,6,negative,r,prefix,x,y);
    _PointAddXYZZ_mm(X,Y,U,V,X,anchor,x,y);
    // Seed's deferred anchor is the FIRST point, i.e. carry14.
    #pragma unroll 1
    for(int j=5;j>=0;j--){
        if(j)hy_get((unsigned)j-1,prefix);
        hy_affine_pair(table,(unsigned)j,negative,r,prefix,x,y);
        if(j==0 && exception){
            if(exception==2){hy_double(x,y);Load256(X,x);Load256(Y,y);U[0]=V[0]=1;}
            else {X[0]=X[1]=X[2]=X[3]=Y[0]=Y[1]=Y[2]=Y[3]=0;U[0]=V[0]=0;}
            U[1]=U[2]=U[3]=V[1]=V[2]=V[3]=0;
        }else{
            _PointAddXYZZT<true>(X,Y,U,V,x,y,anchor);Load256(anchor,y);
        }
    }
    if(!exception){_ModMult(x,anchor,V);_ModSub256(Y,Y,x);}
    uint64_t a[4]={pin_u2rx_words[0],pin_u2rx_words[1],pin_u2rx_words[2],pin_u2rx_words[3]},D[5];
    qsb_recovery_denominator(X,U,Y,V,a,D);
    bool usable=active && ((D[0]|D[1]|D[2]|D[3])!=0);
    if(!usable){D[0]=1;D[1]=D[2]=D[3]=D[4]=0;}
    // Its entry barrier protects prefix reuse AND all earlier reads of roots.
    qsb_packed_prepare(D,U,Y,V,usable,active,n,saved,roots);
    // Infinity is a legitimate scalar result. Reserve (vbar=1,tbar=0)
    // for P=infinity; ordinary unusable lanes remain all-zero.
    if(active && exception==1){qsb_st_v2(saved+i,1,0);}
}
