// SPDX-License-Identifier: GPL-3.0-only
// Balanced affine sum of two eight-term GLV components. Adapted from the
// locally preserved exact affine implementation 4c830add25624f0b477c55d28f965d7da1f729f1.
// Every thread reaches every block collective, including inactive tail lanes.
#pragma once
#ifndef QSB_GLV_AFFINE_LEVELS
#define QSB_GLV_AFFINE_LEVELS 1
#endif
#if QSB_GLV_AFFINE_LEVELS != 1 && QSB_GLV_AFFINE_LEVELS != 4
#error QSB_GLV_AFFINE_LEVELS must be 1 or 4
#endif
__device__ __forceinline__ bool s30_zero(const uint64_t *a) {
    return !(a[0]|a[1]|a[2]|a[3]);
}
__device__ __forceinline__ void s30_one(uint64_t *a) {
    a[0]=1; a[1]=a[2]=a[3]=0;
}
__device__ __forceinline__ void s30_normalize(uint64_t *a) {
    if ((a[1]&a[2]&a[3])==UINT64_MAX && a[0]>=0xFFFFFFFEFFFFFC2FULL) {
        a[0]-=0xFFFFFFFEFFFFFC2FULL; a[1]=a[2]=a[3]=0;
    }
}
__device__ __forceinline__ void s30_mul(uint64_t *out, const uint64_t *a, const uint64_t *b) {
#ifdef __CUDA_ARCH__
    // This inherited raw product includes the final carry fold. Its fifth
    // output word is real storage, not an out-of-bounds write to a field value.
    uint64_t tmp[5];
    qsb_field_mul(tmp,const_cast<uint64_t*>(a),const_cast<uint64_t*>(b));
    Load256(out,tmp);
#else
    // Exact portable branch; host success does not execute the inline PTX.
    uint64_t t[8]={0,0,0,0,0,0,0,0};
    for(int i=0;i<4;i++) {
        uint64_t carry=0;
        for(int j=0;j<4;j++) {
            __uint128_t z=(__uint128_t)a[i]*b[j]+t[i+j]+carry;
            t[i+j]=(uint64_t)z; carry=(uint64_t)(z>>64);
        }
        t[i+4]=carry;
    }
    uint64_t carry=0;
    for(int i=0;i<4;i++) {
        __uint128_t z=(__uint128_t)t[i+4]*0x1000003D1ULL+t[i]+carry;
        out[i]=(uint64_t)z; carry=(uint64_t)(z>>64);
    }
    while(carry) {
        __uint128_t z=(__uint128_t)carry*0x1000003D1ULL+out[0];
        out[0]=(uint64_t)z; carry=(uint64_t)(z>>64);
        for(int i=1;i<4;i++) {
            z=(__uint128_t)out[i]+carry;
            out[i]=(uint64_t)z; carry=(uint64_t)(z>>64);
        }
    }
    s30_normalize(out);
#endif
}

struct s30_point { uint64_t x[4],y[4]; bool inf; };

// Modes: ordinary sum, double, return first, return second, infinity.
__device__ __forceinline__ int s30_denom(uint64_t *d,const s30_point &a,const s30_point &b) {
    if(a.inf || b.inf) { s30_one(d); return a.inf ? (b.inf?4:3) : 2; }
    _ModSub256(d,const_cast<uint64_t*>(b.x),const_cast<uint64_t*>(a.x));
    if(!s30_zero(d))return 0;
    uint64_t dy[4];
    _ModSub256(dy,const_cast<uint64_t*>(b.y),const_cast<uint64_t*>(a.y));
    if(!s30_zero(dy)||s30_zero(a.y)) { s30_one(d);return 4; }
    _ModAdd256(d,const_cast<uint64_t*>(a.y),const_cast<uint64_t*>(a.y));
    return 1;
}
__device__ __forceinline__ void s30_add_inverse(
    s30_point &r,const s30_point &a,const s30_point &b,const uint64_t *inv,int mode) {
    if(mode>=2) {
        if(mode==2)r=a;
        else if(mode==3)r=b;
        else { r.inf=true; r.x[0]=r.x[1]=r.x[2]=r.x[3]=0;
               r.y[0]=r.y[1]=r.y[2]=r.y[3]=0; }
        return;
    }
    uint64_t m[4],xx[4],yy[4],t[4];
    if(mode==1) {
        s30_mul(m,a.x,a.x); _ModAdd256(t,m,m); _ModAdd256(m,t,m);
    } else _ModSub256(m,const_cast<uint64_t*>(b.y),const_cast<uint64_t*>(a.y));
    s30_mul(m,m,inv);
    s30_mul(xx,m,m);
    _ModSub256(xx,xx,const_cast<uint64_t*>(a.x));
    _ModSub256(xx,xx,const_cast<uint64_t*>(b.x));
    _ModSub256(t,const_cast<uint64_t*>(a.x),xx);
    s30_mul(yy,m,t); _ModSub256(yy,yy,const_cast<uint64_t*>(a.y));
    Load256(r.x,xx);Load256(r.y,yy);r.inf=false;
}

template<int N> __device__ __forceinline__ void s30_level(s30_point (&points)[8]) {
    uint64_t prefix[N][4],d[4],product[5];
    #pragma unroll
    for(int i=0;i<N;i++) {
        s30_denom(d,points[2*i],points[2*i+1]);
        if(i==0) { Load256(prefix[i],d); }
        else s30_mul(prefix[i],prefix[i-1],d);
    }
    Load256(product,prefix[N-1]);product[4]=0;
    qsb_block_inverse_tree(product);
    // The inherited tree ends with cross-warp leaf reads. A later invocation
    // reuses its shared arrays, so every reader must finish before any overwrite.
    __syncthreads();
    s30_normalize(product);
    #pragma unroll
    for(int i=N-1;i>=0;i--) {
        int mode=s30_denom(d,points[2*i],points[2*i+1]);
        uint64_t inv[4];
        if(i==0) { Load256(inv,product); }
        else { s30_mul(inv,product,prefix[i-1]);s30_mul(product,product,d); }
        // Writing points[i] here would destroy a lower pair's unread input.
        s30_add_inverse(points[2*i],points[2*i],points[2*i+1],inv,mode);
    }
    #pragma unroll
    for(int i=0;i<N;i++)points[i]=points[2*i];
}


__device__ __forceinline__ void s30_load(s30_point &p,const uint8_t *table,
                                       const int32_t *digits,int c) {
    glv_load<false>(table,(unsigned)c&7u,digits[c],p.x,p.y);
    s30_normalize(p.x);s30_normalize(p.y);p.inf=false;
}
__device__ __forceinline__ void s30_load_x(uint64_t *x,const uint8_t *table,
                                         const int32_t *digits,int c) {
    uint32_t idx;uint64_t neg;gt_digit_idx(digits[c],&idx,&neg);
    const uint64_t *p=(const uint64_t*)(table+((size_t)(((unsigned)c&7u)<<15)+idx)*64);
    Load256(x,p);s30_normalize(x);
}
__device__ __forceinline__ void glv_affine_collective(
    uint64_t *X,uint64_t *Y,uint64_t *ZZ,uint64_t *ZZZ,
    const uint64_t input[4],const uint8_t *table) {
    const uint64_t z[4]={input[0],input[1],input[2],input[3]};
    int32_t digits[16];glv_digits(digits,z);
    s30_point points[8];
    #pragma unroll
    for(int i=0;i<8;i++) {
        uint64_t a[4],b[4],denom[4];
        s30_load_x(a,table,digits,2*i);s30_load_x(b,table,digits,2*i+1);
        _ModSub256(denom,b,a);
        if(s30_zero(denom)) {
            s30_point p,q;s30_load(p,table,digits,2*i);s30_load(q,table,digits,2*i+1);
            s30_denom(denom,p,q);
        }
        if(i==0){Load256(points[i].x,denom);}
        else s30_mul(points[i].x,points[i-1].x,denom);
    }
    uint64_t product[5];Load256(product,points[7].x);product[4]=0;
    qsb_block_inverse_tree(product);__syncthreads();s30_normalize(product);
    #pragma unroll
    for(int i=7;i>=0;i--) {
        s30_point a,b;s30_load(a,table,digits,2*i);s30_load(b,table,digits,2*i+1);
        uint64_t denom[4],inv[4];int mode=s30_denom(denom,a,b);
        if(i==0){Load256(inv,product);}
        else{s30_mul(inv,product,points[i-1].x);s30_mul(product,product,denom);}
        s30_add_inverse(points[i],a,b,inv,mode);
    }
#if QSB_GLV_AFFINE_LEVELS == 4
    s30_level<4>(points);s30_level<2>(points);
    // The two remaining points are uB and vB. Apply phi once to vB.
    const uint64_t beta[4]={0xc1396c28719501eeULL,0x9cf0497512f58995ULL,
                            0x6e64479eac3434e9ULL,0x7ae96a2b657c0710ULL};
    s30_mul(points[1].x,points[1].x,beta);
    s30_level<1>(points);
    Load256(X,points[0].x);Load256(Y,points[0].y);
    ZZ[0]=ZZZ[0]=points[0].inf?0:1;
    ZZ[1]=ZZ[2]=ZZ[3]=ZZZ[1]=ZZZ[2]=ZZZ[3]=0;
#else
    // Adjacent digit pairs are nonzero affine points. Their grouping keeps
    // the original odd-chain exclusion through the penultimate v pair.
    // The final pair is complete, including modular doubling and infinity.
    uint64_t anchor[4];Load256(anchor,points[0].y);
    _PointAddXYZZ_mm_def(X,Y,ZZ,ZZZ,points[0].x,points[0].y,points[1].x,points[1].y);
    #pragma unroll
    for(int i=2;i<8;i++) {
        if(i==4) {
            uint64_t beta2[4]={0x3ec693d68e6afa40ULL,0x630fb68aed0a766aULL,
                               0x919bb86153cbcb16ULL,0x851695d49a83f8efULL};
            _ModMult(X,beta2);
        }
        if(i==7)qsb_complete_last_add(X,Y,ZZ,ZZZ,points[i].x,points[i].y,anchor);
        else {
            _PointAddXYZZ_def<true>(X,Y,ZZ,ZZZ,points[i].x,points[i].y,anchor);
            Load256(anchor,points[i].y);
        }
    }
    uint64_t beta[4]={0xc1396c28719501eeULL,0x9cf0497512f58995ULL,
                      0x6e64479eac3434e9ULL,0x7ae96a2b657c0710ULL};
    _ModMult(X,beta);
#endif
}
