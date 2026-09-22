/* GPL-3.0-only. Signed B8/T16/S2 multi-comb with deferred-anchor XYZZ.
 * Extends the independently checked X26 comb construction. Signed-comb identity:
 * bitcoin-core/secp256k1 ecmult_gen_impl.h 46db787112beabdb5e17e0dc35680716f1057e7b.
 * Group operations adapt VanitySearch GPUMath.h and EFD XYZZ dbl/madd-2008-s.
 * Public synthetic scalars only; not a secret-key implementation.
 */
#pragma once
__device__ __forceinline__ bool hm37_zero(const uint64_t a[4]) {
    return !(a[0]|a[1]|a[2]|a[3]) ||
        (a[0]==0xFFFFFFFEFFFFFC2FULL && a[1]==~0ULL && a[2]==~0ULL && a[3]==~0ULL);
}
__device__ __forceinline__ uint32_t hm37_pack_even(uint32_t x) {
    x&=0x55555555u; x=(x|(x>>1))&0x33333333u; x=(x|(x>>2))&0x0f0f0f0fu;
    x=(x|(x>>4))&0x00ff00ffu; return (x|(x>>8))&0xffffu;
}
/* Each bank word packs row1 then row0 sign/index in two 16-bit halves.
 * d=(k+(2^256-1-n)/2) mod n, carrying bit256 through the reduction. */
__device__ __forceinline__ void hm37_recode(const uint64_t k[4],uint32_t e[8]) {
    const uint32_t n[8]={0xd0364141u,0xbfd25e8cu,0xaf48a03bu,0xbaaedce6u,
                        0xfffffffeu,0xffffffffu,0xffffffffu,0xffffffffu};
    const uint32_t offset[8]={0x97e4df5fu,0x2016d0b9u,0xa85bafe2u,0xa2a8918cu,0,0,0,0};
    uint32_t sum[8],red[8];uint64_t carry=0,borrow=0;
    #pragma unroll
    for(int i=0;i<8;i++){uint64_t w=(uint64_t)(uint32_t)(k[i/2]>>(32*(i&1)))+offset[i]+carry;sum[i]=(uint32_t)w;carry=w>>32;}
    #pragma unroll
    for(int i=0;i<8;i++){uint64_t w=(uint64_t)sum[i]-n[i]-borrow;red[i]=(uint32_t)w;borrow=w>>63;}
    uint32_t select=0u-(uint32_t)(carry||!borrow);
    #pragma unroll
    for(int b=0;b<8;b++){
        uint32_t d=(sum[b]&~select)|(red[b]&select),odd=hm37_pack_even(d>>1),even=hm37_pack_even(d);
        uint32_t no=odd>>15,ne=even>>15;
        e[b]=(((odd^(0u-no))&0x7fffu)|(no<<15))|
             ((((even^(0u-ne))&0x7fffu)|(ne<<15))<<16);
    }
}
__device__ __forceinline__ void hm37_load(const uint8_t *table,int bank,uint32_t digit,
                                        uint64_t x[4],uint64_t y[4]) {
    gt_load_signed_flat_f(table,(unsigned)bank<<15,digit&32767,(digit>>15)&1u,x,y);
}
/* Complete a=0 XYZZ doubling, exact Y input/output; 6M+3S. */
__device__ __forceinline__ void hm37_double(uint64_t X[4],uint64_t Y[4],uint64_t ZZ[4],uint64_t ZZZ[4]) {
    uint64_t u[4],v[4],w[4],s[4],m[4],t[4];
    _ModAdd256(u,Y,Y);_ModSqr(v,u);_ModMult(w,u,v);_ModMult(s,X,v);
    _ModSqr(m,X);_ModAdd256(t,m,m);_ModAdd256(m,t,m);
    _ModSqr(t,m);_ModSub256(t,t,s);_ModSub256(t,t,s);
    _ModMult(ZZ,v);_ModMult(ZZZ,w);_ModMult(w,Y);
    _ModSub256(s,s,t);_ModMult(Y,m,s);_ModSub256(Y,Y,w);Load256(X,t);
}
/* Input stores Yactual+anchor*ZZZ. Output defers relative to new affine y
 * iff defer_y. Handle infinity, equal points and inverse points explicitly. */
__device__ __forceinline__ void hm37_madd(uint64_t X[4],uint64_t Y[4],uint64_t ZZ[4],uint64_t ZZZ[4],
                                        const uint64_t x[4],const uint64_t y[4],const uint64_t anchor[4],bool defer_y) {
    if(hm37_zero(ZZ)){
        Load256(X,x);Load256(Y,y);if(defer_y)_ModAdd256(Y,Y,Y);
        ZZ[0]=ZZZ[0]=1;ZZ[1]=ZZ[2]=ZZ[3]=ZZZ[1]=ZZZ[2]=ZZZ[3]=0;return;
    }
    uint64_t U[4],S[4],P[4],R[4],PP[4],PPP[4],V[4],T[4];
    _ModMult(U,(uint64_t*)x,ZZ);_ModAdd256(S,(uint64_t*)y,(uint64_t*)anchor);
    _ModMult(S,ZZZ);_ModSub256(P,U,X);_ModSub256(R,S,Y);
    if(hm37_zero(P)){
        if(hm37_zero(R)){
            _ModMult(S,(uint64_t*)anchor,ZZZ);_ModSub256(Y,Y,S);
            hm37_double(X,Y,ZZ,ZZZ);
            if(defer_y){_ModMult(S,(uint64_t*)y,ZZZ);_ModAdd256(Y,Y,S);}
        }else{
            #pragma unroll
            for(int j=0;j<4;j++)X[j]=Y[j]=ZZ[j]=ZZZ[j]=0;
        }
        return;
    }
    _ModSqr(PP,P);_ModMult(PPP,PP,P);_ModMult(V,U,PP);_ModMult(ZZ,PP);
    _ModSqr(T,R);_ModAdd256(T,T,PPP);_ModSub256(T,T,V);_ModSub256(T,T,V);
    _ModMult(ZZZ,PPP);_ModSub256(V,V,T);_ModMult(V,R);
    if(defer_y){Load256(Y,V);}
    else {_ModMult(S,(uint64_t*)y,ZZZ);_ModSub256(Y,V,S);}
    Load256(X,T);
}
__device__ void hm83_chain_exact(uint64_t *X,uint64_t *Y,uint64_t *ZZ,uint64_t *ZZZ,
                                        const uint64_t k[4],const uint8_t *table) {
    uint32_t digits[8];hm37_recode(k,digits);
    /* Constant-index packing lets the compiler scalarize recoding. Each row
     * streams two 64-bit words instead of dynamically indexing a local array. */
    uint64_t even_lo=0,even_hi=0,odd_lo=0,odd_hi=0;
    #pragma unroll
    for(int b=0;b<4;b++){
        odd_lo|=(uint64_t)(digits[b]&65535u)<<(16*b);
        odd_hi|=(uint64_t)(digits[b+4]&65535u)<<(16*b);
        even_lo|=(uint64_t)(digits[b]>>16)<<(16*b);
        even_hi|=(uint64_t)(digits[b+4]>>16)<<(16*b);
    }
    uint64_t x0[4],y0[4],x[4],y[4];
    hm37_load(table,0,(uint32_t)odd_lo,x0,y0);hm37_load(table,1,(uint32_t)(odd_lo>>16),x,y);
    odd_lo=(odd_lo>>32)|(odd_hi<<32);odd_hi>>=32;
    // Row1 bank0 and bank1 have disjoint dominating coefficient ranges.
    _PointAddXYZZ_mm_def(X,Y,ZZ,ZZZ,x0,y0,x,y);
    /* All nonfinal sums use distinct signed powers whose absolute coefficient
     * is strictly between zero and n. The highest power exceeds the sum of
     * lower powers; an omitted bit254 keeps the upper bound below n even
     * after the row shift. Thus equal/inverse pairs are impossible here.
     * The final bank covers all 256 bits and must retain complete handling. */
    #pragma unroll 1
    for(int b=2;b<8;b++){
        hm37_load(table,b,(uint32_t)odd_lo,x,y);
        odd_lo=(odd_lo>>16)|(odd_hi<<48);odd_hi>>=16;
        if(b!=7)_PointAddXYZZ_def<true>(X,Y,ZZ,ZZZ,x,y,y0);
        else qsb_complete_last_add(X,Y,ZZ,ZZZ,x,y,y0);
        Load256(y0,y);
    }
    hm37_double(X,Y,ZZ,ZZZ);
    #pragma unroll
    for(int j=0;j<4;j++)y0[j]=0; // row1 ended with the exact Y
    #pragma unroll 1
    for(int b=0;b<7;b++){
        hm37_load(table,b,(uint32_t)even_lo,x,y);
        even_lo=(even_lo>>16)|(even_hi<<48);even_hi>>=16;
        _PointAddXYZZ_def<true>(X,Y,ZZ,ZZZ,x,y,y0);Load256(y0,y);
    }
    hm37_load(table,7,(uint32_t)even_lo,x,y);hm37_madd(X,Y,ZZ,ZZZ,x,y,y0,false);
}

// Current fast field primitives; only output verifier authorizes hits.
__device__ void hm83_chain_filter(uint64_t *X,uint64_t *Y,uint64_t *ZZ,uint64_t *ZZZ,
                                        const uint64_t k[4],const uint8_t *table) {
    uint32_t bad=0;
    uint32_t digits[8];hm37_recode(k,digits);
    /* Constant-index packing lets the compiler scalarize recoding. Each row
     * streams two 64-bit words instead of dynamically indexing a local array. */
    uint64_t even_lo=0,even_hi=0,odd_lo=0,odd_hi=0;
    #pragma unroll
    for(int b=0;b<4;b++){
        odd_lo|=(uint64_t)(digits[b]&65535u)<<(16*b);
        odd_hi|=(uint64_t)(digits[b+4]&65535u)<<(16*b);
        even_lo|=(uint64_t)(digits[b]>>16)<<(16*b);
        even_hi|=(uint64_t)(digits[b+4]>>16)<<(16*b);
    }
    uint64_t x0[4],y0[4],x[4],y[4];
    hm37_load(table,0,(uint32_t)odd_lo,x0,y0);hm37_load(table,1,(uint32_t)(odd_lo>>16),x,y);
    odd_lo=(odd_lo>>32)|(odd_hi<<32);odd_hi>>=32;
    // Row1 bank0 and bank1 have disjoint dominating coefficient ranges.
    qsb_filter_point_seed(X,Y,ZZ,ZZZ,x0,y0,x,y,bad);
    /* All nonfinal sums use distinct signed powers whose absolute coefficient
     * is strictly between zero and n. The highest power exceeds the sum of
     * lower powers; an omitted bit254 keeps the upper bound below n even
     * after the row shift. Thus equal/inverse pairs are impossible here.
     * The final bank covers all 256 bits and must retain complete handling. */
    #pragma unroll 1
    for(int b=2;b<8;b++){
        hm37_load(table,b,(uint32_t)odd_lo,x,y);
        odd_lo=(odd_lo>>16)|(odd_hi<<48);odd_hi>>=16;
        if(b!=7)qsb_filter_point_add<true>(X,Y,ZZ,ZZZ,x,y,y0,bad);
        else qsb_filter_last_add(X,Y,ZZ,ZZZ,x,y,y0,bad);
        Load256(y0,y);
    }
    hm37_double(X,Y,ZZ,ZZZ);
    #pragma unroll
    for(int j=0;j<4;j++)y0[j]=0; // row1 ended with the exact Y
    #pragma unroll 1
    for(int b=0;b<7;b++){
        hm37_load(table,b,(uint32_t)even_lo,x,y);
        even_lo=(even_lo>>16)|(even_hi<<48);even_hi>>=16;
        qsb_filter_point_add<true>(X,Y,ZZ,ZZZ,x,y,y0,bad);Load256(y0,y);
    }
    hm37_load(table,7,(uint32_t)even_lo,x,y);hm37_madd(X,Y,ZZ,ZZZ,x,y,y0,false);
}
