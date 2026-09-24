#pragma once
// Positive-Y XYZZ convention from subset, with its unchanged speculative carries.
// Scalar decomposition and signed digits are exact; field filtering remains speculative.
__device__ __forceinline__ void glv10_load(const uint8_t *table, uint32_t code,
                                         uint64_t *x, uint64_t *y) {
    gt_load_signed_flat_f(table,0,code&0x7fffffffu,code>>31,x,y);
}
__device__ void qsb_filter_chain_trial(uint64_t *X,uint64_t *Y,uint64_t *ZZ,uint64_t *ZZZ,
                                     const uint64_t k[4],const uint8_t *table,uint32_t &bad) {
    uint64_t p[2],q[2]; unsigned sp,sq;
    q9_glv_split(k,p,q,&sp,&sq);
    const bool have_q=(q[0]|q[1])!=0;
    if(!have_q && !(p[0]|p[1])) {
        // Infinity is an explicit sentinel, consumed before any inversion.
        #pragma unroll
        for(int j=0;j<4;j++)X[j]=Y[j]=ZZ[j]=ZZZ[j]=0;
        return;
    }
    uint64_t mag[2]={have_q?q[0]:p[0],have_q?q[1]:p[1]};
    unsigned sign=have_q?sq:sp;
    uint64_t x0[4],anchor[4],x1[4],y1[4];
    glv10_load(table,glv10_code(mag,sign,0),x0,anchor);
    glv10_load(table,glv10_code(mag,sign,1),x1,y1);
    qsb_filter_point_seed(X,Y,ZZ,ZZZ,x0,anchor,x1,y1,bad);
    const int terms=have_q?10:5;
    #pragma unroll 1
    for(int term=2;term<terms;term++) {
        if(term==5) {
            // phi changes X only: Y, ZZ, ZZZ and the deferred affine-Y anchor survive.
            const uint64_t beta[4]={0xC1396C28719501EEULL,0x9CF0497512F58995ULL,
                0x6E64479EAC3434E9ULL,0x7AE96A2B657C0710ULL};
            qsb_filter_mul(X,X,beta,bad);
            mag[0]=p[0];mag[1]=p[1];sign=sp;
        }
        glv10_load(table,glv10_code(mag,sign,term<5?term:term-5),x1,y1);
        if(term==terms-1) qsb_filter_last_add(X,Y,ZZ,ZZZ,x1,y1,anchor,bad);
        else {
            qsb_filter_point_add<true>(X,Y,ZZ,ZZZ,x1,y1,anchor,bad);
#if !QSB_CHAIN_ANCHOR_UPDATE
            Load256(anchor,y1);
#endif
        }
    }
}
