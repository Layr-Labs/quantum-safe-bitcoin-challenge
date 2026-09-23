// GPL-3.0-only. Table-only adaptation of the independently reviewed GLV14
// construction. Retains subset QSB_STARTUP_TRIM and gathered spot-check policy.
// Split/table architecture credit: public GLV40 4b77964f and inherited authors.
#pragma once
static void qsb_glv14_batch_ladder(EC_GROUP *grp, const EC_POINT *step, int count,
                            uint64_t *out, BIGNUM *x, BIGNUM *y,
                            const BIGNUM *alpha, const BIGNUM *beta,
                            const BIGNUM *field_p, BN_CTX *ctx) {
    EC_POINT *points[GT_HI];
    if(count<1 || count>=GT_HI) { fprintf(stderr,"Invalid ladder size\n");exit(2); }
    for(int i=0;i<count;i++) {
        points[i]=EC_POINT_new(grp);
        if(!points[i]) { fprintf(stderr,"Ladder allocation failed\n");exit(2); }
        int ok=i==0 ? EC_POINT_copy(points[i],step)
                    : EC_POINT_add(grp,points[i],points[i-1],step,ctx);
        if(!ok) { fprintf(stderr,"Ladder addition failed\n");exit(2); }
    }
#if QSB_STARTUP_TRIM
    if(!EC_POINTs_make_affine(grp,(size_t)count,points,ctx)) {
        fprintf(stderr,"Ladder batch normalization failed\n");exit(2);
    }
#endif
    for(int i=0;i<count;i++) {
        gt_point_to_limbs(grp,points[i],x,y,alpha,beta,field_p,ctx,
                          out+(size_t)(i+1)*8);
        EC_POINT_free(points[i]);
    }
}

static void qsb_glv14_biased_ladder(EC_GROUP *grp, const EC_POINT *first,
                             const EC_POINT *step, int count,
                             uint64_t *out, BIGNUM *x, BIGNUM *y,
                             const BIGNUM *alpha, const BIGNUM *beta,
                             const BIGNUM *field_p, BN_CTX *ctx) {
    EC_POINT *points[GT_HI];
    if(count<1 || count>GT_HI) { fprintf(stderr,"Invalid biased ladder size\n");exit(2); }
    for(int i=0;i<count;i++) {
        points[i]=EC_POINT_new(grp);
        if(!points[i]) { fprintf(stderr,"Biased ladder allocation failed\n");exit(2); }
        int ok=i==0 ? EC_POINT_copy(points[i],first)
                    : EC_POINT_add(grp,points[i],points[i-1],step,ctx);
        if(!ok) { fprintf(stderr,"Biased ladder addition failed\n");exit(2); }
    }
#if QSB_STARTUP_TRIM
    if(!EC_POINTs_make_affine(grp,(size_t)count,points,ctx)) {
        fprintf(stderr,"Biased ladder normalization failed\n");exit(2);
    }
#endif
    for(int i=0;i<count;i++) {
        gt_point_to_limbs(grp,points[i],x,y,alpha,beta,field_p,ctx,
                          out+(size_t)i*8);
        EC_POINT_free(points[i]);
    }
}

static void qsb_glv14_build_ladders(uint64_t *hL, uint64_t *hH, const uint8_t neg_r_inv[32],
                             const uint64_t alpha_le[4], const uint64_t beta_le[4]) {
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *x=BN_new(),*y=BN_new(),*factor=BN_new(),*order=BN_new(),
           *nri=BN_new(),*bscal=BN_new(),*field_p=BN_new(),
           *alpha=BN_new(),*beta=BN_new(),*bias=BN_new();
    EC_POINT *base=EC_POINT_new(grp),*step=EC_POINT_new(grp),*first=EC_POINT_new(grp);
    EC_GROUP_get_order(grp,order,ctx);
    EC_GROUP_get_curve_GFp(grp,field_p,NULL,NULL,ctx);
    BN_lebin2bn((const uint8_t*)alpha_le,32,alpha);
    BN_lebin2bn((const uint8_t*)beta_le,32,beta);
    BN_lebin2bn(neg_r_inv,32,nri);
    BN_set_word(bias,333126); BN_lshift(bias,bias,108); BN_sub_word(bias,1u<<17);
    memset(hL,0,(size_t)GT_CHUNKS*GT_LO*8*sizeof(uint64_t));
    memset(hH,0,(size_t)GT_CHUNKS*GT_HI*8*sizeof(uint64_t));
    for(int ch=0;ch<GT_CHUNKS;ch++) {
        if(ch==0) {
            /* base=A, L[lo]=(K+lo)A, H[hi]=hi*256A. */
            EC_POINT_mul(grp,base,nri,NULL,NULL,ctx);
            BN_mod_mul(bscal,bias,nri,order,ctx);
            EC_POINT_mul(grp,first,bscal,NULL,NULL,ctx);
            qsb_glv14_biased_ladder(grp,first,base,GT_LO,
                hL,x,y,alpha,beta,field_p,ctx);
        } else {
            /* base=2^(shift-1)A, L[lo]=lo*base. */
            BN_one(factor); BN_lshift(factor,factor,gt_shift(ch)-1);
            BN_mod_mul(bscal,factor,nri,order,ctx);
            EC_POINT_mul(grp,base,bscal,NULL,NULL,ctx);
            qsb_glv14_batch_ladder(grp,base,GT_LO-1,
                hL+(size_t)ch*GT_LO*8,x,y,alpha,beta,field_p,ctx);
        }
        BN_set_word(factor,256);
        EC_POINT_mul(grp,step,NULL,base,factor,ctx);
        unsigned max_m=ch==0?gt_entries(ch)-1:2*(gt_entries(ch)-1)+1;
        int high=(int)(max_m>>8);
        qsb_glv14_batch_ladder(grp,step,high,
            hH+(size_t)ch*GT_HI*8,x,y,alpha,beta,field_p,ctx);
    }
    BN_free(x);BN_free(y);BN_free(factor);BN_free(order);BN_free(nri);
    BN_free(bscal);BN_free(field_p);BN_free(alpha);BN_free(beta);BN_free(bias);
    EC_POINT_free(base);EC_POINT_free(step);EC_POINT_free(first);
    EC_GROUP_free(grp);BN_CTX_free(ctx);
}

static void qsb_glv14_table_scalar(BIGNUM *k,int ch,unsigned index) {
    if(ch==0) {
        BN_set_word(k,333126); BN_lshift(k,k,108);
        BN_sub_word(k,1u<<17); BN_add_word(k,index);
    } else {
        BN_one(k); BN_lshift(k,k,gt_shift(ch)-1);
        BN_mul_word(k,(BN_ULONG)(2*index+1));
    }
}

static void qsb_glv14_compute_gtable(uint8_t *gTable, const uint8_t neg_r_inv[32],
                           const uint64_t alpha_le[4], const uint64_t beta_le[4]) {
    printf("  Computing GLV14 GTable (OpenSSL fallback)...\n");
    EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1); BN_CTX *ctx=BN_CTX_new();
    BIGNUM *x=BN_new(),*y=BN_new(),*k=BN_new(),*stepk=BN_new(),*order=BN_new(),
           *nri=BN_new(),*field_p=BN_new(),*alpha=BN_new(),*beta=BN_new();
    EC_POINT *pt=EC_POINT_new(grp),*step=EC_POINT_new(grp);
    EC_GROUP_get_order(grp,order,ctx); EC_GROUP_get_curve_GFp(grp,field_p,NULL,NULL,ctx);
    BN_lebin2bn((const uint8_t*)alpha_le,32,alpha);
    BN_lebin2bn((const uint8_t*)beta_le,32,beta);
    BN_lebin2bn(neg_r_inv,32,nri);
    for(int ch=0;ch<GT_CHUNKS;ch++) {
        qsb_glv14_table_scalar(k,ch,0); BN_mod_mul(k,k,nri,order,ctx);
        EC_POINT_mul(grp,pt,k,NULL,NULL,ctx);
        if(ch==0) BN_copy(stepk,nri);
        else { BN_one(stepk); BN_lshift(stepk,stepk,gt_shift(ch));
               BN_mod_mul(stepk,stepk,nri,order,ctx); }
        EC_POINT_mul(grp,step,stepk,NULL,NULL,ctx);
        for(unsigned d=0;d<gt_entries(ch);d++) {
            uint64_t limbs[8]; gt_point_to_limbs(grp,pt,x,y,alpha,beta,field_p,ctx,limbs);
            memcpy(gTable+((size_t)gt_offset(ch)+d)*64,limbs,64);
            if(d+1<gt_entries(ch)) EC_POINT_add(grp,pt,pt,step,ctx);
        }
    }
    BN_free(x);BN_free(y);BN_free(k);BN_free(stepk);BN_free(order);BN_free(nri);
    BN_free(field_p);BN_free(alpha);BN_free(beta);EC_POINT_free(pt);EC_POINT_free(step);
    EC_GROUP_free(grp);BN_CTX_free(ctx);
}
