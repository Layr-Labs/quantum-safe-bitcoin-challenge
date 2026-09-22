/* GPL-3.0-only. Adapted from the X26 signed-comb table construction. */
/* HM37 runtime table construction. Public synthetic instance only.
 * Each tooth advances by two bits. Gray traversal changes one sign at a time;
 * the index in memory is ordinary binary, not Gray order. */
static void hm37_fill(EC_GROUP *grp, BN_CTX *ctx, EC_POINT *base,
                     int teeth, int free_bits, uint64_t *out) {
    EC_POINT *powers[16], *steps[16];
    EC_POINT *acc=EC_POINT_new(grp), *delta=EC_POINT_new(grp);
    BIGNUM *x=BN_new(), *y=BN_new();
    EC_POINT_set_to_infinity(grp,acc);
    for(int t=0;t<teeth;t++){
        powers[t]=EC_POINT_new(grp);steps[t]=EC_POINT_new(grp);
        EC_POINT_copy(powers[t],base);
        EC_POINT_add(grp,acc,acc,base,ctx);
        EC_POINT_dbl(grp,steps[t],base,ctx);
        EC_POINT_dbl(grp,base,steps[t],ctx); /* base *= 4 */
    }
    EC_POINT_invert(grp,acc,ctx); /* all signs negative */
    unsigned previous=0;
    for(unsigned j=0;j<(1u<<free_bits);j++){
        unsigned gray=j^(j>>1);
        if(j){
            unsigned changed=gray^previous;
            int bit=__builtin_ctz(changed);
            EC_POINT_copy(delta,steps[bit]);
            if(!(gray&changed)) EC_POINT_invert(grp,delta,ctx);
            EC_POINT_add(grp,acc,acc,delta,ctx);
        }
        gt_point_to_limbs(grp,acc,x,y,ctx,out+(size_t)gray*8);
        previous=gray;
    }
    for(int t=0;t<teeth;t++){EC_POINT_free(powers[t]);EC_POINT_free(steps[t]);}
    EC_POINT_free(acc);EC_POINT_free(delta);BN_free(x);BN_free(y);
}

static void hm37_base(EC_GROUP *grp,BN_CTX *ctx,EC_POINT *base,
                     const uint8_t neg_r_inv[32]){
    BIGNUM *n=BN_new(),*two=BN_new(),*k=BN_new(),*nri=BN_new();
    EC_GROUP_get_order(grp,n,ctx);BN_set_word(two,2);
    BN_mod_inverse(k,two,n,ctx);BN_lebin2bn(neg_r_inv,32,nri);
    BN_mod_mul(k,k,nri,n,ctx);EC_POINT_mul(grp,base,k,NULL,NULL,ctx);
    BN_free(n);BN_free(two);BN_free(k);BN_free(nri);
}

/* 8*(256+128)=3072 affine points. H's top tooth is always negative.
 * None is infinity, and H cannot equal +/-L: the highest coefficient
 * dominates all lower powers and the absolute combined scalar is < n. */
static void gt_build_ladders(uint64_t *hL,uint64_t *hH,const uint8_t neg_r_inv[32]){
    EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx=BN_CTX_new();EC_POINT *base=EC_POINT_new(grp);
    hm37_base(grp,ctx,base,neg_r_inv);
    for(int bank=0;bank<GT_CHUNKS;bank++){
        hm37_fill(grp,ctx,base,8,8,hL+(size_t)bank*GT_LO*8);
        hm37_fill(grp,ctx,base,8,7,hH+(size_t)bank*GT_HI*8);
    }
    EC_POINT_free(base);EC_GROUP_free(grp);BN_CTX_free(ctx);
}

/* Independent scalar-multiplication oracle; does not use the Gray builder. */
static int gt_spot_check(const uint8_t *gTable,int samples,
                         const uint8_t neg_r_inv[32]){
    EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx=BN_CTX_new();
    BIGNUM *x=BN_new(),*y=BN_new(),*k=BN_new(),*power=BN_new(),
           *n=BN_new(),*half=BN_new(),*nri=BN_new();
    EC_POINT *pt=EC_POINT_new(grp);
    EC_GROUP_get_order(grp,n,ctx);BN_set_word(k,2);
    BN_mod_inverse(half,k,n,ctx);BN_lebin2bn(neg_r_inv,32,nri);
    BN_mod_mul(half,half,nri,n,ctx);
    unsigned seed=0x9e3779b9u;int ok=1;
    for(int t=0;t<samples&&ok;t++){
        int bank,index;
        if(t<GT_CHUNKS*4){
            bank=t/4;const int corners[4]={0,1,2,32768-1};
            index=corners[t%4];
        }else{
            seed=seed*1664525u+1013904223u;
            bank=(seed>>28)%GT_CHUNKS;index=(seed>>4)&(32768-1);
        }
        BN_zero(k);
        for(int tooth=0;tooth<16;tooth++){
            BN_one(power);BN_lshift(power,power,(bank*16+tooth)*2);
            if(index&(1<<tooth)) BN_add(k,k,power);
            else BN_sub(k,k,power);
        }
        BN_nnmod(k,k,n,ctx);BN_mod_mul(k,k,half,n,ctx);
        EC_POINT_mul(grp,pt,k,NULL,NULL,ctx);
        uint64_t want[8];gt_point_to_limbs(grp,pt,x,y,ctx,want);
        size_t off=((size_t)bank*32768+index)*64;
        if(memcmp(gTable+off,want,64)){
            fprintf(stderr,"  HM37 GTable spot check FAILED at bank %d entry %d\n",bank,index);ok=0;
        }
    }
    BN_free(x);BN_free(y);BN_free(k);BN_free(power);BN_free(n);BN_free(half);BN_free(nri);
    EC_POINT_free(pt);EC_GROUP_free(grp);BN_CTX_free(ctx);return ok;
}

/* Same complete signed geometry if the GPU table's independent check fails. */
static void compute_gtable(uint8_t *gTable,const uint8_t neg_r_inv[32]){
    printf("  Computing HM37 GTable (OpenSSL fallback)...\n");
    EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx=BN_CTX_new();EC_POINT *base=EC_POINT_new(grp);
    hm37_base(grp,ctx,base,neg_r_inv);
    uint64_t *bank=(uint64_t*)malloc((size_t)32768*64);
    if(!bank){fprintf(stderr,"HM37 table allocation failed\n");exit(1);}
    for(int ch=0;ch<GT_CHUNKS;ch++){
        hm37_fill(grp,ctx,base,16,15,bank);
        for(unsigned d=0;d<32768;d++){
            size_t off=((size_t)ch*32768+d)*64;
            memcpy(gTable+off,bank+(size_t)d*8,64);
        }
    }
    free(bank);EC_POINT_free(base);EC_GROUP_free(grp);BN_CTX_free(ctx);
}
