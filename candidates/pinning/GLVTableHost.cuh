#pragma once
static void q9_phi_host(uint8_t*out,const uint64_t*x,BN_CTX*ctx){
    BN_CTX_start(ctx);BIGNUM*a=BN_CTX_get(ctx),*b=BN_CTX_get(ctx),*p=BN_CTX_get(ctx);
    BN_lebin2bn((const unsigned char*)x,32,a);
    BN_hex2bn(&b,"7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE");
    BN_hex2bn(&p,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    BN_mod_mul(a,a,b,p,ctx);BN_bn2lebinpad(a,out,32);BN_CTX_end(ctx);
}
// Entries are multiples of the problem-dependent A = neg_r_inv * G.
// Chunk 0: (2^127 - 2^16 + index) A, with all 17-bit indices.
// Chunk c>0: (2*index+1) 2^(17*c-1) A, with signed odd decoding.
static void q9_table_scalar(BIGNUM *k,int ch,unsigned index){
    if(ch==7){
        BIGNUM *lambda=nullptr;BN_hex2bn(&lambda,"5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72");
        BN_mul_word(lambda,2*((index>>8)&255)+1);BN_set_word(k,2*(index&255)+1);
        if(index>>16)BN_sub(k,k,lambda);else BN_add(k,k,lambda);BN_lshift(k,k,118);BN_free(lambda);return;
    }
    BN_one(k);
    if(ch==0){BN_lshift(k,k,127);BN_sub_word(k,1u<<16);BN_add_word(k,index);}
    else {BN_lshift(k,k,17*ch-1);BN_mul_word(k,2*index+1);}
}
static void q9_biased_ladder(EC_GROUP *grp,const EC_POINT *first,const EC_POINT *step,int count,
    uint64_t *out,BIGNUM*x,BIGNUM*y,BN_CTX*ctx){
    EC_POINT *points[GT_HI];
    for(int i=0;i<count;i++){
        points[i]=EC_POINT_new(grp);
        int ok=i?EC_POINT_add(grp,points[i],points[i-1],step,ctx):EC_POINT_copy(points[i],first);
        if(!ok){fprintf(stderr,"GLV biased ladder failed\n");exit(2);}
    }
    if(!EC_POINTs_make_affine(grp,count,points,ctx)){fprintf(stderr,"GLV normalization failed\n");exit(2);}
    for(int i=0;i<count;i++){gt_point_to_limbs(grp,points[i],x,y,ctx,out+(size_t)i*8);EC_POINT_free(points[i]);}
}
static void gt_build_ladders(uint64_t*hL,uint64_t*hH,const uint8_t nri_bytes[32]){
    EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);BN_CTX *ctx=BN_CTX_new();
    BIGNUM *x=BN_new(),*y=BN_new(),*nri=BN_lebin2bn(nri_bytes,32,nullptr),*k=BN_new(),*n=BN_new();
    EC_GROUP_get_order(grp,n,ctx);EC_POINT *base=EC_POINT_new(grp),*step=EC_POINT_new(grp),*bias=EC_POINT_new(grp);
    memset(hL,0,(size_t)GT_CHUNKS*GT_LO*64);memset(hH,0,(size_t)GT_CHUNKS*GT_HI*64);
    for(int ch=0;ch<GT_CHUNKS;ch++){
        if(ch==7){
            BN_one(k);BN_lshift(k,k,118);BN_mod_mul(k,k,nri,n,ctx);EC_POINT_mul(grp,base,k,nullptr,nullptr,ctx);
            BN_set_word(k,2);EC_POINT_mul(grp,step,nullptr,base,k,ctx);
            uint64_t *ll=hL+(size_t)7*GT_LO*8,*hh=hH+(size_t)7*GT_HI*8;
            q9_biased_ladder(grp,base,step,256,ll,x,y,ctx);
            for(int i=0;i<256;i++){memcpy(hh+(size_t)i*8,ll+(size_t)i*8,64);q9_phi_host((uint8_t*)(hh+(size_t)i*8),ll+(size_t)i*8,ctx);}
            continue;
        }

        BN_one(k);if(ch)BN_lshift(k,k,17*ch-1);BN_mod_mul(k,k,nri,n,ctx);EC_POINT_mul(grp,base,k,nullptr,nullptr,ctx);
        gt_batch_ladder(grp,base,GT_LO-1,hL+(size_t)ch*GT_LO*8,x,y,ctx);
        BN_set_word(k,256);EC_POINT_mul(grp,step,nullptr,base,k,ctx);
        if(ch==0){
            q9_table_scalar(k,0,0);BN_mod_mul(k,k,nri,n,ctx);EC_POINT_mul(grp,bias,k,nullptr,nullptr,ctx);
            q9_biased_ladder(grp,bias,step,512,hH,x,y,ctx);
        }else{
            int high=(2*(gt_entries(ch)-1)+1)>>8;
            gt_batch_ladder(grp,step,high,hH+(size_t)ch*GT_HI*8,x,y,ctx);
        }
    }
    EC_POINT_free(base);EC_POINT_free(step);EC_POINT_free(bias);EC_GROUP_free(grp);
    BN_free(x);BN_free(y);BN_free(nri);BN_free(k);BN_free(n);BN_CTX_free(ctx);
}
static int gt_spot_check(const uint8_t*table,int samples,const uint8_t nri_bytes[32]){
    EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);BN_CTX *ctx=BN_CTX_new();
    BIGNUM *x=BN_new(),*y=BN_new(),*nri=BN_lebin2bn(nri_bytes,32,nullptr),*k=BN_new(),*n=BN_new();
    EC_GROUP_get_order(grp,n,ctx);EC_POINT *pt=EC_POINT_new(grp);unsigned seed=0x9e3779b9;int ok=1;
    for(int t=0;t<samples&&ok;t++){
        int ch;unsigned i;
        if(t<GT_CHUNKS*4){ch=t/4;unsigned corners[4]={0,1,2,gt_entries(ch)-1};i=corners[t%4];}
        else{seed=seed*1664525u+1013904223u;ch=(seed>>28)%GT_CHUNKS;i=(seed>>4)&(gt_entries(ch)-1);}
        q9_table_scalar(k,ch,i);BN_mod_mul(k,k,nri,n,ctx);EC_POINT_mul(grp,pt,k,nullptr,nullptr,ctx);
        uint64_t expected[8];gt_point_to_limbs(grp,pt,x,y,ctx,expected);
        if(memcmp(expected,table+((size_t)gt_offset(ch)+i)*64,64)){
            fprintf(stderr,"GLV table mismatch chunk=%d index=%u\n",ch,i);ok=0;
        }
    }
    EC_POINT_free(pt);EC_GROUP_free(grp);BN_free(x);BN_free(y);BN_free(nri);BN_free(k);BN_free(n);BN_CTX_free(ctx);return ok;
}
static void compute_gtable(uint8_t*table,const uint8_t nri_bytes[32]){
    printf("  Computing GLV GTable (OpenSSL fallback)...\n");
    EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);BN_CTX *ctx=BN_CTX_new();
    BIGNUM *x=BN_new(),*y=BN_new(),*nri=BN_lebin2bn(nri_bytes,32,nullptr),*k=BN_new(),*n=BN_new();
    EC_GROUP_get_order(grp,n,ctx);EC_POINT *pt=EC_POINT_new(grp),*step=EC_POINT_new(grp);

    for(int ch=0;ch<GT_CHUNKS;ch++){
        if(ch==7){
            for(unsigned i=0;i<gt_entries(ch);i++){
                q9_table_scalar(k,ch,i);BN_mod_mul(k,k,nri,n,ctx);EC_POINT_mul(grp,pt,k,nullptr,nullptr,ctx);
                uint64_t limbs[8];gt_point_to_limbs(grp,pt,x,y,ctx,limbs);memcpy(table+((size_t)gt_offset(ch)+i)*64,limbs,64);
            }
            continue;
        }

        BN_one(k);if(ch)BN_lshift(k,k,17*ch);BN_mod_mul(k,k,nri,n,ctx);EC_POINT_mul(grp,step,k,nullptr,nullptr,ctx);
        q9_table_scalar(k,ch,0);BN_mod_mul(k,k,nri,n,ctx);EC_POINT_mul(grp,pt,k,nullptr,nullptr,ctx);
        for(unsigned i=0;i<gt_entries(ch);i++){
            uint64_t limbs[8];gt_point_to_limbs(grp,pt,x,y,ctx,limbs);memcpy(table+((size_t)gt_offset(ch)+i)*64,limbs,64);
            if(i+1<gt_entries(ch))EC_POINT_add(grp,pt,pt,step,ctx);
        }
    }
    EC_POINT_free(pt);EC_POINT_free(step);EC_GROUP_free(grp);BN_free(x);BN_free(y);BN_free(nri);BN_free(k);BN_free(n);BN_CTX_free(ctx);
}
