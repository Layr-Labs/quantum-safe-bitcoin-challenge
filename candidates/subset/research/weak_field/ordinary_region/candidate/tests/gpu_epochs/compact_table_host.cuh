#pragma once
static void compact_point_to_limbs(EC_GROUP *grp, EC_POINT *pt, BIGNUM *x, BIGNUM *y,
                              BN_CTX *ctx, uint64_t out[8]) {
    uint8_t xb[32], yb[32];
    memset(xb, 0, 32); memset(yb, 0, 32);
    if(EC_POINT_get_affine_coordinates_GFp(grp,pt,x,y,ctx)!=1 ||
       BN_bn2binpad(x,xb,32)!=32 || BN_bn2binpad(y,yb,32)!=32){
        fprintf(stderr,"OpenSSL table builder failure: point serialization\n");exit(2);
    }
    for (int j = 0; j < 16; j++) { uint8_t t = xb[j]; xb[j] = xb[31-j]; xb[31-j] = t; }
    for (int j = 0; j < 16; j++) { uint8_t t = yb[j]; yb[j] = yb[31-j]; yb[31-j] = t; }
    memcpy(out,     xb, 32);
    memcpy(out + 4, yb, 32);
}
static void compact_require(bool ok,const char *where){if(!ok){fprintf(stderr,"OpenSSL table builder failure: %s\n",where);exit(2);}}
static void compact_build_ladders(uint64_t *hL,uint64_t *hH,const uint8_t neg_r_inv[32]){
    EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);BN_CTX *ctx=BN_CTX_new();
    BIGNUM *x=BN_new(),*y=BN_new(),*shift=BN_new(),*order=BN_new(),*inv2=BN_new(),*nri=BN_new(),*coef=BN_new();
    EC_POINT *base=grp?EC_POINT_new(grp):nullptr,*step=grp?EC_POINT_new(grp):nullptr;
    compact_require(grp&&ctx&&x&&y&&shift&&order&&inv2&&nri&&coef&&base&&step,"allocate");
    compact_require(EC_GROUP_get_order(grp,order,ctx),"group order");compact_require(BN_set_word(shift,2),"set2");
    compact_require(BN_mod_inverse(inv2,shift,order,ctx)!=nullptr,"inverse2");
    compact_require(BN_lebin2bn(neg_r_inv,32,nri)!=nullptr,"runtime coefficient");
    compact_require(BN_mod_mul(coef,inv2,nri,order,ctx),"fold base coefficient");
    compact_require(!BN_is_zero(coef),"nonzero runtime base");
    compact_require(EC_POINT_mul(grp,base,coef,nullptr,nullptr,ctx),"base point");
    memset(hL,0,(size_t)COMPACT_CHUNKS*COMPACT_LO*8*sizeof(uint64_t));
    memset(hH,0,(size_t)COMPACT_CHUNKS*COMPACT_HI*8*sizeof(uint64_t));
    for(int ch=0;ch<COMPACT_CHUNKS;++ch){
        if(ch){compact_require(BN_set_word(shift,1u<<mixed_bits(ch-1)),"window shift");compact_require(EC_POINT_mul(grp,base,nullptr,base,shift,ctx),"shifted base");}
        std::vector<EC_POINT*> points;std::vector<uint64_t*> destinations;
        compact_require(EC_POINT_dbl(grp,step,base,ctx),"low step");
        for(int lo=1;lo<COMPACT_LO;lo+=2){
            EC_POINT *pt=EC_POINT_new(grp);compact_require(pt!=nullptr,"low point allocate");
            compact_require(lo==1?EC_POINT_copy(pt,base):EC_POINT_add(grp,pt,points.back(),step,ctx),"low ladder");
            points.push_back(pt);destinations.push_back(hL+((size_t)ch*COMPACT_LO+lo)*8);
        }
        compact_require(BN_set_word(shift,COMPACT_LO),"high step scalar");
        compact_require(EC_POINT_mul(grp,step,nullptr,base,shift,ctx),"high step");
        int high_count=(int)((uint64_t)compact_entries(ch)*2/COMPACT_LO);
        compact_require(high_count>0 && high_count<=COMPACT_HI,"high ladder capacity");
        for(int hi=1;hi<high_count;++hi){
            EC_POINT *pt=EC_POINT_new(grp);compact_require(pt!=nullptr,"high point allocate");
            compact_require(hi==1?EC_POINT_copy(pt,step):EC_POINT_add(grp,pt,points.back(),step,ctx),"high ladder");
            points.push_back(pt);destinations.push_back(hH+((size_t)ch*COMPACT_HI+hi)*8);
        }
        compact_require(EC_POINTs_make_affine(grp,points.size(),points.data(),ctx),"batch normalize");
        for(size_t i=0;i<points.size();++i){compact_point_to_limbs(grp,points[i],x,y,ctx,destinations[i]);EC_POINT_free(points[i]);}
    }
    EC_POINT_free(base);EC_POINT_free(step);BN_free(x);BN_free(y);BN_free(shift);BN_free(order);BN_free(inv2);BN_free(nri);BN_free(coef);EC_GROUP_free(grp);BN_CTX_free(ctx);
}
static int compact_spot_check(const uint8_t *gTable, int samples,
                         const uint8_t neg_r_inv[32]) {
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *x = BN_new(), *y = BN_new(), *k = BN_new(), *inv2 = BN_new(), *order = BN_new(),
           *nri = BN_new(), *half_nri = BN_new();
    EC_POINT *pt = grp?EC_POINT_new(grp):nullptr;
    compact_require(grp&&ctx&&x&&y&&k&&inv2&&order&&nri&&half_nri&&pt,"spot-check allocate");
    uint64_t want[8];
    int ok = 1;
    unsigned seed = 0x9e3779b9u;
    compact_require(EC_GROUP_get_order(grp,order,ctx),"spot-check order");
    compact_require(BN_set_word(k,2),"spot-check set2");
    compact_require(BN_mod_inverse(inv2,k,order,ctx)!=nullptr,"spot-check inverse2");
    compact_require(BN_lebin2bn(neg_r_inv,32,nri)!=nullptr,"spot-check runtime coefficient");
    compact_require(BN_mod_mul(half_nri,inv2,nri,order,ctx),"spot-check half coefficient");              /* (2^-1 * neg_r_inv) mod n = A/2 scalar */
    for (int t = 0; t < samples && ok; t++) {
        /* always include the corners of each chunk, then pseudo-random entries */
        int ch, i;
        if (t < COMPACT_CHUNKS * 8) {
            ch = t / 8;
            const int corner[8] = {0,1,2,COMPACT_LO/2-1,COMPACT_LO/2,COMPACT_LO/2+1,(int)compact_entries(ch)-2,(int)compact_entries(ch)-1};
            i = corner[t % 8];
        } else {
            seed = seed * 1664525u + 1013904223u;
            ch = (int)(seed >> 28) % COMPACT_CHUNKS;
            i  = (int)((seed >> 4) & (compact_entries(ch) - 1));
        }
        /* want = (2i+1) * 2^compact_shift(ch) * (A/2). */
        compact_require(BN_one(k),"spot-check one");
        compact_require(BN_lshift(k,k,compact_shift(ch)),"spot-check window shift");
        compact_require(BN_mul_word(k,(BN_ULONG)(2*i+1)),"spot-check odd scalar");
        compact_require(BN_mod_mul(k,k,half_nri,order,ctx),"spot-check scalar");
        compact_require(EC_POINT_mul(grp,pt,k,NULL,NULL,ctx),"spot-check point");
        compact_point_to_limbs(grp, pt, x, y, ctx, want);
        size_t off = ((size_t)compact_offset(ch) + i) * 64;
        uint8_t sample[64];
        if(cudaMemcpy(sample,gTable+off,64,cudaMemcpyDeviceToHost)!=cudaSuccess){ok=0;break;}
        if (memcmp(sample,want,32)!=0 || memcmp(sample+32,want+4,32)!=0) {
            fprintf(stderr, "  GTable spot check FAILED at chunk %d entry %d\n", ch, i);
            ok = 0;
        }
    }
    BN_free(x); BN_free(y); BN_free(k); BN_free(inv2); BN_free(order); BN_free(nri); BN_free(half_nri);
    EC_POINT_free(pt); EC_GROUP_free(grp); BN_CTX_free(ctx);
    return ok;
}
static void mixed_build_table(uint8_t *table,const uint8_t neg_r_inv[32]){
    struct timespec begin,end;clock_gettime(CLOCK_MONOTONIC,&begin);
    size_t lb=(size_t)COMPACT_CHUNKS*COMPACT_LO*8*sizeof(uint64_t),hb=(size_t)COMPACT_CHUNKS*COMPACT_HI*8*sizeof(uint64_t);
    uint64_t *hL=(uint64_t*)malloc(lb),*hH=(uint64_t*)malloc(hb);
    if(!hL||!hH){fprintf(stderr,"Host ladder allocation failed\n");exit(2);}
    compact_build_ladders(hL,hH,neg_r_inv);
    uint64_t *L=nullptr,*H=nullptr,*roots=nullptr,*tree=nullptr,*super=nullptr,*root_tree=nullptr;
    const uint32_t chunk=1u<<20;const int blocks=chunk/256,groups=blocks/256;
    wide_cuda_require(cudaMalloc(&L,lb),"allocate L ladder");
    wide_cuda_require(cudaMalloc(&H,hb),"allocate H ladder");
    wide_cuda_require(cudaMemcpy(L,hL,lb,cudaMemcpyHostToDevice),"copy L ladder");
    wide_cuda_require(cudaMemcpy(H,hH,hb,cudaMemcpyHostToDevice),"copy H ladder");
    free(hL);free(hH);
    wide_cuda_require(cudaMalloc(&roots,(size_t)blocks*32),"allocate builder roots");
    wide_cuda_require(cudaMalloc(&tree,(size_t)blocks*4*QSB_CHECKPOINT_STRIDE*8),"allocate builder tree");
    wide_cuda_require(cudaMalloc(&super,(size_t)groups*32),"allocate builder super roots");
    wide_cuda_require(cudaMalloc(&root_tree,(size_t)groups*4*QSB_CHECKPOINT_STRIDE*8),"allocate builder root tree");
    for(uint32_t start=0;start<COMPACT_TOTAL_ENTRIES;start+=chunk){
        int count=(int)((COMPACT_TOTAL_ENTRIES-start)<chunk?(COMPACT_TOTAL_ENTRIES-start):chunk);
        int nblocks=(count+255)/256,ngroups=(nblocks+255)/256;
        mixed_table_prepare<<<nblocks,256>>>(L,H,table,start,count,roots,tree);
        wide_cuda_require(cudaGetLastError(),"table prepare launch");
        qsb_root_group_prepare<<<ngroups,256>>>(roots,nblocks,super,root_tree);
        wide_cuda_require(cudaGetLastError(),"table root prepare launch");
        qsb_invert_super_roots<<<1,256>>>(super,ngroups);
        wide_cuda_require(cudaGetLastError(),"table super inverse launch");
        qsb_root_group_finish<<<ngroups,256>>>(roots,nblocks,super,root_tree);
        wide_cuda_require(cudaGetLastError(),"table root finish launch");
        mixed_table_finish<<<nblocks,256>>>(L,H,table,start,count,roots,tree);
        wide_cuda_require(cudaGetLastError(),"table finish launch");
    }
    wide_cuda_require(cudaDeviceSynchronize(),"table build execution");
    wide_cuda_require(cudaFree(L),"free L ladder");wide_cuda_require(cudaFree(H),"free H ladder");
    wide_cuda_require(cudaFree(roots),"free builder roots");wide_cuda_require(cudaFree(tree),"free builder tree");
    wide_cuda_require(cudaFree(super),"free builder super roots");wide_cuda_require(cudaFree(root_tree),"free builder root tree");
    if(!compact_spot_check(table,COMPACT_CHUNKS*8+256,neg_r_inv)){fprintf(stderr,"Resident table OpenSSL validation failed\n");exit(2);}
    clock_gettime(CLOCK_MONOTONIC,&end);
    double seconds=(end.tv_sec-begin.tv_sec)+(end.tv_nsec-begin.tv_nsec)/1e9;
    printf("  Resident table built in %.2fs: %u entries, %.0f MiB; sampled entries verified\n",seconds,COMPACT_TOTAL_ENTRIES,(double)COMPACT_TOTAL_ENTRIES*64/(1024*1024));fflush(stdout);
}
static void compact_build_table(uint8_t **tableX,uint8_t **tableY,const uint8_t nri[32]){
    wide_cuda_require(cudaMalloc(tableX,(size_t)COMPACT_TOTAL_ENTRIES*64),"allocate resident32 table");
    *tableY=nullptr; // Both coordinates occupy the same interleaved allocation.
    mixed_build_table(*tableX,nri);
}
