// CPU/GPU startup uses bounded scratch; no whole-table host copy or disk cache.
#pragma once
static void wide_cuda_require(cudaError_t status,const char *where){
    if(status!=cudaSuccess){fprintf(stderr,"%s: %s\n",where,cudaGetErrorString(status));exit(2);}
}
static void wide_build_table(uint8_t *table,const uint8_t neg_r_inv[32]){
    struct timespec begin,end;clock_gettime(CLOCK_MONOTONIC,&begin);
    size_t lb=(size_t)GT_CHUNKS*GT_LO*8*sizeof(uint64_t),hb=(size_t)GT_CHUNKS*GT_HI*8*sizeof(uint64_t);
    uint64_t *hL=(uint64_t*)malloc(lb),*hH=(uint64_t*)malloc(hb);
    if(!hL||!hH){fprintf(stderr,"Host ladder allocation failed\n");exit(2);}
    gt_build_ladders(hL,hH,neg_r_inv);
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
    for(uint32_t start=0;start<GT_TOTAL_ENTRIES;start+=chunk){
        int count=(int)((GT_TOTAL_ENTRIES-start)<chunk?(GT_TOTAL_ENTRIES-start):chunk);
        int nblocks=(count+255)/256,ngroups=(nblocks+255)/256;
        wide_table_prepare<<<nblocks,256>>>(L,H,table,start,count,roots,tree);
        wide_cuda_require(cudaGetLastError(),"table prepare launch");
        qsb_root_group_prepare<<<ngroups,256>>>(roots,nblocks,super,root_tree);
        wide_cuda_require(cudaGetLastError(),"table root prepare launch");
        qsb_invert_super_roots<<<1,256>>>(super,ngroups);
        wide_cuda_require(cudaGetLastError(),"table super inverse launch");
        qsb_root_group_finish<<<ngroups,256>>>(roots,nblocks,super,root_tree);
        wide_cuda_require(cudaGetLastError(),"table root finish launch");
        wide_table_finish<<<nblocks,256>>>(L,H,table,start,count,roots,tree);
        wide_cuda_require(cudaGetLastError(),"table finish launch");
    }
    wide_cuda_require(cudaDeviceSynchronize(),"table build execution");
    wide_cuda_require(cudaFree(L),"free L ladder");wide_cuda_require(cudaFree(H),"free H ladder");
    wide_cuda_require(cudaFree(roots),"free builder roots");wide_cuda_require(cudaFree(tree),"free builder tree");
    wide_cuda_require(cudaFree(super),"free builder super roots");wide_cuda_require(cudaFree(root_tree),"free builder root tree");
    if(!gt_spot_check(table,GT_CHUNKS*8+256,neg_r_inv)){fprintf(stderr,"Wide table OpenSSL validation failed\n");exit(2);}
    clock_gettime(CLOCK_MONOTONIC,&end);
    double seconds=(end.tv_sec-begin.tv_sec)+(end.tv_nsec-begin.tv_nsec)/1e9;
    printf("  Wide table built in %.2fs: %u entries, %.0f MiB; sampled entries verified\n",seconds,GT_TOTAL_ENTRIES,(double)GT_TOTAL_ENTRIES*64/(1024*1024));fflush(stdout);
}
