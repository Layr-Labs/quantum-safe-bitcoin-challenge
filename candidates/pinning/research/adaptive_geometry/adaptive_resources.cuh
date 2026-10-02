#pragma once
static uint8_t *adaptive_optional_wide(bool eligible,const uint8_t nri[32]){
    if(!eligible)return nullptr;
    size_t free_bytes=0,total_bytes=0;
    wide_cuda_require(cudaMemGetInfo(&free_bytes,&total_bytes),"query optional table memory");
    // Search state/trees have already been allocated. The largest builder
    // working set is below 64 MiB (ladders, one million-entry tree and roots).
    const size_t table_bytes=(size_t)WIDE_TOTAL_ENTRIES*64;
    if(free_bytes<table_bytes+(64ull<<20))return nullptr;
    uint8_t *table=nullptr;
    cudaError_t e=cudaMalloc(&table,table_bytes);
    if(e==cudaErrorMemoryAllocation){cudaGetLastError();return nullptr;}
    wide_cuda_require(e,"allocate optional wide table");
    wide_build_table(table,nri);
    return table;
}
