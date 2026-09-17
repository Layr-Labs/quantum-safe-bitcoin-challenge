#pragma once

// The caller has established its stack and allocated all mandatory buffers,
// including the small table and ranked pipeline. Only allocation exhaustion of
// this optional accelerator is recoverable; other CUDA failures remain fatal.
static uint8_t *qsb_optional_wide_table(bool eligible,const uint8_t base[32]) {
    if(!eligible)return nullptr;
    size_t free_bytes=0,total_bytes=0;
    wide_cuda_require(cudaMemGetInfo(&free_bytes,&total_bytes),"query dual-table free memory");
    const size_t wide_bytes=(size_t)GT_TOTAL_ENTRIES*64;
    const size_t search_reserve=2ull<<30;
    if(free_bytes<wide_bytes+search_reserve){
        printf("  Wide table unavailable within memory budget; using small table\n");
        return nullptr;
    }
    uint8_t *table=nullptr;
    cudaError_t error=cudaMalloc(&table,wide_bytes);
    if(error==cudaErrorMemoryAllocation){
        // Clear that recoverable error before later kernel-launch checks.
        cudaError_t last=cudaGetLastError();
        if(last!=cudaSuccess&&last!=cudaErrorMemoryAllocation)
            wide_cuda_require(last,"unrelated CUDA error during optional allocation");
        printf("  Wide allocation exhausted memory; using small table\n");
        return nullptr;
    }
    wide_cuda_require(error,"allocate optional wide table");
    wide_build_table(table,base);
    return table;
}
