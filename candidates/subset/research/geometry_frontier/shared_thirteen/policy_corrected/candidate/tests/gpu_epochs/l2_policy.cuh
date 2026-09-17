#pragma once
// Optional policy for this experiment's48MiB small-table prefix. A cache
// priority is not residency. Inspired by ercumentyildirim's public e2fd8093
// hardware observation; independently implemented with checked API results.
static bool qsb_l2_unavailable(cudaError_t error){
    bool unavailable=error==cudaErrorNotSupported || error==cudaErrorUnsupportedLimit ||
                     error==cudaErrorInvalidValue;
    if(unavailable){
        cudaError_t last=cudaGetLastError();
        if(last!=cudaSuccess && last!=error)
            wide_cuda_require(last,"unrelated CUDA error during optional L2 policy");
    }
    return unavailable;
}
static bool qsb_enable_small_l2_policy(void *table,int device){
    constexpr size_t hot_bytes=48ull<<20;
    wide_cuda_require((uint64_t)mixed_offset(9)*64==hot_bytes?cudaSuccess:cudaErrorInvalidValue,
                      "L2 policy must cover only the nine small windows");
    int max_persist=0,max_window=0;
    cudaError_t error=cudaDeviceGetAttribute(&max_persist,cudaDevAttrMaxPersistingL2CacheSize,device);
    if(qsb_l2_unavailable(error))return false;
    wide_cuda_require(error,"query persisting L2 capacity");
    error=cudaDeviceGetAttribute(&max_window,cudaDevAttrMaxAccessPolicyWindowSize,device);
    if(qsb_l2_unavailable(error))return false;
    wide_cuda_require(error,"query L2 access window capacity");
    if(max_persist<0 || max_window<0 || (size_t)max_persist<hot_bytes || (size_t)max_window<hot_bytes){
        printf("  Small-table L2 policy unavailable: need48MiB, capacities%d/%d bytes\n",max_persist,max_window);
        return false;
    }
    size_t previous=0;
    error=cudaDeviceGetLimit(&previous,cudaLimitPersistingL2CacheSize);
    if(qsb_l2_unavailable(error))return false;
    wide_cuda_require(error,"read previous persisting L2 limit");
    error=cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,hot_bytes);
    if(qsb_l2_unavailable(error))return false;
    wide_cuda_require(error,"set small-table persisting L2 limit");
    size_t effective=0;
    wide_cuda_require(cudaDeviceGetLimit(&effective,cudaLimitPersistingL2CacheSize),"read effective persisting L2 limit");
    if(effective<hot_bytes){
        wide_cuda_require(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,previous),"restore smaller L2 limit");
        return false;
    }
    cudaStreamAttrValue attr={};
    attr.accessPolicyWindow.base_ptr=table;
    attr.accessPolicyWindow.num_bytes=hot_bytes;
    attr.accessPolicyWindow.hitRatio=1.0f;
    attr.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp=cudaAccessPropertyNormal;
    error=cudaStreamSetAttribute(0,cudaStreamAttributeAccessPolicyWindow,&attr);
    if(qsb_l2_unavailable(error)){
        wide_cuda_require(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,previous),"restore unused L2 limit");
        return false;
    }
    wide_cuda_require(error,"set small-table L2 access policy");
    printf("  Small-table L2 policy enabled for48MiB; effective set-aside%zu bytes, residency not guaranteed\n",effective);
    fflush(stdout);
    return true;
}
