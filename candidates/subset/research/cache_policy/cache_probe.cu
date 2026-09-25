#include <cuda_runtime.h>
#include <stdio.h>
int main(){
 cudaDeviceProp p={}; cudaError_t e=cudaGetDeviceProperties(&p,0);
 if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 1;}
 printf("gpu=%s l2=%d max_persist=%d max_window=%d\n",p.name,p.l2CacheSize,p.persistingL2CacheMaxSize,p.accessPolicyMaxWindowSize);
 size_t actual=0; e=cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,p.persistingL2CacheMaxSize);
 printf("set_limit=%s\n",cudaGetErrorString(e)); if(e!=cudaSuccess)return 2;
 e=cudaDeviceGetLimit(&actual,cudaLimitPersistingL2CacheSize); if(e!=cudaSuccess)return 3;
 void *table=nullptr;e=cudaMalloc(&table,64ULL*1024*1024);if(e!=cudaSuccess)return 4;
 cudaStreamAttrValue attr={};attr.accessPolicyWindow.base_ptr=table;attr.accessPolicyWindow.num_bytes=64ULL*1024*1024;
 attr.accessPolicyWindow.hitRatio=(float)actual/(64ULL*1024*1024);attr.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;attr.accessPolicyWindow.missProp=cudaAccessPropertyNormal;
 e=cudaStreamSetAttribute(0,cudaStreamAttributeAccessPolicyWindow,&attr);
 printf("actual_reserve=%zu ratio=%f set_policy=%s\n",actual,attr.accessPolicyWindow.hitRatio,cudaGetErrorString(e));
 cudaFree(table);return e==cudaSuccess?0:5;
}
