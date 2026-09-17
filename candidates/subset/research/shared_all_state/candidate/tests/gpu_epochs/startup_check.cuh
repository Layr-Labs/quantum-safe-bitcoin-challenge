#pragma once
static void wide_cuda_require(cudaError_t status,const char *where){
    if(status!=cudaSuccess){fprintf(stderr,"%s: %s\n",where,cudaGetErrorString(status));exit(2);}
}
