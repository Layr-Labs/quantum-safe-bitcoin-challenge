// SPDX-License-Identifier: GPL-3.0-only
// Execute the production recovery primitives, including aliasing/carry edges.
#define main qsb_grinder_unused_main
#include "../pinning.cu"
#undef main

__global__ void recovery_field_vectors(const uint64_t *input, uint64_t *output,
                                       unsigned n) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    uint64_t a[4],b[4],r[4];
    for(int k=0;k<4;k++) {a[k]=input[(size_t)i*8+k];b[k]=input[(size_t)i*8+4+k];}
    qsb_recovery_mul(r,a,b);
    for(int k=0;k<4;k++)output[(size_t)i*20+k]=r[k];
    Load256(r,a);qsb_recovery_mul(r,r,b);
    for(int k=0;k<4;k++)output[(size_t)i*20+4+k]=r[k];
    Load256(r,b);qsb_recovery_mul(r,a,r);
    for(int k=0;k<4;k++)output[(size_t)i*20+8+k]=r[k];
    qsb_field_normalize(a);qsb_field_normalize(b);
    _ModAdd256(r,a,b);
    for(int k=0;k<4;k++)output[(size_t)i*20+12+k]=r[k];
    _ModSub256(r,a,b);
    for(int k=0;k<4;k++)output[(size_t)i*20+16+k]=r[k];
}

#define QSB_TEST_CUDA(call) do {cudaError_t e=(call);if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA field test: %s: %s\n",#call,cudaGetErrorString(e));return 2;}} while(0)

int main(int argc,char **argv) {
    if(argc!=3){fprintf(stderr,"usage: native_field_cuda INPUT OUTPUT\n");return 2;}
    FILE *f=fopen(argv[1],"rb");if(!f)return 2;
    uint32_t n=0;
    if(fread(&n,4,1,f)!=1 || !n || n>1000000){fclose(f);return 2;}
    size_t inbytes=(size_t)n*8*sizeof(uint64_t),outbytes=(size_t)n*20*sizeof(uint64_t);
    uint64_t *input=(uint64_t*)malloc(inbytes),*output=(uint64_t*)malloc(outbytes);
    if(!input||!output)return 2;
    if(fread(input,1,inbytes,f)!=inbytes){fclose(f);return 2;}
    fclose(f);
    uint64_t *d_input=nullptr,*d_output=nullptr;
    QSB_TEST_CUDA(cudaMalloc(&d_input,inbytes));
    QSB_TEST_CUDA(cudaMalloc(&d_output,outbytes));
    QSB_TEST_CUDA(cudaMemcpy(d_input,input,inbytes,cudaMemcpyHostToDevice));
    recovery_field_vectors<<<(n+127)/128,128>>>(d_input,d_output,n);
    QSB_TEST_CUDA(cudaGetLastError());
    QSB_TEST_CUDA(cudaDeviceSynchronize());
    QSB_TEST_CUDA(cudaMemcpy(output,d_output,outbytes,cudaMemcpyDeviceToHost));
    QSB_TEST_CUDA(cudaFree(d_input));QSB_TEST_CUDA(cudaFree(d_output));
    f=fopen(argv[2],"wb");if(!f)return 2;
    bool ok=fwrite(output,1,outbytes,f)==outbytes;
    ok=(fclose(f)==0)&&ok;
    free(input);free(output);
    return ok?0:2;
}
