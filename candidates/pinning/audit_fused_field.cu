#define main qsb_unused_main
#include "pinning.cu"
#undef main
#include <vector>
__global__ void audit_fused(const uint64_t *in,uint64_t *out,unsigned n) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    uint64_t a[4],b[4],c[4],r[4];
    #pragma unroll
    for(int j=0;j<4;j++){a[j]=in[12ull*i+j];b[j]=in[12ull*i+4+j];c[j]=in[12ull*i+8+j];}
    _ModSqrAddSub2(r,a,b,c);for(int j=0;j<4;j++)out[16ull*i+j]=r[j];
    Load256(r,a);_ModSqrAddSub2(r,r,b,c);for(int j=0;j<4;j++)out[16ull*i+4+j]=r[j];
    Load256(r,b);_ModSqrAddSub2(r,a,r,c);for(int j=0;j<4;j++)out[16ull*i+8+j]=r[j];
    Load256(r,c);_ModSqrAddSub2(r,a,b,r);for(int j=0;j<4;j++)out[16ull*i+12+j]=r[j];
}
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));return 3;}}while(0)
int main(int argc,char **argv) {
    if(argc!=3)return 2;FILE*f=fopen(argv[1],"rb");if(!f)return 2;unsigned n;
    if(fread(&n,4,1,f)!=1||!n||n>1000000)return 2;
    std::vector<uint64_t>in(12ull*n),out(16ull*n);if(fread(in.data(),96,n,f)!=n)return 2;fclose(f);
    uint64_t *di,*doo;CK(cudaMalloc(&di,in.size()*8));CK(cudaMalloc(&doo,out.size()*8));
    CK(cudaMemcpy(di,in.data(),in.size()*8,cudaMemcpyHostToDevice));
    audit_fused<<<(n+127)/128,128>>>(di,doo,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out.data(),doo,out.size()*8,cudaMemcpyDeviceToHost));f=fopen(argv[2],"wb");if(!f)return 2;
    bool ok=fwrite(out.data(),128,n,f)==n;ok=(fclose(f)==0)&&ok;CK(cudaFree(di));CK(cudaFree(doo));return ok?0:2;
}
