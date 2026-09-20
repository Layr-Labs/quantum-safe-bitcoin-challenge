// Public synthetic QSB benchmark arithmetic audit, never production.
#define main qsb_unused_main
#include "pinning.cu"
#undef main
#include <vector>
__device__ __forceinline__ uint32_t parent_packed_finish(
    const uint64_t *vbar,const uint64_t *tbar,const uint64_t *root_inv,
    const uint64_t *weighted_inv,
    uint64_t *a,uint64_t *b,uint64_t *c,uint64_t *x1,uint64_t *x2) {
    uint64_t u[4],v[4],l[4],m[4],sum[4],t[4],s[4];
    qsb_recovery_mul(u,tbar,weighted_inv);
    qsb_recovery_mul(v,vbar,root_inv);
    _ModSubCanonicalRhs(l,u,v); _ModAdd256(m,u,v); _ModAdd256(sum,l,m);
    _ModSubCanonicalRhs(t,l,c); qsb_recovery_mul(x1,sum,t); _ModAdd256(x1,x1,a);
    _ModSubCanonicalRhs(t,m,c); qsb_recovery_mul(x2,sum,t); _ModAdd256(x2,x2,a);
    _ModSubCanonicalRhs(t,a,x1); qsb_packed_raw_mul(s,l,t); qsb_parity_boundary(s,b);
    uint32_t parity=qsb_difference_parity(s,b);
    _ModSubCanonicalRhs(t,a,x2); qsb_packed_raw_mul(s,m,t); qsb_parity_boundary(s,b);
    return parity|(qsb_difference_parity(b,s)<<1);
}

__global__ void audit_sums(const uint64_t *in, uint64_t *out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    uint64_t x[4],y[4],a[4],b[4],c[4];
    const uint64_t *q=in+28ull*i;
    for(int k=0;k<4;k++){a[k]=q[16+k];b[k]=q[20+k];c[k]=q[24+k];}
    uint32_t flags=qsb_packed_finish(q,q+4,q+8,q+12,a,b,c,x,y);
    for(int k=0;k<4;k++){out[18ull*i+k]=x[k];out[18ull*i+4+k]=y[k];}
    out[18ull*i+8]=flags;
    flags=parent_packed_finish(q,q+4,q+8,q+12,a,b,c,x,y);
    for(int k=0;k<4;k++){out[18ull*i+9+k]=x[k];out[18ull*i+13+k]=y[k];}
    out[18ull*i+17]=flags;
}
static void audit_check(cudaError_t e) {
    if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(3);}
}
int main(int argc,char **argv) {
    if(argc!=3)return 2;
    FILE *f=fopen(argv[1],"rb");if(!f)return 2;
    fseek(f,0,SEEK_END);long bytes=ftell(f);rewind(f);
    if(bytes<=0||bytes%(28*8))return 2;
    int n=bytes/(28*8);std::vector<uint64_t> in(bytes/8),out(18ull*n);
    if(fread(in.data(),1,bytes,f)!=(size_t)bytes)return 2;fclose(f);
    uint64_t *di,*dout;audit_check(cudaMalloc(&di,bytes));
    audit_check(cudaMalloc(&dout,out.size()*8));
    audit_check(cudaMemcpy(di,in.data(),bytes,cudaMemcpyHostToDevice));
    audit_sums<<<(n+127)/128,128>>>(di,dout,n);audit_check(cudaGetLastError());
    audit_check(cudaDeviceSynchronize());
    audit_check(cudaMemcpy(out.data(),dout,out.size()*8,cudaMemcpyDeviceToHost));
    f=fopen(argv[2],"wb");if(!f)return 2;
    if(fwrite(out.data(),8,out.size(),f)!=out.size())return 2;
    if(fclose(f))return 2;audit_check(cudaFree(di));audit_check(cudaFree(dout));
    return 0;
}
