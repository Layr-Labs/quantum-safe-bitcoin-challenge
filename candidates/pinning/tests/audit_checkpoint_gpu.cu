/* GPU validation for the actual checkpoint helpers. Run before claiming a gain:
 * nvcc -O3 -DQSB_ZEROS_N=24 -o /tmp/qsb-checkpoint-audit \
 *   candidates/pinning/tests/audit_checkpoint_gpu.cu -lcrypto -lm
 * /tmp/qsb-checkpoint-audit
 * compute-sanitizer --tool synccheck /tmp/qsb-checkpoint-audit
 * compute-sanitizer --tool racecheck /tmp/qsb-checkpoint-audit
 * GPLv3-governed through inclusion of the existing candidate implementation.
 */
#define main qsb_candidate_main
#include "../pinning.cu"
#undef main
#include <array>
#include <random>
#include <vector>

static void checked(cudaError_t status) {
    if(status!=cudaSuccess) {
        fprintf(stderr,"CUDA audit error: %s\n",cudaGetErrorString(status));exit(1);
    }
}

template<int N>
__global__ void audit_up(const uint64_t *values,uint64_t *roots,uint64_t *checkpoint) {
    int idx=blockIdx.x*N+threadIdx.x;
    uint64_t v[5]={values[4*idx],values[4*idx+1],values[4*idx+2],values[4*idx+3],0};
    qsb_block_product_checkpoint<N>(v,roots,checkpoint);
}

template<int N>
__global__ void audit_down(uint64_t *values,const uint64_t *roots,const uint64_t *checkpoint) {
    int idx=blockIdx.x*N+threadIdx.x;
    uint64_t v[5]={values[4*idx],values[4*idx+1],values[4*idx+2],values[4*idx+3],0};
    qsb_block_inverse_checkpoint<N>(v,roots,checkpoint);
    for(int k=0;k<4;k++)values[4*idx+k]=v[k];
}

template<int N>
static int run_audit(std::mt19937_64 &rng) {
    const int blocks=3,count=N*blocks;
    const int active_cases[]={0,1,31,32,33,N-1,N,N+1,2*N+13,count};
    BN_CTX *ctx=BN_CTX_new();
    BIGNUM *p=NULL,*a=BN_new(),*r=BN_new(),*product=BN_new();
    BN_hex2bn(&p,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    int checked_count=0;
    for(int active:active_cases) {
        std::vector<uint64_t> values(4*count),expected(4*count),roots(4*blocks);
        for(int i=0;i<count;i++) {
            for(int k=0;k<4;k++)values[4*i+k]=rng();
            if(i>=active || i%19==0) {
                values[4*i]=1; values[4*i+1]=values[4*i+2]=values[4*i+3]=0;
            } else if(i%23==0) {
                values[4*i]=0xFFFFFFFEFFFFFC2EULL;
                values[4*i+1]=values[4*i+2]=values[4*i+3]=UINT64_MAX;
            }
            BN_lebin2bn((unsigned char*)(values.data()+4*i),32,a);
            if(!BN_mod_inverse(r,a,p,ctx) || BN_bn2lebinpad(r,(unsigned char*)(expected.data()+4*i),32)!=32)exit(2);
        }
        uint64_t *dv=NULL,*dr=NULL,*dc=NULL;
        checked(cudaMalloc(&dv,values.size()*8));
        checked(cudaMalloc(&dr,roots.size()*8));
        checked(cudaMalloc(&dc,(size_t)blocks*4*N*8));
        checked(cudaMemcpy(dv,values.data(),values.size()*8,cudaMemcpyHostToDevice));
        audit_up<N><<<blocks,N>>>(dv,dr,dc);
        checked(cudaGetLastError());checked(cudaDeviceSynchronize());
        checked(cudaMemcpy(roots.data(),dr,roots.size()*8,cudaMemcpyDeviceToHost));
        for(int b=0;b<blocks;b++) {
            BN_one(product);
            for(int j=0;j<N;j++) {
                BN_lebin2bn((unsigned char*)(values.data()+4*(b*N+j)),32,a);
                if(!BN_mod_mul(product,product,a,p,ctx))exit(2);
            }
            BN_lebin2bn((unsigned char*)(roots.data()+4*b),32,a);
            if(!BN_nnmod(a,a,p,ctx) || BN_cmp(a,product)) {
                fprintf(stderr,"FAIL root width=%d active=%d block=%d\n",N,active,b);exit(3);
            }
            if(!BN_mod_inverse(r,a,p,ctx) || BN_bn2lebinpad(r,(unsigned char*)(roots.data()+4*b),32)!=32)exit(2);
        }
        checked(cudaMemcpy(dr,roots.data(),roots.size()*8,cudaMemcpyHostToDevice));
        audit_down<N><<<blocks,N>>>(dv,dr,dc);
        checked(cudaGetLastError());checked(cudaDeviceSynchronize());
        checked(cudaMemcpy(values.data(),dv,values.size()*8,cudaMemcpyDeviceToHost));
        if(values!=expected) { fprintf(stderr,"FAIL inverse width=%d active=%d\n",N,active);exit(4); }
        checked(cudaFree(dv));checked(cudaFree(dr));checked(cudaFree(dc));
        checked_count+=count;
    }
    BN_free(p);BN_free(a);BN_free(r);BN_free(product);BN_CTX_free(ctx);
    return checked_count;
}
int main() {
    std::mt19937_64 rng(0x7173626261727269ULL);
    int count=run_audit<64>(rng)+run_audit<128>(rng)+run_audit<256>(rng);
    printf("PASS: %d GPU inverses match OpenSSL across 30 cases; widths 64/128/256\n",count);
    return 0;
}
