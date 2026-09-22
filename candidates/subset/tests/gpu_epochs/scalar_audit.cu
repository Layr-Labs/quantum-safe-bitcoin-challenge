// Actual device scalar setup/direct digits versus independent bigint fixtures.
// recode_base_a_model.py emits the QSB_RECODE_BASE_A=1 fixture used by Round 12B;
// compile this audit with the same switch as the fixture convention.
#define main qsb_grinder_main
#include "tree.cu"
#undef main
#include <vector>
#define CHECK(call) do {cudaError_t e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));return 2;}}while(0)
__global__ void scalar_audit_kernel(const uint64_t *input,uint64_t *setup,uint32_t *digits,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    uint64_t k[4],M[4];for(int j=0;j<4;j++)k[j]=input[i*4+j];
    int sign;gt_recode_setup(k,M,&sign);
    for(int j=0;j<4;j++)setup[i*5+j]=M[j];setup[i*5+4]=(uint64_t)(int64_t)sign;
    unsigned pos=1;
    for(int c=0;c<GT_CHUNKS;c++){
        uint32_t idx;uint64_t neg;
        gt_direct_digit(M,(uint64_t)(sign<0),pos,gt_width(c),c==GT_CHUNKS-1,&idx,&neg);
        digits[i*GT_CHUNKS+c]=idx|((uint32_t)neg<<31);pos+=gt_width(c);
    }
}
int main(int argc,char **argv){
    if(argc!=2 || GT_CHUNKS!=15)return 2;
    FILE *f=fopen(argv[1],"rb");if(!f)return 2;
    uint32_t n=0;if(fread(&n,4,1,f)!=1 || !n || n>1000000)return 2;
    std::vector<uint64_t> input(n*4),expected(n*5),out(n*5);
    std::vector<uint32_t> digits(n*15),want_digits(n*15);
    for(unsigned i=0;i<n;i++){
        int32_t sign;
        if(fread(input.data()+i*4,8,4,f)!=4 || fread(expected.data()+i*5,8,4,f)!=4 ||
           fread(&sign,4,1,f)!=1 || fread(want_digits.data()+i*15,4,15,f)!=15)return 2;
        expected[i*5+4]=(uint64_t)(int64_t)sign;
    }
    if(fgetc(f)!=EOF)return 2;fclose(f);
    uint64_t *di,*ds;uint32_t *dd;
    CHECK(cudaMalloc(&di,input.size()*8));CHECK(cudaMalloc(&ds,out.size()*8));CHECK(cudaMalloc(&dd,digits.size()*4));
    CHECK(cudaMemcpy(di,input.data(),input.size()*8,cudaMemcpyHostToDevice));
    scalar_audit_kernel<<<(n+255)/256,256>>>(di,ds,dd,n);CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(),ds,out.size()*8,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(digits.data(),dd,digits.size()*4,cudaMemcpyDeviceToHost));
    int bad=0;
    for(unsigned i=0;i<n;i++){
        if(memcmp(out.data()+i*5,expected.data()+i*5,40))bad++;
        for(int c=0;c<15;c++)if(digits[i*15+c]!=want_digits[i*15+c])bad++;
    }
    printf("Independent scalar fixtures: %u inputs, %u setup/digit checks, mismatches=%d\n",n,n*16,bad);
    cudaFree(di);cudaFree(ds);cudaFree(dd);return bad?1:0;
}
