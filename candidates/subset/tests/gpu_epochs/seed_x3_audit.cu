#define main qsb_subset_program_main
#include "tree.cu"
#undef main
#include <vector>

struct SeedRow {
    uint64_t input[12], canonical[4], raw[4];
    uint32_t guard;
};

__global__ void audit_seed_x3(const SeedRow *rows, unsigned count, unsigned *errors) {
    unsigned task=blockIdx.x*blockDim.x+threadIdx.x;
    if(task>=count*16u)return;
    const SeedRow &row=rows[task/16u];
    const unsigned alias=task&3u;
    const uint32_t prior_values[4]={0,1,0x80000000u,7};
    const uint32_t prior=prior_values[(task>>2)&3u];
    uint64_t a[5],b[5],c[5],out[5];
    for(int j=0;j<4;j++){a[j]=row.input[j];b[j]=row.input[4+j];c[j]=row.input[8+j];}
    const uint64_t sentinel=0x859ab7462301cdfeULL;
    a[4]=b[4]=c[4]=out[4]=sentinel;
    uint64_t *result=alias==0?out:alias==1?a:alias==2?b:c;
    uint32_t flag=prior;
    qsb_replay_seed_x3(result,a,b,c,flag);
    unsigned bad=flag!=(prior|row.guard);
    for(int j=0;j<4;j++) {
        bad+=result[j]!=row.raw[j];
        if(!row.guard)bad+=result[j]!=row.canonical[j];
    }
    bad+=a[4]!=sentinel || b[4]!=sentinel || c[4]!=sentinel || out[4]!=sentinel;
    if(bad)atomicAdd(errors,bad);
}

int main(int argc,char **argv) {
    if(argc!=2)return 2;
    FILE *f=fopen(argv[1],"rb");if(!f)return 2;
    uint32_t count=0;
    if(fread(&count,4,1,f)!=1 || count!=7343)return 2;
    std::vector<SeedRow> rows(count);
    for(auto &row:rows) {
        if(fread(row.input,8,12,f)!=12 || fread(row.canonical,8,4,f)!=4 ||
           fread(row.raw,8,4,f)!=4 || fread(&row.guard,4,1,f)!=1)return 2;
    }
    if(fgetc(f)!=EOF)return 2;
    fclose(f);
    SeedRow *device_rows=nullptr;unsigned *device_errors=nullptr,errors=0;
#define SEED_CUDA(call) do { cudaError_t e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));return 3;} } while(0)
    SEED_CUDA(cudaMalloc(&device_rows,rows.size()*sizeof(SeedRow)));
    SEED_CUDA(cudaMalloc(&device_errors,sizeof(unsigned)));
    SEED_CUDA(cudaMemcpy(device_rows,rows.data(),rows.size()*sizeof(SeedRow),cudaMemcpyHostToDevice));
    SEED_CUDA(cudaMemset(device_errors,0,sizeof(unsigned)));
    audit_seed_x3<<<(count*16u+255u)/256u,256>>>(device_rows,count,device_errors);
    SEED_CUDA(cudaGetLastError());
    SEED_CUDA(cudaMemcpy(&errors,device_errors,sizeof(unsigned),cudaMemcpyDeviceToHost));
    SEED_CUDA(cudaFree(device_rows));SEED_CUDA(cudaFree(device_errors));
    printf("Seed X3: %u fixtures, %u calls (4 output aliases x 4 prior flags), %u errors\n",count,count*16u,errors);
    return errors?1:0;
}
