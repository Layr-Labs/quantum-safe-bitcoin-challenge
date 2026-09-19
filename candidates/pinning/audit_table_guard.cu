#include <vector>
#define main qsb_unused_main
#include "pinning.cu"
#undef main

__global__ void audit_loaders(const uint8_t *table, const uint64_t *masks,
                             uint64_t *out, unsigned count) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    gt_load_signed_flat_m(table,0,i,masks[i],out+16ull*i,out+16ull*i+4);
    gt_load_signed_flat_m_guarded(table,0,i,masks[i],out+16ull*i+8,out+16ull*i+12);
}

int main(int argc, char **argv) {
    if(argc!=3)return 2;
    FILE *f=fopen(argv[1],"rb");if(!f)return 2;
    std::vector<uint64_t> records;uint64_t row[5];
    while(fread(row,sizeof(row),1,f)==1)records.insert(records.end(),row,row+5);
    if(ferror(f)||!feof(f)||records.empty()||records.size()%5)return 2;
    fclose(f);
    unsigned count=records.size()/5;
    std::vector<uint8_t> table(64ull*count),safe;
    std::vector<uint64_t> masks(count),out(16ull*count);
    unsigned fallback=0;
    for(unsigned i=0;i<count;i++) {
        for(int j=0;j<4;j++) {
            uint64_t x=0x123456789abcdef0ULL+(uint64_t)i*17+j;
            memcpy(table.data()+64ull*i+8*j,&x,8);
        }
        memcpy(table.data()+64ull*i+32,records.data()+5ull*i,32);
        masks[i]=records[5ull*i+4];
        bool expected=records[5ull*i]<=0xFFFFFFFEFFFFFC2FULL;
        if(qsb_table_y_range_ok(table.data()+64ull*i,1)!=expected)return 3;
        if(expected)safe.insert(safe.end(),table.data()+64ull*i,table.data()+64ull*(i+1));
        else fallback++;
    }
    if(!fallback||safe.empty()||qsb_table_y_range_ok(table.data(),count)
       ||!qsb_table_y_range_ok(safe.data(),safe.size()/64))return 3;
    for(size_t i:{size_t(0),safe.size()/128,safe.size()/64-1}) {
        uint64_t old,bad=UINT64_MAX;
        memcpy(&old,safe.data()+64*i+32,8);
        memcpy(safe.data()+64*i+32,&bad,8);
        if(qsb_table_y_range_ok(safe.data(),safe.size()/64))return 3;
        memcpy(safe.data()+64*i+32,&old,8);
    }
    uint8_t *dt=nullptr;uint64_t *dm=nullptr,*doo=nullptr;
    if(cudaMalloc(&dt,table.size())!=cudaSuccess
       ||cudaMalloc(&dm,masks.size()*8)!=cudaSuccess
       ||cudaMalloc(&doo,out.size()*8)!=cudaSuccess)return 4;
    if(cudaMemcpy(dt,table.data(),table.size(),cudaMemcpyHostToDevice)!=cudaSuccess
       ||cudaMemcpy(dm,masks.data(),masks.size()*8,cudaMemcpyHostToDevice)!=cudaSuccess)return 4;
    audit_loaders<<<(count+127)/128,128>>>(dt,dm,doo,count);
    if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess
       ||cudaMemcpy(out.data(),doo,out.size()*8,cudaMemcpyDeviceToHost)!=cudaSuccess)return 4;
    f=fopen(argv[2],"wb");if(!f)return 2;
    if(fwrite(out.data(),out.size()*8,1,f)!=1||fclose(f))return 2;
    cudaFree(dt);cudaFree(dm);cudaFree(doo);
    printf("{\"cuda_completed\":true,\"guard_scan_passed\":true,\"cases\":%u,\"fallback\":%u}\n",count,fallback);
    return 0;
}
