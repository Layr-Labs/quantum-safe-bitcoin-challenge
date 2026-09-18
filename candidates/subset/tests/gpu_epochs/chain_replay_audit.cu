// Exact-source field guard and fallback audit. Diagnostic only.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include "../../GPUMath.h"
#include "../../chain_replay_field.cuh"

struct Input { uint64_t a[4], b[4], expected[3][4]; };
struct Output { uint64_t value[3][3][2][4]; uint32_t flags[3][3][2]; uint32_t guards[3][3][2]; };
static_assert(sizeof(Input)==160,"fixture layout");
__global__ void audit_replay_field(const Input *input, Output *output, unsigned count){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    for(unsigned op=0;op<3;op++)for(unsigned alias=0;alias<3;alias++)for(unsigned prior=0;prior<2;prior++){
        uint64_t a[4],b[4],separate[4];
        Load256(a,input[i].a);Load256(b,input[i].b);
        uint64_t *raw=alias==0?separate:(alias==1?a:b);
        uint32_t bad=prior?0x80000000u:0;
        if(op==0)qsb_replay_add(raw,a,b,bad);
        else if(op==1)qsb_replay_mul(raw,a,b,bad);
        else qsb_replay_sqr(raw,a,bad);
        output[i].flags[op][alias][prior]=bad;
        output[i].guards[op][alias][prior]=qsb_replay_guard(raw[3]);
        if(bad){
            // Replay from the preserved original inputs, as the chain does.
            uint64_t saved_a[4],saved_b[4];
            Load256(saved_a,input[i].a);Load256(saved_b,input[i].b);
            if(op==0)_ModAdd256(raw,saved_a,saved_b);
            else if(op==1)_ModMultCore(raw,saved_a,saved_b);
            else _ModSqr(raw,saved_a);
        }
        Load256(output[i].value[op][alias][prior],raw);
    }
}
#define CHECK(call) do{cudaError_t e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));return 2;}}while(0)
int main(int argc,char **argv){
    if(argc!=2)return 2;
    FILE *f=fopen(argv[1],"rb");if(!f)return 2;
    uint32_t n;if(fread(&n,4,1,f)!=1||!n||n>1000000)return 2;
    std::vector<Input> in(n);std::vector<Output> out(n);
    if(fread(in.data(),sizeof(Input),n,f)!=n||fgetc(f)!=EOF)return 2;fclose(f);
    Input *di;Output *dout;
    CHECK(cudaMalloc(&di,in.size()*sizeof(Input)));CHECK(cudaMalloc(&dout,out.size()*sizeof(Output)));
    CHECK(cudaMemcpy(di,in.data(),in.size()*sizeof(Input),cudaMemcpyHostToDevice));
    audit_replay_field<<<(n+127)/128,128>>>(di,dout,n);
    CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(),dout,out.size()*sizeof(Output),cudaMemcpyDeviceToHost));
    unsigned errors=0, natural_guards[3]={0,0,0}, false_guards[3]={0,0,0};
    for(unsigned i=0;i<n;i++)for(unsigned op=0;op<3;op++)for(unsigned alias=0;alias<3;alias++)for(unsigned prior=0;prior<2;prior++){
        bool bad=false;
        for(unsigned j=0;j<4;j++)bad|=out[i].value[op][alias][prior][j]!=in[i].expected[op][j];
        uint32_t guard=out[i].guards[op][alias][prior];
        bad|=guard>1||out[i].flags[op][alias][prior]!=((prior?0x80000000u:0)|guard);
        if(!prior){natural_guards[op]+=guard!=0;false_guards[op]+=guard==0;}
        if(bad){if(errors<8)printf("Replay mismatch case=%u op=%u alias=%u prior=%u\n",i,op,alias,prior);errors++;}
    }
    for(unsigned op=0;op<3;op++)if(!natural_guards[op]||!false_guards[op])errors++;
    printf("Replay field audit: cases=%u values=%u guard_true=%u/%u/%u guard_false=%u/%u/%u errors=%u\n",n,n*18,
        natural_guards[0],natural_guards[1],natural_guards[2],false_guards[0],false_guards[1],false_guards[2],errors);
    cudaFree(di);cudaFree(dout);return errors?1:0;
}
