// Diagnostic only: fused trial formula versus separate C++ field calls.
// Also compares every result and flag with the separately generated CPU fixture.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include "../../GPUMath.h"
#include "../../chain_replay_field.cuh"
__device__ __forceinline__ void qsb_replay_x3_reference(
    uint64_t *r,const uint64_t *a,const uint64_t *b,const uint64_t *c,uint32_t &bad){
#ifdef __CUDA_ARCH__
    uint64_t r0,r1,r2,r3,high;
    // a+b-2c = low+h*2^256, h in {-2,-1,0,1}, for ALL 256-bit inputs.
    // low+h*K lies in (-2K,2^256+K). Retain its signed high word.
    asm("{\n\t.reg .u64 h,t,ext;\n\t"
        "add.cc.u64 %0,%5,%9;\n\t"
        "addc.cc.u64 %1,%6,%10;\n\t"
        "addc.cc.u64 %2,%7,%11;\n\t"
        "addc.cc.u64 %3,%8,%12;\n\t"
        "addc.u64 h,0,0;\n\t"
        "sub.cc.u64 %0,%0,%13;\n\t"
        "subc.cc.u64 %1,%1,%14;\n\t"
        "subc.cc.u64 %2,%2,%15;\n\t"
        "subc.cc.u64 %3,%3,%16;\n\t"
        "subc.u64 h,h,0;\n\t"
        "sub.cc.u64 %0,%0,%13;\n\t"
        "subc.cc.u64 %1,%1,%14;\n\t"
        "subc.cc.u64 %2,%2,%15;\n\t"
        "subc.cc.u64 %3,%3,%16;\n\t"
        "subc.u64 h,h,0;\n\t"
        "shr.s64 ext,h,63;\n\t"
        "mul.lo.u64 t,h,0x1000003d1;\n\t"
        "add.cc.u64 %0,%0,t;\n\t"
        "addc.cc.u64 %1,%1,ext;\n\t"
        "addc.cc.u64 %2,%2,ext;\n\t"
        "addc.cc.u64 %3,%3,ext;\n\t"
        "addc.u64 %4,ext,0;\n\t}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3),"=l"(high)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
          "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]),
          "l"(c[0]),"l"(c[1]),"l"(c[2]),"l"(c[3]));
    bad |= qsb_replay_guard(r3);
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
#else
    _ModAdd256(r,(uint64_t*)a,(uint64_t*)b);
    _ModSub256(r,r,(uint64_t*)c);_ModSub256(r,r,(uint64_t*)c);
#endif
}
template<bool DEFER_Y>
__device__ __forceinline__ void qsb_replay_point_add_reference(
    uint64_t *__restrict__ X1, uint64_t *__restrict__ Y1,
    uint64_t *__restrict__ ZZ1, uint64_t *__restrict__ ZZZ1,
    const uint64_t *__restrict__ X2, const uint64_t *__restrict__ Y2,
    const uint64_t *__restrict__ Yoff, uint32_t &bad)
{
  uint64_t U2[4];
  uint64_t S2[4];
  uint64_t P[4];
  uint64_t R[4];
  uint64_t PP[4];
  uint64_t PPP[4];
  uint64_t Q[4];
  uint64_t T[4];

  qsb_replay_mul(U2, (uint64_t *)X2, ZZ1,bad);   // U2 = X2*ZZ1
  qsb_replay_add(S2, (uint64_t *)Y2, (uint64_t *)Yoff,bad);
  qsb_replay_mul(S2, ZZZ1,bad);                  // S2 = (Y2+Yoff)*ZZZ1
  _ModSub256(P, U2, X1);               // P  = U2 - X1
  _ModSub256(R, S2, Y1);               // R  = S2 - Y1
  qsb_replay_sqr(PP, P,bad);                      // PP = P^2
  qsb_replay_mul(PPP, PP, P,bad);                // PPP = P*PP
  qsb_replay_mul(Q, U2, PP,bad);                 // V  = U2*PP
  qsb_replay_mul(ZZ1, PP,bad);                   // ZZ3; PP dies before the R^2/Y3 tail

  qsb_replay_sqr(T, R,bad);                       // R^2
  qsb_replay_x3_reference(T,T,PPP,Q,bad);

  qsb_replay_mul(ZZZ1, PPP,bad);                 // ZZZ3
  _ModSub256(Q, Q, T);                 // V - X3
  qsb_replay_mul(Q, R,bad);                      // R*(V - X3)
  if (DEFER_Y) {
    Load256(Y1, Q);                    // actual Y3 = Y1 - Y2*ZZZ3
  } else {
    qsb_replay_mul(S2, (uint64_t *)Y2, ZZZ1,bad);// affine Y2*ZZZ3
    _ModSub256(Y1, Q, S2);             // exact Y3
  }

  Load256(X1, T);                      // X3
}
struct Input {uint64_t fields[7][4];};
struct Output {uint64_t fused[4][16],reference[4][16];uint32_t flags[4],ref_flags[4];};
__global__ void audit_point_predicate(const Input*in,Output*out,unsigned n){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const uint32_t priors[4]={0,1,0x80000000u,0xffffffffu};
    for(unsigned mode=0;mode<4;mode++){
        uint64_t a[16],b[16];
        for(unsigned j=0;j<16;j++)a[j]=b[j]=in[i].fields[j/4][j%4];
        uint32_t fa=priors[mode],fb=priors[mode];
        qsb_replay_point_add<true>(a,a+4,a+8,a+12,
            in[i].fields[4],in[i].fields[5],in[i].fields[6],fa);
        qsb_replay_point_add_reference<true>(b,b+4,b+8,b+12,
            in[i].fields[4],in[i].fields[5],in[i].fields[6],fb);
        for(unsigned j=0;j<16;j++){out[i].fused[mode][j]=a[j];out[i].reference[mode][j]=b[j];}
        out[i].flags[mode]=fa;out[i].ref_flags[mode]=fb;
    }
}
#define CHECK(call) do{cudaError_t e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));return 2;}}while(0)
int main(int argc,char**argv){
    if(argc!=2)return 2;FILE*f=fopen(argv[1],"rb");if(!f)return 2;
    uint32_t n;if(fread(&n,4,1,f)!=1||!n||n>1000000)return 2;
    std::vector<Input>in(n);std::vector<Output>out(n);
    std::vector<uint64_t>expected(n*16);std::vector<uint32_t>guards(n);
    for(unsigned i=0;i<n;i++){
        if(fread(in[i].fields,8,28,f)!=28||fread(expected.data()+i*16,8,16,f)!=16||fread(&guards[i],4,1,f)!=1)return 2;
    }
    if(fgetc(f)!=EOF)return 2;fclose(f);
    Input*di;Output*dout;
    CHECK(cudaMalloc(&di,n*sizeof(Input)));CHECK(cudaMalloc(&dout,n*sizeof(Output)));
    CHECK(cudaMemcpy(di,in.data(),n*sizeof(Input),cudaMemcpyHostToDevice));
    audit_point_predicate<<<(n+127)/128,128>>>(di,dout,n);
    CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(),dout,n*sizeof(Output),cudaMemcpyDeviceToHost));
    unsigned errors=0,guard_true=0,guard_false=0;
    const uint32_t priors[4]={0,1,0x80000000u,0xffffffffu};
    for(unsigned i=0;i<n;i++){
        guard_true+=guards[i]==1;guard_false+=guards[i]==0;
        for(unsigned mode=0;mode<4;mode++){
            bool bad=guards[i]>1;
            for(unsigned j=0;j<16;j++)bad|=out[i].fused[mode][j]!=expected[i*16+j]||out[i].reference[mode][j]!=expected[i*16+j];
            bad|=out[i].flags[mode]!=(priors[mode]|guards[i])||out[i].ref_flags[mode]!=(priors[mode]|guards[i]);
            if(bad){if(errors<8)printf("Point predicate mismatch case=%u prior=%u\n",i,mode);errors++;}
        }
    }
    if(!guard_true||!guard_false)errors++;
    printf("Point predicate audit: inputs=%u cases=%u guard_true=%u guard_false=%u errors=%u\n",n,n*4,guard_true,guard_false,errors);
    cudaFree(di);cudaFree(dout);return errors?1:0;
}
