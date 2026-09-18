#define main grinder_main
#include "../pinning.cu"
#undef main
#include <vector>
struct Case {uint64_t raw[4],a[4],b[4],sum[4];uint32_t parity;};
__global__ void audit(Case *cases,int n,unsigned *errors){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;Case c=cases[i];
 uint32_t parity=qsb_difference_parity(c.a,c.b);
 if(parity!=c.parity)atomicAdd(errors,1);
 uint64_t raw[4];Load256(raw,c.raw);qsb_add_boundary(raw,c.a);_ModAdd256(raw,raw,c.a);
 for(int k=0;k<4;k++)if(raw[k]!=c.sum[k])atomicAdd(errors,1);
 // Also preserve the old subtraction parity for unrestricted 256-bit inputs.
 uint64_t old[4];_ModSub256(old,c.raw,c.b);
 if((old[0]&1u)!=qsb_difference_parity(c.raw,c.b))atomicAdd(errors,1);
}
void enc(uint64_t *a,BIGNUM *b){if(BN_bn2lebinpad(b,(unsigned char*)a,32)!=32)abort();}
int main(){const int N=32768;std::vector<Case> cases(N);BN_CTX *ctx=BN_CTX_new();
 BIGNUM *p=NULL;BN_hex2bn(&p,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
 BIGNUM *a=BN_new(),*b=BN_new(),*raw=BN_new(),*t=BN_new();uint64_t rng=0x123456789abcdefULL;
 for(int i=0;i<N;i++){
  uint64_t words[12];for(int j=0;j<12;j++){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;words[j]=rng;}
  BN_lebin2bn((unsigned char*)words,32,raw);BN_lebin2bn((unsigned char*)(words+4),32,a);BN_nnmod(a,a,p,ctx);BN_lebin2bn((unsigned char*)(words+8),32,b);BN_nnmod(b,b,p,ctx);
  if(i<100){
   int k=i%10;if(k<5){BN_copy(raw,p);if(k<2)BN_sub_word(raw,2-k);else BN_add_word(raw,k-2);}else{BN_one(raw);BN_lshift(raw,raw,256);BN_sub_word(raw,k-4);}
   int q=i/10;if(q<4)BN_set_word(a,q);else if(q<8){BN_copy(a,p);BN_sub_word(a,q-3);}else{BN_one(a);BN_lshift(a,a,192);BN_sub(a,p,a);BN_add_word(a,q-8);}
   if(i%3==0)BN_copy(b,a);else if(i%3==1)BN_zero(b);else{BN_copy(b,p);BN_sub_word(b,1);}
  }
  enc(cases[i].raw,raw);enc(cases[i].a,a);enc(cases[i].b,b);
  BN_mod_add(t,raw,a,p,ctx);enc(cases[i].sum,t);BN_mod_sub(t,a,b,p,ctx);cases[i].parity=BN_is_odd(t);
 }
 Case *d;unsigned *err,n=0;cudaMalloc(&d,N*sizeof(Case));cudaMalloc(&err,4);cudaMemset(err,0,4);cudaMemcpy(d,cases.data(),N*sizeof(Case),cudaMemcpyHostToDevice);
 audit<<<N/128,128>>>(d,N,err);cudaError_t status=cudaDeviceSynchronize();cudaMemcpy(&n,err,4,cudaMemcpyDeviceToHost);
 printf("OpenSSL boundary/parity audit: %d cases; %u mismatches; %s\n",N,n,cudaGetErrorString(status));return n||status!=cudaSuccess;
}
