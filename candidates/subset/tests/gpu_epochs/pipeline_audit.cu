#define main grinder_main
#include "../../subset.cu"
#undef main
#include <vector>
__global__ void audit_prepare(const uint64_t*input,uint64_t*co,uint64_t*roots,int n){int i=blockIdx.x*Q4_TREE_N+threadIdx.x;uint64_t v[5]={1,0,0,0,0};if(i<n)for(int k=0;k<4;k++)v[k]=input[4*i+k];__shared__ uint64_t products[4][2*Q4_TREE_N],excluded[4][Q4_TREE_N];q4_cofactor_prepare<Q4_TREE_N>(v,roots,products,excluded);if(i<n)for(int k=0;k<4;k++)co[4*i+k]=v[k];}
__global__ void audit_finish(uint64_t*co,const uint64_t*roots,int n){int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;uint64_t v[4],inv[4];for(int k=0;k<4;k++){v[k]=co[4*i+k];inv[k]=roots[4*(i/Q4_TREE_N)+k];}q4_mul(v,v,inv);for(int k=0;k<4;k++)co[4*i+k]=v[k];}
int main(){BN_CTX*ctx=BN_CTX_new();BIGNUM*p=NULL;BN_hex2bn(&p,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");BIGNUM*a=BN_new(),*want=BN_new(),*actual=BN_new(),*tmp=BN_new();uint64_t rng=0x3158915723456789ULL;unsigned errors=0;int total=0;uint64_t rpoint[8]={1,0,0,0,7,0,0,0};cudaMemcpyToSymbol(QSB_U2R,rpoint,64);
 for(int n: {256,768,33024}){for(int mode=0;mode<3;mode++){
  std::vector<uint64_t>input(n*4),out(n*4),rr(((n+Q4_TREE_N-1)/Q4_TREE_N)*8);
  for(int i=0;i<n;i++){if(mode==0)BN_one(a);else if(mode==1){BN_copy(a,p);BN_sub_word(a,1+i%3);}else{uint64_t words[4];for(auto&x:words){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;x=rng;}BN_lebin2bn((unsigned char*)words,32,a);BN_nnmod(a,a,p,ctx);if(BN_is_zero(a))BN_one(a);}BN_bn2lebinpad(a,(unsigned char*)(input.data()+4*i),32);}
  uint64_t*di,*dc,*dr;cudaMalloc(&di,n*32);cudaMalloc(&dc,n*32);cudaMalloc(&dr,rr.size()*8);cudaMemcpy(di,input.data(),n*32,cudaMemcpyHostToDevice);int count=(n+Q4_TREE_N-1)/Q4_TREE_N;
  audit_prepare<<<count,Q4_TREE_N>>>(di,dc,dr,n);q4_inverse_roots<<<(count+255)/256,256>>>(dr,count);audit_finish<<<(n+255)/256,256>>>(dc,dr,n);auto status=cudaDeviceSynchronize();if(status!=cudaSuccess){printf("CUDA %s\n",cudaGetErrorString(status));return 2;}cudaMemcpy(out.data(),dc,n*32,cudaMemcpyDeviceToHost);cudaMemcpy(rr.data(),dr,rr.size()*8,cudaMemcpyDeviceToHost);
  for(int i=0;i<n;i++){BN_lebin2bn((unsigned char*)(input.data()+4*i),32,a);BN_mod_inverse(want,a,p,ctx);BN_lebin2bn((unsigned char*)(out.data()+4*i),32,actual);if(BN_cmp(want,actual))errors++;}
  for(int i=0;i<count;i++){BN_lebin2bn((unsigned char*)(rr.data()+4*i),32,a);BN_set_word(tmp,7);BN_mod_mul(want,a,tmp,p,ctx);BN_lebin2bn((unsigned char*)(rr.data()+4*(count+i)),32,actual);if(BN_cmp(want,actual))errors++;}
  printf("pipeline audit N=%d mode=%d cumulative_errors=%u\n",n,mode,errors);total+=n;cudaFree(di);cudaFree(dc);cudaFree(dr);
 }}printf("pipeline audit tree=%d cases=%d mismatches=%u\n",Q4_TREE_N,total,errors);return errors?1:0;}
