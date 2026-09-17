// Independent OpenSSL check of the production checkpointed inverse hierarchy.
#define main qsb_grinder_main
#include "tree.cu"
#undef main
#include <vector>
#define CHECK(x) do {cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));return 2;}}while(0)
__global__ void audit_prepare(const uint64_t *in,int n,uint64_t *roots,uint64_t *cp){
 int i=blockIdx.x*256+threadIdx.x;uint64_t v[5]={1,0,0,0,0};
 if(i<n)for(int k=0;k<4;k++)v[k]=in[4*i+k];
 qsb_block_product_checkpoint<256>(v,roots,cp);
}
__global__ void audit_finish(const uint64_t *in,int n,const uint64_t *roots,const uint64_t *cp,uint64_t *out){
 int i=blockIdx.x*256+threadIdx.x;uint64_t v[5]={1,0,0,0,0};
 if(i<n)for(int k=0;k<4;k++)v[k]=in[4*i+k];
 qsb_block_inverse_checkpoint<256>(v,roots,cp);
 if(i<n)for(int k=0;k<4;k++)out[4*i+k]=v[k];
}
int main(){
 BN_CTX *ctx=BN_CTX_new();BIGNUM *p=NULL,*a=BN_new(),*r=BN_new();
 BN_hex2bn(&p,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
 const int sizes[]={1,255,256,257,8191,65535,65536,65537,131073};
 size_t total=0;
 for(int n:sizes){
  std::vector<uint64_t> input((size_t)n*4),want((size_t)n*4),got((size_t)n*4);
  for(int i=0;i<n;i++){
   unsigned char h[32];SHA256((unsigned char*)&i,sizeof(i),h);BN_bin2bn(h,32,a);BN_mod(a,a,p,ctx);
   if(i%257==0)BN_one(a);
   if(i%257==1){BN_copy(a,p);BN_sub_word(a,1);}
   if(i%257==2){BN_copy(a,p);BN_add_word(a,1);}
   if(BN_is_zero(a))BN_one(a);
   BN_bn2lebinpad(a,(unsigned char*)(input.data()+4*i),32);
   if(!BN_mod_inverse(r,a,p,ctx))return 3;
   BN_bn2lebinpad(r,(unsigned char*)(want.data()+4*i),32);
  }
  int blocks=(n+255)/256,groups=(blocks+255)/256;
  uint64_t *di,*doo,*roots,*cp,*super_roots,*rcp;
  CHECK(cudaMalloc(&di,input.size()*8));CHECK(cudaMalloc(&doo,input.size()*8));
  CHECK(cudaMalloc(&roots,(size_t)blocks*4*8));CHECK(cudaMalloc(&cp,(size_t)blocks*4*256*8));
  CHECK(cudaMalloc(&super_roots,(size_t)groups*4*8));CHECK(cudaMalloc(&rcp,(size_t)groups*4*256*8));
  CHECK(cudaMemcpy(di,input.data(),input.size()*8,cudaMemcpyHostToDevice));
  audit_prepare<<<blocks,256>>>(di,n,roots,cp);
  qsb_root_group_prepare<<<groups,256>>>(roots,blocks,super_roots,rcp);
  qsb_invert_super_roots<<<(groups+255)/256,256>>>(super_roots,groups);
  qsb_root_group_finish<<<groups,256>>>(roots,blocks,super_roots,rcp);
  audit_finish<<<blocks,256>>>(di,n,roots,cp,doo);
  CHECK(cudaGetLastError());CHECK(cudaMemcpy(got.data(),doo,got.size()*8,cudaMemcpyDeviceToHost));
  for(int i=0;i<n;i++)if(memcmp(got.data()+4*i,want.data()+4*i,32)){fprintf(stderr,"Mismatch n=%d index=%d\n",n,i);return 1;}
  printf("PASS n=%d, roots=%d, super-roots=%d\n",n,blocks,groups);fflush(stdout);total+=n;
  cudaFree(di);cudaFree(doo);cudaFree(roots);cudaFree(cp);cudaFree(super_roots);cudaFree(rcp);
 }
 printf("PASS: %zu exact OpenSSL inverse comparisons\n",total);
 BN_free(a);BN_free(r);BN_free(p);BN_CTX_free(ctx);return 0;
}
