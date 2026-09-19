#define main grinder_main
#include "../pinning.cu"
#undef main
#include <vector>
struct AuditCase {uint64_t a[4],e[4],q[4],expected[4];};
__global__ void audit_fusion(AuditCase *c,unsigned *errors,int count) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 uint64_t a[4],e[4],q[4],r[4];Load256(a,c[i].a);Load256(e,c[i].e);Load256(q,c[i].q);
 _ModSqrAddSub2(r,a,e,q);qsb_field_normalize(r);
 for(int k=0;k<4;k++)if(r[k]!=c[i].expected[k]){
  if(atomicAdd(errors,1)==0)printf("mismatch case=%d limb=%d got=%016llx expected=%016llx\n",i,k,(unsigned long long)r[k],(unsigned long long)c[i].expected[k]);}
 _ModSqrAddSub2(a,a,e,q);qsb_field_normalize(a);
 for(int k=0;k<4;k++)if(a[k]!=c[i].expected[k])atomicAdd(errors,1);
 _ModSqrAddSub2(e,c[i].a,e,q);qsb_field_normalize(e);
 for(int k=0;k<4;k++)if(e[k]!=c[i].expected[k])atomicAdd(errors,1);
}
int main(){
 const int N=262144;std::vector<AuditCase>c(N);uint64_t rng=0x58402394286edULL;
 BN_CTX*ctx=BN_CTX_new();BIGNUM *p=nullptr;BN_hex2bn(&p,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
 BIGNUM *a=BN_new(),*e=BN_new(),*q=BN_new(),*r=BN_new();
 uint64_t edges[8][4]={{0,0,0,0},{1,0,0,0},{2,0,0,0},{0x1000003d1ULL,0,0,0},{0xfffffffefffffc2eULL,~0ULL,~0ULL,~0ULL},{0xfffffffefffffc2fULL,~0ULL,~0ULL,~0ULL},{0xfffffffefffffc30ULL,~0ULL,~0ULL,~0ULL},{~0ULL,~0ULL,~0ULL,~0ULL}};
 for(int i=0;i<N;i++){
  for(int j=0;j<4;j++){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;c[i].a[j]=rng;rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;c[i].e[j]=rng;rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;c[i].q[j]=rng;}
  if(i<512){for(int j=0;j<4;j++){c[i].a[j]=edges[i&7][j];c[i].e[j]=edges[(i>>3)&7][j];c[i].q[j]=edges[(i>>6)&7][j];}}
  // Small squares force signed-high underflows that random field inputs rarely exercise.
  if(i>=512 && i<32768){c[i].a[3]>>=32;if(i&1)c[i].a[2]=c[i].a[3]=0;}
  BN_lebin2bn((unsigned char*)c[i].a,32,a);BN_lebin2bn((unsigned char*)c[i].e,32,e);BN_lebin2bn((unsigned char*)c[i].q,32,q);
  BN_mod_sqr(r,a,p,ctx);BN_mod_add(r,r,e,p,ctx);BN_mod_sub(r,r,q,p,ctx);BN_mod_sub(r,r,q,p,ctx);BN_bn2lebinpad(r,(unsigned char*)c[i].expected,32);
 }
 AuditCase*d;unsigned*err,errors=0;cudaMalloc(&d,N*sizeof(AuditCase));cudaMalloc(&err,4);cudaMemset(err,0,4);cudaMemcpy(d,c.data(),N*sizeof(AuditCase),cudaMemcpyHostToDevice);
 audit_fusion<<<N/256,256>>>(d,err,N);auto status=cudaDeviceSynchronize();cudaMemcpy(&errors,err,4,cudaMemcpyDeviceToHost);
 printf("fusion audit cases=%d aliases=3 mismatches=%u cuda=%s\n",N,errors,cudaGetErrorString(status));return errors || status!=cudaSuccess;
}
