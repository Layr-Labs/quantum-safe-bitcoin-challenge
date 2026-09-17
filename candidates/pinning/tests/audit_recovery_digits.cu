#define main grinder_main
#include "../pinning.cu"
#undef main
#include <vector>
struct Case { uint64_t y[4],v[4],inv[4],xr[4],yr[4],c[4],xp[4],xm[4]; uint32_t parity; };
__global__ void audit(Case *a,int n,unsigned *errors) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 Case t=a[i];uint64_t p[4],m[4];
 uint32_t parity=qsb_xyzz_finish_symmetric(t.y,t.v,t.inv,t.xr,t.yr,t.c,p,m);
 bool bad=parity!=t.parity;
 for(int j=0;j<4;j++)bad|=p[j]!=t.xp[j]||m[j]!=t.xm[j];
 if(bad)atomicAdd(errors,1);
}
void enc(uint64_t *dst,BIGNUM *b){if(BN_bn2lebinpad(b,(unsigned char*)dst,32)!=32)abort();}
__global__ void digits(unsigned *errors){
 unsigned id=blockIdx.x*blockDim.x+threadIdx.x;
 uint64_t k[4],M[4],copy[4],v=(uint64_t)id+1;
 for(int j=0;j<4;j++){v^=v<<13;v^=v>>7;v^=v<<17;k[j]=v;}
 if(id<4){for(int j=0;j<4;j++)k[j]=id==0?0:~0ULL; k[0]-=id;}
 int sign;gt_recode_setup(k,M,&sign);for(int j=0;j<4;j++)copy[j]=M[j];
 for(int c=0;c<GT_CHUNKS;c++){
   int32_t e=c==0?gt_mixed_step<18>(copy,sign):c==GT_CHUNKS-1?sign*(int32_t)copy[0]:gt_mixed_step<17>(copy,sign);
   uint32_t i1,i2;uint64_t n1,n2;gt_digit_idx(e,&i1,&n1);
   pin_direct_digit(M,sign,gt_shift(c)+1,c==0?18:17,c==GT_CHUNKS-1,&i2,&n2);
   if(i1!=i2||n1!=n2)atomicAdd(errors,1);
 }
}
int main(){
 const int N=16384;std::vector<Case> a(N);
 BN_CTX *ctx=BN_CTX_new();EC_GROUP *g=EC_GROUP_new_by_curve_name(NID_secp256k1);
 BIGNUM *p=BN_new(),*k=BN_new(),*xr=BN_new(),*yr=BN_new(),*x=BN_new(),*y=BN_new(),*z=BN_new(),*z2=BN_new(),*z3=BN_new(),*w=BN_new(),*t=BN_new(),*c=BN_new();
 EC_GROUP_get_curve(g,p,NULL,NULL,ctx);BN_set_word(k,7);
 EC_POINT *R=EC_POINT_new(g),*negR=EC_POINT_new(g),*P=EC_POINT_new(g),*Q=EC_POINT_new(g);
 EC_POINT_mul(g,R,k,NULL,NULL,ctx);EC_POINT_copy(negR,R);EC_POINT_invert(g,negR,ctx);
 EC_POINT_get_affine_coordinates(g,R,xr,yr,ctx);
 BN_mod_sqr(c,xr,p,ctx);BN_mul_word(c,3);BN_mod_add(t,yr,yr,p,ctx);BN_mod_inverse(t,t,p,ctx);BN_mod_mul(c,c,t,p,ctx);
 BN_set_word(k,100);EC_POINT_mul(g,P,k,NULL,NULL,ctx);
 for(int i=0;i<N;i++){
   Case &a0=a[i];EC_POINT_get_affine_coordinates(g,P,x,y,ctx);
   unsigned char digest[32];SHA256((unsigned char*)&i,sizeof(i),digest);BN_bin2bn(digest,32,z);BN_nnmod(z,z,p,ctx);
   if(i<4)BN_set_word(z,i+1);
   BN_mod_sqr(z2,z,p,ctx);BN_mod_mul(z3,z2,z,p,ctx);
   BN_mod_mul(t,y,z3,p,ctx);enc(a0.y,t);enc(a0.v,z3);
   BN_mod_sub(t,xr,x,p,ctx);BN_mod_sqr(w,z3,p,ctx);BN_mod_mul(w,w,t,p,ctx);BN_mod_inverse(w,w,p,ctx);enc(a0.inv,w);
   enc(a0.xr,xr);enc(a0.yr,yr);enc(a0.c,c);
   EC_POINT_add(g,Q,P,R,ctx);EC_POINT_get_affine_coordinates(g,Q,x,y,ctx);enc(a0.xp,x);a0.parity=BN_is_odd(y);
   EC_POINT_add(g,Q,P,negR,ctx);EC_POINT_get_affine_coordinates(g,Q,x,y,ctx);enc(a0.xm,x);a0.parity|=BN_is_odd(y)<<1;
   EC_POINT_add(g,P,P,EC_GROUP_get0_generator(g),ctx);
 }
 Case *d;unsigned *errors,count=0;cudaMalloc(&d,N*sizeof(Case));cudaMalloc(&errors,4);cudaMemset(errors,0,4);cudaMemcpy(d,a.data(),N*sizeof(Case),cudaMemcpyHostToDevice);
 audit<<<(N+127)/128,128>>>(d,N,errors);cudaError_t status=cudaDeviceSynchronize();cudaMemcpy(&count,errors,4,cudaMemcpyDeviceToHost);
 printf("pinning finish OpenSSL audit: %d cases, %u mismatches, CUDA=%s\n",N,count,cudaGetErrorString(status));
 bool bad=count||status!=cudaSuccess;cudaMemset(errors,0,4);digits<<<1024,256>>>(errors);status=cudaDeviceSynchronize();cudaMemcpy(&count,errors,4,cudaMemcpyDeviceToHost);
 printf("pinning direct digits: 3932160 digits, %u mismatches, CUDA=%s\n",count,cudaGetErrorString(status));return bad||count||status!=cudaSuccess;
}
