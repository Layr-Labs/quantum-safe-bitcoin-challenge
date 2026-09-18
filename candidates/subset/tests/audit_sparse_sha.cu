#define main grinder_main
#include "../subset.cu"
#undef main
#include <vector>
__global__ void openssl_inputs(uint32_t *out){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;uint32_t w[9],v=i+1;
 for(int j=0;j<8;j++){v^=v<<13;v^=v>>17;v^=v<<5;w[j]=v;}
 _SHA256TransformDigest32(out+16*i,w);
 w[8]=(v&0xff000000u)|0x00800000u;
 _SHA256TransformPubkey33(out+16*i+8,w);
}
__global__ void audit(unsigned *errors){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 uint32_t w[16],copy[16],a[8],b[8],v=i+1;
 for(int j=0;j<8;j++){v^=v<<13;v^=v>>17;v^=v<<5;w[j]=v;}
 w[8]=0x80000000u;for(int j=9;j<15;j++)w[j]=0;w[15]=256;
 for(int j=0;j<16;j++)copy[j]=w[j];
 _SHA256Initialize(a);_SHA256Transform(a,copy);_SHA256TransformDigest32(b,w);
 for(int j=0;j<8;j++)if(a[j]!=b[j])atomicAdd(errors,1);
 w[8]=(v&0xff000000u)|0x00800000u;w[15]=264;
 for(int j=0;j<16;j++)copy[j]=w[j];
 _SHA256Initialize(a);_SHA256Transform(a,copy);_SHA256TransformPubkey33(b,w);
 for(int j=0;j<8;j++)if(a[j]!=b[j])atomicAdd(errors,1);
}
__global__ void fieldbits_audit(unsigned *errors){
 uint64_t m[4],v=blockIdx.x*blockDim.x+threadIdx.x+1;
 for(int j=0;j<4;j++){v^=v<<13;v^=v>>7;v^=v<<17;m[j]=v;}
 for(unsigned pos=0;pos<256;pos++){
  unsigned li=pos>>6,sh=pos&63;
  uint64_t lo=m[li],hi=li<3?m[li+1]:0;
  uint32_t want=(uint32_t)((lo>>sh)|((hi<<1)<<(63-sh)));
  if(gt_field_bits_v(m,pos)!=want)atomicAdd(errors,1);
 }
}
int main(){unsigned *d,n=0;cudaMalloc(&d,4);cudaMemset(d,0,4);audit<<<1024,256>>>(d);cudaError_t s=cudaDeviceSynchronize();cudaMemcpy(&n,d,4,cudaMemcpyDeviceToHost);printf("subset sparse SHA audit: 524288 digests, %u mismatching words, CUDA=%s\n",n,cudaGetErrorString(s));bool bad=n||s!=cudaSuccess;
 const unsigned count=8192;uint32_t *gpu;std::vector<uint32_t> out(count*16);cudaMalloc(&gpu,out.size()*4);openssl_inputs<<<count/256,256>>>(gpu);s=cudaDeviceSynchronize();cudaMemcpy(out.data(),gpu,out.size()*4,cudaMemcpyDeviceToHost);n=0;
 for(unsigned i=0;i<count;i++){
  uint32_t v=i+1;unsigned char bytes[33],hash[32];
  for(int j=0;j<8;j++){v^=v<<13;v^=v>>17;v^=v<<5;for(int k=0;k<4;k++)bytes[4*j+k]=(unsigned char)(v>>(24-8*k));}
  bytes[32]=(unsigned char)(v>>24);
  for(int mode=0;mode<2;mode++){SHA256(bytes,32+mode,hash);for(int j=0;j<8;j++){uint32_t word=((uint32_t)hash[4*j]<<24)|((uint32_t)hash[4*j+1]<<16)|((uint32_t)hash[4*j+2]<<8)|hash[4*j+3];if(word!=out[16*i+8*mode+j])n++;}}
 }
 printf("subset OpenSSL SHA audit: 16384 digests, %u mismatching words, CUDA=%s\n",n,cudaGetErrorString(s));bad|=n||s!=cudaSuccess;
 cudaMemset(d,0,4);fieldbits_audit<<<16,256>>>(d);s=cudaDeviceSynchronize();cudaMemcpy(&n,d,4,cudaMemcpyDeviceToHost);printf("subset field extraction audit: 1048576 fields, %u mismatches, CUDA=%s\n",n,cudaGetErrorString(s));return bad||n||s!=cudaSuccess;}
