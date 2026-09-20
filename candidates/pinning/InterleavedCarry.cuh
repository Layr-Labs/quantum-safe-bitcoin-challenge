// Interleaved grouped collective: caller maps physical lane t to logical candidate
// blockIdx.x*(N*G)+(t%G)*N+t/G; digit planes remain physical-lane indexed.
#pragma once
template<int N,int G> __device__ __forceinline__ void qsb_interleaved_carry(
 uint64_t *value,uint64_t *roots,uint64_t (*pr)[2*N*G],size_t root_count) {
 static_assert(N==128 && (G==2 || G==4),"audited geometry");
 const int t=threadIdx.x,g=t%G,i=t/G;
 #pragma unroll
 for(int k=0;k<4;k++)pr[k][t]=value[k];
 __syncthreads();int off=0;
 #pragma unroll 1
 for(int count=N;count>2;count>>=1){
  int half=count/2;
  if(i<half){
   uint64_t b[5],out[5];
   #pragma unroll
   for(int k=0;k<4;k++)b[k]=pr[k][(off+half)*G+t];
   b[4]=0;qsb_field_mul_sc(out,value,b);
   #pragma unroll
   for(int k=0;k<4;k++){value[k]=out[k];pr[k][(off+count)*G+t]=out[k];}
  }
  off+=count;
  if(G*half>32)__syncthreads();else __syncwarp();
 }
 if(i<5){
  uint64_t a[5],b[5],out[5];
  int ia=(2*N-4+(i<4?((i&1)^1):0))*G+g;
  int ib=(i<4?2*N-8+(i^2):2*N-3)*G+g;
  #pragma unroll
  for(int k=0;k<4;k++){a[k]=pr[k][ia];b[k]=pr[k][ib];}
  a[4]=b[4]=0;qsb_field_mul_sc(out,a,b);
  if(i<4){
   #pragma unroll
   for(int k=0;k<4;k++){value[k]=out[k];pr[k][ib]=out[k];}
  }else if((size_t)blockIdx.x*G+g<root_count){
   #pragma unroll
   for(int k=0;k<4;k++)roots[((size_t)blockIdx.x*G+g)*4+k]=out[k];
  }
 }
 __syncwarp();off=2*N-16;
 #pragma unroll 1
 for(int count=8;count<N;count<<=1){
  int half=count/2;
  if(i<count){
   uint64_t a[5],b[5],out[5];
   #pragma unroll
   for(int k=0;k<4;k++){
    a[k]=i<half?value[k]:pr[k][(off+count+((i&(half-1))^(half/2)))*G+g];
    b[k]=pr[k][(off+(i^half))*G+g];
   }
   a[4]=b[4]=0;qsb_field_mul_sc(out,a,b);
   #pragma unroll
   for(int k=0;k<4;k++){value[k]=out[k];pr[k][(off+(i^half))*G+g]=out[k];}
  }
  off-=count*2;
  if(G*count*2>32)__syncthreads();else __syncwarp();
 }
 uint64_t a[5],b[5];
 #pragma unroll
 for(int k=0;k<4;k++){
  a[k]=i<N/2?value[k]:pr[k][(N+((i&(N/2-1))^(N/4)))*G+g];
  b[k]=pr[k][(i^(N/2))*G+g];
 }
 a[4]=b[4]=0;qsb_field_mul_sc(value,a,b);value[4]=0;
}
