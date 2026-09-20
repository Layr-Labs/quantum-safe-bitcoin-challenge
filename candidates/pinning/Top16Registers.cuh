// Register-resident tree top. Frontier association retained; duplicated upward products
// exchange operand order only. qsb_field_mul_sc is a function of a*b.
#pragma once
template<int N> __device__ __forceinline__ void qsb_top16_duplicate(
 uint64_t* roots,uint64_t (*products)[2*N],uint64_t (*excluded)[N]) {
 static_assert(N>=32 && !(N&(N-1)),"full first warp required");
 const int tid=threadIdx.x;
 if(tid<32) {
  uint64_t node[4];
  #pragma unroll
  for(int k=0;k<4;k++)node[k]=products[k][2*N-32+(tid&15)];
  #pragma unroll
  for(int wave=0;wave<6;wave++) {
   bool work;int src;
   if(wave<3){
    const int start=wave==0?16:(wave==1?24:28);
    const int bit=wave==0?8:(wave==1?4:2);
    work=tid>=start;src=tid^bit;
   }else if(wave==3){
    work=(tid>=24&&tid<28)||tid==31;
    src=tid==31?30:28+(((tid-24)&1)^1);
   }else if(wave==4){work=tid>=16&&tid<24;src=24+(((tid-16)&3)^2);}
   else{work=tid<16;src=16+((tid&7)^4);}
   uint64_t a[5],b[5],out[5];
   #pragma unroll
   for(int k=0;k<4;k++) {
    uint32_t lo=(uint32_t)node[k],hi=(uint32_t)(node[k]>>32);
    uint32_t ol=__shfl_sync(0xffffffffu,lo,src),oh=__shfl_sync(0xffffffffu,hi,src);
    uint64_t other=(uint64_t)ol|((uint64_t)oh<<32);
    a[k]=wave<3?node[k]:other;b[k]=wave<3?other:node[k];
   }
   a[4]=b[4]=0;
   if(work){
    qsb_field_mul_sc(out,a,b);
    #pragma unroll
    for(int k=0;k<4;k++)node[k]=out[k];
   }
  }
  if(tid<16){
   #pragma unroll
   for(int k=0;k<4;k++)excluded[k][N-32+(tid^8)]=node[k];
  }else if(tid==31){
   #pragma unroll
   for(int k=0;k<4;k++)roots[(size_t)blockIdx.x*4+k]=node[k];
  }
  __syncwarp();
 }
}
