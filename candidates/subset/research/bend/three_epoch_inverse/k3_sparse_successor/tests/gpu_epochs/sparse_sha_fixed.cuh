#pragma once

/* Fixed-padding schedules copied from the promoted pinning lineage cited by
 * pending SUBSET submission f63b274. */
__device__ __forceinline__ void _SHA256TransformDigest32(uint32_t out[8],const uint32_t m[8]){
 uint32_t t1,t2;
 uint32_t a=0x6a09e667u,b=0xbb67ae85u,c=0x3c6ef372u,d=0xa54ff53au;
 uint32_t e=0x510e527fu,f=0x9b05688cu,g=0x1f83d9abu,h=0x5be0cd19u,w[16];
#pragma unroll
 for(int i=0;i<8;i++)w[i]=m[i];
 S2Round(a,b,c,d,e,f,g,h,K[0],w[0]);S2Round(h,a,b,c,d,e,f,g,K[1],w[1]);
 S2Round(g,h,a,b,c,d,e,f,K[2],w[2]);S2Round(f,g,h,a,b,c,d,e,K[3],w[3]);
 S2Round(e,f,g,h,a,b,c,d,K[4],w[4]);S2Round(d,e,f,g,h,a,b,c,K[5],w[5]);
 S2Round(c,d,e,f,g,h,a,b,K[6],w[6]);S2Round(b,c,d,e,f,g,h,a,K[7],w[7]);
 S2Round(a,b,c,d,e,f,g,h,K[8],0x80000000u);S2Round(h,a,b,c,d,e,f,g,K[9],0u);
 S2Round(g,h,a,b,c,d,e,f,K[10],0u);S2Round(f,g,h,a,b,c,d,e,K[11],0u);
 S2Round(e,f,g,h,a,b,c,d,K[12],0u);S2Round(d,e,f,g,h,a,b,c,K[13],0u);
 S2Round(c,d,e,f,g,h,a,b,K[14],0u);S2Round(b,c,d,e,f,g,h,a,K[15],256u);
 w[0]+=s0(w[1]);w[1]+=s1(256u)+s0(w[2]);w[2]+=s1(w[0])+s0(w[3]);
 w[3]+=s1(w[1])+s0(w[4]);w[4]+=s1(w[2])+s0(w[5]);w[5]+=s1(w[3])+s0(w[6]);
 w[6]+=s1(w[4])+256u+s0(w[7]);w[7]+=s1(w[5])+w[0]+s0(0x80000000u);
 w[8]=0x80000000u+s1(w[6])+w[1];w[9]=s1(w[7])+w[2];w[10]=s1(w[8])+w[3];
 w[11]=s1(w[9])+w[4];w[12]=s1(w[10])+w[5];w[13]=s1(w[11])+w[6];
 w[14]=s1(w[12])+w[7]+s0(256u);w[15]=256u+s1(w[13])+w[8]+s0(w[0]);
 SHA256_RND(16);WMIX();SHA256_RND(32);WMIX();SHA256_RND(48);
 out[0]=0x6a09e667u+a;out[1]=0xbb67ae85u+b;out[2]=0x3c6ef372u+c;out[3]=0xa54ff53au+d;
 out[4]=0x510e527fu+e;out[5]=0x9b05688cu+f;out[6]=0x1f83d9abu+g;out[7]=0x5be0cd19u+h;
}

__device__ __forceinline__ void _SHA256TransformPubkey33(uint32_t out[8],const uint32_t m[9]){
 uint32_t t1,t2;
 uint32_t a=0x6a09e667u,b=0xbb67ae85u,c=0x3c6ef372u,d=0xa54ff53au;
 uint32_t e=0x510e527fu,f=0x9b05688cu,g=0x1f83d9abu,h=0x5be0cd19u,w[16];
#pragma unroll
 for(int i=0;i<9;i++)w[i]=m[i];
 S2Round(a,b,c,d,e,f,g,h,K[0],w[0]);S2Round(h,a,b,c,d,e,f,g,K[1],w[1]);
 S2Round(g,h,a,b,c,d,e,f,K[2],w[2]);S2Round(f,g,h,a,b,c,d,e,K[3],w[3]);
 S2Round(e,f,g,h,a,b,c,d,K[4],w[4]);S2Round(d,e,f,g,h,a,b,c,K[5],w[5]);
 S2Round(c,d,e,f,g,h,a,b,K[6],w[6]);S2Round(b,c,d,e,f,g,h,a,K[7],w[7]);
 S2Round(a,b,c,d,e,f,g,h,K[8],w[8]);S2Round(h,a,b,c,d,e,f,g,K[9],0u);
 S2Round(g,h,a,b,c,d,e,f,K[10],0u);S2Round(f,g,h,a,b,c,d,e,K[11],0u);
 S2Round(e,f,g,h,a,b,c,d,K[12],0u);S2Round(d,e,f,g,h,a,b,c,K[13],0u);
 S2Round(c,d,e,f,g,h,a,b,K[14],0u);S2Round(b,c,d,e,f,g,h,a,K[15],0x108u);
 w[0]+=s0(w[1]);w[1]+=s1(0x108u)+s0(w[2]);w[2]+=s1(w[0])+s0(w[3]);
 w[3]+=s1(w[1])+s0(w[4]);w[4]+=s1(w[2])+s0(w[5]);w[5]+=s1(w[3])+s0(w[6]);
 w[6]+=s1(w[4])+0x108u+s0(w[7]);w[7]+=s1(w[5])+w[0]+s0(w[8]);
 w[8]+=s1(w[6])+w[1];w[9]=s1(w[7])+w[2];w[10]=s1(w[8])+w[3];
 w[11]=s1(w[9])+w[4];w[12]=s1(w[10])+w[5];w[13]=s1(w[11])+w[6];
 w[14]=s1(w[12])+w[7]+s0(0x108u);w[15]=0x108u+s1(w[13])+w[8]+s0(w[0]);
 SHA256_RND(16);WMIX();SHA256_RND(32);WMIX();SHA256_RND(48);
 out[0]=0x6a09e667u+a;out[1]=0xbb67ae85u+b;out[2]=0x3c6ef372u+c;out[3]=0xa54ff53au+d;
 out[4]=0x510e527fu+e;out[5]=0x9b05688cu+f;out[6]=0x1f83d9abu+g;out[7]=0x5be0cd19u+h;
}
