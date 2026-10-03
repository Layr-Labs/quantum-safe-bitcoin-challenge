/* Host-only safe raw LZ4 block decoder. No third-party runtime dependency.
 * Used solely to restore byte-exact native image before CUDA loads it.
 * Own implementation of the public LZ4 block format; GPL-3 project license.
 */
#pragma once
static bool qsb_lz4_block(const unsigned char *in,size_t n,unsigned char *out,size_t cap) {
size_t ip=0,op=0;
while(ip<n) {
unsigned tok=in[ip++];size_t lit=tok>>4;
if(lit==15){unsigned x;do{if(ip>=n)return false;x=in[ip++];if(x>cap || lit>cap-x)return false;lit+=x;}while(x==255);}
if(lit>n-ip || lit>cap-op)return false;
memcpy(out+op,in+ip,lit);ip+=lit;op+=lit;
if(ip==n)return op==cap;
if(n-ip<2)return false;
size_t off=(size_t)in[ip]|((size_t)in[ip+1]<<8);ip+=2;
if(off==0 || off>op)return false;
size_t match=(tok&15)+4;
if((tok&15)==15){unsigned x;do{if(ip>=n)return false;x=in[ip++];if(x>cap || match>cap-x)return false;match+=x;}while(x==255);}
if(match>cap-op)return false;
for(size_t j=0;j<match;j++){out[op]=out[op-off];op++;}
}
return op==cap;
}
