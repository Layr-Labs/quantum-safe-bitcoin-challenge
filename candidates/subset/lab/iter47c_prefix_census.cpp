// Work-count census of the active six-segment, 16-record/32-stride builder.
// Counts repeated additions; not a substitute for exact GPU verification.
#include <cstdint>
#include <cstdio>
int main(){
 const unsigned entries[6]={262144,262144,131072,131072,67108864,85279885};
 const unsigned offsets[6]={0,262144,524288,655360,786432,67895296};
 const uint64_t total=153175181,groups=(total+511)/512;
 uint64_t eligible[6]={},miss[6]={},hit[6]={},stored[6]={};
 for(uint64_t group=0;group<groups;group++)for(unsigned lane=0;lane<32;lane++){
  int cc=-1,ci=-1,ck=-1;
  for(unsigned j=0;j<16;j++){
   uint64_t t=group*512+lane+32*j;if(t>=total)break;
   int ch=-1;for(int c=0;c<6;c++)if(t>=offsets[c] && t<uint64_t(offsets[c])+entries[c])ch=c;
   if(ch<0)continue;stored[ch]++;
   unsigned d=t-offsets[ch],m=ch==0?d:2*d+1,hi=(m>>12)&4095,h2=m>>24;
   if(hi && h2){eligible[ch]++;
    if(cc==ch && ci==int(hi) && ck==int(h2))hit[ch]++;
    else{miss[ch]++;cc=ch;ci=hi;ck=h2;}
   }
  }
 }
 puts("{\"segments\":[");uint64_t e=0,m=0,h=0,s=0;
 for(int c=0;c<6;c++){
  if(stored[c]!=entries[c])return 2;
  std::printf("%s{\"chunk\":%d,\"records\":%llu,\"original_prefix_adds\":%llu,\"cached_prefix_adds\":%llu,\"avoided_prefix_adds\":%llu}",c?",\n":"",c,(unsigned long long)stored[c],(unsigned long long)eligible[c],(unsigned long long)miss[c],(unsigned long long)hit[c]);
  e+=eligible[c];m+=miss[c];h+=hit[c];s+=stored[c];
 }
 if(s!=total || e!=m+h)return 3;
 std::printf("\n],\"records\":%llu,\"original_prefix_adds\":%llu,\"cached_prefix_adds\":%llu,\"avoided_prefix_adds\":%llu,\"avoided_field_multiplies\":%llu,\"avoided_field_squares\":%llu}\n",(unsigned long long)s,(unsigned long long)e,(unsigned long long)m,(unsigned long long)h,(unsigned long long)(9*h),(unsigned long long)(2*h));
}
