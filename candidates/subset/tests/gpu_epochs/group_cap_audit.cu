// Exhaustively audit production allocation helper against host rank/unrank.
#define QSB_GROUP_CAP_TIGHT 1
#define main qsb_candidate_main
#include "tree.cu"
#undef main
int main(){
 unsigned long long checked=0;const uint64_t total=binom_u64(137,6);
 for(size_t cap:{1048576ULL,2097152ULL,4194304ULL}){
  size_t maxgroups=0,at=0;
  for(uint64_t b=0;b<total;b+=cap){
   uint8_t a[6],z[6];const uint64_t end=std::min(total,b+cap)-1;
   qsb_host_unrank(b,137,6,a);qsb_host_unrank(end,137,6,z);
   if(qsb_host_rank(a,6,137)!=b || qsb_host_rank(z,6,137)!=end)return 2;
   size_t n=qsb_host_rank(z,5,137)-qsb_host_rank(a,5,137)+1;
   if(n>qsb_group_capacity(137,6,cap))return 3;
   if(n>maxgroups){maxgroups=n;at=b;}
   checked++;
  }
  if(maxgroups!=qsb_group_capacity(137,6,cap))return 4;
  printf("cap=%zu max_groups=%zu base=%zu PASS\n",cap,maxgroups,at);
 }
 if(qsb_group_capacity(136,6,2097152)!=2*2097152+4)return 5;
 if(qsb_group_capacity(137,5,2097152)!=2*2097152+4)return 6;
 if(qsb_group_capacity(137,6,3)!=10)return 7;
 printf("PRODUCTION_GROUP_CAP_AUDIT: PASS launches=%llu fallbacks=3\n",checked);return 0;
}
