#!/usr/bin/env python3
"""Execute extracted host scheduling and hit codecs on CPU, with CUDA calls stubbed.

No hashing, elliptic-curve arithmetic, device synchronization or timing is tested.
The launch/copy stubs record scheduling; the real writers and host decoder are
compared with independent expected locktime, recovery ID and hash choice.
"""
# Adapted from Meganpark980320 PR70, commit066f47d3909f06544180b173b521f6ae09a58b27.
# https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/70
# Original SHA256: 4b7118413c3f28f459f75f1e7c5be27302bd4945a7a675541fa677ff599e0673
# Changes: candidate path; timing-call stubs (policy tested independently).
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def definition(source, signature):
    start=source.index(signature); opening=source.index('{',start)
    depth=1; i=opening+1
    while depth:
        depth+=(source[i]=='{')-(source[i]=='}'); i+=1
    return source[start:i]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args()
    source=(Path(__file__).parent/'candidate/pinning.cu').read_text()
    tuner=(Path(__file__).parent/'candidate/wide_search_tuner.cuh').read_text()
    policy=definition(tuner,'struct WideSchedulePolicy')+';'
    group_limit=definition(tuner,'    uint32_t group_limit(')
    kernel=definition(source,'template<bool FAST_TAIL, int STAGE>')
    first=definition(kernel,'if(vv){')
    tail=kernel[kernel.index(first)+len(first):]
    second=definition(tail,'if(vv){')
    # Preserve the real decoder's continue and invalid-slot handling.
    begin=source.index('uint32_t raw = hits[h];')
    decoder=source[begin:source.index('fprintf(f,',begin)]
    launcher=definition(source,'template<bool FAST_TAIL>\nstatic void launch_pinning_pipeline(')
    signature=launcher[:launcher.index('{')]
    start=source.index('static const uint32_t MAX_READBACK_GROUP')
    stop=source.index('            if (h_hit > 0)',start)
    scheduling=source[start:stop]+'\n}\n'
    batch=int(re.search(r'int BATCH = (\d+);',source).group(1))
    ltmin=int(re.search(r'uint32_t LT_MIN = (\d+);',source).group(1))
    ltmax=int(re.search(r'uint32_t LT_MAX = (\d+);',source).group(1))
    code=r'''
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <vector>
#include <set>
#include <math.h>
struct ulonglong2 {uint64_t x,y;};
struct Launch {uint32_t lt,slot;int size;};
static std::vector<std::vector<Launch>> groups;
static unsigned copies;
// Actual policy/group-limit expressions; event ordering is checked separately.
@POLICY@
struct WideSearchTuner {
 bool enabled=false;WideSchedulePolicy policy;
 void begin(){} void end(){}
 void observe(uint64_t n){if(enabled)policy.observe(100,n);}
 @GROUP_LIMIT@
};
static WideSearchTuner schedule_tuner;
using cudaError_t=int;
static constexpr int cudaSuccess=0,cudaMemcpyDeviceToHost=1;
static const char *cudaGetErrorString(int){return "mock";}
static int cudaMemsetAsync(void*p,int v,size_t n){
 groups.emplace_back();memset(p,v,n);return 0;
}
static int cudaMemcpy(void*d,const void*s,size_t n,int){
 memcpy(d,s,n);copies++;return 0;
}
static uint32_t atomicAdd(uint32_t*p,uint32_t n){uint32_t old=*p;*p+=n;return old;}
'''+signature+r'''{
 assert(batch_size>0);groups.back().push_back({start_lt,group_slot,batch_size});
}
template<bool FAST_TAIL>
static void write_first(uint32_t idx,uint32_t lt,uint32_t group_slot,int ri,
 uint32_t*d_hit_cnt,uint32_t*d_hit_idx){int vv=1;
'''+first+r'''
}
static void write_second(uint32_t idx,uint32_t lt,uint32_t group_slot,int ri,
 uint32_t*d_hit_cnt,uint32_t*d_hit_idx){int vv=1;
'''+second+r'''
}
static bool decode(bool fast_tail,uint32_t raw_value,uint32_t group_count,
 uint32_t*group_lt,uint32_t*out){
 uint32_t hits[1]={raw_value};
 for(int h=0;h<1;h++){
'''+decoder+r'''
 out[0]=lt;out[1]=ri;out[2]=hc;return true;
 }
 return false;
}
static int schedule(bool fast_tail,uint32_t lt_range,bool tune){
 groups.clear();copies=0;
 schedule_tuner.enabled=tune&&fast_tail;schedule_tuner.policy=WideSchedulePolicy{};
 uint32_t LT_MIN=@LTMIN@;int BATCH=@BATCH@;
 uint64_t total_searched=0;
 const uint32_t*d_mid=nullptr;uint8_t*d_suffix=nullptr;
 int gpu_suffix_len=75;struct {int seq_offset=31,lt_offset=67,total_preimage_len=9995;} pp;
 uint32_t seq=0x80000000;
 const uint64_t*d_nri=nullptr,*d_u2rx=nullptr,*d_u2ry=nullptr;
 const uint64_t*d_neg2u2rx=nullptr,*d_neg2u2ry=nullptr;
 uint8_t*d_gt=nullptr;uint32_t counter=0,records[1024],*d_hit_cnt=&counter,*d_hit_idx=records;
 int easy=0,single_hash=1;ulonglong2*d_pipeline_state=nullptr;
 uint64_t*d_pipeline_roots=nullptr,*d_pipeline_tree=nullptr;
 uint64_t*d_super_roots=nullptr,*d_root_checkpoint=nullptr;
'''+scheduling+r'''
 assert(total_searched==lt_range);
 return 0;
}
int main(){
 const uint32_t B=@BATCH@,MIN=@LTMIN@,RANGE=@LTRANGE@;
 const uint32_t spans[]={0,1,B-1,B,B+1,64*B-1,64*B,64*B+1,RANGE};
 unsigned schedules=0,launches=0,records=0,rejected=0,capacity_checks=0;
 unsigned production_groups[2][2]={},production_launches[2][2]={};
 for(int tune=0;tune<2;tune++)for(int fast=0;fast<2;fast++)for(uint32_t span:spans){
  assert(schedule(fast,span,tune)==0);assert(copies==groups.size());
  uint64_t next=MIN;
  if(span==RANGE)production_groups[tune][fast]=groups.size();
  for(const auto&g:groups){
   assert(g.size()<=unsigned(fast?128:64));uint32_t bases[128]={};
   for(unsigned j=0;j<g.size();j++)bases[j]=g[j].lt;
   for(unsigned j=0;j<g.size();j++){
    const auto&l=g[j];assert(l.lt==next && l.slot==j && l.size>0 && unsigned(l.size)<=B);
    next+=l.size;launches++;if(span==RANGE)production_launches[tune][fast]++;
    std::set<uint32_t> indices={0u,unsigned(l.size)-1,unsigned(l.size)/2};
    for(uint32_t idx:indices)for(int ri=0;ri<2;ri++)for(int hc=0;hc<(fast?1:2);hc++){
     uint32_t count=0,raw[1025]={},decoded[3]={};
     if(hc)write_second(idx,l.lt+idx,l.slot,ri,&count,raw);
     else if(fast)write_first<true>(idx,l.lt+idx,l.slot,ri,&count,raw);
     else write_first<false>(idx,l.lt+idx,l.slot,ri,&count,raw);
     assert(count==1 && decode(fast,raw[0],g.size(),bases,decoded));
     assert(decoded[0]==l.lt+idx && decoded[1]==unsigned(ri) && decoded[2]==unsigned(hc));
     records++;
    }
   }
   if(!fast && g.size()<64){
    uint32_t out[3];assert(!decode(false,uint32_t(g.size())<<24,g.size(),bases,out));rejected++;
   }
  }
  assert(next==uint64_t(MIN)+span);schedules++;
 }
 assert(production_groups[0][0]==2 && production_groups[0][1]==1);
 assert(production_groups[1][0]==2 && production_groups[1][1]==7);
 for(int t=0;t<2;t++)for(int f=0;f<2;f++)assert(production_launches[t][f]==75);
 for(int fast=0;fast<2;fast++){
  uint32_t count=1023,raw[1025]={};raw[1024]=0x12345678;
  for(int n=0;n<2;n++){
   if(fast)write_first<true>(7,MIN+7,0,1,&count,raw);
   else write_first<false>(7,MIN+7,0,1,&count,raw);
  }
  assert(count==1025 && raw[1024]==0x12345678);capacity_checks++;
 }
 std::printf("{\"schedules\":%u,\"launches\":%u,\"records\":%u,"
             "\"invalid_slots_rejected\":%u,\"capacity_checks\":%u,"
             "\"production_pipelines\":75,\"fast_readbacks_steady\":1,\"fast_readbacks_first_tuning_sequence\":7,\"fallback_readbacks\":2}\n",
             schedules,launches,records,rejected,capacity_checks);
}
'''
    code=code.replace('@BATCH@',str(batch)).replace('@LTMIN@',str(ltmin)).replace('@LTRANGE@',str(ltmax-ltmin))
    code=code.replace('@POLICY@',policy).replace('@GROUP_LIMIT@',group_limit)
    with tempfile.TemporaryDirectory(prefix='qsb-grouped-records-') as td:
        cpp=Path(td)/'audit.cpp';binary=Path(td)/'audit';cpp.write_text(code)
        subprocess.run(['g++','-std=c++17','-O2','-fsanitize=undefined',
                        '-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True)
        result=json.loads(subprocess.check_output([str(binary)],text=True))
    result.update(status='PASS',gpu_executed=False,ubsan=True,
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  tuner_sha256=hashlib.sha256(tuner.encode()).hexdigest(),
                  scope='actual host scheduling and both hit writers/decoder; CUDA launch/copy stubs; no crypto')
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
