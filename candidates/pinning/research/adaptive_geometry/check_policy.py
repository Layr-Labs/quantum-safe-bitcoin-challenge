#!/usr/bin/env python3
"""Execute the actual four-arm selector/event lifetime with synthetic CUDA stubs."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
base=HERE/'candidate'
code=r'''
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <initializer_list>
using cudaEvent_t=int;
static bool wide_use_fused=false,adaptive_use_wide=false;
static int created=0,destroyed=0,recorded=0;
static float next_ms=1;
static void wide_cuda_require(int s,const char*){assert(!s);}
static int cudaEventCreate(int *e){*e=++created;return 0;}
static int cudaEventDestroy(int){++destroyed;return 0;}
static int cudaEventRecord(int){++recorded;return 0;}
static int cudaEventElapsedTime(float *ms,int,int){*ms=next_ms;return 0;}
#include "adaptive_tuner.cuh"
static void simulate(bool wide,const double rates[4],int expected){
 AdaptiveSearchTuner t(true,wide);int n=wide?4:2;int measures[4]={};
 for(int i=0;i<3*n+2;i++){
  assert(t.group_limit(128)==unsigned(i<3*n?4:128));
  int want=i<n?i:(i<2*n?2*n-i-1:(i<3*n?i-2*n:expected));
  t.begin();int m=2*adaptive_use_wide+wide_use_fused;assert(m==want);
  uint64_t count=1000000ull*(1+(i%3));
  next_ms=float(rates[m]*count);
  if(i<n)next_ms=i%2?0.001f:100000.f; // warmups intentionally misleading
  if(i>=n&&i<3*n)measures[m]++;
  t.end();t.observe(count);
 }
 assert(!t.policy.pending()&&t.policy.chosen==expected);
 for(int j=0;j<n;j++)assert(measures[j]==2);
}
int main(){
 const double baseline[4]={1e-5,1.1e-5,1.2e-5,1.3e-5};
 for(bool wide:{false,true}){
  int n=wide?4:2;
  for(int winner=0;winner<n;winner++){
   double rate[4];for(int j=0;j<4;j++)rate[j]=baseline[j];rate[winner]=8e-6;
   simulate(wide,rate,winner);
  }
  const double tie[4]={1e-5,1e-5,1e-5,1e-5};simulate(wide,tie,0);
  const double margin[4]={1e-5,9.9e-6,9.85e-6,9.9e-6};simulate(wide,margin,0);
  for(float bad:{0.f,-1.f,float(NAN),float(INFINITY)}){
   AdaptiveSearchTuner t(true,wide);t.begin();next_ms=bad;t.end();t.observe(7);
   t.begin();assert(!adaptive_use_wide&&!wide_use_fused&&!t.policy.pending());
  }
  {AdaptiveSearchTuner t(true,wide);t.begin();next_ms=1;t.end();t.observe(0);
   t.begin();assert(!adaptive_use_wide&&!wide_use_fused&&!t.policy.pending());}
  {AdaptiveSearchTuner t(true,wide);t.begin();} // early destruction
 }
 assert(created==destroyed);
 int before=created,old_recorded=recorded;
 {AdaptiveSearchTuner t(false,true);for(int i=0;i<20;i++){
  assert(t.group_limit(64)==64);t.begin();t.end();t.observe(1);
  assert(!adaptive_use_wide&&!wide_use_fused);
 }}
 assert(created==before&&recorded==old_recorded&&created==destroyed);
 printf("PASS: four/two arms, every winner, ties, margin, count normalization, warmups, invalid timing, event lifetime, disabled mode\n");
}
'''
with tempfile.TemporaryDirectory(prefix='qsb-adaptive-policy-') as td:
 p=Path(td);(p/'audit.cpp').write_text(code)
 subprocess.run(['c++','-std=c++17','-O2','-fsanitize=undefined','-fno-sanitize-recover=all','-I'+str(base),str(p/'audit.cpp'),'-o',str(p/'audit')],check=True)
 output=subprocess.check_output([str(p/'audit')],text=True)
src=(base/'pinning.cu').read_text()
assert src.index('pipeline_err=cudaMalloc')<src.index('d_wide=adaptive_optional_wide')
start=src.index('uint64_t group_start_count=total_searched;')
assert start<src.index('schedule_tuner.begin();',start)<src.index('uint8_t *d_gt=adaptive_use_wide?',start)<src.index('cudaMemsetAsync',start)
end=src.index('schedule_tuner.end();',start)
assert src.index('total_searched += batch_sz;',start)<end<src.index('cudaMemcpy(&h_hit',end)<src.index('schedule_tuner.observe(',end)<src.index('if (h_hit > 0)',end)
result={'status':'PASS','gpu_executed':False,'ubsan':True,'summary':output.splitlines()[-1],
 'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [base/n for n in ('pinning.cu','adaptive_policy.cuh','adaptive_tuner.cuh','adaptive_resources.cuh')]},
 'limits':'Synthetic timings and stub CUDA events; source ordering check. No GPU event execution, table allocation or throughput measurement.'}
(HERE/'policy-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
