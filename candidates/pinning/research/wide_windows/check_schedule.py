#!/usr/bin/env python3
"""Run the actual runtime schedule selector with CUDA event stubs and UBSan.

Also bind fused prepare/finish expressions to the split source. No GPU timing,
collective execution or throughput claim. Timing inputs below are synthetic.
"""
import hashlib,json,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
from fused_screen import make_fused

def main():
    base=HERE/'candidate';src=(base/'pinning.cu').read_text()
    fused=(base/'wide_fused_kernel.cuh').read_text()
    assert fused.endswith(make_fused(src)), 'Fused expressions drifted from split source'
    header=(base/'wide_search_tuner.cuh').read_text()
    start=src.index('            uint64_t group_start_count=total_searched;')
    assert start<src.index('schedule_tuner.begin();',start)<src.index('cudaMemsetAsync(d_hit_cnt',start)
    end=src.index('            schedule_tuner.end();',start)
    assert src.index('total_searched += batch_sz;',start)<end
    assert end<src.index('cudaMemcpy(&h_hit',end)<src.index('schedule_tuner.observe(',end)<src.index('if (h_hit > 0)',end)
    code=r'''
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>
using cudaEvent_t=int;
static bool wide_use_fused=false;
static int created=0,destroyed=0,recorded=0,readings=0;
static float next_ms=1;
static void wide_cuda_require(int status,const char*){assert(status==0);}
static int cudaEventCreate(int*e){*e=++created;return 0;}
static int cudaEventDestroy(int){++destroyed;return 0;}
static int cudaEventRecord(int){++recorded;return 0;}
static int cudaEventElapsedTime(float*ms,int,int){*ms=next_ms;++readings;return 0;}
'''+header+r'''
static int simulate(double split,double fused,bool expected){
 WideSearchTuner t(true);int before=recorded;
 const bool modes[6]={false,true,true,false,false,true};
 for(int i=0;i<8;i++){
  assert(t.group_limit(128)==unsigned(i<6?4:128));
  t.begin();assert(wide_use_fused==(i<6?modes[i]:expected));
  // Different candidate counts test rate normalization, not raw elapsed time.
  uint64_t count=(i%2)?1000000:2000000;
  next_ms=float((wide_use_fused?fused:split)*double(count));
  // Excluded warmups deliberately have misleading and unbalanced costs.
  if(i<2)next_ms=i?0.001f:100000.0f;
  t.end();t.observe(count);
 }
 assert(!t.policy.pending()&&t.policy.chosen_fused==expected);
 assert(recorded-before==12);return 1;
}
int main(){
 unsigned tests=0;
 tests+=simulate(1e-5,8e-6,true);
 tests+=simulate(1e-5,12e-6,false);
 tests+=simulate(1e-5,1e-5,false);
 tests+=simulate(1e-5,9.9e-6,false);
 tests+=simulate(1e-5,9.7e-6,true);
 // Invalid event results retain the external schedule and release events.
 for(float bad:{0.0f,-1.0f,float(NAN),float(INFINITY)}){
  WideSearchTuner t(true);t.begin();next_ms=bad;t.end();t.observe(7);
  t.begin();assert(!wide_use_fused&&!t.policy.pending());++tests;
 }
 {WideSearchTuner t(true);t.begin();next_ms=1;t.end();t.observe(0);
  t.begin();assert(!wide_use_fused&&!t.policy.pending());++tests;}
 assert(created==destroyed);
 int old_created=created,old_recorded=recorded,old_readings=readings;
 {WideSearchTuner t(false);for(int i=0;i<10;i++){
   assert(t.group_limit(64)==64);
   t.begin();assert(!wide_use_fused);t.end();t.observe(100);
 }}
 assert(created==old_created&&recorded==old_recorded&&readings==old_readings);++tests;
 std::printf("AUDIT %u %d %d %d\n",tests,created,destroyed,readings);
}
'''
    with tempfile.TemporaryDirectory(prefix='qsb-wide-schedule-') as td:
        cpp=Path(td)/'audit.cpp';binary=Path(td)/'audit';cpp.write_text(code)
        subprocess.run(['c++','-std=c++17','-O2','-fsanitize=undefined','-fno-sanitize-recover=all','-Wno-pragma-once-outside-header',str(cpp),'-o',str(binary)],check=True)
        output=subprocess.check_output([str(binary)],text=True)
    result={'status':'PASS','source_sha256':{name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in ('pinning.cu','wide_fused_kernel.cuh','wide_search_tuner.cuh')},
        'gpu_executed':False,'ubsan':True,'summary':output.strip().splitlines()[-1],
        'synthetic_timing_cases':'fast/slow/tie/inside and outside2% margin; distinct work counts; ignored warmups; invalid/zero timings; disabled generic mode; event lifetime',
        'limits':'CPU timing-policy behavior and exact split/fused source-expression correspondence only; no CUDA events or curve collective executed.'}
    (HERE/'schedule-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
