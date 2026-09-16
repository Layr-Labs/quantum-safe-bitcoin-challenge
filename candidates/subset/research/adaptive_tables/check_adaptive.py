#!/usr/bin/env python3
"""Execute real table policy, launch dispatch and ranked host loop on CPU mocks.

This validates orchestration only. Separate OpenSSL audits test the actual EC
branches; native compiler reports do not imply GPU timing or execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent.parent))
from preflight import source_identity
from check_candidate import function

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--source',type=Path,default=HERE/'candidate')
ap.add_argument('--report',type=Path,default=HERE/'host-flow-results.json')
args=ap.parse_args()
root=args.source.resolve();src=root/'tests/gpu_epochs'
identity=source_identity(root)
tree=(src/'tree.cu').read_text();pipe=(src/'ranked_pipeline.cuh').read_text()
compact=(src/'compact_table_device.cuh').exists()
if compact:pipe=pipe.replace('qsb_ranked_prepare_compact','qsb_ranked_prepare_small')
dispatch=function(pipe,'static cudaError_t qsb_launch_ranked_pipeline(')
dispatch=re.sub(r'<<<.*?>>>','',dispatch)
section=tree[tree.index('    /* Short-epoch path: producer/consumer'):]
loop=function(section,'while (1)')
loop=re.sub(r'<<<.*?>>>','',loop)
policy=(src/'table_policy.cuh').read_text()
tuner=(src/'table_tuner.cuh').read_text().replace('#include "table_policy.cuh"','')
resources=(src/'table_resources.cuh').read_text()

STUB=r'''
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <stdexcept>
#define MAX_T 16
#define QSB_SE_PER_EPOCH 256
#define QSB_SE_LAUNCH_BLOCKS 32768
using cudaError_t=int;using cudaEvent_t=int;
const int cudaSuccess=0,cudaErrorInvalidValue=1,cudaErrorMemoryAllocation=2,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
#define GT_TOTAL_ENTRIES (1ull<<28)
struct epoch_desc_t{int dummy;};struct ulonglong2{uint64_t x,y;};
static uint8_t smallX[1],smallY[1],wideX[1];
static uint64_t current_epoch,observed_epochs,g_total_searched,g_hit_counter;
static unsigned iterations,creations,destructions,records,releases;
static bool available,expect_wide_wins,invalid_time;
static std::vector<bool> modes;
static int stage;
static void wide_cuda_require(int e,const char*){if(e!=cudaSuccess)throw std::runtime_error("CUDA mock failure");}
static int cudaFree(void*p){assert(p==wideX);++releases;return 0;}
static size_t mock_free_bytes;
static int alloc_error,last_error,query_error;
static unsigned queries,allocations,builds,error_clears;
static int cudaMemGetInfo(size_t*f,size_t*t){++queries;*f=mock_free_bytes;*t=24ull<<30;return query_error;}
static int cudaMalloc(uint8_t**p,size_t n){++allocations;assert(n==(16ull<<30));if(!alloc_error)*p=wideX;return alloc_error;}
static void wide_build_table(uint8_t*p,const uint8_t*){assert(p==wideX);++builds;}
static int cudaEventCreate(int *e){*e=++creations;return 0;}
static int cudaEventDestroy(int){++destructions;return 0;}
static int cudaEventRecord(int){++records;return 0;}
static int cudaEventElapsedTime(float *ms,int,int){
 *ms=invalid_time?std::numeric_limits<float>::quiet_NaN():
     (modes.back()?(expect_wide_wins?0.7f:1.3f):1.f);
 return 0;
}
static int cudaMemcpy(void*d,const void*s,size_t n,int){memcpy(d,s,n);return 0;}
static int cudaGetLastError(){int e=last_error;last_error=0;++error_clears;return e;}
static int cudaDeviceSynchronize(){return 0;}
static const char*cudaGetErrorString(int){return "mock";}
static void kernel_build_epochs(uint64_t first,uint64_t total,int,int,
 const uint32_t*,const uint8_t*,int,const uint8_t*,epoch_desc_t*){
 assert(first==observed_epochs&&first<total&&stage==0);current_epoch=first;
}
static void prepare(bool wide,const uint8_t*x,const uint8_t*y,int count){
 assert(stage++==0&&count>0&&count%256==0);
 assert(x==(wide?wideX:smallX)&&y==(wide?nullptr:smallY));
 bool expected=available&&(!invalid_time||iterations==0)&&
   (iterations<6?(iterations==1||iterations==2||iterations==5):expect_wide_wins);
 assert(wide==expected);modes.push_back(wide);
}
static void qsb_ranked_prepare(const epoch_desc_t*,const uint8_t*x,const uint8_t*y,
 const uint64_t*,ulonglong2*,uint64_t*,uint64_t*,int count){prepare(true,x,y,count);}
static void qsb_ranked_prepare_small(const epoch_desc_t*,const uint8_t*x,const uint8_t*y,
 const uint64_t*,ulonglong2*,uint64_t*,uint64_t*,int count){prepare(false,x,y,count);}
static void qsb_root_group_prepare(const uint64_t*,int,uint64_t*,uint64_t*){assert(stage++==1);}
static void qsb_invert_super_roots(uint64_t*,int){assert(stage++==2);}
static void qsb_root_group_finish(uint64_t*,int,const uint64_t*,const uint64_t*){assert(stage++==3);}
static void qsb_ranked_finish(const epoch_desc_t*,const uint64_t*,const uint64_t*,
 const ulonglong2*,const uint64_t*,const uint64_t*,int count,uint32_t*hits,uint32_t*index,uint8_t*combos){
 assert(stage++==4);*hits=2;index[0]=0;index[1]=(uint32_t)(count-1)|(1u<<30);
 for(int h=0;h<2;++h)for(int j=0;j<MAX_T;++j)combos[h*MAX_T+j]=(uint8_t)(j+h);
 observed_epochs+=count/256;++iterations;stage=0;
}
'''
RUN=r'''
static int flow(bool avail,bool wins,bool bad,uint64_t epochs){
 available=avail;expect_wide_wins=wins;invalid_time=bad;
 current_epoch=observed_epochs=g_total_searched=g_hit_counter=0;
 iterations=creations=destructions=records=releases=0;stage=0;last_error=0;modes.clear();
 uint8_t*d_gtX=smallX,*d_gtY=smallY,*d_wide=avail?wideX:nullptr;
 uint32_t countbuf=0,indexbuf[1024]={};uint8_t combobuf[1024*MAX_T]={};
 uint32_t*d_hit_cnt=&countbuf,*d_hit_idx=indexbuf;uint8_t*d_hit_combos=combobuf;
 uint32_t*d_mid=nullptr;uint8_t*d_prem=nullptr,*d_dsigs=nullptr;epoch_desc_t*d_epochs=nullptr;
 uint64_t*d_u2rx=nullptr,*d_u2ry=nullptr,*d_pipe_roots=nullptr,*d_pipe_tree=nullptr,
 *d_pipe_super=nullptr,*d_pipe_root_tree=nullptr;ulonglong2*d_pipe_state=nullptr;
 struct{int prefix_remainder_len=0;}dp;
 int window_start=137,s_early=6,gpu_index=0,calibrate=0,t_sel=9;
 uint64_t n_epochs=epochs,epoch_base=0,total_searched=0,hit_counter=0,global_total=epochs*256;
 timespec t0,t_last_se;clock_gettime(CLOCK_MONOTONIC,&t0);t_last_se=t0;
 FILE*summary_f=nullptr;
 QsbTableTuner table_tuner(d_wide!=nullptr);
 HOST_LOOP
 assert(epoch_base==epochs&&observed_epochs==epochs);
 assert(total_searched==epochs*256&&g_total_searched==total_searched);
 assert(hit_counter==2*iterations&&g_hit_counter==hit_counter);
 if(avail){assert(creations==2);assert(destructions==(bad||iterations>=6?2u:0u));}
 else assert(creations==0&&destructions==0&&records==0);
 assert(releases==(avail&&(bad||(!wins&&iterations>=6))?1u:0u));
 return 0;
}
static void allocation_checks(){
 for(int c=0;c<7;++c){
  queries=allocations=builds=error_clears=0;
  mock_free_bytes=(18ull<<30)-(c==1?1:0);query_error=c==6?9:0;
  alloc_error=(c==3||c==5)?cudaErrorMemoryAllocation:(c==4?9:0);
  last_error=c==3?cudaErrorMemoryAllocation:(c==5?9:0);
  bool fatal=false;uint8_t*out=nullptr,base[32]={};
  try{out=qsb_optional_wide_table(c!=0,base);}catch(const std::runtime_error&){fatal=true;}
  assert(fatal==(c>=4));
  assert(out==(c==2?wideX:nullptr));
  assert(queries==(c==0?0u:1u));
  assert(allocations==((c>=2&&c<=5)?1u:0u));
  assert(builds==(c==2?1u:0u));
  if(c==3){assert(error_clears==1&&last_error==0);}
 }
}
static void policy_checks(){
 const double samples[][6]={{900,800,1,1,1,1},{99,99,0.5,1,1,0.5},
                          {99,99,2,1,1,2},{99,99,0.98,1,1,0.98}};
 const bool expected[]={false,true,false,false};
 for(int c=0;c<4;++c){
  QsbTablePolicy p;
  for(int i=0;i<6;++i){assert(p.wide()==(i==1||i==2||i==5));p.observe(samples[c][i],100);}
  assert(!p.pending()&&p.wide()==expected[c]);p.observe(1e-9,999);assert(p.wide()==expected[c]);
 }
 QsbTablePolicy weighted;
 for(int i=0;i<6;++i)weighted.observe(i==2?10:1,i==2?1000:100);
 assert(!weighted.wide()); // equal time per candidate despite unequal batches
 const double bad[]={0,-1,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()};
 for(double v:bad){QsbTablePolicy p;p.observe(v,100);assert(!p.pending()&&!p.wide());}
 QsbTablePolicy zero;zero.observe(1,0);assert(!zero.pending()&&!zero.wide());
}
int main(){
 policy_checks();allocation_checks();
 for(int mode=0;mode<4;++mode)
  for(uint64_t epochs: {1ull,32768ull,32769ull,6ull*32768,9ull*32768+17})
   {assert(flow(mode!=0,mode==1,mode==3,epochs)==0);assert(creations==destructions);}
 fprintf(stderr,"PASS:20 extracted host-loop scenarios; no missing/duplicate ranges or lost hits; both dispatch layouts and policy cases checked\n");
}
'''
code=STUB+policy+tuner+resources+dispatch+RUN.replace('HOST_LOOP',loop)
if compact:
    code=code.replace('wide?nullptr:smallY','nullptr')
    code=code.replace('*d_gtY=smallY','*d_gtY=nullptr')
with tempfile.TemporaryDirectory(prefix='qsb-adaptive-flow-') as td:
    d=Path(td);cpp=d/'flow.cpp';exe=d/'flow';cpp.write_text(code)
    subprocess.run(['c++','-std=c++17','-O2',str(cpp),'-o',str(exe)],check=True)
    with (d/'stdout.txt').open('w') as output:
        subprocess.run([str(exe)],cwd=d,stdout=output,check=True)
assert source_identity(root)==identity
report={'status':'PASS','validation_level':'CPU execution of extracted policy, launch dispatch and ranked host loop with mocked GPU computation/events',
        'host_scenarios':20,'policy_cases':10,'allocation_scenarios':7,'source_fingerprint':identity['source_fingerprint'],
        'projection_sha256':hashlib.sha256(code.encode()).hexdigest(),'gpu_executed':False,
        'limits':'Does not execute CUDA, table building, EC, SHA, GPU timing, or real rare-hit behavior. Actual branches have separate OpenSSL audits. Mocks exercise every host hit path with two records per batch and check ordered, non-overlapping full/tail epoch ranges.'}
report['compact_layout']='interleaved64MiB' if compact else 'planar32MiB'
args.report.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
