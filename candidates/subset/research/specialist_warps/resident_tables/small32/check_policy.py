#!/usr/bin/env python3
"""Execute the actual optional cache-policy host code against a fault-injected API.

Checks bounds, pointer/stream ownership, rollback and visible unexpected errors.
This does not simulate cache residency, CUDA runtime behavior or GPU performance.
"""
import argparse,json,hashlib,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT))
from preflight import source_identity
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,default=HERE/'candidate')
parser.add_argument('--report',type=Path,default=HERE/'policy-host-results.json')
args=parser.parse_args()
base=args.source.resolve();identity=source_identity(base);h=base/'tests/gpu_epochs'
component_sha256={name:hashlib.sha256((h/name).read_bytes()).hexdigest() for name in ['compact_geometry.cuh','l2_policy.cuh']}
tree=(h/'tree.cu').read_text();build=tree.index('    compact_build_table(&d_gt,&unused_table_y,dp.neg_r_inv);')
assert tree.index('    if(se_mode)qsb_enable_small_l2_policy(d_gt,gpu_index);')==build+len('    compact_build_table(&d_gt,&unused_table_y,dp.neg_r_inv);\n')
code=r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cassert>
#include <initializer_list>
using cudaError_t=int;
enum{cudaSuccess=0,cudaErrorNotSupported=1,cudaErrorUnsupportedLimit=2,cudaErrorInvalidValue=3,unexpected=4};
enum{cudaDevAttrMaxPersistingL2CacheSize=0,cudaDevAttrMaxAccessPolicyWindowSize=1,cudaLimitPersistingL2CacheSize=2,
     cudaStreamAttributeAccessPolicyWindow=3,cudaAccessPropertyPersisting=4,cudaAccessPropertyNormal=5};
struct cudaStreamAttrValue{struct{void*base_ptr;size_t num_bytes;float hitRatio;int hitProp,missProp;}accessPolicyWindow;};
static constexpr size_t HOT=32ull<<20;
static int cap,window,faults[8],calls[8],last_error,forced_last,clears;
static size_t initial_limit,limit,effective;
static bool wrote_policy;
static int fail(int site){++calls[site];int e=faults[site];if(e)last_error=forced_last?forced_last:e;return e;}
static int cudaGetLastError(){++clears;int e=last_error;last_error=0;return e;}
static int cudaDeviceGetAttribute(int*out,int attr,int device){
 assert(device==3);int e=fail(attr==cudaDevAttrMaxPersistingL2CacheSize?1:2);
 if(!e)*out=attr==cudaDevAttrMaxPersistingL2CacheSize?cap:window;return e;
}
static int cudaDeviceGetLimit(size_t*out,int kind){
 assert(kind==cudaLimitPersistingL2CacheSize);int site=calls[3]?5:3;int e=fail(site);
 if(!e)*out=site==3?initial_limit:effective;return e;
}
static int cudaDeviceSetLimit(int kind,size_t value){
 assert(kind==cudaLimitPersistingL2CacheSize);int site=calls[4]?7:4;int e=fail(site);
 if(!e)limit=value;return e;
}
static int cudaStreamSetAttribute(int stream,int attr,cudaStreamAttrValue*a){
 assert(stream==0&&attr==cudaStreamAttributeAccessPolicyWindow);
 assert(a->accessPolicyWindow.base_ptr==(void*)uintptr_t(0x1000));
 assert(a->accessPolicyWindow.num_bytes==HOT && a->accessPolicyWindow.num_bytes<=size_t(window));
 assert(a->accessPolicyWindow.hitRatio==1.0f && a->accessPolicyWindow.hitProp==cudaAccessPropertyPersisting);
 assert(a->accessPolicyWindow.missProp==cudaAccessPropertyNormal);
 int e=fail(6);if(!e)wrote_policy=true;return e;
}
static void wide_cuda_require(int e,const char*why){if(e)throw std::runtime_error(why);}
'''+(h/'compact_geometry.cuh').read_text()+(h/'l2_policy.cuh').read_text()+r'''
static void reset(int c,int w,size_t p,size_t eff){
 cap=c;window=w;initial_limit=limit=p;effective=eff;wrote_policy=false;last_error=forced_last=clears=0;
 for(int i=0;i<8;++i)faults[i]=calls[i]=0;
}
int main(){
 assert(MIXED_CHUNKS==16 && MIXED_TOTAL_ENTRIES==(1ull<<19));
 unsigned offset=0,shift=0;
 for(int c=0;c<16;++c){
  assert(mixed_bits(c)==16 && mixed_entries(c)==(1u<<15));
  assert(mixed_offset(c)==offset && mixed_shift(c)==int(shift));
  offset+=1u<<15;shift+=16;
 }
 assert(shift==256 && uint64_t(offset)*64==HOT);
 assert(mixed_offset(16)==offset);
 int normal=0,fault=0;
 for(int c:{0,int(HOT)-1,int(HOT),50<<20,64<<20})
 for(int w:{0,int(HOT)-1,int(HOT),128<<20})
 for(size_t p:{size_t(0),size_t(16<<20),HOT})
 for(size_t eff:{HOT-1,HOT,HOT+(2<<20)}){
  reset(c,w,p,eff);bool enabled=qsb_enable_small_l2_policy((void*)uintptr_t(0x1000),3);
  bool want=size_t(c)>=HOT&&size_t(w)>=HOT&&eff>=HOT;
  assert(enabled==want&&wrote_policy==want);
  assert(limit==(want?HOT:p));
  assert(last_error==0);++normal;
 }
 for(int site=1;site<=6;++site)for(int error=1;error<=4;++error){
  reset(50<<20,128<<20,16<<20,HOT);faults[site]=error;
  bool threw=false,enabled=false;
  try{enabled=qsb_enable_small_l2_policy((void*)uintptr_t(0x1000),3);}catch(const std::runtime_error&){threw=true;}
  bool mandatory_failure=site==5||error==unexpected;
  assert(threw==mandatory_failure&&!enabled&&!wrote_policy);
  if(!threw){assert(last_error==0);assert(limit==initial_limit);}
  ++fault;
 }
 // Expected optional failure must not conceal a distinct runtime error.
 for(int site:{1,2,3,4,6}){
  reset(50<<20,128<<20,16<<20,HOT);faults[site]=cudaErrorNotSupported;forced_last=unexpected;
  bool threw=false;try{qsb_enable_small_l2_policy((void*)uintptr_t(0x1000),3);}catch(const std::runtime_error&){threw=true;}
  assert(threw&&!wrote_policy);++fault;
 }
 // Failed rollback must be visible too.
 reset(50<<20,128<<20,16<<20,HOT);faults[6]=cudaErrorNotSupported;faults[7]=unexpected;
 bool threw=false;try{qsb_enable_small_l2_policy((void*)uintptr_t(0x1000),3);}catch(const std::runtime_error&){threw=true;}
 assert(threw&&!wrote_policy);++fault;
 int extra=0;
 for(int bad:{-1,-2147483647})for(int which:{0,1}){
  reset(which?int(HOT):bad,which?bad:int(HOT),16<<20,HOT);
  assert(!qsb_enable_small_l2_policy((void*)uintptr_t(0x1000),3));
  assert(!wrote_policy&&limit==initial_limit&&!calls[3]);++extra;
 }
 // A failed restore after a clamped effective limit is fatal, including
 // normally optional error codes. These differ from stream-set rollback.
 for(int error:{1,2,3,4}){
  reset(int(HOT),int(HOT),16<<20,HOT-1);faults[7]=error;
  bool failed=false;try{qsb_enable_small_l2_policy((void*)uintptr_t(0x1000),3);}
  catch(const std::runtime_error&){failed=true;}
  assert(failed&&!wrote_policy);++extra;
 }
 assert(extra==8);
 fprintf(stderr,"PASS %d capacity/limit cases, %d injected failures, %d extra paths\n",normal,fault,extra);
}
'''
with tempfile.TemporaryDirectory(prefix='qsb-resident32-policy-') as tmp:
 cpp=Path(tmp)/'check.cpp';binary=Path(tmp)/'check';cpp.write_text(code)
 compile_result=subprocess.run(['c++','-std=c++17','-O2',str(cpp),'-o',str(binary)],capture_output=True,text=True)
 if compile_result.returncode:print(compile_result.stderr,file=sys.stderr)
 compile_result.check_returncode()
 result=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
 assert 'PASS 180 capacity/limit cases, 30 injected failures, 8 extra paths' in result.stderr,result.stderr
 positive_stderr=result.stderr
 # Execute mutations of the actual extracted policy/geometry in the same
 # independently specified host oracle. A mutant is detected only at runtime.
 mutants={
  'old_cold_prefix_boundary':code.replace('MIXED_TOTAL_ENTRIES*64==hot_bytes &&\n                      (uint64_t)mixed_offset(MIXED_CHUNKS)==MIXED_TOTAL_ENTRIES', '(uint64_t)mixed_offset(9)*64==hot_bytes'),
  'wrong_hot_bytes_48MiB':code.replace('constexpr size_t hot_bytes=32ull<<20;', 'constexpr size_t hot_bytes=48ull<<20;'),
  'wrong_window_pointer':code.replace('base_ptr=table;', 'base_ptr=(char*)table+64;'),
  'missing_stream_failure_restore':code.replace('wide_cuda_require(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,previous),"restore unused L2 limit");', '(void)previous;'),
  'missing_clamped_limit_restore':code.replace('wide_cuda_require(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,previous),"restore smaller L2 limit");', '(void)previous;'),
 }
 negative_controls={}
 for name,mutation in mutants.items():
  assert mutation!=code,name
  cpp.write_text(mutation)
  built=subprocess.run(['c++','-std=c++17','-O2',str(cpp),'-o',str(binary)],capture_output=True,text=True)
  assert built.returncode==0,(name,built.stderr)
  ran=subprocess.run([str(binary)],capture_output=True,text=True)
  assert ran.returncode!=0,(name,'mutant unexpectedly passed')
  negative_controls[name]={'detected':True,'returncode':ran.returncode,'stderr':ran.stderr.strip()}
 assert 'L2 policy must cover exactly the complete32MiB table' in negative_controls['old_cold_prefix_boundary']['stderr']
assert source_identity(base)==identity
report={'status':'PASS','source_fingerprint':identity['source_fingerprint'],'host_cases':180,'injected_failures':30,
 'source_sha256':identity['source_sha256'],'component_sha256':component_sha256,'independent_geometry':[16]*16,
 'actual_after_build_call_verified':True,'extra_path_cases':8,'negative_controls':negative_controls,
 'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'checks':'Exact32MiBfulltable and sentinel, independently specified sixteen16-bit windows, defaultstream/device/pointer ownership, insufficient/unsupported limits, effective limit, rollback, unrelated/fatal error visibility, after-build placement.',
 'projection_sha256':hashlib.sha256(code.encode()).hexdigest(),'gpu_executed':False,'limits':__doc__}
args.report.parent.mkdir(parents=True,exist_ok=True)
args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
