#!/usr/bin/env python3
"""Execute the actual optional cache-policy host code against a fault-injected API.

Checks bounds, pointer/stream ownership, rollback and visible unexpected errors.
This does not simulate cache residency, CUDA runtime behavior or GPU performance.
"""
import json,hashlib,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT))
from preflight import source_identity
base=HERE/'policy_candidate';identity=source_identity(base);h=base/'tests/gpu_epochs'
assert (h/'l2_policy.cuh').read_bytes()==(HERE/'l2_policy.cuh').read_bytes()
tree=(h/'tree.cu').read_text();build=tree.index('    compact_build_table(&d_gtX,&d_gtY,dp.neg_r_inv);')
assert tree.index('    if(se_mode)qsb_enable_small_l2_policy(d_gtX,gpu_index);')==build+len('    compact_build_table(&d_gtX,&d_gtY,dp.neg_r_inv);\n')
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
static constexpr size_t HOT=48ull<<20;
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
static void wide_cuda_require(int e,const char*){if(e)throw std::runtime_error("visible CUDA failure");}
'''+(h/'compact_geometry.cuh').read_text()+(h/'l2_policy.cuh').read_text()+r'''
static void reset(int c,int w,size_t p,size_t eff){
 cap=c;window=w;initial_limit=limit=p;effective=eff;wrote_policy=false;last_error=forced_last=clears=0;
 for(int i=0;i<8;++i)faults[i]=calls[i]=0;
}
int main(){
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
 fprintf(stderr,"PASS %d capacity/limit cases, %d injected failures\n",normal,fault);
}
'''
with tempfile.TemporaryDirectory(prefix='qsb-small48-policy-') as tmp:
 cpp=Path(tmp)/'check.cpp';binary=Path(tmp)/'check';cpp.write_text(code)
 compile_result=subprocess.run(['c++','-std=c++17','-O2',str(cpp),'-o',str(binary)],capture_output=True,text=True)
 if compile_result.returncode:print(compile_result.stderr,file=sys.stderr)
 compile_result.check_returncode()
 result=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
 assert 'PASS 180 capacity/limit cases, 30 injected failures' in result.stderr,result.stderr
assert source_identity(base)==identity
report={'status':'PASS','source_fingerprint':identity['source_fingerprint'],'host_cases':180,'injected_failures':30,
 'checks':'Exact48MiBprefix, defaultstream/device/pointer ownership, insufficient/unsupported limits, effective limit, rollback, unrelated/fatal error visibility, after-build placement.',
 'projection_sha256':hashlib.sha256(code.encode()).hexdigest(),'gpu_executed':False,'limits':__doc__}
(HERE/'policy-host-results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
