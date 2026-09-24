"""Compile real GLV12 stack/cache setup with a fake CUDA runtime; no GPU execution."""
from pathlib import Path
import subprocess,tempfile,os,json
HERE=Path(__file__).resolve().parent
SUBSET=HERE.parents[1]
s=(SUBSET/'glv10_build.cuh').read_text()
a=s.index('static void glv10_select_stack()');b=s.index('__global__ void kernel_build_glv10',a)
setup=s[a:b]
mock=r'''
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "glv10_geometry.h"
static const char *testcase;
static bool mode(const char*x){return strcmp(testcase,x)==0;}
enum {cudaLimitStackSize,cudaLimitPersistingL2CacheSize,cudaStreamAttributeAccessPolicyWindow,cudaAccessPropertyPersisting,cudaAccessPropertyStreaming};
struct cudaDeviceProp {int l2CacheSize,persistingL2CacheMaxSize,accessPolicyMaxWindowSize;};
struct Window {void*base_ptr;size_t num_bytes;float hitRatio;int hitProp,missProp;};
struct cudaStreamAttrValue {Window accessPolicyWindow;};
static cudaStreamAttrValue saved={};
static unsigned calls=0;
static void called(const char*s){++calls;printf("CALL %s\n",s);}
namespace qsb_native {
static constexpr auto consumer_stream=nullptr;
static bool is_active(){return !mode("source");}
static void check(int e,const char*s){if(e){fprintf(stderr,"CUDA failure %s\n",s);exit(3);}}
}
static void glv10_require(bool ok,const char*s){if(!ok){fprintf(stderr,"require failure %s\n",s);exit(2);}}
static void glv10_memory(const char*s){printf("MEM %s\n",s);}
static int cudaDeviceSetLimit(int which,size_t value){called("setlimit");
 if(which==cudaLimitStackSize){if(value!=32768)abort();return mode("stack_api");}
 if(value!=50331648)abort();return mode("cache_api");}
static int cudaDeviceGetLimit(size_t*out,int which){called("getlimit");
 if(which==cudaLimitStackSize){*out=mode("stack_low")?16384:mode("stack_rounded")?65536:32768;return mode("stack_read_api");}
 *out=mode("cache_low")?16777216:mode("cache_high")?67108864:50331648;return mode("cache_read_api");}
static int cudaStreamSetAttribute(decltype(nullptr) stream,int attr,const cudaStreamAttrValue*p){called("setattr");
 if(stream!=nullptr||attr!=cudaStreamAttributeAccessPolicyWindow)abort();saved=*p;return mode("stream_api");}
static int cudaStreamGetAttribute(decltype(nullptr) stream,int attr,cudaStreamAttrValue*p){called("getattr");
 if(stream!=nullptr||attr!=cudaStreamAttributeAccessPolicyWindow)abort();*p=saved;
 if(mode("wrong_base"))p->accessPolicyWindow.base_ptr=nullptr;
 if(mode("wrong_bytes"))p->accessPolicyWindow.num_bytes--;
 if(mode("wrong_ratio"))p->accessPolicyWindow.hitRatio=.5;
 if(mode("wrong_property"))p->accessPolicyWindow.hitProp=cudaAccessPropertyStreaming;
 if(mode("wrong_miss_property"))p->accessPolicyWindow.missProp=cudaAccessPropertyPersisting;
 return mode("stream_read_api");}
'''
main=r'''
int main(int argc,char**argv){if(argc!=2)return 9;testcase=argv[1];
 glv10_select_stack();
 cudaDeviceProp prop={75497472,50331648,134217728};
 if(mode("l2_cap"))prop.l2CacheSize=16777216;
 if(mode("persist_cap"))prop.persistingL2CacheMaxSize=16777216;
 if(mode("window_cap"))prop.accessPolicyMaxWindowSize=16777216;
 glv10_cache_policy(reinterpret_cast<uint8_t*>(UINT64_C(0x100000000)),prop);
 if(calls!=6||saved.accessPolicyWindow.base_ptr!=reinterpret_cast<void*>(UINT64_C(0x100000000))||saved.accessPolicyWindow.num_bytes!=50331648)return 8;
 puts("PASS");return 0;}
'''
cases={'success':0,'stack_rounded':0,'source':0,'stack_api':3,'stack_low':2,'l2_cap':2,'persist_cap':2,'window_cap':2,'cache_api':3,'cache_low':2,'cache_high':2,'stream_api':3,'wrong_base':2,'wrong_bytes':2,'wrong_ratio':2,'wrong_property':2,'wrong_miss_property':2,'stack_read_api':3,'cache_read_api':3,'stream_read_api':3}
with tempfile.TemporaryDirectory(prefix='qsb-glv12-policy-') as td:
 p=Path(td);(p/'test.cpp').write_text(mock+setup+main)
 subprocess.run([os.environ.get('CXX','c++'),'-std=c++14','-O2','-I',str(SUBSET),str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 for case,code in cases.items():
  r=subprocess.run([str(p/'test'),case],capture_output=True,text=True)
  assert r.returncode==code,(case,r.returncode,r.stdout,r.stderr)
  
  if code==0:assert 'PASS' in r.stdout
print(json.dumps({'status':'PASS','mock_runtime_cases':len(cases),'real_setup_bodies':True,'source_and_native_policy_tested':True,'gpu_executed':False},indent=2))
