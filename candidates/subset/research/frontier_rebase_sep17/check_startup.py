#!/usr/bin/env python3
"""Execute the actual donor host startup decisions, not CUDA/table arithmetic.
No L2-policy checker is applicable to this source: that policy is absent.
"""
import argparse,hashlib,importlib.util,json,re,shutil,subprocess,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('preflight',ROOT/'preflight.py');pre=importlib.util.module_from_spec(spec);spec.loader.exec_module(pre)

def portions(source):
    t=(source/'tests/gpu_epochs/tree.cu').read_text()
    geometry=t[t.index('#ifndef ZLAB_T14'):t.index('/* n = secp256k1 group order')]
    startup=t[t.index('    size_t gt_sz = (size_t)GT_TOTAL_ENTRIES*64;'):t.index('    /* Upload params */')]
    return t,geometry,startup

PREFIX=r'''
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cassert>
#include <ctime>
#define __host__
#define __device__
#define __forceinline__ inline
static int hmfail,cmfail,copyfail,syncfail,kernel_error,spot_result;
static int hm,cm,copies,launches,spots,fallbacks,frees;
static size_t alloc_sizes[3];
static int fake_printf(const char*,...){return 0;}
static int fake_fprintf(FILE*,const char*,...){return 0;}
static int fake_fflush(FILE*){return 0;}
static void *fake_malloc(size_t){++hm;return hm==hmfail?nullptr:reinterpret_cast<void*>(uintptr_t(0x1000+hm*256));}
static void fake_free(void*){++frees;}
using cudaError_t=int;
enum{cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2};
template<class T>int cudaMalloc(T**out,size_t n){alloc_sizes[cm]=n;++cm;*out=cm==cmfail?nullptr:reinterpret_cast<T*>(uintptr_t(0x2000+cm*256));return cm==cmfail?1:0;}
static int cudaMemcpy(void*,const void*,size_t,int){return ++copies==copyfail?1:0;}
static int cudaFree(void*){return 0;}
static int cudaDeviceSynchronize(){return syncfail;}
static int cudaGetLastError(){return kernel_error;}
static const char*cudaGetErrorString(int){return "injected";}
static void gt_build_ladders(uint64_t*,uint64_t*,const uint64_t*){}
static void fake_launch(int blocks,int threads,uint64_t*,uint64_t*,uint8_t*){assert(blocks==4096&&threads==256);++launches;}
static int gt_spot_check(uint8_t*,int n,const uint64_t*){assert(n==252);++spots;return spot_result;}
static void compute_gtable(uint8_t*,const uint64_t*){++fallbacks;}
static struct{uint64_t neg_r_inv[4];}dp;
#define malloc fake_malloc
#define free fake_free
#define printf fake_printf
#define fprintf fake_fprintf
#define fflush fake_fflush
'''
SUFFIX=r'''
return 0;
}
static void reset(){hmfail=cmfail=copyfail=syncfail=kernel_error=0;spot_result=1;hm=cm=copies=launches=spots=fallbacks=frees=0;}
int main(){
 assert(GT_CHUNKS==15&&GT_LO==256&&GT_HI==1024&&GT_TOTAL_ENTRIES==1048576);
 unsigned off=0,shift=0;
 for(int c=0;c<15;c++){
  assert(gt_offset(c)==off&&gt_shift(c)==int(shift));
  unsigned entries=1u<<(c==0?17:16);assert(gt_entries(c)==entries);off+=entries;shift+=(c==0?18:17);
 }
 assert(off==GT_TOTAL_ENTRIES&&shift==256);
 int cases=0;
 for(int ke:{0,1})for(int sp:{0,1}){
  reset();kernel_error=ke;spot_result=sp;assert(startup()==0);
  assert(alloc_sizes[0]==64ull*1024*1024&&alloc_sizes[1]==245760&&alloc_sizes[2]==983040);
  assert(launches==1&&spots==(ke?0:1)&&fallbacks==int(ke||!sp));++cases;
 }
 for(int fault:{1,2,3}){reset();hmfail=fault;assert(startup()==1);++cases;}
 // These cases document unchecked return values, NOT safe error recovery.
 for(int fault:{1,2,3}){reset();cmfail=fault;assert(startup()==0&&fallbacks==0);++cases;}
 for(int fault:{1,2,3}){reset();copyfail=fault;assert(startup()==0&&fallbacks==0);++cases;}
 reset();copyfail=3;kernel_error=1;assert(startup()==0&&fallbacks==1&&copies==3);++cases;
 reset();syncfail=1;assert(startup()==0&&fallbacks==0);++cases;
 ::puts("startup_decision_cases=15 unchecked_status_cases=8");
 assert(cases==15);return 0;
}
'''

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=HERE/'candidate');p.add_argument('--base',type=Path,default=HERE/'pending217');p.add_argument('--report',type=Path,default=HERE/'startup-results.json');a=p.parse_args()
    src=a.source.resolve();base=a.base.resolve();identity=pre.source_identity(src)
    t,g,h=portions(src);bt,bg,bh=portions(base)
    assert g==bg and h==bh,'startup/geometry differs from reviewed pending217'
    assert 'tests/gpu_epochs/l2_policy.cuh' not in identity['source_sha256']
    assert not re.search(r'cudaStreamSetAttribute|cudaLimitPersistingL2CacheSize|qsb_enable_small_l2_policy',t)
    gate=(ROOT/'preflight.py').read_text();assert 'if "tests/gpu_epochs/l2_policy.cuh" in report["source_sha256"]:' in gate
    old='kernel_build_gtable<<<(gt_total+255)/256,256>>>(dL,dH,d_gt);'
    assert h.count(old)==1
    projection=h.replace(old,'fake_launch((gt_total+255)/256,256,dL,dH,d_gt);')
    code='#include <initializer_list>\n'+PREFIX+g+'\nstatic int startup(){\n'+projection+SUFFIX
    compiler=shutil.which('clang++') or shutil.which('g++');assert compiler,'C++ compiler required'
    negative={}
    with tempfile.TemporaryDirectory(prefix='qsb-frontier-startup-') as tmp:
        tmp=Path(tmp)
        for name,body in [('actual',code),('missing-host-fallback',code.replace('compute_gtable(chk_table,dp.neg_r_inv);','/* removed fallback */')),('wrong-first-width',code.replace('return c == 0 ? (1u << 17) : (1u << 16);','return (1u << 16);'))]:
            path=tmp/(name+'.cpp');path.write_text(body);exe=tmp/name
            subprocess.run([compiler,'-std=c++17','-O2',str(path),'-o',str(exe)],check=True,capture_output=True,timeout=60)
            r=subprocess.run([str(exe)],capture_output=True,text=True,timeout=20)
            if name=='actual':assert r.returncode==0,r.stderr;result=r.stdout.strip()
            else:assert r.returncode!=0;negative[name]={'compiled':True,'rejected':True,'returncode':r.returncode}
    assert identity==pre.source_identity(src),'source changed during check'
    report={'status':'PASS_UNCHANGED_STARTUP_PROJECTION_POLICY_NOT_APPLICABLE',**identity,'base_identity':pre.source_identity(base),'policy_gate_applicable':False,'actual_after_build_call_verified':False,'policy_capacity_cases':0,'policy_injected_failures':0,'startup_geometry_byte_identical_to_base':True,'host_startup_byte_identical_to_base':True,'component_sha256':{'geometry':hashlib.sha256(g.encode()).hexdigest(),'host_startup':hashlib.sha256(h.encode()).hexdigest()},'startup_decision_cases':15,'unchecked_status_cases':8,'execution':result,'negative_controls':negative,'geometry':{'widths':[18]+[17]*14,'table_bytes':67108864,'low_ladder_bytes':245760,'high_ladder_bytes':983040,'host_spot_samples':252},'limits':['No L2 policy exists, so no 180+30 policy matrix or after-policy-build call can truthfully be claimed.','Actual extracted startup decisions execute with mocked CUDA, allocation and curve-builder APIs; GPU kernels and curve construction are not executed.','Source ignores cudaMalloc/cudaMemcpy/cudaDeviceSynchronize return values at this startup site; injected ignored errors expose that inherited limitation.','Policy gate absence is not proof of safe CUDA error handling.','Main preflight CLI binds its own directory; use source_identity(root) for isolated snapshots, or copy preflight to actual staged candidates/subset before package checks.'],'audit_commands':['nvcc -O3 -DQSB_ZEROS_N=24 -o /tmp/qsb-frontier-audit "$SRC/tests/gpu_epochs/tree_audit.cu" -lcrypto -lm','nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24 -o /tmp/qsb-frontier-audit-sm89 "$SRC/tests/gpu_epochs/tree_audit.cu" -lcrypto -lm'],'gpu_executed':False,'cuda_compile_executed_by_this_checker':False}
    a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['status','source_fingerprint','startup_decision_cases','unchecked_status_cases','policy_gate_applicable']}))
if __name__=='__main__':main()
