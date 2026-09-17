#!/usr/bin/env python3
"""Actual EC192 tree source with cooperative CPU lanes and ownership checks.

OpenSSL implements canonical field operations; Python pow supplies independent
inverses. This is not CUDA arithmetic, GPU scheduling or a PTX barrier test.
"""
import argparse
import ast
import ctypes as C
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HEADER = ROOT/'tests/gpu_epochs/tree_inverse.cuh'
BASE = HERE.parent/'one_seven/candidate/tests/gpu_epochs/tree_inverse.cuh'
SIG = '__device__ __forceinline__ void qsb_ec192_inverse_tree_scratch('


def sha(value):
    return hashlib.sha256(value).hexdigest()


def function(text, signature):
    start = text.index(signature)
    stop = text.index('{', start)+1
    depth = 1
    while depth:
        depth += (text[stop] == '{')-(text[stop] == '}')
        stop += 1
    return text[start:stop]


HOST = r'''
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <openssl/bn.h>
#define __device__
#define __forceinline__ inline
[[noreturn]] static void fail(const char* why){fprintf(stderr,"QSB_EC192_FAILURE %s\n",why);fflush(stderr);std::_Exit(86);}
static void require(bool b){if(!b)fail("field_or_invariant");}
static thread_local int owner=-1,physical=-1,phase=0,warp_phase=0,group_joins=0,warp_joins=0;
struct Barrier{
 std::mutex m;std::condition_variable cv;int left,total,generation=0;
 explicit Barrier(int n):left(n),total(n){}
 void wait(){std::unique_lock<std::mutex> lock(m);int g=generation;
  if(--left==0){left=total;++generation;cv.notify_all();}
  else if(!cv.wait_for(lock,std::chrono::seconds(10),[&]{return generation!=g;}))fail("barrier_timeout");
 }
};
static Barrier *group_barrier,*warps[8];
static uint64_t shuffle_values[8][32];
static std::atomic<int> ownership_writes{0},cross_warp_reads{0};
static void subgroup_join(){
 if(physical<64||physical>=256||owner!=physical-64)fail("subgroup_membership");
 group_barrier->wait();++phase;++group_joins;
}
static void __syncwarp(){warps[physical/32]->wait();++warp_phase;++warp_joins;}
static unsigned long long __shfl_sync(unsigned mask,unsigned long long v,int lane){
 if(mask!=0xffffffffu||lane<0||lane>=32)fail("shuffle_mask_or_lane");
 int w=physical/32;shuffle_values[w][physical%32]=v;warps[w]->wait();
 uint64_t result=shuffle_values[w][lane];warps[w]->wait();return result;
}
struct Cell{
 uint64_t v=0x9ad34e17b62105c8ULL;int writer=-1,write_phase=-1,write_warp_phase=-1;
 std::mutex lock;
 operator uint64_t(){
  std::lock_guard<std::mutex> guard(lock);
  if(writer<0)fail("uninitialized_tree_read");
  if(writer!=owner && write_phase>=phase){
   if(writer/32!=owner/32)fail("cross_warp_read_without_subgroup_join");
   if(write_warp_phase>=warp_phase)fail("same_warp_read_without_warp_join");
  }
  if(writer/32!=owner/32)++cross_warp_reads;
  return v;
 }
 void store(uint64_t value,int row,int col){
  std::lock_guard<std::mutex> guard(lock);
  if(owner<0||owner>=192)fail("invalid_ecid");
  if(phase==0){
   int expected=-1;
   if(col>=256&&col<448)expected=col-256;
   else if(col>=448)expected=col-448;
   else if(col>=128&&col<224)expected=2*(col-128);
   else if(col>=224&&col<256)expected=col-224;
   if(expected!=owner)fail("initial_tree_writer_ownership");
   if((col>=448||(col>=224&&col<256)) && value!=(row==0?1ULL:0ULL))fail("padding_not_identity");
   ++ownership_writes;
  }
  v=value;writer=owner;write_phase=phase;write_warp_phase=warp_phase;
 }
};
struct AuditTree{
 Cell cells[4][512];
 struct Ref{Cell*cell;int row,col;operator uint64_t(){return (uint64_t)*cell;}
  void operator=(uint64_t value){cell->store(value,row,col);}};
 struct Row{AuditTree*tree;int row;Ref operator[](int col){
  if(col<0||col>=512)fail("tree_column_bounds");return {&tree->cells[row][col],row,col};}};
 Row operator[](int row){if(row<0||row>=4)fail("tree_row_bounds");return {this,row};}
};
'''

WRAPPER = r'''
extern "C" void audit_ec192(const uint64_t*input,uint64_t*out,int*counts){
 multiply_count=0;inverse_count=0;ownership_writes=0;cross_warp_reads=0;
 group_barrier=new Barrier(192);
 for(int w=2;w<8;w++)warps[w]=new Barrier(32);
 AuditTree tree;std::vector<std::thread> lanes;int joins[192],wjoins[192];
 for(int ecid=0;ecid<192;ecid++)lanes.emplace_back([&,ecid]{
  owner=ecid;physical=ecid+64;phase=warp_phase=group_joins=warp_joins=0;
  struct GuardedValue{uint64_t before,value[5],after;} guarded;
  guarded.before=guarded.after=0x719fd875ec4123abULL;
  uint64_t *value=guarded.value;memcpy(value,input+5*ecid,40);
  uint32_t sync[2]={0,0}; // Unused by the numerical barrier abstraction.
  qsb_ec192_inverse_tree_scratch(value,tree,ecid,sync);
  if(guarded.before!=0x719fd875ec4123abULL||guarded.after!=0x719fd875ec4123abULL)fail("value_canary");
  memcpy(out+5*ecid,value,40);joins[ecid]=group_joins;wjoins[ecid]=warp_joins;
 });
 for(auto&t:lanes)t.join();
 for(int i=0;i<192;i++)if(joins[i]!=5||wjoins[i]!=11)fail("join_counts");
 for(int row=0;row<4;row++){
  if(tree.cells[row][0].writer!=-1)fail("node_zero_written");
  for(int i=0;i<192;i++)if(tree.cells[row][256+i].v!=input[5*i+row])fail("original_leaf_overwritten");
  for(int i=448;i<512;i++)if(tree.cells[row][i].writer<0||tree.cells[row][i].v!=(row==0?1ULL:0ULL))fail("padding_leaf_missing");
 }
 counts[0]=multiply_count;counts[1]=inverse_count;counts[2]=5;counts[3]=11;
 counts[4]=ownership_writes;counts[5]=cross_warp_reads;
 delete group_barrier;for(int w=2;w<8;w++)delete warps[w];
}
'''


def build_source(mutation):
    header = HEADER.read_text()
    assert header.startswith(BASE.read_text()), 'Original APIs changed'
    barrier = function(header, '__device__ __forceinline__ void qsb_ec192_barrier(')
    assert 'asm' not in barrier and '__syncthreads' not in barrier
    assert barrier.count('__syncwarp();') == 2
    assert barrier.count('__threadfence_block();') == 3
    assert 'const uint32_t generation=atomicAdd(sync+1,0u);' in barrier
    assert 'const uint32_t ticket=atomicAdd(sync,1u);' in barrier
    assert 'if(ticket==5u)' in barrier
    assert barrier.index('atomicExch(sync,0u);') < barrier.index('atomicAdd(sync+1,1u);')
    assert 'while(atomicAdd(sync+1,0u)==generation){}' in barrier
    body = function(header, SIG)
    assert '__syncthreads' not in body and 'blockDim' not in body and 'threadIdx' not in body
    assert body.count('qsb_ec192_barrier(sync);') == 4  # One ascent and two descent dynamic joins.
    backend_source = (ROOT/'check_candidate.py').read_text()
    module = ast.parse(backend_source)
    backend = next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='BACKEND' for t in n.targets))
    field = backend[backend.index('struct Field{'):backend.index('static uint32_t __byte_perm')]
    # The algorithm is unchanged. Only scratch parameter type gets an instrumented
    # array proxy and the PTX rendezvous gets a cooperative host implementation.
    projected = body.replace('uint64_t tree[4][512]', 'AuditTree &tree', 1)
    mutations = {
        'missing-padding-leaves': ('if(ecid<64){', 'if(ecid<0){'),
        'wrong-padding-pairs': ('tree[k][224+ecid]=(k==0?1ULL:0ULL);', 'tree[k][224+ecid]=0;'),
        'missing-first-join': ('    qsb_ec192_barrier(sync);\n\n    #pragma unroll 1\n    for(int width=n>>2;',
                               '    /* omitted subgroup publication join */\n\n    #pragma unroll 1\n    for(int width=n>>2;'),
        'missing-ascent-join': ('if(width>32)qsb_ec192_barrier(sync);else __syncwarp();', 'if(width>32){}else __syncwarp();'),
        'missing-final-descent-join': ('if((width<<1)>32)qsb_ec192_barrier(sync);else __syncwarp();', 'if((width<<1)>32){if(width!=64)qsb_ec192_barrier(sync);}else __syncwarp();'),
        'wrong-final-sibling': ('tree[k][n+(tid^1)]', 'tree[k][n+tid]'),
    }
    if mutation:
        old,new=mutations[mutation]
        assert projected.count(old)==1
        projected=projected.replace(old,new,1)
    projected_barrier='__device__ __forceinline__ void qsb_ec192_barrier(uint32_t*){subgroup_join();}'
    code=HOST+field+projected_barrier+'\n'+projected+WRAPPER
    return code, {'header_sha256':sha(header.encode()),'actual_helper_sha256':sha(body.encode()),
                  'actual_barrier_sha256':sha(barrier.encode()),'field_backend_source_sha256':sha(backend_source.encode()),
                  'generated_cpp_sha256':sha(code.encode()),'original_apis_preserved':True,'mutation':mutation}


def worker(library, mutation, result_path):
    lib=C.CDLL(str(library));u64=C.c_uint64
    lib.audit_ec192.argtypes=[C.POINTER(u64),C.POINTER(u64),C.POINTER(C.c_int)]
    p=2**256-2**32-977;rng=random.Random(202609171192)
    tails=[0,1,2,31,32,33,63,64,65,95,96,127,128,159,160,161,175,190,191,192]
    if mutation:tails=[192]
    evidence=[]
    for case,active in enumerate(tails):
        values=[rng.randrange(1,p) for _ in range(active)]+[1]*(192-active)
        if active==192:
            values=([1,2,p-1,p-2,p-65537,2**255,p//2,17]*24)
        raw=[word for v in values for word in [(v>>(64*k))&((1<<64)-1) for k in range(4)]+[123]]
        inputs=(u64*960)(*raw);out=(u64*960)();counts=(C.c_int*6)()
        lib.audit_ec192(inputs,out,counts)
        assert tuple(counts[:5])==(669,1,5,11,1536),tuple(counts)
        assert counts[5]>0
        for lane,v in enumerate(values):
            actual=sum(out[5*lane+k]<<(64*k) for k in range(4))
            if actual!=pow(v,-1,p) or out[5*lane+4]!=0:
                print(f'QSB_EC192_FAILURE inverse_mismatch case={case} lane={lane}',file=sys.stderr,flush=True)
                raise SystemExit(86)
        evidence.append({'active':active,'values_checked':192,'multiply':counts[0],'inverse':counts[1],
                         'subgroup_joins_per_lane':counts[2],'warp_joins_per_lane':counts[3],
                         'initial_limb_writes_checked':counts[4],'cross_warp_reads_checked':counts[5]})
    result_path.write_text(json.dumps({'status':'PASS','cases':evidence},indent=2)+'\n')


def main():
    global HEADER
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        worker(Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4]));return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=HERE/'inverse-check-results.json')
    parser.add_argument('--source',type=Path,default=ROOT)
    args=parser.parse_args();HEADER=args.source.resolve()/'tests/gpu_epochs/tree_inverse.cuh';initial=sha(HEADER.read_bytes());results=[]
    with tempfile.TemporaryDirectory(prefix='qsb-ec192-inverse-') as tmp:
        tmp=Path(tmp)
        for mutation in ['', 'missing-padding-leaves','wrong-padding-pairs','missing-first-join',
                         'missing-ascent-join','missing-final-descent-join','wrong-final-sibling']:
            code,evidence=build_source(mutation);tag=mutation or 'positive'
            cpp=tmp/(tag+'.cpp');library=tmp/(tag+'.so');cpp.write_text(code)
            subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC','-pthread',
                            '-I/opt/homebrew/opt/openssl@3/include','-L/opt/homebrew/opt/openssl@3/lib',
                            str(cpp),'-lcrypto','-o',str(library)],check=True,capture_output=True,text=True)
            result_path=tmp/(tag+'.json')
            run=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(library),mutation,str(result_path)],
                               capture_output=True,text=True,timeout=180)
            diagnostic=[line for line in run.stderr.splitlines() if line.startswith('QSB_EC192_FAILURE ')]
            if mutation:
                assert run.returncode==86 and diagnostic,(tag,run.returncode,run.stderr)
            else:
                assert run.returncode==0,(run.returncode,run.stderr)
                evidence['positive_cases']=json.loads(result_path.read_text())['cases']
            evidence.update(compilation_completed=True,exit_code=run.returncode,diagnostics=diagnostic)
            results.append(evidence)
            print(tag,run.returncode,diagnostic,flush=True)
    assert sha(HEADER.read_bytes())==initial,'Owned header changed during checks'
    report={'status':'PASS_ACTUAL_SOURCE_COOPERATIVE_CPU_INVERSE','header_sha256':initial,
            'checker_sha256':sha(Path(__file__).read_bytes()),'source_preserved':True,
            'participants':{'physical':[64,255],'ecid':[0,191],'producer_threads_launched':0},
            'values_checked':20*192,'value_canaries_checked':2*20*192,'padding_leaves':list(range(448,512)),
            'padding_pairs':list(range(224,256)),'work':{'real_pairs':96,'upper':127,'descent':254,'expansion':192,'multiply':669,'inverse':1},
            'barrier_contract':{'implementation':'Six warp leaders, legacy atomicAdd/atomicExch and block fences, two warp joins per rendezvous',
                                'subgroup_joins':5,'tree_warp_joins':11,'additional_barrier_warp_joins':10,
                                'numerical_projection':'Abstract192-thread rendezvous; exact body hash and shape bound, atomic protocol tested separately'},
            'results':results,'gpu_executed':False,'cuda_compiled':False,
            'limits':__doc__+' Actual helper executes with an instrumented tree-reference proxy and cooperative host barriers. Read epochs check cross-warp publication, writes check initial ownership/padding and bounds. No raw zero is supported: callers must substitute identity for inactive/singular lanes. Kernel queue, phase joins, native field/PTX semantics and full prototype are outside this component check.'}
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'values_checked':report['values_checked'],'header_sha256':initial},indent=2))


if __name__=='__main__':main()
