#!/usr/bin/env python3
"""Actual EC192 tree plus actual cooperative HM43 root; CPU-only projection."""
import argparse,ast,ctypes as C,hashlib,importlib.util,json,random,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'research/specialist_warps/two_six_ring/check_inverse.py'
spec=importlib.util.spec_from_file_location('old_ec192_check',OLD);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
function=old.function;sha=old.sha
TAILS=[0,1,31,32,33,191,192]


def source_code(source,mutation):
 hpath=source/'tests/gpu_epochs/tree_inverse.cuh';header=hpath.read_text()
 base=(ROOT/'research/weak_field/ordinary_region/candidate/tests/gpu_epochs/tree_inverse.cuh').read_text()
 barrier=function(header,'__device__ __forceinline__ void qsb_ec192_barrier(')
 assert barrier==function(base,'__device__ __forceinline__ void qsb_ec192_barrier(')
 body=function(header,old.SIG);basebody=function(base,old.SIG)
 start=body.index('    // Only the first EC warp');end=body.index('    qsb_ec192_barrier(sync);',start)
 root=body[start:end];bs=basebody.index('    if(tid==0){');be=basebody.index('    qsb_ec192_barrier(sync);',bs)
 assert body[:start]+basebody[bs:be]+body[end:]==basebody
 assert 'if(ecid<32){' in root and root.count('if(ecid==0){')==2
 assert 'uint64_t root[5]={0,0,0,0,0};' in root
 assert root.count('hm43_warp_inverse(root,lane);')==1
 assert body.count('qsb_ec192_barrier(sync);')==4
 if mutation=='wrong-physical-argument':root=root.replace('root,lane','root,physical')
 elif mutation=='leader-only-caller':root=root.replace('if(ecid<32){','if(ecid==0){',1)
 elif mutation=='missing-guard-lane':root=root.replace('if(ecid<32){','if(ecid<31){',1)
 elif mutation=='active-only-root':root=root.replace('if(ecid<32){','if(ecid<32 && ecid<active_count){',1)
 projected=(body[:start]+root+body[end:]).replace('uint64_t tree[4][512]','AuditTree &tree',1)
 projected=projected.replace('hm43_warp_inverse(root,','checked_hm43(root,')
 if mutation=='missing-root-publication':
  marker='    qsb_ec192_barrier(sync);\n    #pragma unroll 1\n    for(int width=1;'
  assert projected.count(marker)==1;projected=projected.replace(marker,'    /* omitted root publication */\n    #pragma unroll 1\n    for(int width=1;',1)
 backend_source=(ROOT/'check_candidate.py').read_text();module=ast.parse(backend_source)
 backend=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BACKEND' for t in n.targets))
 field=backend[backend.index('struct Field{'):backend.index('static uint32_t __byte_perm')]
 inv=function(field,'static void _ModInv(')
 field=field.replace(inv,'static void _ModInv(uint64_t*){fail("forbidden_OpenSSL_root_stub");}')
 support='\n#include "'+str(HERE/'host_support.hpp')+'"\n#include "'+str(source/'tests/gpu_epochs/hm43_warp_inverse.cuh')+'"\n'
 instrumentation=r'''
static std::atomic<int> root_calls{0};static int active_count;
static void checked_hm43(uint64_t *r,int lane){
 if(physical<64||physical>=96||owner!=physical-64)fail("root_physical_membership");
 if(owner!=0)for(int k=0;k<5;k++)if(r[k])fail("uninitialized_nonleader_root");
 ++root_calls;if(owner==0)++inverse_count;
 hm43_warp_inverse(r,lane);
}
static void qsb_ec192_barrier(uint32_t*){subgroup_join();}
'''
 wrapper=old.WRAPPER.replace('int*counts){','int*counts,int active){',1)
 wrapper=wrapper.replace(' multiply_count=0;', ' active_count=active;root_calls=0;qsb_hm43_host::Warp root_warp;\n multiply_count=0;',1)
 wrapper=wrapper.replace('owner=ecid;physical=ecid+64;', 'owner=ecid;physical=ecid+64;qsb_hm43_host::current_warp=&root_warp;qsb_hm43_host::current_lane=ecid&31;',1)
 wrapper=wrapper.replace(' for(int i=0;i<192;i++)if(joins', ' if(root_calls!=32||root_warp.observed_mask!=0xffffffffu)fail("root_participation_count");\n for(int lane=0;lane<32;lane++)if(root_warp.lane_calls[lane]!=root_warp.generation)fail("root_collective_lane_count");\n for(int i=0;i<192;i++)if(joins',1)
 wrapper=wrapper.replace(' counts[0]=multiply_count;', ' counts[6]=root_calls;counts[7]=root_warp.exchanges;counts[8]=root_warp.ballots;counts[9]=root_warp.generation;\n counts[0]=multiply_count;',1)
 code=old.HOST+field+support+instrumentation+projected+wrapper
 return code,{'actual_tree_helper_sha256':sha(body.encode()),'actual_root_region_sha256':sha(body[start:end].encode()),'actual_barrier_sha256':sha(barrier.encode()),'unchanged_nonroot_helper_bytes':True,'generated_cpp_sha256':sha(code.encode()),'mutation':mutation}


def worker(library,mutation,result_path):
 lib=C.CDLL(str(library));u64=C.c_uint64
 lib.audit_ec192.argtypes=[C.POINTER(u64),C.POINTER(u64),C.POINTER(C.c_int),C.c_int]
 p=2**256-2**32-977;rng=random.Random(2026091743192);evidence=[]
 tails=TAILS if not mutation else ([1] if mutation=='active-only-root' else [192])
 for case,active in enumerate(tails):
  values=[rng.randrange(1,p) for _ in range(active)]+[1]*(192-active)
  if active==192:values=[1,2,p-1,p-2,p-65537,2**255,p//2,17]*24
  raw=[word for v in values for word in [(v>>(64*k))&((1<<64)-1) for k in range(4)]+[123]]
  inputs=(u64*960)(*raw);out=(u64*960)();counts=(C.c_int*10)()
  lib.audit_ec192(inputs,out,counts,active)
  assert tuple(counts[:5])==(669,1,5,11,1536),tuple(counts)
  assert counts[5]>0 and counts[6]==32 and counts[7]>0 and counts[8]>0
  for lane,v in enumerate(values):
   actual=sum(out[5*lane+k]<<(64*k) for k in range(4))
   if actual!=pow(v,-1,p) or out[5*lane+4]!=0:
    print(f'QSB_EC192_FAILURE inverse_mismatch case={case} lane={lane}',file=sys.stderr,flush=True);raise SystemExit(86)
  evidence.append({'active':active,'outputs':192,'canaries':384,'multiply':counts[0],'cooperative_root_inverse':counts[1],'subgroup_joins':counts[2],'tree_warp_joins':counts[3],'initial_limb_writes':counts[4],'cross_warp_reads':counts[5],'root_callers':counts[6],'hm43_exchanges':counts[7],'hm43_ballots':counts[8],'root_collective_generations':counts[9]})
 result_path.write_text(json.dumps({'status':'PASS','cases':evidence},indent=2)+'\n')


def main():
 if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4]));return
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=HERE/'candidate');parser.add_argument('--output',type=Path,default=HERE/'tree-check');args=parser.parse_args()
 source=args.source.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
 prepared=json.loads((HERE/'prepared-source.json').read_text());assert len(prepared['source_sha256'])==19
 for name,digest in prepared['source_sha256'].items():assert sha((source/name).read_bytes())==digest,name
 for name in ['host_support.hpp','host_warp.hpp','field_projection.hpp']:assert (HERE/name).exists(),name
 support_hashes={name:sha((HERE/name).read_bytes()) for name in ['host_support.hpp','host_warp.hpp','field_projection.hpp']}
 results=[]
 for mutation in ['', 'wrong-physical-argument','leader-only-caller','missing-guard-lane','active-only-root','missing-root-publication']:
  tag=mutation or 'positive';code,entry=source_code(source,mutation);cpp=out/(tag+'.cpp');so=out/(tag+'.so');cpp.write_text(code)
  build=subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC','-pthread','-fwrapv','-I/opt/homebrew/opt/openssl@3/include','-L/opt/homebrew/opt/openssl@3/lib',str(cpp),'-lcrypto','-o',str(so)],capture_output=True,text=True)
  (out/(tag+'-build.log')).write_text(build.stdout+build.stderr);assert build.returncode==0,(tag,build.stderr)
  try:run=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(so),mutation,str(out/(tag+'.json'))],capture_output=True,text=True,timeout=150)
  except subprocess.TimeoutExpired as e:raise AssertionError(('worker_timeout_not_pass',tag)) from e
  (out/(tag+'-run.log')).write_text(run.stdout+run.stderr)
  diagnostics=[x for x in run.stderr.splitlines() if x.startswith(('QSB_EC192_FAILURE','COLLECTIVE_'))]
  if mutation:assert run.returncode!=0 and diagnostics,(tag,run.returncode,run.stderr)
  else:
   assert run.returncode==0,(tag,run.returncode,run.stderr)
   entry['cases']=json.loads((out/(tag+'.json')).read_text())['cases']
  entry.update(compiled=True,returncode=run.returncode,diagnostics=diagnostics);results.append(entry);print(tag,run.returncode,diagnostics,flush=True)
 for name,digest in prepared['source_sha256'].items():assert sha((source/name).read_bytes())==digest,name
 for name,digest in support_hashes.items():assert sha((HERE/name).read_bytes())==digest,name
 report={'status':'PASS_ACTUAL_EC192_TREE_WITH_COOPERATIVE_HM43_HOST_ROOT','source_fingerprint':prepared['source_fingerprint'],'source_sha256':prepared['source_sha256'],'support_sha256':support_hashes,'checker_sha256':sha(Path(__file__).read_bytes()),'inherited_checker_sha256':sha(OLD.read_bytes()),'physical_participants':[64,255],'root_physical_participants':[64,95],'tails':TAILS,'values_checked':len(TAILS)*192,'canaries_checked':len(TAILS)*384,'results':results,'limits':'Actual tree and actual HM43 header execute on host threads. OpenSSL multiplies tree nodes; Python pow independently checks all results. _ModInv is a fail-only stub, never the root implementation. EC barriers abstract exact unchanged protocol as192-thread rendezvous; not CUDA memory/scheduling or GPU performance evidence. Macros/intrinsics projected by separately checked host support.','gpu_executed':False,'native_cuda_compiled':False,'candidate_changed':False}
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(report['status'],flush=True)
if __name__=='__main__':main()
