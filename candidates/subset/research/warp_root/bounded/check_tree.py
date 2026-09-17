#!/usr/bin/env python3
"""Actual EC192 tree plus actual cooperative HM43 root; CPU-only projection."""
import argparse,ast,ctypes as C,hashlib,importlib.util,json,random,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];SUPPORT=HERE.parent
OLD=ROOT/'research/specialist_warps/two_six_ring/check_inverse.py'
spec=importlib.util.spec_from_file_location('old_ec192_check',OLD);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
function=old.function;sha=old.sha
TAILS=[0,1,31,32,33,191,192]


LEGACY_SCALARS=r"""
#define __noinline__
#define UADDO(r,a,b) r=qsb_host_add(a,b,false,true)
#define UADDC(r,a,b) r=qsb_host_add(a,b,true,true)
#define UADD(r,a,b) r=qsb_host_add(a,b,true,false)
#define USUBO(r,a,b) r=qsb_host_sub(a,b,false,true)
#define USUBC(r,a,b) r=qsb_host_sub(a,b,true,true)
#define USUB(r,a,b) r=qsb_host_sub(a,b,true,false)
#define UMULLO(r,a,b) r=(uint64_t)((__uint128_t)(uint64_t)(a)*(uint64_t)(b))
#define UMULHI(r,a,b) r=(uint64_t)(((__uint128_t)(uint64_t)(a)*(uint64_t)(b))>>64)
#define MADDO(r,a,b,c) r=qsb_host_add((uint64_t)(((__uint128_t)(uint64_t)(a)*(uint64_t)(b))>>64),c,false,true)
#define MADDC(r,a,b,c) r=qsb_host_add((uint64_t)(((__uint128_t)(uint64_t)(a)*(uint64_t)(b))>>64),c,true,true)
#define MADD(r,a,b,c) r=qsb_host_add((uint64_t)(((__uint128_t)(uint64_t)(a)*(uint64_t)(b))>>64),c,true,false)
#define MADDS(r,a,b,c) r=qsb_host_add((uint64_t)(((__int128)(int64_t)(a)*(int64_t)(b))>>64),c,true,false)
"""

def legacy_projection(source):
 text=(source/'GPUMath.h').read_text();blocks=[]
 for name in ['NBBLOCK','_IsZero(a)','_IsOne(a)','Neg(r)','Load(r, a)','__sleft128(a,b,n)','SWAP(tmp,x,y)']:
  begin=text.index('#define '+name+' ');lines=text[begin:].splitlines();block=[]
  for line in lines:
   block.append(line)
   if not line.endswith('\\'):break
  blocks.append('\n'.join(block))
 functions=[]
 for sig in ['__device__ void _ShiftR62(uint64_t *r)','__device__ void _ShiftR62(uint64_t dest[5]',
             '__device__ void _IMult(', '__device__ uint64_t _IMultC(', '__device__ void _MulP(',
             '__device__ void _DivStep62(', '__device__ void _MatrixVecMulHalf(', '__device__ void _MatrixVecMul(',
             '__device__ uint64_t _AddCh(', '__device__ __noinline__ void _ModInv(']:
  functions.append(function(text,sig))
 original='\n'.join(functions)
 projected=original.replace('*uu <<= zeros;', '*uu = (int64_t)((uint64_t)*uu << zeros);').replace('*uv <<= zeros;', '*uv = (int64_t)((uint64_t)*uv << zeros);')
 projected=projected.replace('void _ModInv(', 'void actual_legacy_ModInv(')
 return LEGACY_SCALARS+'\n'+'\n'.join(blocks)+'\n'+projected,{'legacy_function_bundle_sha256':sha(original.encode()),'legacy_macro_bundle_sha256':sha('\n'.join(blocks).encode()),'legacy_projection_sha256':sha(projected.encode()),'projection_changes':'CUDA scalar extended arithmetic uses explicit uint128 carry/borrow; signed coefficient left shifts use unsigned bit-preserving shifts; functions otherwise exact; -fwrapv.'}


def source_code(source,mutation,poison):
 header=(source/'tests/gpu_epochs/tree_inverse.cuh').read_text()
 base=(ROOT/'research/weak_field/ordinary_region/candidate/tests/gpu_epochs/tree_inverse.cuh').read_text()
 barrier=function(header,'__device__ __forceinline__ void qsb_ec192_barrier(')
 assert barrier==function(base,'__device__ __forceinline__ void qsb_ec192_barrier(')
 body=function(header,old.SIG);basebody=function(base,old.SIG)
 start=body.index('    // Only the first EC warp');end=body.index('    qsb_ec192_barrier(sync);',start)
 root=body[start:end];bs=basebody.index('    if(tid==0){');be=basebody.index('    qsb_ec192_barrier(sync);',bs)
 assert body[:start]+basebody[bs:be]+body[end:]==basebody
 assert root.count('hm43_warp_inverse(root,lane);')==1 and 'root[4]=0;' in root
 assert 'if(!root_complete){' in root and body.count('qsb_ec192_barrier(sync);')==4
 reload='for(int k=0;k<4;k++)root[k]=tree[k][1];'
 assert root.count(reload)==2
 if mutation=='no-reload':
  first=root.index(reload);i=root.index(reload,first+len(reload));root=root[:i]+root[i:].replace(reload,'for(int k=0;k<4;k++){}',1)
 elif mutation=='missing-clear':root=root.replace('root[4]=0;','/* omitted explicit fifth-word clear */',1)
 elif mutation=='publish-partial-root':root=root.replace('if(!root_complete){','if(false){',1)
 projected=(body[:start]+root+body[end:]).replace('uint64_t tree[4][512]','AuditTree &tree',1).replace('hm43_warp_inverse(root,','checked_hm43(root,')
 backend_source=(ROOT/'check_candidate.py').read_text();module=ast.parse(backend_source)
 backend=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BACKEND' for t in n.targets))
 field=backend[backend.index('struct Field{'):backend.index('static uint32_t __byte_perm')]
 field=field.replace(function(field,'static void _ModInv('),'')
 support='\n#include "'+str(SUPPORT/'host_support.hpp')+'"\n#include "'+str(source/'tests/gpu_epochs/hm43_warp_inverse.cuh')+'"\n'
 legacy,legacy_evidence=legacy_projection(source)
 instrumentation=r"""
static std::atomic<int> root_calls{0},fallback_calls{0},false_calls{0};static int active_count;
static bool checked_hm43(uint64_t *r,int lane){
 if(physical<64||physical>=96||owner!=physical-64)fail("root_physical_membership");
 if(owner!=0)for(int k=0;k<5;k++)if(r[k])fail("uninitialized_nonleader_root");
 ++root_calls;if(owner==0)++inverse_count;
 const bool ok=hm43_warp_inverse(r,lane);
 if(!ok){++false_calls;POISON_STATEMENT}
 return ok;
}
static void _ModInv(uint64_t *r){
 if(owner!=0||physical!=64)fail("fallback_owner");
 if(r[4]!=0)fail("fallback_word4_not_zero");
 ++fallback_calls;actual_legacy_ModInv(r);
}
static void qsb_ec192_barrier(uint32_t*){subgroup_join();}
extern "C" void audit_legacy(uint64_t *r){actual_legacy_ModInv(r);}
""".replace('POISON_STATEMENT','for(int k=0;k<5;k++)r[k]=0x7193a1ULL+k;' if poison else '')
 wrapper=old.WRAPPER.replace('int*counts){','int*counts,int active){',1)
 wrapper=wrapper.replace(' multiply_count=0;', ' active_count=active;root_calls=0;fallback_calls=0;false_calls=0;qsb_hm43_host::Warp root_warp;\n multiply_count=0;',1)
 wrapper=wrapper.replace('owner=ecid;physical=ecid+64;', 'owner=ecid;physical=ecid+64;qsb_hm43_host::current_warp=&root_warp;qsb_hm43_host::current_lane=ecid&31;',1)
 wrapper=wrapper.replace(' for(int i=0;i<192;i++)if(joins', ' if(root_calls!=32||root_warp.observed_mask!=0xffffffffu)fail("root_participation_count");\n for(int lane=0;lane<32;lane++)if(root_warp.lane_calls[lane]!=root_warp.generation)fail("root_collective_lane_count");\n if(false_calls!=0 && false_calls!=32)fail("nonuniform_HM43_status");\n for(int i=0;i<192;i++)if(joins',1)
 wrapper=wrapper.replace(' counts[0]=multiply_count;', ' counts[6]=root_calls;counts[7]=root_warp.exchanges;counts[8]=root_warp.ballots;counts[9]=root_warp.generation;counts[10]=fallback_calls;counts[11]=false_calls;\n counts[0]=multiply_count;',1)
 code=old.HOST+field+support+legacy+instrumentation+projected+wrapper
 return code,{'actual_tree_helper_sha256':sha(body.encode()),'actual_root_region_sha256':sha(body[start:end].encode()),'actual_barrier_sha256':sha(barrier.encode()),'unchanged_nonroot_helper_bytes':True,'generated_cpp_sha256':sha(code.encode()),'mutation':mutation,'false_return_private_state_poison':poison,**legacy_evidence}


def worker(library,mutation,result_path,cap):
 lib=C.CDLL(str(library));u64=C.c_uint64
 lib.audit_ec192.argtypes=[C.POINTER(u64),C.POINTER(u64),C.POINTER(C.c_int),C.c_int]
 lib.audit_legacy.argtypes=[C.POINTER(u64)]
 p=2**256-2**32-977;rng=random.Random(2026091743192);evidence=[]
 legacy_roots=[0,1,2,p-1,p-2,p-65537,2**255,p//2]+[rng.randrange(1,p) for _ in range(512)]
 for v in legacy_roots:
  memory=(u64*7)(0x7182,*[(v>>(64*k))&((1<<64)-1) for k in range(4)],0,0x8127)
  lib.audit_legacy(C.cast(C.byref(memory,8),C.POINTER(u64)))
  want=pow(v,-1,p) if v else 0;actual=sum(memory[k+1]<<(64*k) for k in range(5))
  assert actual==want and memory[0]==0x7182 and memory[6]==0x8127,('legacy_projection_oracle',v,actual,want)
 tails=TAILS if not mutation else [33]
 for case,active in enumerate(tails):
  values=[rng.randrange(1,p) for _ in range(active)]+[1]*(192-active)
  if active==192:values=[1,2,p-1,p-2,p-65537,2**255,p//2,17]*24
  raw=[word for v in values for word in [(v>>(64*k))&((1<<64)-1) for k in range(4)]+[123]]
  inputs=(u64*960)(*raw);out=(u64*960)();counts=(C.c_int*12)()
  lib.audit_ec192(inputs,out,counts,active)
  assert tuple(counts[:5])==(669,1,5,11,1536),tuple(counts)
  assert counts[5]>0 and counts[6]==32
  if cap<=1 and mutation!='publish-partial-root':assert counts[10]==1 and counts[11]==32,('forced_fallback_not_executed',tuple(counts))
  if cap==16:assert counts[10]==0 and counts[11]==0,('unexpected_baseline_fallback',tuple(counts))
  for lane,v in enumerate(values):
   actual=sum(out[5*lane+k]<<(64*k) for k in range(4))
   if actual!=pow(v,-1,p) or out[5*lane+4]!=0:
    print(f'QSB_EC192_FAILURE inverse_mismatch case={case} lane={lane}',file=sys.stderr,flush=True);raise SystemExit(86)
  evidence.append({'active':active,'outputs':192,'canaries':384,'multiply':counts[0],'cooperative_root_attempt':counts[1],'subgroup_joins':counts[2],'tree_warp_joins':counts[3],'initial_limb_writes':counts[4],'cross_warp_reads':counts[5],'root_callers':counts[6],'hm43_exchanges':counts[7],'hm43_ballots':counts[8],'root_collective_generations':counts[9],'actual_scalar_fallback_calls':counts[10],'false_return_lanes':counts[11]})
 result_path.write_text(json.dumps({'status':'PASS','legacy_projection_independent_roots':len(legacy_roots),'cases':evidence},indent=2)+'\n')


def main():
 if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4]),int(sys.argv[5]));return
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=HERE/'candidate');parser.add_argument('--output',type=Path,default=HERE/'tree-check');args=parser.parse_args()
 source=args.source.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
 prepared=json.loads((HERE/'prepared-source.json').read_text());assert len(prepared['source_sha256'])==19
 for name,digest in prepared['source_sha256'].items():assert sha((source/name).read_bytes())==digest,name
 support_hashes={name:sha((SUPPORT/name).read_bytes()) for name in ['host_support.hpp','host_warp.hpp','field_projection.hpp']}
 results=[]
 configurations=[(16,'',False),(0,'',False),(1,'',False),(0,'',True),(1,'',True),(1,'no-reload',True),(1,'missing-clear',True),(1,'publish-partial-root',True)]
 for cap,mutation,poison in configurations:
  tag=f'cap{cap}-'+(mutation or 'positive')+('-poison' if poison else '')
  code,entry=source_code(source,mutation,poison);cpp=out/(tag+'.cpp');so=out/(tag+'.so');cpp.write_text(code)
  build=subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC','-pthread','-fwrapv',f'-DHM43_WARP_MAX_BATCHES={cap}','-I/opt/homebrew/opt/openssl@3/include','-L/opt/homebrew/opt/openssl@3/lib',str(cpp),'-lcrypto','-o',str(so)],capture_output=True,text=True)
  (out/(tag+'-build.log')).write_text(build.stdout+build.stderr);assert build.returncode==0,(tag,build.stderr)
  try:run=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(so),mutation,str(out/(tag+'.json')),str(cap)],capture_output=True,text=True,timeout=150)
  except subprocess.TimeoutExpired as e:raise AssertionError(('worker_timeout_not_pass',tag)) from e
  (out/(tag+'-run.log')).write_text(run.stdout+run.stderr)
  diagnostics=[x for x in run.stderr.splitlines() if x.startswith(('QSB_EC192_FAILURE','COLLECTIVE_'))]
  if mutation:assert run.returncode!=0 and diagnostics,(tag,run.returncode,run.stderr)
  else:
   assert run.returncode==0,(tag,run.returncode,run.stderr)
   evidence=json.loads((out/(tag+'.json')).read_text());entry['cases']=evidence['cases'];entry['legacy_projection_independent_roots']=evidence['legacy_projection_independent_roots']
  entry.update(cap=cap,compiled=True,returncode=run.returncode,diagnostics=diagnostics);results.append(entry);print(tag,run.returncode,diagnostics,flush=True)
 for name,digest in prepared['source_sha256'].items():assert sha((source/name).read_bytes())==digest,name
 for name,digest in support_hashes.items():assert sha((SUPPORT/name).read_bytes())==digest,name
 report={'status':'PASS_BOUNDED_HM43_TREE_AND_ACTUAL_LEGACY_FALLBACK_PROJECTION','source_fingerprint':prepared['source_fingerprint'],'source_sha256':prepared['source_sha256'],'support_sha256':support_hashes,'checker_sha256':sha(Path(__file__).read_bytes()),'inherited_checker_sha256':sha(OLD.read_bytes()),'physical_participants':[64,255],'root_physical_participants':[64,95],'tails':TAILS,'natural_values_checked':3*len(TAILS)*192,'fault_injected_positive_values_checked':2*len(TAILS)*192,'canaries_checked':5*len(TAILS)*384,'results':results,'limits':'Actual bounded HM43/tree and extracted legacy scalar inverse/functions execute on host. CUDA scalar ops projected with uint128 carry, validated against520 independent Python inverse roots before every run. OpenSSL only multiplies tree nodes. Additional poison false-return tests are defensive fault injection, not natural cap behavior. EC barriers abstract unchanged protocol as192-thread rendezvous. No CUDA/GPU/race/performance evidence.','gpu_executed':False,'native_cuda_compiled':False,'candidate_changed':False}
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(report['status'],flush=True)
if __name__=='__main__':main()
