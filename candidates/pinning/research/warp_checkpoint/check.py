#!/usr/bin/env python3
"""Actual checkpoint/pipeline CPU bodies with OpenSSL and separate warp barriers."""
import argparse,ast,ctypes as C,hashlib,json,os,random,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;PIN=HERE.parents[1]
sys.path.insert(0,str(PIN.parent/'subset'))
from check_candidate import BACKEND,function
from check_deferred_source import FIELD
import audit_integrated as ref
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--variant',choices=['warp_full','warp_compact','warp_compact_fused'],required=True)
ap.add_argument('--sanitizer',action='store_true')
args=ap.parse_args();BASE=HERE/args.variant
src=(BASE/'pinning.cu').read_text();helper=(BASE/'warp_checkpoint.cuh').read_text()
identities={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.iterdir() if p.is_file()}
backend=BACKEND.replace('multiply_count++;','multiply_count.fetch_add(1,std::memory_order_relaxed);')
backend=backend.replace('inverse_count++;','inverse_count.fetch_add(1,std::memory_order_relaxed);')
backend+='\nstatic thread_local int warp_syncs;\nstatic void __syncwarp(){warp_syncs++;warp_barriers[threadIdx.x/32]->wait();}\n'
old=ast.parse((HERE.parent/'compact_state/check_pipeline_full.py').read_text())
stubs=next(ast.literal_eval(n.value) for n in old.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='STUBS' for x in n.targets))
stubs=stubs.replace('#define QSB_CHECKPOINT_NODES 254\n','').replace('#define QSB_CHECKPOINT_STRIDE 256\n','')
functions=[function(src,sig) for sig in (
 '__device__ __forceinline__ void qsb_field_normalize(',
 '__device__ __forceinline__ void qsb_block_inverse(')]
roots=[function(src,sig) for sig in (
 '__global__ void __launch_bounds__(256,2) qsb_root_group_prepare(',
 '__global__ void __launch_bounds__(256,1) qsb_invert_super_roots(',
 '__global__ void __launch_bounds__(256,2) qsb_root_group_finish(')]
tree=r'''
extern "C" int direct_inverse(const uint64_t*in,uint64_t*out,int*counts){
 multiply_count=0;inverse_count=0;
 launch(256,[&](int i){uint64_t v[5]={};Load256(v,in+4*i);warp_syncs=0;
  qsb_block_inverse(v);memcpy(out+4*i,v,32);
  if(i==0){counts[0]=block_syncs;counts[1]=warp_syncs;}});
 counts[2]=multiply_count;return inverse_count;
}
extern "C" int tree(const uint64_t*in,uint64_t*out,int*counts){
 uint64_t root[5]={};std::vector<uint64_t> checkpoint(4*QSB_CHECKPOINT_STRIDE+2,0xcafe);
 multiply_count=0;inverse_count=0;
 launch(256,[&](int i){uint64_t v[5]={};Load256(v,in+4*i);warp_syncs=0;
  qsb_block_product_checkpoint(v,root,checkpoint.data());
  if(i==0){counts[2]=block_syncs;counts[3]=warp_syncs;}});
 counts[0]=multiply_count;_ModInv(root);multiply_count=0;
 launch(256,[&](int i){uint64_t v[5]={};Load256(v,in+4*i);warp_syncs=0;
  qsb_block_inverse_checkpoint(v,root,checkpoint.data());memcpy(out+4*i,v,32);
  if(i==0){counts[4]=block_syncs;counts[5]=warp_syncs;}});
 counts[1]=multiply_count;
 require(checkpoint[4*QSB_CHECKPOINT_STRIDE]==0xcafe&&checkpoint.back()==0xcafe);
 for(int k=0;k<4;k++)for(int n=QSB_CHECKPOINT_NODES;n<QSB_CHECKPOINT_STRIDE;n++)
  require(checkpoint[k*QSB_CHECKPOINT_STRIDE+n]==0xcafe);
 return inverse_count;
}
'''
pipeline=r'''
extern "C" int pipeline(const uint64_t *points,const uint64_t *rx,const uint64_t *ry,int count,uint32_t *hits,uint32_t *hit_count){
 input_points=points;memcpy(pin_u2rx_words,rx,32);memcpy(pin_u2ry_words,ry,32);
 int blocks=(count+255)/256,groups=(blocks+255)/256;
 std::vector<ulonglong2> saved(count*8+2);saved.back()={0x12345678,0x87654321};
 std::vector<uint64_t> roots(blocks*4),super(groups*4),root_tree(groups*4*QSB_CHECKPOINT_STRIDE+1,0xbeef);
 std::vector<uint64_t> checkpoint(blocks*4*QSB_CHECKPOINT_STRIDE+1,0xcafe);
 uint32_t mid[8]={};*hit_count=0;inverse_count=0;
 for(int b=0;b<blocks;b++)launch(256,[&](int){blockIdx.x=b;kernel_pinning_pipeline<true,0>(mid,nullptr,0,0,0,0,0x80000000,500000000,nullptr,rx,ry,nullptr,nullptr,nullptr,hit_count,hits,count,0,1,0,saved.data(),roots.data(),checkpoint.data());});
 for(int b=0;b<groups;b++)launch(256,[&](int){blockIdx.x=b;qsb_root_group_prepare(roots.data(),blocks,super.data(),root_tree.data());});
 launch(256,[&](int){qsb_invert_super_roots(super.data(),groups);});
 for(int b=0;b<groups;b++)launch(256,[&](int){blockIdx.x=b;qsb_root_group_finish(roots.data(),blocks,super.data(),root_tree.data());});
 for(int b=0;b<blocks;b++)launch(256,[&](int){blockIdx.x=b;kernel_pinning_pipeline<true,2>(mid,nullptr,0,0,0,0,0x80000000,500000000,nullptr,rx,ry,nullptr,nullptr,nullptr,hit_count,hits,count,0,1,0,saved.data(),roots.data(),checkpoint.data());});
 require(saved.back().x==0x12345678&&saved.back().y==0x87654321);
 require(checkpoint.back()==0xcafe&&root_tree.back()==0xbeef);
 return inverse_count;
}
'''
common='\n'.join([backend,FIELD,stubs,*functions,helper])
flags=['c++','-std=c++17','-pthread','-Wno-pragma-once-outside-header',
 '-I/opt/homebrew/opt/openssl@3/include','-L/opt/homebrew/opt/openssl@3/lib']
report={'source_sha256':identities,'gpu_executed':False,'variant':args.variant}
with tempfile.TemporaryDirectory(prefix='qsb-warp-checkpoint-') as td:
 p=Path(td);cpp=p/'audit.cpp'
 if args.sanitizer:
  # Avoid counter synchronization concealing missing cross-warp happens-before.
  # OpenSSL operands are thread-local; actual shared arrays remain instrumented.
  main=r'''
int main(){for(int r=0;r<3;r++){
 uint64_t input[1024]={},output[1024]={};int counts[6];
 for(int i=0;i<256;i++)input[4*i]=1+(i*17+r*193)%997;
 for(int mode=0;mode<2;mode++){
 if(mode)direct_inverse(input,output,counts);else tree(input,output,counts);
 for(int i=0;i<256;i++){uint64_t v[5]={},w[5]={},z[5]={};Load256(v,input+4*i);Load256(w,output+4*i);qsb_field_mul(z,v,w);require(z[0]==1&&!(z[1]|z[2]|z[3]));}}
}return 0;}
'''
  variants={'correct':common,
   'missing_up_crosswarp':common.replace('if(count>64)__syncthreads();','if(count>128)__syncthreads();'),
   'missing_down_crosswarp':common.replace('if(count>=32)__syncthreads();','if(count>=64)__syncthreads();')}
  outcomes={}
  for name,body in variants.items():
   if name!='correct':assert body!=common
   cpp.write_text(body+tree+main);exe=p/name
   subprocess.run(flags+['-O1','-g','-fsanitize=thread',str(cpp),'-lcrypto','-o',str(exe)],check=True)
   env=dict(os.environ,TSAN_OPTIONS='halt_on_error=1:abort_on_error=0:exitcode=66:symbolize=0')
   result=subprocess.run([str(exe)],env=env,capture_output=True,text=True,timeout=120)
   (HERE/f'{args.variant}-{name}-tsan.log').write_text(result.stdout+result.stderr)
   if name=='correct':assert result.returncode==0,result.stderr[-3000:]
   else:assert result.returncode==66 and 'ThreadSanitizer: data race' in result.stderr,result.stderr[-3000:]
   outcomes[name]={'exit_code':result.returncode,'data_race_detected':'ThreadSanitizer: data race' in result.stderr}
  report['sanitizer']=outcomes;report['scope']='Actual tree helper bodies on 256 CPU threads, OpenSSL fields, distinct warp/block barriers and two deliberately invalid cross-warp boundaries. C++ race screen only.'
 else:
  sha=(BASE/'GPUHash.h').read_text().split('//Modified SHA256 function')[0]
  recovery=[function(src,sig) for sig in (
   '__device__ __forceinline__ void qsb_xyzz_finish_prepare(',
   '__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed(',
   '__device__ int gpu_is_der_easy(',
   '__device__ __forceinline__ int gpu_bench_valid_words(')]
  kernel='template<bool FAST_TAIL,int STAGE>\n'+function(src,'__global__ void __launch_bounds__(256, STAGE == 0 ? 2 : 3) kernel_pinning_pipeline(')
  cpp.write_text('\n'.join([common,sha,*roots,*recovery,kernel,tree,pipeline]));so=p/'audit.so'
  subprocess.run(flags+['-O2','-shared','-fPIC',str(cpp),'-lcrypto','-o',str(so)],check=True)
  lib=C.CDLL(str(so));U64=C.c_uint64;U32=C.c_uint32;ptr=C.POINTER(U64)
  lib.tree.argtypes=[ptr,ptr,C.POINTER(C.c_int)]
  lib.direct_inverse.argtypes=[ptr,ptr,C.POINTER(C.c_int)]
  lib.pipeline.argtypes=[ptr,ptr,ptr,C.c_int,C.POINTER(U32),C.POINTER(U32)]
  def buf(values):return (U64*(4*len(values)))(*[x>>(64*i)&((1<<64)-1) for x in values for i in range(4)])
  def integer(v):return sum(int(x)<<(64*i) for i,x in enumerate(v))
  rng=random.Random(9862);P=ref.P
  expected_counts=[255,510,3,5,4,4] if args.variant=='warp_full' else [255,702,3,5,5,4]
  direct_counts=[7,9,765] if args.variant=='warp_compact_fused' else [16,0,765]
  for i in range(8):
   values=[rng.randrange(1,P) for _ in range(256)];values[0]=1;values[-1]=P-1
   if i==0:values=[1]*256
   output=(U64*1024)();counts=(C.c_int*6)()
   assert lib.tree(buf(values),output,counts)==1
   assert list(counts)==expected_counts,list(counts)
   assert all(integer(output[4*j:4*j+4])==pow(v,-1,P) for j,v in enumerate(values))
   assert lib.direct_inverse(buf(values),output,counts)==1
   assert list(counts)[:3]==direct_counts,list(counts)
   assert all(integer(output[4*j:4*j+4])==pow(v,-1,P) for j,v in enumerate(values))
  R=ref.scalar_mult(1984321);pool=[ref.scalar_mult(rng.randrange(1,ref.N)) for _ in range(32)]
  sizes=[1,31,32,255,256,257,511,513]
  for n in sizes:
   points=[];expected=[]
   for idx in range((n+255)//256*256):
    q=R if idx%31==0 else pool[idx%32];z=rng.randrange(1,P);a=z*z%P;b=a*z%P
    points.extend([q[0]*a%P,q[1]*b%P,a,b])
    if idx>=n or q==R:continue
    for ri,rr in enumerate((R,(R[0],-R[1]%P))):
     result=ref.affine_add(q,rr);digest=hashlib.sha256(bytes([2+(result[1]&1)])+result[0].to_bytes(32,'big')).digest()
     if digest[0]>>2==0:expected.append((500000000+idx)|(ri<<31));break
   hits=(U32*1024)();nh=U32()
   assert lib.pipeline(buf(points),buf([R[0]]),buf([R[1]]),n,hits,C.byref(nh))==1
   assert sorted(hits[:nh.value])==sorted(expected),(n,list(hits[:nh.value]),expected)
  report['counts']={'leaf_inverses':2048,'prepare_multiplications':expected_counts[0],'finish_multiplications':expected_counts[1],
   'prepare_block_barriers':expected_counts[2],'prepare_warp_barriers':expected_counts[3],
   'finish_block_barriers':expected_counts[4],'finish_warp_barriers':expected_counts[5],
   'pipeline_sizes':sizes,'pipeline_active_candidates':sum(sizes),'canaries':'PASS',
   'direct_inverse_cases':2048,'direct_inverse_block_warp_multiply_counts':direct_counts}
  report['scope']='Actual tree, root hierarchy and pinning pipeline bodies; OpenSSL fields and controlled XYZZ producer; actual SHA header CPU path and independent hit oracle. No PTX/GPU execution or timing.'
assert all(hashlib.sha256((BASE/n).read_bytes()).hexdigest()==h for n,h in identities.items())
report['status']='PASS';report_path=HERE/(args.variant+('-tsan.json' if args.sanitizer else '-cpu.json'))
report_path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
