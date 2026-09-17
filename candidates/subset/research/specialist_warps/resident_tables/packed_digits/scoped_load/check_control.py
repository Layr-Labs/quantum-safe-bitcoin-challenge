#!/usr/bin/env python3
"""Execute the actual specialist kernel control flow with CPU protocol mocks.

256 host threads exercise real loops, role splits, tickets, payload copies,
192-way joins, padding, and hit identity. Hash/field/recovery/gate math are
explicit deterministic mocks; actual recode, pack and unpack functions execute.
Independent Python big-integer peel supplies packet identity expectations; component tests verify actual math separately.
Actual legacy atomic/fence bodies execute with host atomics and warp joins.
A configurable host-only poll delay avoids oversubscription; no CUDA execution.
"""
import argparse,time
from pathlib import Path
import subprocess,tempfile,hashlib,json,sys,re
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'research/wide_windows')]
from preflight import source_identity
from audit_support import function
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cases',default='1,2,3,4,5,6,7,8,11,12,13,32')
parser.add_argument('--timeout',type=int,default=180)
parser.add_argument('--positive-only',action='store_true')
parser.add_argument('--poll-us',type=int,default=500)
parser.add_argument('--source',type=Path,default=ROOT)
parser.add_argument('--output',type=Path,default=HERE/'control-check-results.json')
parser.add_argument('--debug-output',type=Path,default=Path(tempfile.mkdtemp(prefix='qsb-resident-control-')))
args=parser.parse_args()
case_values=[int(v) for v in args.cases.split(',')]
assert case_values and all(1<=v<=64 for v in case_values)
assert 0<=args.poll_us<=10000
DEBUG=args.debug_output.resolve();DEBUG.mkdir(parents=True,exist_ok=True)
source=args.source.resolve();identity=source_identity(source)
tree=(source/'tests/gpu_epochs/tree.cu').read_text()
device=(source/'tests/gpu_epochs/compact_table_device.cuh').read_text()
setup=function(tree,'__device__ __forceinline__ void gt_recode_setup(')
order_constant=re.search(r'__device__ __constant__ uint64_t GT_ORDER_N\[4\] = \{.*?\};',tree,re.S).group()
pack_scalar=function(device,'__device__ __forceinline__ void qsb_pack_scalar16(')
pack_digest=function(device,'__device__ __forceinline__ void qsb_pack_digest16(')
unpack=function(device,'__device__ __forceinline__ void qsb_ec192_packed_digit(')
actual_pack_sources=order_constant+'\n'+setup+'\n'+pack_scalar+'\n'+pack_digest+'\n'+unpack
# Independent integer recurrence, not the producer's bit-sliced formula.
order=0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141
packet_rows=[];seen_packets=set()
for idx in range(max(case_values)*256):
    words=[idx if k==7 else ((idx+1)*0x91741+(k+1)*0x713719)&0xffffffff for k in range(8)]
    scalar=sum(words[k]<<(32*(7-k)) for k in range(8))
    twice=(2*(scalar%order))%order
    sign=1 if twice&1 else -1
    remaining=twice if sign==1 else order-twice
    digits=[]
    for c in range(15):
        digit=(remaining%131072)-65536
        assert digit&1 and abs(digit)<65536
        remaining=(remaining-digit)//65536
        digits.append(sign*digit)
    digits.append(sign*remaining)
    assert sum(digit<<(16*c) for c,digit in enumerate(digits))%order==twice
    cells=[((abs(d)-1)//2)|((d<0)<<15) for d in digits]
    packet=tuple(sum(cells[4*k+j]<<(16*j) for j in range(4)) for k in range(4))
    assert packet not in seen_packets;seen_packets.add(packet)
    packet_rows.append('{'+','.join(hex(v)+'ULL' for v in packet)+'}')
expected_packet_declaration='static const uint64_t EXPECTED_PACKETS[][4]={'+',\n'.join(packet_rows)+'};'

kernel=function(tree,'__global__ void __launch_bounds__(256,3) kernel_digest_specialist(')
load=function(tree,'__device__ __forceinline__ uint32_t qsb_ticket_acquire(')
store=function(tree,'__device__ __forceinline__ void qsb_ticket_release(')
wait=function(tree,'__device__ __forceinline__ void qsb_ticket_wait(')
assert 'if(lane==0){while(qsb_ticket_acquire(ticket)!=expected){}}' in wait
assert wait.index('__syncwarp();')<wait.rindex('__threadfence_block();')
assert 'atomicAdd(const_cast<uint32_t*>(ticket),0u)' in load and '__threadfence_block();' in load
assert store.index('__threadfence_block();')<store.index('atomicExch(ticket,value);')
barrier=function((source/'tests/gpu_epochs/tree_inverse.cuh').read_text(),'__device__ __forceinline__ void qsb_ec192_barrier(')
producer_barrier=function((source/'tests/gpu_epochs/window_schedule_shared.cuh').read_text(),'__device__ __forceinline__ void qsb_producer64_barrier(')
assert kernel.count('qsb_producer64_barrier(arena.producer_sync);')==2
assert kernel.count('qsb_pack_digest16(digest);')==1
assert kernel.index('qsb_producer_window_hash(digest,')<kernel.index('qsb_pack_digest16(digest);')<kernel.index('arena.digest[slot][k][lane]=digest[k]')
assert kernel.count('compact_ec192_fixed_xyzz_packed_shared(')==1
assert 'if(epoch>0)qsb_producer64_barrier' in kernel
assert 'uint32_t free_ticket[6];' in kernel and 'uint32_t producer_sync[2];' in kernel
assert kernel.count('__syncthreads();')==1 and kernel.count('qsb_ec192_barrier(arena.ec_sync);')==2
assert kernel.count('return;')==2
assert '__syncwarp();\n                if(lane==0)qsb_ticket_release(&arena.ready_ticket[slot]' in kernel
assert '__syncwarp();\n            if(lane==0)qsb_ticket_release(&arena.free_ticket' in kernel
assert 'uint32_t digest[6][8][32]' in kernel and 'sizeof(QsbSpecialistArena)==32832' in kernel
launch='kernel_digest_specialist<<<(nblk+5)/6,256>>>'
assert tree.count(launch)==1
segment=tree[tree.index(launch):tree.index('            epoch_base += nblk;',tree.index(launch))]
assert segment.index('cudaDeviceSynchronize();')<segment.index('cudaGetLastError()')<segment.index('total_searched += batch_pos;')
assert 'd_gt,d_hit_cnt,d_hit_idx,d_hit_combos,nblk,d_epochs' in segment
cpp=r'''
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <array>
#include <map>
#define QSB_RANKED_ONLY 1
#define MAX_T 9
#define __constant__
#define __device__
#define __forceinline__ inline
#define __global__
#define __launch_bounds__(a,b)
#define __shared__ static
struct Dim {int x;};
thread_local Dim threadIdx,blockIdx;
thread_local unsigned actual_idx=0,gate_ri=0;
struct Barrier {
 std::mutex mu;std::condition_variable cv;unsigned count=0,generation=0,target;
 Barrier(unsigned n):target(n){}
 void wait(){std::unique_lock<std::mutex> lock(mu);unsigned old=generation;
  if(++count==target){count=0;++generation;cv.notify_all();}
  else if(!cv.wait_for(lock,std::chrono::seconds(8),[&]{return old!=generation;})){
   fprintf(stderr,"BARRIER_TIMEOUT target=%u arrived=%u thread=%d\n",target,count,threadIdx.x);std::_Exit(90);
  }
 }
};
static Barrier cta(256),ec(192),warps[8]={{32},{32},{32},{32},{32},{32},{32},{32}};
static void require(bool yes){if(!yes){fprintf(stderr,"CONTROL_MISMATCH thread=%d candidate=%u\n",threadIdx.x,actual_idx);std::_Exit(86);}}
static void __syncthreads(){cta.wait();}
static std::atomic<unsigned> joins_enter[256],joins_exit[256],poll_values[256],poll_count[256];
static std::atomic<uintptr_t> poll_addresses[256];
static void __syncwarp(){joins_enter[threadIdx.x]++;warps[threadIdx.x/32].wait();joins_exit[threadIdx.x]++;}
static void __threadfence_block(){std::atomic_thread_fence(std::memory_order_seq_cst);}
static uint32_t atomicAdd(uint32_t*p,uint32_t v){
 uint32_t r=__atomic_fetch_add(p,v,__ATOMIC_RELAXED);
 if(v==0){poll_addresses[threadIdx.x]=(uintptr_t)p;poll_values[threadIdx.x]=r;poll_count[threadIdx.x]++;}
 if(v==0)std::this_thread::sleep_for(std::chrono::microseconds(5));
 return r;
}
static uint32_t atomicExch(uint32_t*p,uint32_t v){return __atomic_exchange_n(p,v,__ATOMIC_RELAXED);}
__SOURCE_PROTOCOL__
struct epoch_desc_t {uint32_t mid[8];uint32_t remW[2];uint8_t early[6];};
static uint64_t QSB_U2R[8]={1,2,3,4,5,6,7,8};
static uint8_t WIN3[256][3];
static std::atomic<unsigned> counts[64*256],hashes[64*256];
static unsigned expected_n=0;
static uint32_t token(unsigned idx,unsigned word){return word==7?idx:(idx+1)*0x91741u+(word+1)*0x713719u;}
static void qsb_producer64_prepare_first(const epoch_desc_t*epoch,uint32_t f[8][64],int producer_tid){
 for(int i=producer_tid;i<64;i+=64)for(int k=0;k<8;k++)f[k][i]=epoch->mid[0]+k;
}
static void qsb_producer_window_hash(uint32_t*out,const uint32_t f[8][64],int choice){
 unsigned slot=(choice*37+13)&63,epoch=f[0][slot];
 for(int k=0;k<8;k++)require(f[k][slot]==epoch+k);
 unsigned idx=epoch*256+choice;require(idx<expected_n*256);hashes[idx].fetch_add(1);
 for(int k=0;k<8;k++)out[k]=token(idx,k);
 if((idx%29)==0)std::this_thread::yield();
}
__ACTUAL_PACK_SOURCES__
__EXPECTED_PACKET_DECLARATION__
static std::map<std::array<uint64_t,4>,unsigned> packet_identity;
static void compact_ec192_fixed_xyzz_packed_shared(uint64_t*x,uint64_t*y,uint64_t*zz,uint64_t*zzz,const uint64_t z[4],const uint8_t*,const uint8_t*,volatile uint64_t a[4][768],unsigned owner){
 const auto &identities=packet_identity;
 auto found=identities.find({z[0],z[1],z[2],z[3]});require(found!=identities.end());
 actual_idx=found->second;require(actual_idx<expected_n*256);require(owner==(unsigned)threadIdx.x-64);
 for(int k=0;k<4;k++)a[k][owner]=z[k];
 for(unsigned c=0;c<16;c++){
  uint32_t index;uint64_t negative;qsb_ec192_packed_digit(a,owner,c,&index,&negative);
  unsigned cell=(EXPECTED_PACKETS[actual_idx][c/4]>>(16*(c%4)))&65535;
  require(index==(cell&32767)&&negative==(cell>>15));
 }
 require(counts[actual_idx].fetch_add(1)==0);
 for(int k=0;k<4;k++)for(int s=0;s<4;s++)a[k][owner+192*s]=token(actual_idx,k)+s;
 for(int k=0;k<4;k++){x[k]=a[k][owner];y[k]=a[k][192+owner];zz[k]=a[k][384+owner];zzz[k]=a[k][576+owner];}
 if((actual_idx%31)==0)std::this_thread::yield();
}
static void qsb_xyzz_finish_prepare(uint64_t*,uint64_t*,uint64_t*,uint64_t*,uint64_t*p){p[0]=(actual_idx%17)?actual_idx+1:0;for(int k=1;k<5;k++)p[k]=0;}
static void Load256(uint64_t*d,const uint64_t*s){memcpy(d,s,32);}
static void qsb_ec192_inverse_tree_scratch(uint64_t *v,uint64_t tree[4][512],int id,uint32_t*sync){
 require(id==threadIdx.x-64);
 // Deliberate cross-lane overwritten arena, five mock collective phases.
 for(int k=0;k<4;k++)tree[k][256+id]=v[k];
 qsb_ec192_barrier(sync);qsb_ec192_barrier(sync);qsb_ec192_barrier(sync);qsb_ec192_barrier(sync);qsb_ec192_barrier(sync);
 if(id%11==0)std::this_thread::yield();
 for(int k=0;k<4;k++)require(tree[k][256+id]==v[k]);
}
static uint32_t qsb_xyzz_finish_precomputed(uint64_t*,uint64_t*,uint64_t*,uint64_t*,uint64_t*,uint64_t*,uint64_t*,uint64_t*x,uint64_t*y){
 gate_ri=0;for(int k=0;k<4;k++){x[k]=actual_idx+k;y[k]=actual_idx+k+1;}return 2;
}
static uint32_t __byte_perm(uint32_t a,uint32_t b,uint32_t s){return a^b^s;}
static void _SHA256Initialize(uint32_t*h){for(int k=0;k<8;k++)h[k]=0;}
static void _SHA256Transform(uint32_t*h,uint32_t*){h[0]=actual_idx;h[1]=gate_ri++;}
static int gpu_bench_valid_words(uint32_t*h){return (h[0]*7+h[1])%19==0;}
static int gpu_is_der_relaxed(uint8_t*,int){require(false);return 0;}
static int gpu_is_der_easy(uint8_t*,int){require(false);return 0;}
'''.replace('__SOURCE_PROTOCOL__',load+'\n'+store+'\n'+wait+'\n'+barrier+'\n'+producer_barrier)+kernel+r'''
int main(){
 for(unsigned i=0;i<sizeof(EXPECTED_PACKETS)/sizeof(EXPECTED_PACKETS[0]);++i){
  const uint64_t*p=EXPECTED_PACKETS[i];require(packet_identity.emplace(std::array<uint64_t,4>{p[0],p[1],p[2],p[3]},i).second);
 }
 std::thread watchdog([]{
  for(int tick=1;tick<=300;tick++){
   std::this_thread::sleep_for(std::chrono::seconds(2));
   fprintf(stderr,"DEBUG_TICK seconds=%d\n",tick*2);
   for(int w=0;w<8;w++){
    int t=w*32;unsigned emin=~0u,emax=0,xmin=~0u,xmax=0,pmin=~0u,pmax=0;
    for(int l=0;l<32;l++){emin=std::min(emin,joins_enter[t+l].load());emax=std::max(emax,joins_enter[t+l].load());xmin=std::min(xmin,joins_exit[t+l].load());xmax=std::max(xmax,joins_exit[t+l].load());pmin=std::min(pmin,poll_values[t+l].load());pmax=std::max(pmax,poll_values[t+l].load());}
    fprintf(stderr,"DEBUG_WARP w=%d join_enter=%u..%u exit=%u..%u leader_poll=%u address=%llx polls=%u values=%u..%u\n",w,emin,emax,xmin,xmax,poll_values[t].load(),(unsigned long long)poll_addresses[t].load(),poll_count[t].load(),pmin,pmax);
   }
   fflush(stderr);
  }
 });watchdog.detach();
 for(int choice=0;choice<256;choice++)for(int k=0;k<3;k++)WIN3[choice][k]=137+choice%10+k;
 unsigned total=0,hit_total=0,ctas=0;
 for(unsigned n:{1u,2u,6u,7u,8u,14u,15u,32u}){
  fprintf(stderr,"CASE_BEGIN epochs=%u\n",n);fflush(stderr);
  expected_n=n;std::vector<epoch_desc_t> epochs(n);
  for(unsigned e=0;e<n;e++){epochs[e].mid[0]=e;for(int k=0;k<6;k++)epochs[e].early[k]=e+k;}
  for(unsigned i=0;i<n*256;i++){counts[i]=0;hashes[i]=0;}
  uint32_t hit_cnt=0;uint32_t hit_idx[1024]={};uint8_t hit_combos[1024*MAX_T]={};
  for(unsigned b=0;b<(n+5)/6;b++){
   fprintf(stderr,"CTA_BEGIN block=%u epochs=%u\n",b,n);fflush(stderr);
   std::vector<std::thread> threads;
   for(int t=0;t<256;t++)threads.emplace_back([&,t,b]{threadIdx.x=t;blockIdx.x=b;kernel_digest_specialist(nullptr,&hit_cnt,hit_idx,hit_combos,n,epochs.data());});
   for(auto &t:threads)t.join();++ctas;
  }
  unsigned expected_hits=0;
  for(unsigned i=0;i<n*256;i++){require(counts[i]==1&&hashes[i]==1);if(i%17 && ((i*7)%19==0||(i*7+1)%19==0))++expected_hits;}
  require(hit_cnt==expected_hits && hit_cnt<1024);
  std::vector<unsigned> seen(n*256,0);
  for(unsigned h=0;h<hit_cnt;h++){
   unsigned idx=hit_idx[h]&0x3fffffffu,ri=(hit_idx[h]>>30)&1,hc=hit_idx[h]>>31;
   require(idx<n*256 && idx%17 && hc==0 && (idx*7+ri)%19==0);require(!seen[idx]++);
   unsigned epoch=idx/256,choice=idx%256;
   for(int k=0;k<6;k++)require(hit_combos[h*MAX_T+k]==epoch+k);
   for(int k=0;k<3;k++)require(hit_combos[h*MAX_T+6+k]==137+choice%10+k);
  }
  total+=n*256;hit_total+=hit_cnt;
  fprintf(stderr,"CASE_PASS epochs=%u candidates=%u hits=%u\n",n,n*256,hit_cnt);
 }
 printf("CONTROL_PASS candidates=%u hits=%u ctas=%u\n",total,hit_total,ctas);
}
'''
cpp=cpp.replace('__ACTUAL_PACK_SOURCES__',actual_pack_sources).replace('__EXPECTED_PACKET_DECLARATION__',expected_packet_declaration)
cpp=cpp.replace('1u,2u,6u,7u,8u,14u,15u,32u',','.join(str(v)+'u' for v in case_values),1)
cpp=cpp.replace('std::chrono::microseconds(5)','std::chrono::microseconds('+str(args.poll_us)+')',1)
(DEBUG/'positive-generated.cpp').write_text(cpp)
with tempfile.TemporaryDirectory(prefix='qsb-specialist-control-') as tmp:
 p=Path(tmp);src=p/'check.cpp';binary=p/'check';src.write_text(cpp)
 def compile_source(text):
  src.write_text(text);subprocess.run(['c++','-std=c++17','-O1','-pthread',str(src),'-o',str(binary)],check=True,capture_output=True,text=True)
 run_summaries={}
 def execute(label,timeout):
  stdout=DEBUG/(label+'.stdout.log');stderr=DEBUG/(label+'.stderr.log')
  started=time.monotonic()
  with stdout.open('w') as out,stderr.open('w') as err:
   proc=subprocess.Popen([str(binary)],stdout=out,stderr=err)
   try:code=proc.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    proc.kill();proc.wait();code=124
  result=subprocess.CompletedProcess([str(binary)],code,stdout.read_text(),stderr.read_text())
  summary={'label':label,'exit_code':code,'elapsed_seconds':time.monotonic()-started,'stdout_log':stdout.name,'stderr_log':stderr.name,'source_fingerprint':identity['source_fingerprint'],'generated_cpp_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'poll_delay_us':args.poll_us}
  run_summaries[label]=summary
  (DEBUG/(label+'-run.json')).write_text(json.dumps(summary,indent=2)+'\n')
  print(json.dumps(summary),flush=True)
  return result
 compile_source(cpp);run=execute('positive',args.timeout)
 if run.returncode:
  print(run.stdout,run.stderr[-3000:]);raise RuntimeError('positive control failed; full logs retained at '+str(DEBUG))
 assert 'CONTROL_PASS' in run.stdout
 mutants={
  'producer_omits_pack':cpp.replace('                qsb_pack_digest16(digest);','                /* mutant: no packed handoff */',1),
  'producer_wrong_word_order':cpp.replace('digest[6-2*k]=(uint32_t)(words[k]>>32);','digest[6-2*k]=(uint32_t)(words[3-k]>>32);',1),
  'producer_wrong_sign_bit':cpp.replace('idx|(neg<<15)','idx|((neg^1u)<<15)',1),
  'consumer_wrong_limb_order':cpp.replace('z[0]=((uint64_t)s2[6]<<32)|s2[7];','z[0]=((uint64_t)s2[7]<<32)|s2[6];',1),
  'consumer_wrong_cell_sign':cpp.replace('*neg=cell>>15;','*neg=(cell>>15)^1u;',1),
  'consumer_uses_physical_thread_choice':cpp.replace('const int choice=(packet&7)*32+lane;\n        const int idx=','const int choice=tid;\n        const int idx=',1),
  'inactive_warp_returns_before_inverse':cpp.replace('        // All field reads precede reuse of the overlapping inverse view.','        if(!active)return;\n        // All field reads precede reuse of the overlapping inverse view.',1),
 }
 results={}
 for name,text in ({} if args.positive_only else mutants).items():
  assert text!=cpp
  compile_source(text);r=execute(name,35)
  expected_code,expected_text=(86,'CONTROL_MISMATCH') if name!='inactive_warp_returns_before_inverse' else (90,'BARRIER_TIMEOUT target=32 arrived=31')
  assert r.returncode==expected_code and expected_text in r.stderr,(name,r.returncode,r.stderr)
  results[name]={'exit_code':r.returncode,'compilation_completed':True,'diagnostic':'\n'.join(line for line in r.stderr.splitlines() if line.startswith(('CONTROL_MISMATCH','BARRIER_TIMEOUT'))),'generated_cpp_sha256':run_summaries[name]['generated_cpp_sha256']}
assert source_identity(source)==identity
report={'status':'PASS',**identity,'actual_pack_source_sha256':hashlib.sha256(actual_pack_sources.encode()).hexdigest(),'independent_packet_table_sha256':hashlib.sha256(expected_packet_declaration.encode()).hexdigest(),'independent_packet_identities':len(packet_rows),'actual_pack_functions':['gt_recode_setup','qsb_pack_scalar16','qsb_pack_digest16','qsb_ec192_packed_digit'],'actual_kernel_sha256':hashlib.sha256(kernel.encode()).hexdigest(),'generated_cpu_sha256':hashlib.sha256(cpp.encode()).hexdigest(),'positive':run.stdout.strip(),'cases':'\n'.join(line for line in run.stderr.splitlines() if line.startswith(('CASE_BEGIN','CASE_PASS','CTA_BEGIN'))),'actual_barrier_sha256':hashlib.sha256(barrier.encode()).hexdigest(),'actual_producer_barrier_sha256':hashlib.sha256(producer_barrier.encode()).hexdigest(),'actual_ticket_acquire_sha256':hashlib.sha256(load.encode()).hexdigest(),'actual_ticket_release_sha256':hashlib.sha256(store.encode()).hexdigest(),'actual_ticket_wait_sha256':hashlib.sha256(wait.encode()).hexdigest(),'run_summaries':run_summaries,'negative_controls':results,'requested_epoch_cases':case_values,'host_poll_delay_us':args.poll_us,'debug_logs':str(DEBUG),'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'ticket_projection':'Actual ticket helpers and EC192/producer64 barriers execute through host legacy atomics, sequential fences and warp joins. Host atomics sleep the reported configurable delay after polling to provide scheduling opportunities under256-thread oversubscription; no target sleep introduced.','host_accounting':'Launch ceil(nblk/6), nblk epoch bound, counts only after checked synchronization; source order audited.','scope':'Actual kernel role/packet/cohort/tail/hit-control and ring statements. Hash,field,inverse,gating primitives mocked with independent identity tokens. Actual producer recode/pack and consumer packed-digit extraction execute. Expected packet identities come from independent Python big-integer scalar reduction and signed-odd peeling. Actual numerical curve helpers have separate source-bound tests.','gpu_executed':False}
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
