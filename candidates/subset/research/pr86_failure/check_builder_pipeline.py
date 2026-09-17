#!/usr/bin/env python3
"""Execute both actual table-builder kernels and the complete inverse hierarchy.

Uses sparse virtual table storage, CPU threads and OpenSSL arithmetic. No GPU
execution, timing or whole-resident-table claim. Includes the official failed
run's public synthetic base coefficient.
"""
import argparse,ctypes as C,hashlib,json,random,re,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from check_candidate import BACKEND,function
from check_deferred_source import FIELD
from preflight import source_identity
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--compact',action='store_true')
ap.add_argument('--fused',action='store_true',help='use the fused candidate startup-only checkpoint helpers and warp-synchronized inverse tree')
ap.add_argument('--source',type=Path,help='explicit isolated successor; default remains the preserved failed source')
ap.add_argument('--widths',help='comma-separated independently expected widths')
ap.add_argument('--low-bits',type=int)
ap.add_argument('--report',type=Path)
args=ap.parse_args();base=args.source.resolve() if args.source else ROOT;h=base/'tests/gpu_epochs';identity=source_identity(base)
if not args.source:assert identity['source_fingerprint']=='6bfe0e61fbbcabce237877c95e238f0bd22914ed6ceade6baf97e058c5e51ac6','Failed source changed; explicitly select a new comparison.'
if args.source:assert args.report,'A successor must write its own report, preserving the failed-source evidence.'
src=(h/'tree.cu').read_text()
checkpoint_header='builder_checkpoint.cuh' if args.fused else 'ranked_pipeline.cuh'
rank=(h/checkpoint_header).read_text()
prefix='compact' if args.compact else 'gt';devprefix='mixed' if args.compact else 'wide';macro='COMPACT' if args.compact else 'GT'
host=(h/'compact_table_host.cuh').read_text() if args.compact else src
geometry=(h/('compact_geometry.cuh' if args.compact else 'wide_geometry.cuh')).read_text()
decl=(h/'compact_table_device.cuh').read_text().split('__device__ __forceinline__ int32_t compact_recode_step(')[0] if args.compact else src[src.index('#define GT_CHUNKS'):src.index('/* n = secp256k1')]
decl=re.sub(r'^#include "[^"\n]+"\n','',decl,flags=re.M)
common=BACKEND+FIELD+r'''
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <sys/mman.h>
#define __host__
#define __launch_bounds__(...)
static void _ModSub256(uint64_t*r,uint64_t*b){field_op(r,r,b,2);}
'''
if args.fused:
 common+='\nstatic void __syncwarp(unsigned mask=0xffffffffu){require(mask==0xffffffffu);warp_barriers[threadIdx.x/32]->wait();}\n'
pipeline=rank.split('/* Shared-denominator recovery',1)[0]
pipeline=pipeline.replace('#pragma once','')
cache_header=h/'cache_ops.cuh'
if cache_header.exists():
 common+='\nstruct ulonglong2{uint64_t x,y;};\n'+cache_header.read_text()
 pipeline=pipeline.replace('#include "cache_ops.cuh"','')
code='\n'.join([common,geometry,decl,(h/'tree_inverse.cuh').read_text(),pipeline,
 function(host,f'static void {prefix}_point_to_limbs('),
 function(host,f'static void {prefix}_require('),
 function(host,f'static void {prefix}_build_ladders('),
 (h/(devprefix.replace('mixed','compact')+'_table_kernels.cuh')).read_text()])
wrapper=r'''
static std::vector<uint64_t>L,H;
static EC_GROUP *group;static BN_CTX *ctx;static BIGNUM *scale,*order,*k,*x,*y;static EC_POINT *point;
static uint8_t *table;
extern "C" void init(const uint64_t*coef){
 L.resize((size_t)TABLE_CHUNKS*TABLE_LO*8);H.resize((size_t)TABLE_CHUNKS*TABLE_HI*8);
 HOST_build_ladders(L.data(),H.data(),(const uint8_t*)coef);
 table=(uint8_t*)mmap(nullptr,(size_t)TABLE_TOTAL_ENTRIES*64,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
 require(table!=MAP_FAILED);
 group=EC_GROUP_new_by_curve_name(NID_secp256k1);ctx=BN_CTX_new();scale=BN_lebin2bn((const uint8_t*)coef,32,nullptr);
 order=BN_new();k=BN_new();x=BN_new();y=BN_new();point=EC_POINT_new(group);
 require(EC_GROUP_get_order(group,order,ctx));BN_set_word(k,2);require(BN_mod_inverse(k,k,order,ctx)!=nullptr);require(BN_mod_mul(scale,scale,k,order,ctx));
}
extern "C" int audit(uint32_t start,int count){
 int blocks=(count+255)/256,groups=(blocks+255)/256;
 require(start+(uint64_t)count<=TABLE_TOTAL_ENTRIES);
 std::vector<uint64_t> roots(blocks*4),super(groups*4),tree(blocks*4*QSB_CHECKPOINT_STRIDE+1,0xcafe),root_tree(groups*4*QSB_CHECKPOINT_STRIDE+1,0xbeef);
 if(start)memset(table+(size_t)(start-1)*64,0xa7,64);
 if((uint64_t)start+count<TABLE_TOTAL_ENTRIES)memset(table+(size_t)(start+count)*64,0xb8,64);
 inverse_count=0;
 for(int b=0;b<blocks;b++)launch(256,[&](int){blockIdx.x=b;DEVICE_table_prepare(L.data(),H.data(),table,start,count,roots.data(),tree.data());});
 for(int b=0;b<groups;b++)launch(256,[&](int){blockIdx.x=b;qsb_root_group_prepare(roots.data(),blocks,super.data(),root_tree.data());});
 launch(256,[&](int){qsb_invert_super_roots(super.data(),groups);});
 for(int b=0;b<groups;b++)launch(256,[&](int){blockIdx.x=b;qsb_root_group_finish(roots.data(),blocks,super.data(),root_tree.data());});
 for(int b=0;b<blocks;b++)launch(256,[&](int){blockIdx.x=b;DEVICE_table_finish(L.data(),H.data(),table,start,count,roots.data(),tree.data());});
 require(inverse_count==1&&tree.back()==0xcafe&&root_tree.back()==0xbeef);
 if(start)for(int j=0;j<64;j++)require(table[(size_t)(start-1)*64+j]==0xa7);
 if((uint64_t)start+count<TABLE_TOTAL_ENTRIES)for(int j=0;j<64;j++)require(table[(size_t)(start+count)*64+j]==0xb8);
 for(int i=0;i<count;i++){
  uint32_t t=start+i,hi,lo;int ch;DEVICE_decode_entry(t,&ch,&hi,&lo);
  BN_set_word(k,2ull*(t-HOST_offset(ch))+1);require(BN_lshift(k,k,HOST_shift(ch)));require(BN_mod_mul(k,k,scale,order,ctx));
  require(EC_POINT_mul(group,point,k,nullptr,nullptr,ctx));uint64_t want[8];HOST_point_to_limbs(group,point,x,y,ctx,want);
  if(memcmp(want,table+(size_t)t*64,64)){fprintf(stderr,"Integrated builder mismatch at %u\n",t);return 0;}
 }
 return 1;
}
extern "C" void cleanup(){
 munmap(table,(size_t)TABLE_TOTAL_ENTRIES*64);L.clear();H.clear();EC_POINT_free(point);EC_GROUP_free(group);BN_CTX_free(ctx);BN_free(scale);BN_free(order);BN_free(k);BN_free(x);BN_free(y);
}
'''.replace('TABLE_',macro+'_').replace('HOST_',prefix+'_').replace('DEVICE_',devprefix+'_')
code+='\n'+wrapper
widths=[18]+[17]*14 if args.compact else [26]*6+[25]*4
if args.widths:widths=[int(b) for b in args.widths.split(',')]
assert sum(widths)==256 and all(2<=b<=30 for b in widths)
offsets=[sum(1<<(b-1) for b in widths[:i]) for i in range(len(widths))];total=sum(1<<(b-1) for b in widths)
low=256 if args.compact else 8192
if args.low_bits is not None:low=1<<args.low_bits
cases=[(0,1),(0,257),(low//2-1,257),(total-513,513)]
for off in offsets[1:]:cases.append((off-1,257))
N=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
official=int(json.loads((HERE/'problem.json').read_text())['neg_r_inv'],16)
counts=0
with tempfile.TemporaryDirectory(prefix='qsb-builder-pipeline-') as td:
 p=Path(td);cpp=p/'audit.cpp';so=p/'audit.so';cpp.write_text(code)
 subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC','-pthread','-Wno-deprecated-declarations','-Wno-pragma-once-outside-header','-I/opt/homebrew/opt/openssl@3/include','-L/opt/homebrew/opt/openssl@3/lib',str(cpp),'-lcrypto','-o',str(so)],check=True)
 lib=C.CDLL(str(so));lib.init.argtypes=[C.POINTER(C.c_uint64)];lib.audit.argtypes=[C.c_uint32,C.c_int]
 for coefficient in [official,1,N-1]:
  lib.init((C.c_uint64*4)(*[coefficient>>(64*j)&((1<<64)-1) for j in range(4)]))
  for start,count in cases:assert lib.audit(start,count),(start,count);counts+=count
  lib.cleanup()
assert source_identity(base)==identity
report={'status':'PASS',**identity,'geometry_bits':widths,'table_bytes':total*64,
 'checkpoint_source':checkpoint_header,'fused_warp_inverse':args.fused,
 'table_entries_checked':counts,'runtime_bases':3,'includes_failed_run_base':True,'cases':cases,
 'projection_sha256':hashlib.sha256(code.encode()).hexdigest(),'gpu_executed':False,
 'scope':'Actual prepare/finish table kernels, complete root inversion hierarchy and actual host ladders; CPU threads/OpenSSL, sparse virtual table, independent EC oracle and adjacent canaries. No native PTX/CUDA execution or full resident table. Passing does not diagnose the official early exit.'}
out=args.report or HERE/('compact-builder-pipeline.json' if args.compact else 'wide-builder-pipeline.json');out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
