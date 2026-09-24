#!/usr/bin/env python3
"""Exercise the actual host routing header with mocked CUDA calls; no GPU execution.

python3 candidates/subset/tests/native_runtime/check.py --docker qsb-cuda
Or omit --docker to use local g++ and OpenSSL development files.
The existing container must mount this repository at /work (or --mount-root).
"""
import argparse
import hashlib
import json
from pathlib import Path
import pathlib
import re
import subprocess
import tempfile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--docker', help='Existing CUDA container; no provisioning')
parser.add_argument('--mount-root', default='/work', help='Repository path inside container')
parser.add_argument('--cxx', default='g++')
parser.add_argument('--out', type=Path, help='Optional JSON result path')
options=parser.parse_args()
HERE=Path(__file__).resolve().parent
root=next(p for p in HERE.parents if (p/'.git').exists())
track=HERE.parents[1]
manifest=json.loads((track/'native_manifest.json').read_text())
assert len(manifest['entries'])==11 and len(manifest['globals'])==20 and len(manifest['host_globals'])==13
assert hashlib.sha256((track/'native_sm89.cubin').read_bytes()).hexdigest()==manifest['cubin_sha256']
abi={'entries':[{'symbol':e['symbol'],'parameters':[{'bytes':n} for n in e['bytes']]} for e in manifest['entries']], 'globals':manifest['globals']}
workspace=tempfile.TemporaryDirectory(prefix='.native-test-',dir=HERE)
out=Path(workspace.name)
def mapped(path):
    path=Path(path).resolve()
    return str(Path(options.mount_root)/path.relative_to(root)) if options.docker else str(path)
def run(cmd, **kwargs):
    if options.docker: cmd=['docker','exec',options.docker]+cmd
    return subprocess.run(cmd,**kwargs)
texts='\n'.join((track/p).read_text() for p in ['tests/gpu_epochs/tree.cu','tests/gpu_epochs/window_schedule_shared.cuh','tests/gpu_epochs/epoch_groups.cuh','glv10_build.cuh'])
texts=re.sub(r'/\*.*?\*/|//[^\n]*','',texts,flags=re.S)
prototypes=[];tests=[]
for idx,e in enumerate(abi['entries']):
 name=manifest['entries'][idx]['name']
 m=re.search(r'\b'+name+r'\s*\((.*?)\)\s*\{',texts,re.S);assert m,name
 params=re.sub(r'^\s*#.*$','',m.group(1),flags=re.M).replace('__restrict__','').strip()
 ps=[p.strip() for p in params.split(',')];assert len(ps)==len(e['parameters']),(name,ps)
 prototypes.append('void '+name+'('+params+') {}')
 args=[];prepare=[]
 for j,p in enumerate(ps):
  typ=re.sub(r'\b\w+\s*$','',p).strip();v=f'v{idx}_{j}'
  if '*' in typ:
   expr='0' if j%3==0 else f'reinterpret_cast<{typ}>(uintptr_t(0x10000+{idx*100+j}))'
  else:expr=f'uint64_t(0xffffffff00000000ULL+{idx*100+j})'
  prepare.append(f'{typ} {v}=({typ})({expr}); expect({v});')
  args.append(expr)
 tests.append('{ expected.clear(); '+''.join(prepare)+f' expected_handle={idx+1}; expected_source=(const void*){name}; QSB_LAUNCH({name}, dim3(17,2,1), dim3(128,1,1), '+','.join(args)+'); }')
header='''#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <string>
#include <stdlib.h>
#include <unistd.h>
#define CUDART_VERSION 12080
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
struct cudaDeviceProp { int major,minor; };
typedef void* cudaStream_t; typedef int cudaError_t; typedef void* cudaLibrary_t; typedef void* cudaKernel_t;
const int cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
static int launches=0,source_copies=0,load_calls=0,get_kernels=0,get_globals=0;
static uintptr_t expected_handle; static const void* expected_source;
static std::vector<std::vector<unsigned char>> expected;
static std::string mode;
template<class T> void expect(T v){ const unsigned char*p=(const unsigned char*)&v; expected.emplace_back(p,p+sizeof(v)); }
static const char* cudaGetErrorString(int){return "MOCK ERROR";}
static cudaError_t cudaLibraryLoadData(cudaLibrary_t *p,const void*,void*,void*,unsigned,void*,void*,unsigned){++load_calls;*p=(void*)1;return mode=="loaderror"?99:0;}
static cudaError_t cudaLibraryGetKernel(cudaKernel_t *p,cudaLibrary_t,const char*){*p=(void*)(uintptr_t)(++get_kernels);return 0;}
static cudaError_t cudaLibraryGetGlobal(void** p,size_t* n,cudaLibrary_t,const char*name){ ++get_globals;
'''
for g in abi['globals']:header+=f'if(!strcmp(name,"{g["symbol"]}")){{*n={g["bytes"]};*p=calloc(1,*n);return 0;}}\n'
header+='''return 99; }
static cudaError_t cudaMemcpy(void*d,const void*s,size_t n,int){memcpy(d,s,n);return 0;}
template<class T> static cudaError_t cudaMemcpyToSymbol(const T&s,const void*p,size_t n){++source_copies;memcpy((void*)&s,p,n);return 0;}
template<class T> static cudaError_t cudaMemcpyFromSymbol(void*p,const T&s,size_t n){++source_copies;memcpy(p,&s,n);return 0;}
static cudaError_t cudaLaunchKernel(const void*fn,dim3 grid,dim3 block,void**args,size_t shared,void*stream){
 assert(fn==(mode=="source"||mode=="missing"?expected_source:(void*)expected_handle));
 assert(grid.x==17&&grid.y==2&&grid.z==1&&block.x==128&&shared==0&&stream==NULL);
 for(size_t i=0;i<expected.size();++i)assert(!memcmp(args[i],expected[i].data(),expected[i].size()));
 ++launches;return mode=="launcherror"?99:0;
}
static ssize_t fake_readlink(const char*,char*p,size_t n){ const char*base=getenv("QSB_TEST_EXE");assert(base);size_t len=strlen(base);assert(len<n);memcpy(p,base,len);return len;}
'''
(out/'cuda_runtime.h').write_text(header)
code='''#include "cuda_runtime.h"
#define readlink fake_readlink
#define QSB_NATIVE_MODULE 1
#define QSB_ZEROS_N 24
#include "'''+'native_runtime.cuh'+'''"
#undef readlink
struct epoch_desc_t{}; struct qsb_group_t{};
'''+ '\n'.join(prototypes)+'''
int main(int argc,char**argv){
 assert(argc==2);mode=argv[1];
 qsb_native::environment();assert(!strcmp(getenv("CUDA_MODULE_LOADING"),"LAZY"));
 if(mode!="unfrozen")qsb_native::select({8,mode=="source"?6:9});
 if(mode=="repeat")qsb_native::select({8,9});
 if(mode=="abi")qsb_native::kernels[0].sizes[0]=4;
 uint32_t K[64]={},original[64],copy[64]={}; for(unsigned i=0;i<64;++i)original[i]=i*777;
 if(mode=="bounds")QSB_TO_SYMBOL(K,original,sizeof(K)+1);
 if(mode=="declared"){uint32_t small[63];qsb_native::to_symbol("K",small,original,sizeof(small));}
 if(mode!="unfrozen"){
 QSB_TO_SYMBOL(K,original,sizeof(K));QSB_FROM_SYMBOL(copy,K,sizeof(K));assert(!memcmp(copy,original,sizeof(K)));
 }
 for(int iteration=0;iteration<2;++iteration){
'''+ '\n'.join(tests)+'''
 }
 assert(launches==22);
 if(mode=="source"||mode=="missing")assert(source_copies==2 && load_calls==0);
 else assert(source_copies==0 && load_calls==1&&get_kernels==11&&get_globals==13);
 puts("PASS native-header mock: 22 launches, all arguments, round-trip symbols, selection");
}
'''
symbol_tests=[]
for g in abi['globals']:
 if g['symbol'] not in manifest['host_globals']: continue
 n=g['bytes'];name=g['symbol']
 symbol_tests.append('{unsigned char symbol['+str(n)+'], input['+str(n)+'], output['+str(n)+']; memset(input,0x5a,sizeof(input)); qsb_native::to_symbol("'+name+'",symbol,input,sizeof(input)); qsb_native::from_symbol("'+name+'",output,symbol,sizeof(output)); assert(!memcmp(input,output,sizeof(input)));}')
code=code.replace('assert(launches==22);','assert(launches==22); if(mode=="native"){'+''.join(symbol_tests)+'}')
(out/'test.cc').write_text(code)


try:
    command=[options.cxx,'-std=c++14','-Werror','-Wno-overflow','-Wno-deprecated-declarations',
             '-I'+mapped(out),'-I'+mapped(track),mapped(out/'test.cc'),'-lcrypto','-o',mapped(out/'test')]
    run(command,check=True)
    badlength=out/'badlength';badlength.mkdir();(badlength/'native_sm89.cubin').write_bytes(b'bad')
    corrupt=out/'corrupt';corrupt.mkdir()
    damaged=bytearray((track/'native_sm89.cubin').read_bytes());damaged[0]^=1
    (corrupt/'native_sm89.cubin').write_bytes(damaged)
    records=[]
    for mode in ['native','source','missing','abi','bounds','declared','launcherror','unfrozen','repeat','loaderror','badlength','corrupt']:
        location=track/'test'
        if mode=='missing':location=out/'nonexistent/test'
        if mode in ('badlength','corrupt'):location=out/mode/'test'
        command=['env','QSB_TEST_EXE='+mapped(location),mapped(out/'test'),mode]
        proc=run(command,text=True,capture_output=True)
        expected=0 if mode in ('native','source','missing') else 1
        assert proc.returncode==expected,(mode,proc.returncode,proc.stdout,proc.stderr)
        records.append({'case':mode,'exit':proc.returncode,'stdout':proc.stdout.strip(),'stderr':proc.stderr.strip()})
    result={'gpu_executed':False,'status':'PASS','cases':records,'native_header_sha256':hashlib.sha256((track/'native_runtime.cuh').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((track/'native_manifest.cuh').read_bytes()).hexdigest()}
    encoded=json.dumps(result,indent=2)+'\n'
    if options.out: options.out.write_text(encoded)
    print(encoded,end='')
finally:
    workspace.cleanup()
