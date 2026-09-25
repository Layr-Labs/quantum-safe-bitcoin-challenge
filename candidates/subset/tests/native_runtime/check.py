#!/usr/bin/env python3
"""Exercise the actual host routing header with mocked CUDA calls; no GPU execution.

python3 candidates/subset/tests/native_runtime/check.py --docker qsb-cuda
Or omit --docker to use local g++ and OpenSSL development files.
The existing container must mount this repository at /work (or --mount-root).
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import pathlib
import re
import subprocess
import tempfile
import shutil
import struct

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--adapt', action='store_true', help='Exercise mirrored globals, dual attributes and explicit digest routing')
parser.add_argument('--mock-image', action='store_true', help='Use a synthetic CPU-only payload fixture; no CUDA generation')
parser.add_argument('--docker', help='Existing CUDA container; no provisioning')
parser.add_argument('--mount-root', default='/work', help='Repository path inside container')
parser.add_argument('--cxx', default='g++')
parser.add_argument('--out', type=Path, help='Optional JSON result path')
options=parser.parse_args()
HERE=Path(__file__).resolve().parent
root=HERE.parents[3]
track=root/'candidates/subset'
workspace=tempfile.TemporaryDirectory(prefix='.native-test-',dir=HERE)
out=Path(workspace.name)
contract=json.loads((track/'native_contract.json').read_text())
if options.mock_image:
    manifest=json.loads((HERE/'expected_abi.json').read_text())
    payload=b'CPU mock fixture: not a CUDA image'
    manifest.update(cubin_bytes=len(payload), cubin_sha256=hashlib.sha256(payload).hexdigest())
    header=['#define QSB_NATIVE_BYTES '+str(len(payload)),
            'static const unsigned char payload_sha256[32] = {'+','.join(str(x) for x in hashlib.sha256(payload).digest())+'};',
            'static Kernel kernels[] = {']
    for e in manifest['entries']:
        header.append('{"'+e['name']+'", "'+e['symbol']+'", '+str(len(e['bytes']))+', {'+','.join(map(str,e['bytes']))+'}, NULL, false},')
    header+=['};','static Global globals[] = {']
    header += ['{"'+g['symbol']+'", '+str(g['bytes'])+', NULL},' for g in manifest['globals'] if g['symbol'] in manifest['host_globals']]
    header += ['};']
    (out/'native_manifest.cuh').write_text('\n'.join(header)+'\n')
    (out/'native_image.cuh').write_text('static const char *native_b64[] = {"'+base64.b64encode(payload).decode()+'"};\n')
    # Exercise byte-for-byte actual routing header against fixtures only in the temporary directory.
    shutil.copyfile(track/'native_runtime.cuh',out/'native_runtime.cuh')
    shutil.copyfile(track/'native_contract.cuh',out/'native_contract.cuh')
    metadata_root=out
else:
    manifest=json.loads((track/'native_manifest.json').read_text())
    metadata_root=track
assert len(manifest['entries'])==6 and len(manifest['host_globals'])==16
embedded=base64.b64decode(''.join(re.findall(r'"([A-Za-z0-9+/=]+)"', (metadata_root/'native_image.cuh').read_text())))
assert len(embedded)==manifest['cubin_bytes']
assert hashlib.sha256(embedded).hexdigest()==manifest['cubin_sha256']
abi={'entries':[{'symbol':e['symbol'],'parameters':[{'bytes':n} for n in e['bytes']]} for e in manifest['entries']], 'globals':manifest['globals']}
def mapped(path):
    path=Path(path).resolve()
    return str(Path(options.mount_root)/path.relative_to(root)) if options.docker else str(path)
def run(cmd, **kwargs):
    if options.docker: cmd=['docker','exec',options.docker]+cmd
    return subprocess.run(cmd,**kwargs)
texts='\n'.join((root/p).read_text() for p in ['candidates/subset/tests/gpu_epochs/tree.cu','candidates/subset/tests/gpu_epochs/window_schedule_shared.cuh','candidates/subset/tests/gpu_epochs/epoch_groups.cuh'])
texts=re.sub(r'/\*.*?\*/|//[^\n]*','',texts,flags=re.S)
prototypes=[];tests=[]
for idx,e in enumerate(abi['entries']):
 # Names are taken from the known small entry set, never from ABI demangling guesses.
 names=['kernel_build_first_flat','kernel_build_epochs','kernel_epoch_groups','kernel_build_epochs_inc','kernel_verify_pair_hits','kernel_digest','kernel_build_gtable']
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
 tests.append('{ expected.clear(); '+''.join(prepare)+f' expected_handle={idx+1}; expected_source=(const void*){name}; QSB_LAUNCH({name}, dim3(17,2,1), dim3(128,1,1), expected_shared, expected_stream, '+','.join(args)+'); }')
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
typedef int cudaError_t; typedef void* cudaLibrary_t; typedef void* cudaKernel_t; typedef void* cudaStream_t;
enum cudaFuncAttribute { cudaFuncAttributePreferredSharedMemoryCarveout=9 };
const int cudaErrorInvalidValue=1;
static size_t expected_shared; static void* expected_stream; static int attr_calls=0;
const int cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
static int launches=0,source_copies=0,load_calls=0,get_kernels=0,get_globals=0;
static uintptr_t expected_handle; static const void* expected_source; static bool explicit_source=false;
static std::vector<std::vector<unsigned char>> expected;
static std::string mode;
template<class T> void expect(T v){ const unsigned char*p=(const unsigned char*)&v; expected.emplace_back(p,p+sizeof(v)); }
static const char* cudaGetErrorString(int){return "MOCK ERROR";}
static cudaError_t cudaLibraryLoadData(cudaLibrary_t *p,const void*,void*,void*,unsigned,void*,void*,unsigned){++load_calls;*p=(void*)1;return mode=="loaderror"?99:0;}
static cudaError_t cudaLibraryGetKernel(cudaKernel_t *p,cudaLibrary_t,const char*){*p=(void*)(uintptr_t)(++get_kernels);return 0;}
static cudaError_t cudaLibraryGetGlobal(void** p,size_t* n,cudaLibrary_t,const char*name){ ++get_globals;
'''
for g in abi['globals']:
 if g['symbol']=='QSB_S3_DESC':
  data=struct.pack('<44I',*contract['descriptor_words'])
  header+='if(!strcmp(name,"QSB_S3_DESC")){*n=mode=="old_desc_size"?192:176;*p=calloc(1,*n); const unsigned char expected_desc[]={'+','.join(map(str,data))+'};memcpy(*p,expected_desc,176);if(mode=="wrong_desc")((unsigned char*)*p)[0]^=1;return 0;}\n'
 else:header+=f'if(!strcmp(name,"{g["symbol"]}")){{*n={g["bytes"]};*p=calloc(1,*n);return 0;}}\n'
header+='''return 99; }
static cudaError_t cudaMemcpy(void*d,const void*s,size_t n,int){memcpy(d,s,n);return 0;}
template<class T> static cudaError_t cudaMemcpyToSymbol(const T&s,const void*p,size_t n){++source_copies;memcpy((void*)&s,p,n);return 0;}
template<class T> static cudaError_t cudaMemcpyFromSymbol(void*p,const T&s,size_t n){++source_copies;memcpy(p,&s,n);return 0;}
static cudaError_t cudaLaunchKernel(const void*fn,dim3 grid,dim3 block,void**args,size_t shared,void*stream){
 assert(fn==((mode=="source"||explicit_source)?expected_source:(void*)expected_handle));
 assert(grid.x==17&&grid.y==2&&grid.z==1&&block.x==128&&shared==expected_shared&&stream==expected_stream);
 for(size_t i=0;i<expected.size();++i)assert(!memcmp(args[i],expected[i].data(),expected[i].size()));
 ++launches;return mode=="launcherror"?99:0;
}
struct cudaFuncAttributes {int unused;};
static cudaError_t cudaFuncGetAttributes(cudaFuncAttributes*,const void*p){assert(p==expected_source);return 0;}
static cudaError_t cudaFuncSetAttribute(const void *p,cudaFuncAttribute attr,int value){ assert(p==expected_source&&attr==cudaFuncAttributePreferredSharedMemoryCarveout&&value==100);++attr_calls;return 0;}
static cudaError_t cudaKernelSetAttributeForDevice(cudaKernel_t p,cudaFuncAttribute attr,int value,int device){ assert(p==(void*)expected_handle&&attr==cudaFuncAttributePreferredSharedMemoryCarveout&&value==100&&device==3);++attr_calls;return mode=="attribute_error"?99:0;}
'''
(out/'cuda_runtime.h').write_text(header)
code='''#include "cuda_runtime.h"
#define QSB_NATIVE_MODULE 1
#define QSB_ZEROS_N 24
#include "'''+'native_runtime.cuh'+'''"
struct epoch_desc_t{}; struct qsb_group_t{};
'''+ '\n'.join(prototypes)+'''
int main(int argc,char**argv){
 assert(argc==2);mode=argv[1];
 qsb_native::environment();assert(!strcmp(getenv("CUDA_MODULE_LOADING"),"LAZY"));
 if(mode=="badlength")qsb_native::native_b64[0]="";
 if(mode=="corrupt"){char *p=strdup(qsb_native::native_b64[0]);p[0]=p[0]=='A'?'B':'A';qsb_native::native_b64[0]=p;}
 if(mode=="encoding")qsb_native::native_b64[0]="!";
 uint32_t source_desc[44],source_banks[24];
 memcpy(source_desc,qsb_native::native_contract_desc,sizeof(source_desc));
 memcpy(source_banks,qsb_native::native_contract_banks,sizeof(source_banks));
 if(mode=="source_desc")source_desc[0]^=1;
 if(mode=="source_bank")source_banks[20]^=1;
 if(mode!="unfrozen")qsb_native::select({8,mode=="source"?6:9},source_desc,sizeof(source_desc),source_banks,sizeof(source_banks));
 if(mode=="repeat")qsb_native::select({8,9},source_desc,sizeof(source_desc),source_banks,sizeof(source_banks));
 if(mode=="abi")qsb_native::kernels[0].sizes[0]=4;
 uint32_t K[64]={},original[64],copy[64]={}; for(unsigned i=0;i<64;++i)original[i]=i*777;
 if(mode=="bounds")QSB_TO_SYMBOL(K,original,sizeof(K)+1);
 if(mode=="declared"){uint32_t small[63];qsb_native::to_symbol("K",small,original,sizeof(small));}
 if(mode!="unfrozen"){
 QSB_TO_SYMBOL(K,original,sizeof(K));QSB_FROM_SYMBOL(copy,K,sizeof(K));assert(!memcmp(copy,original,sizeof(K)));
 }
 for(int iteration=0;iteration<3;++iteration){ expected_stream=iteration?(void*)(uintptr_t)(0x1000+iteration):NULL; expected_shared=iteration*16;
'''+ '\n'.join(tests)+'''
 }
 assert(launches==18);
 if(mode=="source")assert(source_copies==2 && load_calls==0);
 else assert(source_copies==0 && load_calls==1&&get_kernels==6&&get_globals==17);
 puts("PASS native-header mock: 18 launches, all arguments, round-trip symbols, selection");
}
'''
digest_index=next(i for i,e in enumerate(manifest['entries']) if e['name']=='kernel_digest')
code=code.replace('assert(launches==18);', 'assert(launches==18); expected_handle='+str(digest_index+1)+'; expected_source=(const void*)kernel_digest; assert(QSB_ATTRIBUTE(kernel_digest,cudaFuncAttributePreferredSharedMemoryCarveout,100,3)==0); assert(attr_calls==1);')
symbol_tests=[]
for g in abi['globals']:
 if g['symbol'] not in manifest['host_globals']: continue
 n=g['bytes'];name=g['symbol']
 symbol_tests.append('{unsigned char symbol['+str(n)+'], input['+str(n)+'], output['+str(n)+']; memset(input,0x5a,sizeof(input)); qsb_native::to_symbol("'+name+'",symbol,input,sizeof(input)); qsb_native::from_symbol("'+name+'",output,symbol,sizeof(output)); assert(!memcmp(input,output,sizeof(input)));}')
code=code.replace('assert(launches==18);','assert(launches==18); if(mode=="native"){'+''.join(symbol_tests)+'}')
if not options.adapt:
    code=code.replace('#define QSB_NATIVE_MODULE 1','#define QSB_NATIVE_ADAPT 0\n#define QSB_NATIVE_MODULE 1')
if options.adapt:
    code=code.replace('#define QSB_NATIVE_MODULE 1','#define QSB_NATIVE_ADAPT 1\n#define QSB_NATIVE_MODULE 1')
    code=code.replace('assert(source_copies==0 && load_calls==1','assert(source_copies==17 && load_calls==1')
    code=code.replace('assert(attr_calls==1);','assert(attr_calls==(mode=="source"?1:2));')
    digest_test=tests[digest_index]
    source_test=digest_test.replace('QSB_LAUNCH(kernel_digest,','QSB_DIGEST_LAUNCH(false, kernel_digest,')
    native_test=digest_test.replace('QSB_LAUNCH(kernel_digest,','QSB_DIGEST_LAUNCH(true, kernel_digest,')
    code=code.replace('puts("PASS native-header mock:', 'if(mode=="native"){explicit_source=true;'+source_test+'explicit_source=false;'+native_test+'assert(launches==20);} puts("PASS native-header mock:')
(out/'test.cc').write_text(code)


try:
    command=[options.cxx,'-std=c++14','-Werror','-Wno-overflow','-Wno-deprecated-declarations',
             '-I'+mapped(out),'-I'+mapped(track),mapped(out/'test.cc'),'-lcrypto','-o',mapped(out/'test')]
    run(command,check=True)
    records=[]
    for mode in ['native','source','abi','bounds','declared','launcherror','unfrozen','repeat','loaderror','badlength','corrupt','encoding','source_desc','source_bank','old_desc_size','wrong_desc','attribute_error']:
        command=[mapped(out/'test'),mode]
        proc=run(command,text=True,capture_output=True)
        expected=0 if mode in ('native','source') else 1
        assert proc.returncode==expected,(mode,proc.returncode,proc.stdout,proc.stderr)
        records.append({'case':mode,'exit':proc.returncode,'stdout':proc.stdout.strip(),'stderr':proc.stderr.strip()})
    result={'gpu_executed':False,'status':'PASS','cases':records,'native_header_sha256':hashlib.sha256((track/'native_runtime.cuh').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((metadata_root/'native_manifest.cuh').read_bytes()).hexdigest(),'synthetic_payload':options.mock_image}
    encoded=json.dumps(result,indent=2)+'\n'
    if options.out: options.out.write_text(encoded)
    print(encoded,end='')
finally:
    workspace.cleanup()
