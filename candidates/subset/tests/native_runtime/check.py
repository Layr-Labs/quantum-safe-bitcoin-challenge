#!/usr/bin/env python3
"""Compile the real carrier adapters against CUDA mocks; never execute GPU code.

Run from any directory: python3 path/to/tests/native_runtime/check.py --docker qsb-cuda
The existing container must mount the repository at /work (override --mount-root).
Without --docker, use local g++ plus OpenSSL headers/libraries.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--docker')
parser.add_argument('--mount-root', default='/work')
parser.add_argument('--cxx', default='g++')
parser.add_argument('--out', type=Path)
opts = parser.parse_args()
HERE = Path(__file__).resolve().parent
TRACK = HERE.parents[1]
ROOT = next(p for p in HERE.parents if (p / '.git').exists())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((TRACK / 'native_manifest.json').read_text())
assert len(manifest['entries']) == 7 and len(manifest['globals']) == 20 and len(manifest['host_globals']) == 13
assert (TRACK / 'native_sm89.cubin').stat().st_size == 933536
assert sha(TRACK / 'native_sm89.cubin') == manifest['cubin_sha256'] == '994014fb153b73eb4338b013d373734dc8350041378b2c39ed0199987cc38ae8'
source_paths = ['tests/gpu_epochs/tree.cu', 'tests/gpu_epochs/window_schedule_shared.cuh', 'tests/gpu_epochs/epoch_groups.cuh']
texts = '\n'.join((TRACK / p).read_text() for p in source_paths)
clean = re.sub(r'/\*.*?\*/|//[^\n]*', '', texts, flags=re.S)
checks, launch_tests = [], []
for idx, entry in enumerate(manifest['entries']):
    name = entry['name']
    match = re.search(r'\b' + name + r'\s*\((.*?)\)\s*\{', clean, re.S)
    assert match, name
    params = re.sub(r'^\s*#.*$', '', match[1], flags=re.M).replace('__restrict__', '').strip()
    params = [p.strip() for p in params.split(',')]
    assert len(params) == len(entry['bytes']), (name, params)
    types = [re.sub(r'\b\w+\s*$', '', p).strip() for p in params]
    checks.append('static_assert(std::is_same<typename std::remove_cv<decltype(qsb_native::abi::' + name + ')>::type,qsb_native::KernelTag<' + ','.join(types) + '>>::value,"source/token parameter type mismatch: ' + name + '");')
    args, prepare = [], []
    for j, typ in enumerate(types):
        if '*' in typ:
            expr = '0' if j % 3 == 0 else ('NULL' if j % 3 == 1 else 'reinterpret_cast<' + typ + '>(uintptr_t(0x100000000ULL+' + str(256 * idx + 16 * j) + '))')
        else:
            expr = '-7' if j % 3 == 0 else 'uint64_t(0xffffffff00000000ULL+' + str(idx * 100 + j) + ')'
        prepare.append(typ + ' v' + str(j) + '=(' + typ + ')(' + expr + ');expect(v' + str(j) + ');')
        args.append(expr)
    launch_tests.append('{ expected.clear();' + ''.join(prepare) + 'expected_handle=' + str(idx + 1) + ';QSB_LAUNCH(' + name + ',dim3(17,2,1),dim3(128,1,1),' + ','.join(args) + ');}')

symbol_types = {
    'BINOM_C': 'uint64_t[151][10]', 'K': 'uint32_t[64]', 'QSB_CONST_SCHEDULE': 'uint32_t[4][64]',
    'QSB_PUSH_WORDS': 'uint4[151]', 'QSB_U2R': 'uint64_t[8]', 'QSB_U2R_C': 'uint64_t[4]',
    'WIN3': 'uint8_t[128][3]', 'QSB_WINDOW_FIRST': 'uint32_t[14][128]', 'QSB_WINDOW_SECOND': 'uint32_t[64][128]',
    'QSB_WINDOW_CLASS': 'uint32_t[128]', 'QSB_FIRST_CLASS': 'uint32_t[128]',
    'QSB_FIRST_UNIQUE': 'uint32_t[14][16]', 'QSB_FIRST_COUNT': 'int',
}
assert set(symbol_types) == set(manifest['host_globals'])
for name, typ in symbol_types.items():
    checks.append('static_assert(std::is_same<typename std::remove_cv<decltype(qsb_native::abi::' + name + ')>::type,qsb_native::SymbolTag<' + typ + '>>::value,"global token type mismatch: ' + name + '");')
checks += [
    'static_assert(sizeof(epoch_desc_t)==64 && alignof(epoch_desc_t)==4,"epoch POD size/alignment");',
    'static_assert(offsetof(epoch_desc_t,mid)==0 && offsetof(epoch_desc_t,remW)==32 && offsetof(epoch_desc_t,early)==40 && offsetof(epoch_desc_t,pad)==46,"epoch POD offsets");',
    'static_assert(sizeof(qsb_group_t)==128 && alignof(qsb_group_t)==16,"group POD size/alignment");',
    'static_assert(offsetof(qsb_group_t,st)==0 && offsetof(qsb_group_t,w)==32 && offsetof(qsb_group_t,pos)==96 && offsetof(qsb_group_t,acc)==100 && offsetof(qsb_group_t,nb)==104 && offsetof(qsb_group_t,last)==108 && offsetof(qsb_group_t,base_lo)==112 && offsetof(qsb_group_t,base_hi)==116 && offsetof(qsb_group_t,o)==120,"group POD offsets");',
    'static_assert(sizeof(uint4)==16 && alignof(uint4)==16,"uint4 wire layout");',
]

def function(text, name):
    start = text.index(name + '(')
    start = text.rfind('\n', 0, start) + 1
    end = text.index('{', start) + 1
    depth = 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]

tree = (TRACK / source_paths[0]).read_text()
helpers = '\n'.join(function(tree, name) for name in ['qsb_prepare_push_words', 'qsb_host_rotr', 'qsb_prepare_constant_schedule'])
start = tree.index('        int cnt = 0;', tree.index('uint8_t h_win3'))
end = tree.index('\n        QSB_TO_SYMBOL(WIN3', start)
selector = tree[start:end]

header = r'''#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <string>
#include <type_traits>
#include <stdlib.h>
#include <unistd.h>
#ifndef CUDART_VERSION
#define CUDART_VERSION 12080
#endif
#define __align__(n) alignas(n)
struct alignas(16) uint4 {uint32_t x,y,z,w;};
struct dim3 {unsigned x,y,z;dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
struct cudaDeviceProp {int major,minor;};
typedef int cudaError_t;typedef void* cudaLibrary_t;typedef void* cudaKernel_t;
const int cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
static int launches=0,load_calls=0,get_kernels=0,get_globals=0,uploads=0,downloads=0,k_reads=0;
static uintptr_t expected_handle;
static std::vector<std::vector<unsigned char>> expected;
static std::string mode;
struct MockGlobal {const char* name;size_t bytes;void* pointer;};
static MockGlobal mock_globals[]={
'''
for g in manifest['globals']:
    if g['symbol'] in symbol_types:
        header += '{"' + g['symbol'] + '",' + str(g['bytes']) + ',nullptr},\n'
header += r'''};
template<class T> void expect(T v){const unsigned char*p=(const unsigned char*)&v;expected.emplace_back(p,p+sizeof(v));}
static const char* cudaGetErrorString(int){return "MOCK ERROR";}
static cudaError_t cudaLibraryLoadData(cudaLibrary_t*p,const void*,void*,void*,unsigned,void*,void*,unsigned){++load_calls;*p=(void*)1;return mode=="loaderror"?99:0;}
static cudaError_t cudaLibraryGetKernel(cudaKernel_t*p,cudaLibrary_t,const char*name){++get_kernels;if(mode=="kernel_lookup"&&get_kernels==3)return 99;
'''
for idx, entry in enumerate(manifest['entries']):
    header += 'if(!strcmp(name,"' + entry['symbol'] + '")){*p=(void*)uintptr_t(' + str(idx + 1) + ');return 0;}\n'
header += r'''return 99;}
static cudaError_t cudaLibraryGetGlobal(void**p,size_t*n,cudaLibrary_t,const char*name){++get_globals;
 if(mode=="global_lookup"&&get_globals==3)return 99;
 for(auto &g:mock_globals)if(!strcmp(name,g.name)){
  g.pointer=calloc(1,g.bytes);*p=g.pointer;*n=g.bytes;
  if(!strcmp(name,"K"))for(unsigned i=0;i<64;i++)((uint32_t*)g.pointer)[i]=0x12345678u+i*777u;
  if(mode=="global_size"&&get_globals==1)--*n;
  if(mode=="global_null"&&get_globals==1)*p=nullptr;
  return 0;
 }return 99;}
static cudaError_t cudaMemcpy(void*d,const void*s,size_t n,int kind){
 if(kind==cudaMemcpyHostToDevice){++uploads;if(mode=="copy_h2d")return 99;}
 else if(kind==cudaMemcpyDeviceToHost){++downloads;if(mode=="copy_d2h")return 99;
  for(auto &g:mock_globals)if(!strcmp(g.name,"K")&&s==g.pointer)++k_reads;
 }else abort();
 memcpy(d,s,n);return 0;}
static cudaError_t cudaLaunchKernel(const void*fn,dim3 grid,dim3 block,void**args,size_t shared,void*stream){
 assert(fn==(void*)expected_handle);assert(grid.x==17&&grid.y==2&&grid.z==1&&block.x==128&&block.y==1&&block.z==1&&shared==0&&stream==nullptr);
 for(size_t i=0;i<expected.size();i++)assert(!memcmp(args[i],expected[i].data(),expected[i].size()));
 ++launches;return mode=="launcherror"?99:0;}
static ssize_t fake_readlink(const char*,char*p,size_t n){
 if(mode=="readlink_error")return -1;
 if(mode=="readlink_length")return n;
 const char*base=getenv("QSB_TEST_EXE");assert(base);size_t len=strlen(base);assert(len<n);memcpy(p,base,len);return len;}
'''
# No cudaMemcpyToSymbol/FromSymbol or source kernel declarations are supplied.
# Accidentally selecting those operations in the carrier cannot link this mock.
code = r'''#include "cuda_runtime.h"
#define readlink fake_readlink
#ifndef QSB_NATIVE_MODULE
#define QSB_NATIVE_MODULE 1
#endif
#ifndef QSB_ZEROS_N
#define QSB_ZEROS_N 24
#endif
#include "native_runtime.cuh"
#undef readlink
#define QSB_SE_WINDOWS 128
#define QSB_SE_PER_EPOCH 128
#define QSB_SE_CUT 137
#define QSB_SE_TWIN 3
static uint32_t qsb_host_rotr(uint32_t,int);
#include "tests/gpu_epochs/window_schedule_shared.cuh"
''' + '\n'.join(checks) + '\n' + helpers + r'''
static int actual_host_transfers(){
 uint8_t rows[1510];for(unsigned i=0;i<sizeof(rows);i++)rows[i]=(uint8_t)(i*13+7);
 uint32_t words[69];for(unsigned i=0;i<69;i++)words[i]=i*0x1020304u+17;
 uint8_t h_win3[128][3];
''' + selector + r'''
 assert(cnt==128);assert(qsb_prepare_push_words(rows,151)==0);
 assert(qsb_prepare_constant_schedule(words,69)==0);
 QSB_TO_SYMBOL(WIN3,h_win3,sizeof(h_win3));
 assert(qsb_prepare_window_schedule(rows,h_win3,words)==0);
 uint64_t h_u2r[8]={1,2,3,4,5,6,7,8},h_c[4]={9,10,11,12};
 uint64_t h_binom[151][10]={};
 QSB_TO_SYMBOL(QSB_U2R,h_u2r,sizeof(h_u2r));QSB_TO_SYMBOL(QSB_U2R_C,h_c,sizeof(h_c));QSB_TO_SYMBOL(BINOM_C,h_binom,sizeof(h_binom));
 assert(uploads==12&&downloads==2&&k_reads==2);return 0;
}
int main(int argc,char**argv){assert(argc==2);mode=argv[1];
 qsb_native::environment();assert(!strcmp(getenv("CUDA_MODULE_LOADING"),"LAZY"));
 if(mode=="unfrozen_write"){uint32_t z[64]={};QSB_TO_SYMBOL(K,z,sizeof(z));return 90;}
 if(mode=="unfrozen_read"){uint32_t z[64];QSB_FROM_SYMBOL(z,K,sizeof(z));return 91;}
 if(mode!="unfrozen_launch")qsb_native::select({mode=="architecture_major"?9:8,mode=="architecture_minor"?6:9});
 if(mode=="repeat")qsb_native::select({8,9});
 if(mode=="abi_width")qsb_native::kernels[0].sizes[0]=4;
 if(mode=="abi_count")++qsb_native::kernels[0].count;
 uint32_t scratch[65]={};
 if(mode=="bounds")QSB_TO_SYMBOL(K,scratch,257);
 if(mode=="declared")qsb_native::to_symbol(qsb_native::SymbolTag<uint32_t[63]>{"K"},scratch,252);
 if(mode=="unknown_global")qsb_native::to_symbol(qsb_native::SymbolTag<uint32_t>{"not_in_image"},scratch,4);
 if(mode=="unknown_kernel")qsb_native::launch(qsb_native::KernelTag<int>{"not_in_image"},dim3(1),dim3(1),0);
 if(mode!="unfrozen_launch")assert(actual_host_transfers()==0);
 for(int iteration=0;iteration<2;iteration++){
''' + '\n'.join(launch_tests) + r'''
 }
 assert(launches==14&&load_calls==1&&get_kernels==7&&get_globals==13);
'''
for g in manifest['globals']:
    if g['symbol'] in symbol_types:
        name, n = g['symbol'], g['bytes']
        code += '{unsigned char input[' + str(n) + '],output[' + str(n) + '];memset(input,0x5a,sizeof(input));QSB_TO_SYMBOL(' + name + ',input,sizeof(input));QSB_FROM_SYMBOL(output,' + name + ',sizeof(output));assert(!memcmp(input,output,sizeof(input)));}\n'
code += 'puts("PASS carrier: 14 typed launches; 14 host transfer sites including both K reads; 13 global roundtrips; exact token types and POD layouts");}\n'

cases = ['native', 'architecture_minor', 'architecture_major', 'missing', 'badlength', 'extra_bytes', 'corrupt',
         'readlink_error', 'readlink_length', 'repeat', 'unfrozen_launch', 'unfrozen_write', 'unfrozen_read',
         'loaderror', 'kernel_lookup', 'global_lookup', 'global_size', 'global_null',
         'abi_width', 'abi_count', 'bounds', 'declared', 'unknown_global', 'unknown_kernel',
         'copy_h2d', 'copy_d2h', 'launcherror']
with tempfile.TemporaryDirectory(prefix='.carrier-test-', dir=HERE) as td:
    build = Path(td)
    def mapped(path):
        path = Path(path).resolve()
        return str(Path(opts.mount_root) / path.relative_to(ROOT)) if opts.docker else str(path)
    def run(command, **kw):
        return subprocess.run((['docker', 'exec', opts.docker] if opts.docker else []) + command, **kw)
    (build / 'cuda_runtime.h').write_text(header)
    (build / 'test.cc').write_text(code)
    compile_cmd = [opts.cxx, '-std=c++14', '-Werror', '-Wno-overflow', '-Wno-deprecated-declarations',
                   '-I' + mapped(build), '-I' + mapped(TRACK), mapped(build / 'test.cc'), '-lcrypto']
    run(compile_cmd + ['-o', mapped(build / 'test')], check=True)
    payload = (TRACK / 'native_sm89.cubin').read_bytes()
    for mode, data in [('badlength', b'bad'), ('extra_bytes', payload + b'x'), ('corrupt', bytes([payload[0] ^ 1]) + payload[1:])]:
        folder = build / mode
        folder.mkdir()
        (folder / 'native_sm89.cubin').write_bytes(data)
    records = []
    for case in cases:
        location = TRACK / 'mock-executable'
        if case == 'missing':
            location = build / 'missing' / 'mock-executable'
        elif case in ('badlength', 'extra_bytes', 'corrupt'):
            location = build / case / 'mock-executable'
        proc = run(['env', 'QSB_TEST_EXE=' + mapped(location), mapped(build / 'test'), case], text=True, capture_output=True)
        expected_exit = 0 if case == 'native' else 1
        assert proc.returncode == expected_exit, (case, proc.returncode, proc.stdout, proc.stderr)
        if case == 'native':
            assert 'PASS carrier:' in proc.stdout
        else:
            assert 'QSB native:' in proc.stderr or 'QSB CUDA' in proc.stderr, (case, proc.stderr)
        records.append({'case': case, 'exit': proc.returncode, 'stdout': proc.stdout.strip(), 'stderr': proc.stderr.strip()})
    compile_failures = []
    for name, flags, message in [
        ('unsupported_headers', ['-DCUDART_VERSION=12070'], 'Native carrier requires CUDA'),
        ('wrong_N', ['-DQSB_ZEROS_N=3'], 'Native carrier requires CUDA'),
        ('inconsistent_roles', ['-DQSB_HOST_CARRIER=0'], 'Select carrier/module together'),
        ('nonboolean_role', ['-DQSB_NATIVE_MODULE=2'], 'Select carrier/module together'),
    ]:
        proc = run(compile_cmd + flags + ['-o', mapped(build / 'invalid')], text=True, capture_output=True)
        assert proc.returncode != 0 and message in proc.stderr, (name, proc.returncode, proc.stderr)
        compile_failures.append({'case': name, 'rejected': True})
    result = {'status': 'PASS', 'gpu_executed': False, 'runtime_cases': len(records), 'cases': records,
              'compile_rejections': compile_failures, 'kernel_token_types_checked': 7, 'global_token_types_checked': 13,
              'typed_launches': 14, 'actual_host_transfer_sites_exercised': 14, 'K_reads': 2, 'global_roundtrips': 13,
              'POD_layouts_checked': ['epoch_desc_t', 'qsb_group_t', 'uint4'],
              'payload_sha256': sha(TRACK / 'native_sm89.cubin'),
              'source_sha256': {p: sha(TRACK / p) for p in ['native_runtime.cuh', 'native_abi.cuh', 'native_types.cuh', 'native_role.cuh', 'native_manifest.cuh'] + source_paths},
              'test_sha256': sha(Path(__file__)),
              'limits': 'Host CUDA mocks only; no real driver, GPU execution, throughput or JIT observation.'}
    encoded = json.dumps(result, indent=2) + '\n'
    if opts.out:
        opts.out.write_text(encoded)
    print(encoded, end='')
