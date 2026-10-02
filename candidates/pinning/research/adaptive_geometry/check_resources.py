#!/usr/bin/env python3
"""Exercise the actual optional table allocator with bounded CUDA stubs."""
from pathlib import Path
import hashlib,json,subprocess,tempfile
HERE=Path(__file__).resolve().parent;BASE=HERE/'candidate'
code=r'''
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cstdio>
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaErrorMemoryAllocation=2;
constexpr uint64_t WIDE_TOTAL_ENTRIES=1ull<<28;
static size_t available=0;static int query_error=0,alloc_error=0,queries=0,allocations=0,cleared=0,built=0;
static uint8_t backing,coefficient[32]={17};
static void wide_cuda_require(int e,const char*){if(e)throw e;}
static int cudaMemGetInfo(size_t*f,size_t*t){++queries;*f=available;*t=24ull<<30;return query_error;}
static int cudaMalloc(uint8_t**p,size_t n){++allocations;assert(n==16ull<<30);*p=alloc_error?nullptr:&backing;return alloc_error;}
static int cudaGetLastError(){++cleared;return alloc_error;}
static void wide_build_table(uint8_t*p,const uint8_t*n){assert(p==&backing&&n==coefficient);++built;}
#include "adaptive_resources.cuh"
int main(){
 constexpr size_t threshold=(16ull<<30)+(64ull<<20);
 assert(!adaptive_optional_wide(false,coefficient));assert(!queries&&!allocations&&!built);
 for(size_t n:{size_t(0),threshold-1}){available=n;assert(!adaptive_optional_wide(true,coefficient));}
 assert(queries==2&&!allocations&&!built);
 available=threshold;assert(adaptive_optional_wide(true,coefficient)==&backing);assert(allocations==1&&built==1);
 alloc_error=cudaErrorMemoryAllocation;assert(!adaptive_optional_wide(true,coefficient));assert(cleared==1&&built==1);
 alloc_error=7;try{adaptive_optional_wide(true,coefficient);assert(false);}catch(int e){assert(e==7);}
 assert(cleared==1&&built==1);
 query_error=9;int old=allocations;try{adaptive_optional_wide(true,coefficient);assert(false);}catch(int e){assert(e==9);}
 assert(old==allocations);
 puts("PASS: disabled, insufficient memory, exact reserve, success, OOM-only fallback, non-OOM and query errors");
}
'''.replace('#include <cstdio>','#include <cstdio>\n#include <initializer_list>')
with tempfile.TemporaryDirectory(prefix='qsb-adaptive-resources-') as td:
 p=Path(td);(p/'audit.cpp').write_text(code)
 subprocess.run(['c++','-std=c++17','-O2','-fsanitize=undefined','-fno-sanitize-recover=all','-I'+str(BASE),str(p/'audit.cpp'),'-o',str(p/'audit')],check=True)
 output=subprocess.check_output([str(p/'audit')],text=True).strip()
# Actual source-bound builder workspace: two ladders, a 4096-CTA tree,
# CTA roots, 16 super roots and the corresponding root tree.
scratch=2*10*8192*64+4096*4*256*8+4096*32+16*32+16*4*256*8
assert scratch<64<<20
result={'status':'PASS','gpu_executed':False,'ubsan':True,'summary':output,
 'wide_builder_scratch_bytes':scratch,'reserve_bytes':64<<20,
 'source_sha256':{n:hashlib.sha256((BASE/n).read_bytes()).hexdigest() for n in ('adaptive_resources.cuh','wide_table_host.cuh','wide_geometry.cuh','pinning.cu')},
 'limits':'Stub allocation/error behavior and source-derived explicit allocation budget. No actual GPU allocation; runtime/context overhead and startup timing are unmeasured.'}
(HERE/'resource-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
