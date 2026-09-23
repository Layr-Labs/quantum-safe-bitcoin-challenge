#!/usr/bin/env python3
"""Run the real search kernel on CPU threads with sparse real OpenSSL tables."""
from pathlib import Path
import hashlib,json,random,subprocess,sys,tempfile
from audit_search_primitives import extract

HERE=Path(__file__).resolve().parent
CANDIDATE=HERE.parent
ROOT=CANDIDATE.parents[1]
sys.path.insert(0,str(ROOT/'harness'))
import gen_problem as GEN


def main():
    source=(CANDIDATE/'pinning.cu').read_text()
    search=(CANDIDATE/'affine_search.cuh').read_text()
    # Reuse the same thread, barrier, warp-vote and exact field shims audited
    # with the synthetic probe. The actual new kernel is inserted below.
    preamble=(HERE/'audit_persistent_cpu.cpp').read_text().split('struct CpuObservation')[0]
    preamble=preamble.replace('cpp_int K=', 'cpp_int field_K=').replace('cpp_int P=B-K;', 'cpp_int P=B-field_K;')
    preamble=preamble.replace('uint64_t products[4][256]{},inverses[4][128]{};',
                              'uint64_t products[4][256]{},inverses[4][128]{}; uint32_t codes[15][128]{};')
    preamble=preamble.replace('#include "inverse_tree.cuh"',f'#include "{HERE}/inverse_tree.cuh"')
    preamble=preamble.replace('#include "inverse_service.cuh"',f'#include "{HERE}/inverse_service.cuh"')
    preamble=preamble.replace('#include "shifted_tail.cuh"',f'#include "{HERE}/shifted_tail.cuh"')
    preamble+='''
#include <openssl/sha.h>
#include <fstream>
#include <iterator>
#define __constant__
#define QSB_SHA_OPT 1
#define QSB_SPARSE_D 1
#define QSB_ZEROS_N 4
#define QSB_ISO_XR 0
#define QSB_YOFF 0
#define GT_CHUNKS 15
#define GT_TOTAL_ENTRIES 1048576
static uint32_t pin_tail_words[3];
inline uint32_t atomicOr(uint32_t* where,uint32_t value){return std::atomic_ref<uint32_t>(*where).fetch_or(value);}
inline uint32_t __umulhi(uint32_t a,uint32_t b){return uint32_t((uint64_t(a)*b)>>32);}
inline uint32_t __byte_perm(uint32_t a,uint32_t b,uint32_t selector){
 uint64_t both=uint64_t(a)|(uint64_t(b)<<32);uint32_t out=0;
 for(int i=0;i<4;i++){unsigned s=(selector>>(4*i))&15;unsigned v=(both>>(8*(s&7)))&255;
 if(s&8)v=(v&128)?255:0;out|=v<<(8*i);}return out;
}
inline void gt_load_signed_flat_m(const uint8_t* table,int offset,uint32_t index,uint64_t negative,uint64_t x[4],uint64_t y[4]){
 memcpy(x,table+64ull*(offset+index),32);memcpy(y,table+64ull*(offset+index)+32,32);
 if(negative)limbs(y,(P-integer(y))%P);
}
'''
    preamble+=(CANDIDATE/'GPUHash.h').read_text().split('//Take the last 8 bytes')[0]
    preamble+=(CANDIDATE/'sha_schedule_interleaved.cuh').read_text()
    optimized=(CANDIDATE/'sha_pinsha.cuh').read_text()
    optimized=optimized.replace(extract(optimized,'qsb_fadd'),'inline uint32_t qsb_fadd(uint32_t a,uint32_t one,uint32_t b){return a*one+b;}')
    preamble+=optimized
    for marker in ['struct qsb_tail_pre {','__device__ __constant__ uint64_t GT_ORDER_N[4]']:
        start=source.index(marker);preamble+=source[start:source.index('};',start)+2]+'\n'
    for name in ['qsb_h_ror','qsb_make_tail_pre','_SHA256TransformFastTail11Q','qsb_signed_recode_setup','gt_offset','gt_shift']:
        preamble+=extract(source,name)+'\n'
    start=source.index('typedef struct {',source.index('/* Params loader for pinning2.bin */'))
    preamble+=source[start:source.index('} pinning2_params_t;',start)+len('} pinning2_params_t;')]+'\n'
    preamble+=extract(source,'load_pinning2')+'\n'
    old=extract(search,'sub')
    search=search.replace(old,'static void sub(uint64_t out[4],const uint64_t a[4],const uint64_t b[4]){cpp_int v=(integer(a)-integer(b))%P;if(v<0)v+=P;limbs(out,v);}')
    search='\n'.join(line for line in search.splitlines() if not line.startswith('#include "research_affine/'))
    search=search.replace('__shared__ uint64_t products[4][2*QSB_AFFINE_SEARCH_N],inverses[4][QSB_AFFINE_SEARCH_N];',
                          'auto &products=cpu_block->products;auto &inverses=cpu_block->inverses;')
    search=search.replace('__shared__ uint32_t codes[GT_CHUNKS][QSB_AFFINE_SEARCH_N];','auto &codes=cpu_block->codes;')
    body=preamble+'\n'+search+'\n'+(HERE/'audit_search_cpu_main.cpp').read_text()
    problem,blob=GEN.gen_pinning(random.Random(20260923))
    with tempfile.TemporaryDirectory(prefix='.audit-search-cpu-',dir=HERE) as directory:
        temporary=Path(directory);cpp=temporary/'audit.cpp';exe=temporary/'audit'
        cpp.write_text(body);(temporary/'problem.bin').write_bytes(blob)
        (temporary/'prefix.bin').write_bytes(bytes.fromhex(problem['pin_prefix']))
        compile_result=subprocess.run(['rtk','proxy','g++','-std=c++20','-O2','-pthread','-Wno-deprecated-declarations',
            '-I/tmp/qsb-boost-audit/usr/include',str(cpp),'-lcrypto','-o',str(exe)],capture_output=True,text=True)
        if compile_result.returncode:raise RuntimeError(compile_result.stderr)
        run=subprocess.run(['rtk','proxy',str(exe),str(temporary/'problem.bin'),str(temporary/'prefix.bin')],
                           capture_output=True,text=True,timeout=180,check=True)
    result=json.loads(run.stdout.splitlines()[-1])
    result['source_sha256']={name:hashlib.sha256((CANDIDATE/name).read_bytes()).hexdigest() for name in ['pinning.cu','affine_search.cuh','GPUHash.h','sha_pinsha.cuh','research_affine/inverse_tree.cuh','research_affine/inverse_service.cuh','research_affine/shifted_tail.cuh']}
    result['extracted_translation_sha256']=hashlib.sha256(body.encode()).hexdigest()
    (HERE/'search_cpu_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
