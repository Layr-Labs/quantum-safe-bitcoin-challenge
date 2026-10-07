"""Build a Windows C++ audit from the real planner and SHA-NI source.

This does not compile the whole Linux/CUDA program. Extracted functions are
unchanged; surrounding Ctx is the minimal host layout those functions use.
"""
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmark.json").is_file())
(ROOT / "research").mkdir(exist_ok=True)
src = (ROOT / 'candidates/subset/CpuGrindSubset.h').read_text()

def function(start):
    i = src.index(start)
    a = src.index('{', i)
    depth = 1
    b = a + 1
    while depth:
        depth += (src[b] == '{') - (src[b] == '}')
        b += 1
    return src[i:b] + '\n'

sha = src[src.index('#define QSHA '):src.index('#if QSB_CPU_FENCE\n/* QSB_CPU_FENCE:')]
# Remove the final #endif matching QCPU_SHANI outside this excerpt.
assert sha.rstrip().endswith('#endif')
sha = sha.rstrip()[:-len('#endif')]
ctx = '''
struct digest_params_t {
    uint32_t n=150, t=9, total_preimage_len=9906;
    uint32_t prefix_remainder_len=42, tail_section_len=218, tx_suffix_len=44;
    uint8_t *dummy_sigs, *tail_section, *tx_suffix;
};
template<class T> using qalloc64 = std::allocator<T>;
struct Ctx {
    const digest_params_t *dp;
    uint8_t cwin[CWIN_MAX][CWIN_BYTES];
    int ncwin=0, cut=137, early=CEARLY;
    uint64_t mid_bytes=8192;
    bool hplan=false;
    int h_nb=0, h_ng=0;
    uint8_t h_g0[CWIN_MAX];
    const uint32_t *h_wkp[CWIN_MAX][16];
    uint64_t h_binom[256][8];
    std::vector<uint8_t> h_gblk;
    std::vector<uint32_t, qalloc64<uint32_t>> h_wk;
};
static uint64_t binom_u64(int n,int k) {
    if(k<0 || k>n) return 0;
    uint64_t r=1;
    for(int i=1;i<=k;++i) r=r*(n-i+1)/i;
    return r;
}
'''
intro = '''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <memory>
#include <immintrin.h>
#include <cpuid.h>
#include "../candidates/subset/CpuSubsetFamily.h"
#define QCPU_VEC 0
#define QCPU_PFQ 0
#define QSB_CPU_KH16 0
#define QSB_CPU_SHA4 1
#define QSB_CPU_SHC 1
#define QSB_CPU_X4PS 1
#define QSB_ZEROS_N 24
#define SIG_PUSH_SIZE 10
struct fe { uint64_t v[4]; };
'''
constants = src[src.index('static const int CWIN_MAX'):src.index('static const int NWMAX')]
generated = intro + constants + sha + ctx
generated += function('static void hash_plan(Ctx &c)')
generated += '#define QSB_CPU_PREFIX100 1\n' + function('static void hash_plan_cpu_patterns(Ctx &c)')
generated += Path(__file__).with_name('family_audit_main.cpp').read_text()
(ROOT / 'research/family_audit_generated.cpp').write_text(generated)
print('Extracted the actual enumeration constants, hash planner, and SHA-NI functions.')
