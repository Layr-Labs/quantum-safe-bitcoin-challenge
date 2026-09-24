#!/usr/bin/env python3
"""CPU tests of the actual loader, trial state and sequence-boundary integration.

Only the load instructions and CUDA scheduling are stubbed. No GPU timing or
device correctness claim follows from this test.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
SOURCE = (HERE / 'pinning.cu').read_text()
GLV = (HERE / 'GLVScalar.cuh').read_text()


def function(text, signature):
    a = text.index(signature)
    b = text.index('{', a)
    depth = 1
    e = b + 1
    while depth:
        depth += (text[e] == '{') - (text[e] == '}')
        e += 1
    return text[a:e]


loader = function(SOURCE, '__device__ __forceinline__ void gt_load_signed_flat_m(')
geometry = GLV[:GLV.index('// QSB/VanitySearch')].replace('#pragma once', '')
boundary_start = SOURCE.index('#if QSB_COLD_STREAM == 2\n        // Complete all work')
boundary_end = SOURCE.index('        /* Progress every 10 sequences */', boundary_start)
boundary = SOURCE[boundary_start:boundary_end]

prefix = r'''
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <vector>
#include <sys/mman.h>
#include <time.h>
#include "ColdStreamTrial.h"
#define __host__
#define __device__
#define __forceinline__ inline
#define QSB_YOFF 1
struct alignas(16) ulonglong2 { uint64_t x, y; };
static uintptr_t allocation;
static std::vector<uint64_t> reads;
static std::vector<bool> policies;
static ulonglong2 get(const ulonglong2 *p, bool streaming) {
    uint64_t off = (uintptr_t)p - allocation;
    assert(off % 16 == 0 && off + 16 <= UINT64_C(9803211584));
    reads.push_back(off); policies.push_back(streaming);
    return {off ^ UINT64_C(0xCA74B52D162EE39F),
            (off + 8) ^ UINT64_C(0x58AE6CF020C921D3)};
}
static ulonglong2 __ldg(const ulonglong2 *p) { return get(p, false); }
static ulonglong2 qsb_cold_ld_v2(const ulonglong2 *p) { return get(p, true); }
'''
cpp = prefix + geometry + '\n#define QSB_COLD_STREAM 0\n' + loader.replace(
    'gt_load_signed_flat_m(', 'baseline_load(') + '\n#undef QSB_COLD_STREAM\n#define QSB_COLD_STREAM 2\n'
cpp += 'template<bool COLD_STREAM>\n' + loader
cpp += r'''
static uint64_t rng = 0x958DB16F41A77C25ULL;
static uint64_t random_word() {
    rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
    return rng * UINT64_C(0x2545F4914F6CDD1D);
}
static unsigned loader_test() {
    void *p = mmap(nullptr, UINT64_C(9803211584), PROT_NONE,
                   MAP_PRIVATE | MAP_ANON, -1, 0);
    assert(p != MAP_FAILED); allocation = (uintptr_t)p;
    unsigned cases = 0;
    for (int bank=0; bank<6; ++bank) {
        for (unsigned it=0; it<1024; ++it) {
            const unsigned entries = q9_bigtbl_entries(bank);
            const unsigned local = it==0 ? 0 : it==1 ? entries-1 : random_word()%entries;
            const unsigned record = q9_bigtbl_offset(bank)+local;
            for (unsigned sign=0; sign<2; ++sign) {
                for (unsigned split=0; split<2; ++split) {
                    const unsigned base = split ? q9_bigtbl_offset(bank) : 0;
                    const unsigned idx = record-base;
                    uint64_t expected_x[4], expected_y[4], x[4], y[4];
                    const uint64_t mask = UINT64_C(0)-sign;
                    reads.clear(); policies.clear();
                    baseline_load((uint8_t*)p, base, idx, mask, expected_x, expected_y);
                    const auto expected_reads = reads;
                    assert(reads.size()==4);
                    for (unsigned k=0; k<4; ++k) {
                        assert(reads[k] == (uint64_t)record*64 + 16*k);
                        assert(!policies[k]);
                    }
                    reads.clear(); policies.clear();
                    gt_load_signed_flat_m<false>((uint8_t*)p, base, idx, mask, x, y);
                    assert(reads == expected_reads);
                    assert(!memcmp(x,expected_x,32) && !memcmp(y,expected_y,32));
                    for (bool flag : policies) assert(!flag);
                    reads.clear(); policies.clear();
                    gt_load_signed_flat_m<true>((uint8_t*)p, base, idx, mask, x, y);
                    assert(reads == expected_reads);
                    assert(!memcmp(x,expected_x,32) && !memcmp(y,expected_y,32));
                    for (bool flag : policies) assert(flag == (bank>=4));
                    ++cases;
                }
            }
        }
    }
    assert(munmap(p, UINT64_C(9803211584))==0);
    return cases;
}
static void decision(const double arm_time[8], bool selected) {
    qsb::ColdStreamTrial t;
    const bool plan[8]={false,true,true,false,true,false,false,true};
    double now=100;
    unsigned drains=0, starts=0, samples=0;
    for (unsigned arm=0; arm<8; ++arm) {
        for (unsigned seq=0; seq<6; ++seq) {
            assert(t.arm()==arm && t.streaming()==plan[arm]);
            assert(t.drain_next()==(seq==1 || seq==5));
            drains += t.drain_next();
            now += seq<2 ? 7.0 : arm_time[arm]/4;
            auto event=t.complete_sequence(now, 1244600001);
            starts += event==qsb::ColdStreamTrial::Start;
            samples += event==qsb::ColdStreamTrial::Sample;
            if(seq==1) assert(event==qsb::ColdStreamTrial::Start);
            else if(seq==5) assert(event==(arm==7?qsb::ColdStreamTrial::Selected:qsb::ColdStreamTrial::Sample));
            else assert(event==qsb::ColdStreamTrial::Continue);
        }
    }
    assert(drains==16 && starts==8 && samples==7);
    assert(!t.active() && !t.drain_next() && t.streaming()==selected);
    for(unsigned i=0;i<20;++i) assert(t.complete_sequence(0,0)==qsb::ColdStreamTrial::Continue);
}
static void invalid_tests() {
    for(unsigned position=0;position<48;++position) {
        for(unsigned fault=0;fault<6;++fault) {
            qsb::ColdStreamTrial t; double now=100;
            for(unsigned i=0;i<position;++i) {
                assert(t.complete_sequence(now,100)!=qsb::ColdStreamTrial::Invalid);
                now+=1;
            }
            uint64_t work=100;
            if(fault==0) now=std::numeric_limits<double>::quiet_NaN();
            if(fault==1) now=std::numeric_limits<double>::infinity();
            if(fault==2) now=-1;
            if(fault==3) work=0;
            if(fault==4) { if(!position) continue; now-=1; }
            if(fault==5) { if(!position) continue; work=101; }
            assert(t.complete_sequence(now,work)==qsb::ColdStreamTrial::Invalid);
            assert(!t.active() && !t.streaming());
        }
    }
}

// Execute the literal production sequence-boundary block. Fake asynchronous
// batches stay pending in two slots until collected/drained. The fake clock
// refuses a measurement timestamp while either slot has pending work.
static double fake_now;
static bool require_empty;
static int slots[2];
static int fake_clock_gettime(int, struct timespec *out) {
    if (require_empty) assert(slots[0]==0 && slots[1]==0);
    out->tv_sec=(time_t)fake_now;
    out->tv_nsec=(long)((fake_now-out->tv_sec)*1e9);
    return 0;
}
#define clock_gettime fake_clock_gettime
#define QSB_SLOTS 2
#define QSB_OVERLAP_SEQUENCES 1
#define QSB_TAIL_TAB 0
static int integration(double stream_time, bool expected_selection) {
    qsb::ColdStreamTrial cold_trial;
    bool qsb_cold_stream_selected=false;
    uint64_t submitted=0, published=0, drains=0;
    const unsigned lt_range=401;
    unsigned batch_no=0;
    fake_now=100;
    slots[0]=slots[1]=0;
    auto drain_slot = [&](int s) {
        published+=slots[s]; slots[s]=0; ++drains; return 0;
    };
    for(unsigned seq=0;seq<50;++seq) {
        if(cold_trial.active()) assert(qsb_cold_stream_selected==cold_trial.streaming());
        for(unsigned n=0;n<5;++n) {
            unsigned s=batch_no++%2;
            published+=slots[s]; slots[s]=n==4?1:100;
            submitted+=slots[s];
        }
        fake_now+=qsb_cold_stream_selected?stream_time:1.0;
        require_empty=cold_trial.drain_next();
'''
cpp += boundary
cpp += r'''
        assert(submitted-published==(unsigned)(slots[0]+slots[1]));
    }
    assert(!cold_trial.active() && qsb_cold_stream_selected==expected_selection);
    assert(drains==32 && submitted==50*401);
    // Once selected, normal cross-sequence overlap resumes.
    assert(slots[0]+slots[1]>0);
    drain_slot(0); drain_slot(1);
    assert(submitted==published);
    return 0;
}
#undef clock_gettime
int main() {
    const unsigned cases=loader_test();
    const double equal[8]={4,4,4,4,4,4,4,4};
    const double gain[8]={4,3.8,3.8,4,3.8,4,4,3.8};
    const double loss[8]={4,4.2,4.2,4,4.2,4,4,4.2};
    const double small[8]={4,3.97,3.97,4,3.97,4,4,3.97};
    const double inconsistent[8]={4,3.8,3.8,4,4.1,4,4,4.1};
    const double drift[8]={4,4.1,4.2,4.3,4.4,4.5,4.6,4.7};
    decision(equal,false); decision(gain,true); decision(loss,false);
    decision(small,false); decision(inconsistent,false); decision(drift,false);
    invalid_tests();
    assert(integration(.95,true)==0);
    assert(integration(1.05,false)==0);
    printf("PASS loader=%u decisions=6 invalid=286 integrations=2\n",cases);
}
'''

with tempfile.TemporaryDirectory(prefix='qsb-cold-stream-') as temp:
    root = Path(temp)
    fixture = root / 'test.cpp'
    fixture.write_text(cpp)
    exe = root / 'test'
    subprocess.run(['c++', '-std=c++17', '-O2', '-fsanitize=undefined',
                    '-fno-sanitize-recover=all', '-I', str(HERE), str(fixture), '-o', str(exe)], check=True)
    run = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    assert 'PASS loader=24576 decisions=6 invalid=286 integrations=2' in run.stdout
    # A misplaced cold-bank boundary must be caught, not silently accepted.
    mutant = root / 'mutant.cpp'
    mutant.write_text(cpp.replace('base + idx >= q9_bigtbl_offset(4)',
                                 'base + idx >= q9_bigtbl_offset(5)'))
    subprocess.run(['c++', '-std=c++17', '-O2', '-I', str(HERE), str(mutant), '-o', str(root/'mutant')], check=True)
    bad = subprocess.run([str(root/'mutant')], capture_output=True)
    assert bad.returncode != 0, 'cold-boundary negative control escaped'

print(json.dumps({
    'actual_loader_cases': 24576, 'trial_decisions': 6,
    'invalid_time_or_work_cases': 286, 'actual_boundary_integrations': 2,
    'cold_boundary_negative_control': 'rejected', 'ubsan': 'pass',
    'gpu_executed': False,
    'limits': 'Load instructions and CUDA scheduling are stubbed; no GPU timing or hit-set comparison.',
    'source_sha256': {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                      for name in ['pinning.cu', 'ColdStreamTrial.h', 'GLVScalar.cuh']},
}, indent=2))
