#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Differential-test and microbenchmark the actual extracted host hash_record.

No CUDA is needed. The comparison source is read from the parent repository's
pinned baseline; no benchmark/judge files are modified. Requires g++/libcrypto.
"""
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[4]
BASE = "efef868ab78ff8d1229cdc1591b90797ee3be197"
TRACK = ROOT / "candidates/pinning"
source = (TRACK / "pinning.cu").read_text()
original = subprocess.check_output(
    ["git", "show", f"{BASE}:candidates/pinning/pinning.cu"], cwd=ROOT, text=True
)

def host_part(s):
    p = s[s.index("static inline void message(uint32_t w[16]"):
          s.index("/* claim and hash records of plane b;")]
    p = p.replace('#include "pksha_host_h0.h"', '')
    return p.replace("static void hash_record(", "__attribute__((noinline)) static void hash_record(")

support = '''
static int mode=1;
struct Job {
    std::atomic<uint32_t> nhit{0};
    const uint8_t *plane=nullptr;
    uint32_t batch_sz=0;
    uint32_t hits[64];
};
'''
cpp = '''
#define _GNU_SOURCE 1
#define QSB_ZEROS_N 6
#define QSB_FIN_BAL2 3
#define QSB_HOST_PKSHA 4
#define QSB_PK_LANES 128
#define QSB_PK_REC ((size_t)QSB_PK_LANES * 68u)
#include <atomic>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <chrono>
#include <sched.h>
#include <unistd.h>
#include <openssl/sha.h>
#include "cg_sha.h"
#include "pksha_host_h0.h"
namespace baseline {
''' + support + host_part(original) + '''
}
namespace optimized {
''' + support + host_part(source) + '''
}
static uint64_t rng=0xAC04202620260001ULL;
static uint64_t rand64(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;}
static uint32_t exact_h0(const uint64_t x[4],uint32_t prefix){
    unsigned char pub[33],h[32];pub[0]=prefix;
    for(int i=0;i<32;i++)pub[i+1]=(unsigned char)(x[3-i/8]>>(56-8*(i%8)));
    SHA256(pub,33,h);
    return ((uint32_t)h[0]<<24)|((uint32_t)h[1]<<16)|((uint32_t)h[2]<<8)|h[3];
}
static volatile uint64_t sink=0;
int main(){
    if(!__builtin_cpu_supports("avx2")){fprintf(stderr,"AVX2 unavailable\\n");return 77;}
    printf("compiler=%s cpu_avx2=1 cpu_sha=%d\\n",__VERSION__,__builtin_cpu_supports("sha")!=0);
    cpu_set_t allowed;CPU_ZERO(&allowed);
    if(sched_getaffinity(0,sizeof allowed,&allowed)==0){
        for(int c=0;c<CPU_SETSIZE;c++)if(CPU_ISSET(c,&allowed)){
            cpu_set_t one;CPU_ZERO(&one);CPU_SET(c,&one);
            if(sched_setaffinity(0,sizeof one,&one)==0)printf("pinned_cpu=%d\\n",c);
            break;
        }
    }
    unsigned checked=0;
    for(unsigned r=0;r<2048;r++){
        uint64_t x0[8][4],x1[8][4];uint32_t p0[8],p1[8];
        uint32_t w0[16][8],w1[16][8],o0[8],o1[8];
        for(int k=0;k<16;k++)for(int t=0;t<8;t++)w0[k][t]=rand64(),w1[k][t]=rand64();
        for(int t=0;t<8;t++){
            for(int i=0;i<4;i++)x0[t][i]=rand64(),x1[t][i]=rand64();
            if(r<16)for(int i=0;i<4;i++)x0[t][i]=(r&1)?UINT64_MAX:0;
            p0[t]=2+(rand64()&1);p1[t]=2+(rand64()&1);
            optimized::message_live_column(w0,t,x0[t],p0[t]);
            optimized::message_live_column(w1,t,x1[t],p1[t]);
            uint32_t old[16];baseline::message(old,x0[t],p0[t]);
            for(int k=0;k<9;k++)if(old[k]!=w0[k][t]){fprintf(stderr,"message mismatch\\n");return 1;}
        }
        optimized::hash8_avx2(w0,w1,o0,o1);
        for(int t=0;t<8;t++){
            if(o0[t]!=exact_h0(x0[t],p0[t]) || o1[t]!=exact_h0(x1[t],p1[t])){
                fprintf(stderr,"OpenSSL digest mismatch at round=%u lane=%d\\n",r,t);return 1;
            }
            checked+=2;
        }
    }
    const unsigned nr=128;
    std::vector<uint64_t> planes((nr*QSB_PK_REC+7)/8);
    for(auto &v:planes)v=rand64();
    unsigned records=0;
    for(int m=0;m<3;m++){
        if(m==2&&!__builtin_cpu_supports("sha"))continue;
        baseline::mode=optimized::mode=m;
        for(unsigned trial=0;trial<512;trial++){
            auto *bytes=(uint8_t*)planes.data();
            for(unsigned j=0;j<nr;j++){
                auto *yp=(uint32_t*)(bytes+j*QSB_PK_REC+64u*QSB_PK_LANES);
                for(unsigned l=0;l<QSB_PK_LANES;l++)yp[l]=(trial%5 && (rand64()&7)==0)?0:
                    0x80000202u|((rand64()&1)<<8)|(rand64()&1);
            }
            const uint32_t bs=trial<128?trial:((trial*977u)%(nr*4u*QSB_PK_LANES));
            for(unsigned j=0;j<nr;j++){
                baseline::Job a;optimized::Job b;a.plane=b.plane=bytes;a.batch_sz=b.batch_sz=bs;
                baseline::hash_record(a,j);optimized::hash_record(b,j);
                unsigned na=a.nhit.load(),nb=b.nhit.load();
                if(na!=nb||na>64||memcmp(a.hits,b.hits,na*sizeof(uint32_t))){
                    fprintf(stderr,"hit mismatch mode=%d trial=%u record=%u batch=%u\\n",m,trial,j,bs);return 1;
                }
                records++;
            }
        }
    }
    printf("verified_digest_count=%u verified_record_comparisons=%u (includes inactive lanes and partial batches)\\n",checked,records);
    auto *bytes=(uint8_t*)planes.data();
    for(unsigned j=0;j<nr;j++){
        auto *yp=(uint32_t*)(bytes+j*QSB_PK_REC+64u*QSB_PK_LANES);
        for(unsigned l=0;l<QSB_PK_LANES;l++)yp[l]=0x80000202u|((rand64()&1)<<8)|(rand64()&1);
    }
    baseline::mode=optimized::mode=1;
    baseline::Job a;optimized::Job b;a.plane=b.plane=bytes;a.batch_sz=b.batch_sz=nr*4u*QSB_PK_LANES;
    for(int k=0;k<8;k++)for(unsigned j=0;j<nr;j++){a.nhit=0;b.nhit=0;baseline::hash_record(a,j);optimized::hash_record(b,j);}
    const int order[12]={0,1,1,0,1,0,0,1,0,1,1,0};
    double rates[12];
    for(int e=0;e<12;e++){
        const auto t0=std::chrono::steady_clock::now();uint64_t done=0;
        double elapsed=0;
        do{
            for(unsigned j=0;j<nr;j++){
                if(order[e]){b.nhit=0;optimized::hash_record(b,j);sink+=b.nhit.load();}
                else{a.nhit=0;baseline::hash_record(a,j);sink+=a.nhit.load();}
            }
            done+=nr*QSB_PK_LANES;
            elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        }while(elapsed<0.75);
        rates[e]=done/elapsed;
        printf("epoch=%d path=%s candidate_per_s=%.3f seconds=%.6f\\n",e,order[e]?"optimized":"baseline",rates[e],elapsed);fflush(stdout);
    }
    std::vector<double> base,opt;
    for(int e=0;e<12;e++)(order[e]?opt:base).push_back(rates[e]);
    std::sort(base.begin(),base.end());std::sort(opt.begin(),opt.end());
    const double bm=(base[2]+base[3])/2,om=(opt[2]+opt[3])/2;
    printf("median_baseline=%.3f median_optimized=%.3f throughput_gain_pct=%+.4f min_max_baseline=%.3f/%.3f min_max_optimized=%.3f/%.3f sink=%llu\\n",
           bm,om,100*(om/bm-1),base.front(),base.back(),opt.front(),opt.back(),(unsigned long long)sink);
}
'''
with tempfile.TemporaryDirectory(prefix="qsb-hostsha-") as d:
    p=pathlib.Path(d);(p/"check.cpp").write_text(cpp)
    subprocess.run(["g++","-O3","-std=c++17","-I",str(TRACK),str(p/"check.cpp"),"-lcrypto","-o",str(p/"check")],check=True)
    subprocess.run([str(p/"check")],check=True)
