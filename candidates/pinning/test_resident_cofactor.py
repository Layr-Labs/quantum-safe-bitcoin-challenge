#!/usr/bin/env python3
"""Execute the actual collective source with CPU warp/barrier emulation.

A deliberately non-associative, non-commutative four-word operation detects
changes to operand order and parenthesization. This tests routing, not CUDA
field arithmetic, racecheck, or device performance.
"""
from pathlib import Path
import os
import subprocess
import tempfile

SOURCE = r'''
#include <array>
#include <barrier>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>
#include <atomic>
#include <cstdlib>
#define __device__
#define __forceinline__ inline
#define QSB_TREE_TOP2 1
struct Index { int x; };
thread_local Index threadIdx;
Index blockIdx{0};
std::barrier<> *block_barrier;
std::array<std::unique_ptr<std::barrier<>>,8> warp_barriers;
uint64_t exchange_words[256];
std::atomic<unsigned> operations;
void __syncthreads(){block_barrier->arrive_and_wait();}
void __syncwarp(){warp_barriers[threadIdx.x/32]->arrive_and_wait();}
uint64_t __shfl_sync(unsigned mask,uint64_t value,int owner){
    if(mask!=0xffffffffu || owner<0 || owner>=32)std::abort();
    exchange_words[threadIdx.x]=value;
    __syncwarp();
    auto out=exchange_words[(threadIdx.x/32)*32+owner];
    __syncwarp();
    return out;
}
void Load256(uint64_t *r,const uint64_t *a){for(int k=0;k<4;k++)r[k]=a[k];}
void qsb_field_mul_sc(uint64_t *r,const uint64_t *a,const uint64_t *b){
    uint64_t out[4];
    for(int k=0;k<4;k++)
        out[k]=std::rotl(a[k]+0x9e3779b97f4a7c15ULL,11+7*k)
               ^ ((b[k]^a[(k+1)%4])*0xd6e8feb86659fd93ULL);
    for(int k=0;k<4;k++)r[k]=out[k];
    r[4]=0;
    ++operations;
}
#include "cofactor_checkpoint.h"
template<int N> void run(unsigned seed,int active){
    uint64_t products[4][2*N]={},excluded[4][N]={},roots[4]={};
    uint64_t values[N][5];
    uint64_t rng=seed+1;
    for(int i=0;i<N;i++){
        for(int k=0;k<4;k++){
            rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;
            values[i][k]=i<active ? rng : uint64_t(k==0);
        }
        values[i][4]=0;
    }
    std::barrier<> bb(N);block_barrier=&bb;
    for(int i=0;i<N/32;i++)warp_barriers[i]=std::make_unique<std::barrier<>>(32);
    operations=0;
    std::vector<std::thread> threads;
    for(int i=0;i<N;i++)threads.emplace_back([&,i]{
        threadIdx.x=i;
        qsb_cofactor_prepare<N>(values[i],roots,products,excluded);
    });
    for(auto &t:threads)t.join();
    printf("N=%d seed=%u active=%d operations=%u\n",N,seed,active,operations.load());
    for(auto x:roots)printf("%016lx",x);
    puts("");
    for(auto &v:values){for(auto x:v)printf("%016lx",x);puts("");}
}
int main(){
    for(unsigned seed=0;seed<3;seed++){
        run<64>(seed,seed==0?0:seed==1?31:64);
        run<128>(seed,seed==0?1:seed==1?93:128);
        run<256>(seed,seed==0?255:seed==1?129:256);
    }
}
'''


def main():
    here = Path(__file__).resolve().parent
    outputs = []
    with tempfile.TemporaryDirectory(prefix="pinning-resident-") as tmp:
        tmp = Path(tmp)
        source = tmp / "audit.cpp"
        source.write_text(SOURCE)
        for enabled in (0, 1):
            exe = tmp / f"audit-{enabled}"
            subprocess.run([
                "g++", "-std=c++20", "-O1", "-pthread",
                "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                f"-DQSB_TOP32_RESIDENT={enabled}", "-I", str(here),
                str(source), "-o", str(exe),
            ], check=True)
            env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0")
            outputs.append(subprocess.check_output([str(exe)], env=env, timeout=90))
        assert outputs[0] == outputs[1], "routing, operand order or product count differs"
    cases = [line for line in outputs[0].decode().splitlines() if line.startswith("N=")]
    print("PASS: actual header, 9 full trees, 1,344 leaves, roots and operation counts identical")
    print("PASS: AddressSanitizer and UndefinedBehaviorSanitizer")
    print("\n".join(cases))


if __name__ == "__main__":
    main()
