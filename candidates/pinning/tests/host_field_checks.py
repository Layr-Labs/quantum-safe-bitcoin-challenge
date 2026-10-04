#!/usr/bin/env python3
"""Compile production host helpers and compare against independent integer arithmetic.
Only the arithmetic prefix is extracted, avoiding CUDA/device dependencies.
Run: python3 candidates/pinning/tests/host_field_checks.py
Requires g++, OpenSSL development headers, AVX2; IFMA checks run only on supported CPUs.
"""
from pathlib import Path
import subprocess, tempfile, os
root = Path(__file__).resolve().parents[1]
ifma = (root / 'cpu_cogrind3_ifma.h').read_text()
ifma = ifma[ifma.index('namespace v4i {'):ifma.index('/* r = a + 2p - b')] + '\n}\n'
source = r'''
#include <cstdint>
#include <immintrin.h>
#include <random>
#include <iostream>
#include <chrono>
#include <cassert>
#include <openssl/bn.h>
''' + ifma + r'''
BIGNUM* P;
BN_CTX* ctx;
BIGNUM* words(const uint64_t* limbs,int n,int bits){
    BIGNUM* x=BN_new();BN_zero(x);
    for(int k=n-1;k>=0;--k){assert(BN_lshift(x,x,bits));assert(BN_add_word(x,limbs[k]));}
    return x;
}
__attribute__((target("avx2,avx512f,avx512vl,avx512ifma"))) void check_square() {
    std::mt19937_64 rng(1891);
    unsigned checked=0;
    for(unsigned i=0;i<65536;++i) {
        v4i::vfe a, squared, generic;
        for(int k=0;k<5;++k)for(int l=0;l<4;++l)
            a.n[k][l]=rng()&((1ULL<<(k==4 ? 49:52))-1);
        if(i<4)for(int k=0;k<5;++k)for(int l=0;l<4;++l)
            a.n[k][l]=(i==0?0:i==1?1:i==2?((1ULL<<(k==4?49:52))-1):((1ULL<<(k==4?48:51))+l));
        v4i::fsqr(&squared,&a); v4i::fmul(&generic,&a,&a);
        v4i::vfe alias=a;v4i::fsqr(&alias,&alias);
        for(int l=0;l<4;++l) {
            uint64_t al[5],sl[5];for(int k=0;k<5;++k){al[k]=a.n[k][l];sl[k]=squared.n[k][l];}
            BIGNUM* x=words(al,5,52);BIGNUM* y=words(sl,5,52);BIGNUM* expected=BN_new();
            assert(BN_mod_sqr(expected,x,P,ctx));assert(BN_nnmod(y,y,P,ctx));assert(BN_cmp(y,expected)==0);
            BN_free(x);BN_free(y);BN_free(expected);
            for(int k=0;k<5;++k){assert(squared.n[k][l]==generic.n[k][l]);assert(alias.n[k][l]==squared.n[k][l]);}
            ++checked;
        }
    }
    std::cout<<"IFMA square: "<<checked<<" lanes, integer oracle, generic multiply and in-place alias agree\n";
}
__attribute__((target("avx2,avx512f,avx512vl,avx512ifma"),noinline))
void square_step(v4i::vfe* r,const v4i::vfe* a){v4i::fsqr(r,a);}
__attribute__((target("avx2,avx512f,avx512vl,avx512ifma"),noinline))
void multiply_step(v4i::vfe* r,const v4i::vfe* a){v4i::fmul(r,a,a);}
__attribute__((target("avx2,avx512f,avx512vl,avx512ifma")))
void bench_square() {
    for(int rep=0;rep<6;++rep){
        v4i::vfe x;for(int k=0;k<5;++k)x.n[k]=v4i::vs1(k+1);
        auto start=std::chrono::steady_clock::now();
        for(int i=0;i<2000000;++i){if(rep%2)square_step(&x,&x);else multiply_step(&x,&x);}
        double ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/2000000;
        std::cout<<(rep%2?"symmetric_square":"generic_square")<<" ns/op="<<ns<<" checksum="<<x.n[0][0]<<"\n";
    }
}
int main(){
    ctx=BN_CTX_new();assert(BN_hex2bn(&P,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"));
    __builtin_cpu_init();
    if(__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512vl")&&__builtin_cpu_supports("avx512ifma")){
        check_square();bench_square();
    }else std::cout<<"IFMA unavailable: square test skipped\n";
}
'''
with tempfile.TemporaryDirectory(prefix='qsb-host-fields-') as directory:
    p=Path(directory);(p/'test.cpp').write_text(source)
    subprocess.run(['g++','-std=c++17','-O3','-I',str(root),str(p/'test.cpp'),'-o',str(p/'test'),'-lcrypto'],check=True)
    # Pin one available CPU to limit migration in the diagnostic microbenchmark.
    if hasattr(os,'sched_getaffinity'):
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    subprocess.run([str(p/'test')],check=True)
