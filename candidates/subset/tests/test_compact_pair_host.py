#!/usr/bin/env python3
"""Execute the source's paired SHA and host class builder against OpenSSL.

Requires a C++17 compiler and OpenSSL development headers. No CUDA or GPU.
This checks SHA semantics and class mapping, not PTX or speculative EC recall.
"""
import os
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
header = (root / "tests/gpu_epochs/window_schedule_shared.cuh").read_text()
sha = (root / "GPUHash.h").read_text()


def function(text, name):
    start = text.index(name)
    start = text.rfind("\n", 0, start) + 1
    opening = text.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


macros = []
for name in ["ROR", "S0", "S1", "s0", "s1", "Maj", "Ch", "S2Round"]:
    match = re.search(r"^#define " + name + r"\(", sha, re.M)
    if not match:
        raise RuntimeError(name)
    lines = sha[match.start():].splitlines()
    definition = [lines[0]]
    while definition[-1].endswith("\\"):
        definition.append(lines[len(definition)])
    macros.append("\n".join(definition))
k = re.search(r"uint32_t K\[\] = \{.*?\};", sha, re.S).group()
prefix = header[:header.index("/* First-block states")]
pair = function(header, "qsb_scheduled_window_hash_pair(")
source = r'''
#include <openssl/sha.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <random>
#define __device__
#define __constant__
#define __forceinline__ inline
#define QSB_SE_WINDOWS 128
#define QSB_SE_PER_EPOCH 128
#define QSB_950_PACK 1
#define QSB_PAIR_SHA_UNROLL_CONST 0
#define QSB_PAIR_SHA_UNROLL_CONST_INNER 0
struct alignas(16) uint4 { uint32_t x,y,z,w; };
const int cudaSuccess=0;
template<class T> int cudaMemcpyToSymbol(T& d,const void* s,size_t n) { memcpy(&d,s,n); return 0; }
template<class T> int cudaMemcpyFromSymbol(void* d,const T& s,size_t n) { memcpy(d,&s,n); return 0; }
uint32_t qsb_host_rotr(uint32_t x,int n) { return (x>>n)|(x<<(32-n)); }
uint32_t QSB_CONST_SCHEDULE[4][64];
''' + k + "\n" + "\n".join(macros) + "\n" + prefix + "\n" + pair + r'''
void require(bool ok,const char* message) { if(!ok) { fprintf(stderr,"FAIL: %s\n",message); exit(1); } }
void expand(const uint8_t *bytes,uint32_t *w) {
    for(int i=0;i<16;i++) w[i]=(uint32_t(bytes[4*i])<<24)|(uint32_t(bytes[4*i+1])<<16)|(uint32_t(bytes[4*i+2])<<8)|bytes[4*i+3];
    for(int i=16;i<64;i++) {
        const uint32_t a=w[i-15],b=w[i-2];
        w[i]=w[i-16]+(qsb_host_rotr(a,7)^qsb_host_rotr(a,18)^(a>>3))+w[i-7]+(qsb_host_rotr(b,17)^qsb_host_rotr(b,19)^(b>>10));
    }
    for(int i=0;i<64;i++) w[i]+=K[i];
}
int main() {
    std::mt19937 rng(0x24c0ffee);
    uint8_t blocks[128][64],suffix[4][64];
    alignas(16) uint32_t firstA[16*8],firstB[16*8];
    // Execute the actual host builder on 128 valid lane windows, then compare
    // every uploaded block and packed record with direct byte reconstruction.
    uint8_t rows[1500],windows[128][3]; uint32_t constant[5];
    for(auto& x:rows) x=rng(); for(auto& x:constant) x=rng();
    int lane=0;
    while(lane<128) for(int a=143;a<148 && lane<128;a++)
      for(int b=a+1;b<149 && lane<128;b++) for(int c=b+1;c<150 && lane<128;c++) {
        windows[lane][0]=a;windows[lane][1]=b;windows[lane++][2]=c;
      }
    require(qsb_prepare_window_schedule(rows,windows,constant)==0,"host prepare");
    for(int l=0;l<128;l++) {
        require(QSB_LANE_CLASS[l]==((QSB_FIRST_CLASS[l]<<16)|QSB_WINDOW_CLASS[l]),"packed mapping");
        uint8_t bytes[128]={}; int pos=8;
        for(int i=137;i<150;i++) if(i!=windows[l][0] && i!=windows[l][1] && i!=windows[l][2]) {memcpy(bytes+pos,rows+10*i,10);pos+=10;}
        for(int j=0;j<5;j++) for(int b=0;b<4;b++) bytes[pos++]=constant[j]>>(24-8*b);
        uint32_t words[64];expand(bytes+64,words);
        for(int j=0;j<64;j++) require(QSB_WINDOW_SECOND[j][QSB_WINDOW_CLASS[l]]==words[j],"host schedule");
        for(int j=0;j<14;j++) {
            const int p=4*(j+2);
            uint32_t word=(uint32_t(bytes[p])<<24)|(uint32_t(bytes[p+1])<<16)|(uint32_t(bytes[p+2])<<8)|bytes[p+3];
            require(QSB_FIRST_UNIQUE[j][QSB_FIRST_CLASS[l]]==word,"host first class");
        }
    }
    // All 16*128 admissible class pairs are exercised, then randomized again
    // with refreshed independent inputs. Arbitrary states test feed-forward.
    for(int trial=0;trial<10000;trial++) {
        if(trial%2048==0) {
            for(auto& x:firstA) x=rng();for(auto& x:firstB) x=rng();
            for(int s=0;s<128;s++) {
                for(auto& x:blocks[s]) x=rng();
                uint32_t w[64];expand(blocks[s],w);
                for(int j=0;j<64;j++) QSB_WINDOW_SECOND[j][s]=w[j];
            }
            for(int b=0;b<4;b++) {for(auto& x:suffix[b]) x=rng();expand(suffix[b],QSB_CONST_SCHEDULE[b]);}
        }
        int fs=(trial/128)%16,ss=trial%128,l=trial%128;
        QSB_LANE_CLASS[l]=(fs<<16)|ss;
        require((QSB_LANE_CLASS[l]>>16)==unsigned(fs) && (QSB_LANE_CLASS[l]&65535)==unsigned(ss),"packing roundtrip");
        uint32_t a[8],b[8];
        qsb_scheduled_window_hash_pair(a,b,l,firstA,firstB);
        SHA256_CTX ca={},cb={};
        memcpy(ca.h,firstA+fs*8,32);memcpy(cb.h,firstB+fs*8,32);
        SHA256_Transform(&ca,blocks[ss]);SHA256_Transform(&cb,blocks[ss]);
        for(int j=0;j<4;j++){SHA256_Transform(&ca,suffix[j]);SHA256_Transform(&cb,suffix[j]);}
        require(memcmp(a,ca.h,32)==0 && memcmp(b,cb.h,32)==0,"paired SHA versus OpenSSL");
    }
    puts("PASS: actual host schedule for 128 lanes; 10000 paired SHA cases (20000 states) versus OpenSSL; all 2048 packed class pairs");
}
'''
with tempfile.TemporaryDirectory(prefix="qsb-pair-test-") as directory:
    path = Path(directory)
    (path / "test.cpp").write_text(source)
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-fno-strict-aliasing", "-Wno-deprecated-declarations", str(path / "test.cpp"), "-lcrypto", "-o", str(path / "test")], check=True)
    subprocess.run([str(path / "test")], check=True)
