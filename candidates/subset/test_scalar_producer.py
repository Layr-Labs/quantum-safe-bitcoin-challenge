#!/usr/bin/env python3
"""Run production paired SHA, producer stores and consumer loads on the CPU.

OpenSSL compression is the independent hash oracle. CUDA launch/memory timing
is not simulated, and this test reports no throughput prediction.
"""
from pathlib import Path
import re
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def function(source, marker):
    start = source.index(marker)
    opening = source.index('{', start)
    depth = 0
    tokens = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|//[^\n]*|/\*.*?\*/|[{}]', re.S)
    for token in tokens.finditer(source, opening):
        if token[0] in ('{', '}'):
            depth += 1 if token[0] == '{' else -1
            if depth == 0:
                return source[start:token.end()]
    raise AssertionError('unclosed function: ' + marker)


def main():
    gpu_hash = (HERE / 'GPUHash.h').read_text()
    pair = (HERE / 'tests/gpu_epochs/pair_shared.cuh').read_text()
    window = (HERE / 'tests/gpu_epochs/window_schedule_shared.cuh').read_text()
    tree = (HERE / 'tests/gpu_epochs/tree.cu').read_text()
    scalar = (HERE / 'subset_sha_flags.h').read_text() + '\n' + (HERE / 'tests/gpu_epochs/scalar_producer.cuh').read_text()
    shim = r'''
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>
#include <openssl/sha.h>
#define __host__
#define __device__
#define __global__
#define __constant__
#define __restrict__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define DEF(x,y) uint32_t x=output[y]
#define QSB_ZEROS_N 24
#define ZLAB_DUAL_EPOCH_SHA 1
#define ZLAB_K2S3M 1
#define QSB_PAIR_SHARED 1
#define QSB_SE_BLOCK 256
#define QSB_PAIR_MUL (2*(QSB_SE_BLOCK/QSB_SE_WINDOWS))
#define QSB_FIRST_SLOTS (QSB_SE_WINDOWS==128?16:64)
#define QSB_PAIR_SHA_UNROLL_WINDOW 1
#define QSB_PAIR_SHA_UNROLL_CONST 1
#define QSB_SHA_UNROLL_CONST 1
struct epoch_desc_t;
struct Index {unsigned x;};
Index blockIdx,threadIdx;
uint32_t QSB_FIRST_CLASS[QSB_SE_WINDOWS], QSB_WINDOW_CLASS[QSB_SE_WINDOWS];
uint32_t QSB_WINDOW_SECOND[64][QSB_SE_WINDOWS], QSB_CONST_SCHEDULE[4][64];
struct QsbPairEpochZ {uint64_t a[4], b[4];};
'''
    source = shim + gpu_hash.split('//Take the last 8 bytes')[0]
    source += function(gpu_hash, '__device__ void _SHA256Transform(') + '\n'
    source += function(tree, 'template<int block> __device__ __forceinline__ void qsb_compress_constant(') + '\n'
    source += function(window, '__device__ __forceinline__ void qsb_scheduled_window_hash(') + '\n'
    source += function(window, '__device__ __forceinline__ void qsb_scheduled_window_hash_pair(') + '\n'
    source += function(pair, '__device__ __forceinline__ void qsb_pair_second_sha_z(') + '\n'
    source += function(pair, '__device__ __forceinline__ QsbPairEpochZ qsb_pair_epoch_z_value(') + '\n'
    source += scalar + "\n#include \"subset_tile_plan.h\"\n"
    source += r'''
uint64_t random_state=0x259758efa6956ULL;
uint32_t random32(){random_state^=random_state<<13;random_state^=random_state>>7;random_state^=random_state<<17;return random_state>>17;}
void expand(const uint32_t *input,uint32_t*out){
    uint32_t w[64];memcpy(w,input,64);
    for(int i=16;i<64;i++)w[i]=w[i-16]+s0(w[i-15])+w[i-7]+s1(w[i-2]);
    for(int i=0;i<64;i++)out[i]=w[i]+K[i];
}
void transform(SHA256_CTX& ctx,const uint32_t*w){
    unsigned char bytes[64];
    for(int i=0;i<16;i++)for(int j=0;j<4;j++)bytes[4*i+j]=w[i]>>(24-8*j);
    SHA256_Transform(&ctx,bytes);
}
int main(){
    uint32_t messages[QSB_SE_WINDOWS][16],constants[4][16];
    for(auto&row:messages)for(auto&v:row)v=random32();
    for(auto&row:constants)for(auto&v:row)v=random32();
    for(int lane=0;lane<QSB_SE_WINDOWS;lane++){
        QSB_FIRST_CLASS[lane]=(lane*7)%8;
        QSB_WINDOW_CLASS[lane]=(lane*11)%QSB_SE_WINDOWS;
        uint32_t expanded[64];expand(messages[lane],expanded);
        for(int r=0;r<64;r++)QSB_WINDOW_SECOND[r][lane]=expanded[r];
    }
    for(int b=0;b<4;b++)expand(constants[b],QSB_CONST_SCHEDULE[b]);
    unsigned batches=0;uint64_t pairs=0,hashes=0;
    for(unsigned epochs: {0u,1u,2u,3u,4u,5u,7u,8u,9u,31u,32u,33u,65u,67u}){
        unsigned blocks=(epochs+QSB_PAIR_MUL-1)/QSB_PAIR_MUL;
        std::vector<uint32_t> first((size_t)epochs*QSB_FIRST_SLOTS*8);
        for(auto&v:first)v=random32();
        const uint64_t canary=0x51a7d63f7eaaa153ULL;
        std::vector<uint64_t> storage((size_t)blocks*8*QSB_SE_BLOCK+2,canary);
        uint64_t*scalars=storage.data()+1;
        std::vector<unsigned char> covered((size_t)epochs*QSB_SE_WINDOWS,0);
        for(blockIdx.x=0;blockIdx.x<blocks;blockIdx.x++)for(threadIdx.x=0;threadIdx.x<256;threadIdx.x++)
            kernel_subset_scalars(first.data(),scalars,epochs);
        assert(storage.front()==canary&&storage.back()==canary);
        const auto paired_storage=storage;
        std::fill(storage.begin(),storage.end(),canary);
        for(blockIdx.x=0;blockIdx.x<4*blocks;blockIdx.x++)for(threadIdx.x=0;threadIdx.x<128;threadIdx.x++)
            kernel_subset_scalars_single(first.data(),scalars,epochs);
        assert(storage==paired_storage);
        // Exercise the production planner, producers, and local-to-batch tag mapping.
        for(unsigned single=0;single<2;single++){
            std::vector<uint64_t> tile_storage((size_t)qsb_subset_tile_capacity(blocks)*8*256+2,canary);
            std::fill(covered.begin(),covered.end(),0);
            for(unsigned first_block=0;first_block<blocks;){
                const auto tile=qsb_subset_next_tile(first_block,blocks,epochs,QSB_PAIR_MUL);
                uint64_t *tile_scalars=tile_storage.data()+1;
                const uint32_t *tile_first=first.data()+(size_t)tile.first_epoch*QSB_FIRST_SLOTS*8;
                for(blockIdx.x=0;blockIdx.x<tile.blocks*(single?4:1);blockIdx.x++)
                    for(threadIdx.x=0;threadIdx.x<(single?128:256);threadIdx.x++){
                        if(single)kernel_subset_scalars_single(tile_first,tile_scalars,tile.epochs);
                        else kernel_subset_scalars(tile_first,tile_scalars,tile.epochs);
                    }
                assert(tile_storage.front()==canary&&tile_storage.back()==canary);
                for(blockIdx.x=0;blockIdx.x<tile.blocks;blockIdx.x++)for(threadIdx.x=0;threadIdx.x<256;threadIdx.x++){
                    const unsigned local_ep=blockIdx.x*QSB_PAIR_MUL+2*(threadIdx.x/QSB_SE_WINDOWS);
                    const unsigned lane=threadIdx.x&(QSB_SE_WINDOWS-1);
                    const auto got=qsb_load_scalar_pair(tile_scalars);
                    for(unsigned which=0;which<2;which++){
                        const bool active=which?qsb_scalar_active<true>(tile.blocks*256,tile.epochs):qsb_scalar_active<false>(tile.blocks*256,tile.epochs);
                        if(!active)continue; // Inactive aliases can select a different safe epoch.
                        const unsigned global_ep=tile.first_epoch+local_ep+which;
                        const size_t identity=(size_t)global_ep*QSB_SE_WINDOWS+lane;
                        assert(identity<covered.size()&&!covered[identity]);covered[identity]=1;
                        const uint64_t *actual=which?got.b:got.a;
                        for(unsigned k=0;k<4;k++)assert(actual[k]==paired_storage[1+qsb_scalar_plane_index(tile.first_block+blockIdx.x,k+4*which,threadIdx.x)]);
                    }
                }
                first_block+=tile.blocks;
            }
            for(auto v:covered)assert(v==1);
        }
        std::fill(covered.begin(),covered.end(),0);
        // Reverse consumer block order to detect accidental cross-block ownership.
        for(unsigned b=blocks;b-->0;){blockIdx.x=b;
          for(threadIdx.x=0;threadIdx.x<256;threadIdx.x++){
            unsigned lane=threadIdx.x&(QSB_SE_WINDOWS-1),half=threadIdx.x/QSB_SE_WINDOWS;
            unsigned e=QSB_PAIR_MUL*b+2*half;
            unsigned selected[2]={e<epochs?e:0,(e+1<epochs)?e+1:(e<epochs?e:0)};
            QsbPairEpochZ got=qsb_load_scalar_pair(scalars);
            for(int which=0;which<2;which++){
                bool active=which?qsb_scalar_active<true>(blocks*256,epochs):qsb_scalar_active<false>(blocks*256,epochs);
                if(active){
                    const size_t identity=(size_t)(e+which)*QSB_SE_WINDOWS+lane;
                    assert(identity<covered.size()&&!covered[identity]);covered[identity]=1;
                }
                SHA256_CTX ctx;SHA256_Init(&ctx);
                auto*initial=first.data()+((size_t)selected[which]*QSB_FIRST_SLOTS+QSB_FIRST_CLASS[lane])*8;
                for(int j=0;j<8;j++)ctx.h[j]=initial[j];
                transform(ctx,messages[QSB_WINDOW_CLASS[lane]]);
                for(int c=0;c<4;c++)transform(ctx,constants[c]);
                unsigned char digest[32],want[32];
                for(int j=0;j<8;j++)for(int k=0;k<4;k++)digest[j*4+k]=ctx.h[j]>>(24-k*8);
                SHA256(digest,32,want);
                const uint64_t*actual=which?got.b:got.a;
                for(int j=0;j<32;j++)assert(((actual[j/8]>>(8*(j%8)))&255)==want[31-j]);
                hashes++;
            }
            pairs++;
          }
        }
        for(auto v:covered)assert(v==1);
        // No output index may exceed the 4 GiB default allocation.
        assert(qsb_scalar_plane_index(262143,7,255)==size_t(262144)*8*256-1);
        batches++;
    }
    printf("{\"windows\":%d,\"tail_batches\":%u,\"pairs\":%llu,\"openssl_hash_comparisons\":%llu,\"mismatches\":0}\n",
           QSB_SE_WINDOWS,batches,(unsigned long long)pairs,(unsigned long long)hashes);
}
'''
    with tempfile.TemporaryDirectory(prefix='qsb-subset-scalar-test-') as tmp:
        path = Path(tmp)
        cpp = path / 'audit.cpp'
        cpp.write_text(source)
        for windows in (128, 256):
            binary = path / f'audit-{windows}'
            cmd = ['clang++', '-std=c++17', '-O2', '-Wno-deprecated-declarations',
                   '-Wno-unknown-pragmas', '-fsanitize=undefined,bounds',
                   f'-DQSB_SE_WINDOWS={windows}', '-DQSB_SUBSET_TILE_BLOCKS=3', '-I'+str(HERE), '-I/opt/homebrew/opt/openssl@3/include',
                   '-L/opt/homebrew/opt/openssl@3/lib', str(cpp), '-lcrypto', '-o', str(binary)]
            subprocess.run(cmd, check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
