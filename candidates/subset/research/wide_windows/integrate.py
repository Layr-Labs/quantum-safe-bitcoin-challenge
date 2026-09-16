#!/usr/bin/env python3
"""Stage a full ten-window subset prototype, preserving the pending entry.

Reads the exact PR60 subset and PR74 pinning sources; writes only this research
directory. The output is an experimental grinder, not a GPU performance claim.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SUBSET = HERE.parent.parent
PINNING = SUBSET.parent / 'pinning'
sys.path.insert(0, str(SUBSET))
from preflight import source_identity, source_files
from check_candidate import function


def main():
    identity = source_identity()
    assert identity['source_fingerprint'] == '44f9d5808add05e204b169194af8893dc91a9c01d31448362b0e01c667c9fd06', 'PR60 control changed'
    pin_record = json.loads((PINNING/'research/wide_windows/submitted-source.json').read_text())
    # Verify the helper donor independently of unrelated research files.
    pin_hashes = {
        'pinning.cu': 'a30d5228ed06d01807d3197275b3998f0b4cd3f91ad103c82fbf1df2d39faf03',
        'wide_geometry.cuh': '985e935e371842eceabe05245b8d28237bfd19e2987c75894c59e389766eabbb',
        'wide_table_kernels.cuh': '811668b4a00b8fb35716392b7c60d79e952c511eeb3f08a5a6c9321b4057c8d2',
        'wide_table_host.cuh': 'ac27e68bbc52f7cb8d9c00faba61f96f75545af0aa22d2a8d0adf4ffd41e8e4b',
    }
    for name, digest in pin_hashes.items():
        assert hashlib.sha256((PINNING/name).read_bytes()).hexdigest() == digest, name
    out = HERE/'candidate'
    control = HERE/'control'
    for root in (out, control):
        root.mkdir(exist_ok=True)
        for src in source_files() + [SUBSET/'COPYING']:
            dst = root/src.relative_to(SUBSET)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if root == control and dst.exists():
                assert dst.read_bytes() == src.read_bytes(), 'Frozen control differs'
            else:
                shutil.copyfile(src, dst)

    tree_path = out/'tests/gpu_epochs/tree.cu'
    s = tree_path.read_text()
    donor = (PINNING/'pinning.cu').read_text()
    s = s.replace('#include <cuda_runtime.h>', '#include <cuda_runtime.h>\n#include <vector>')
    start = s.index('/* ============================================================\n * Signed-digit fixed-base geometry')
    end = s.index('/* n = secp256k1', start)
    s = s[:start] + '''/* Experimental ten-window table: [26x6,25x4], 16 GiB.
 * Runtime base A/2 and all ranked subset enumeration remain unchanged.
 * Derived from our public pinning PR74; no GPU speed claim. */
#include "wide_geometry.cuh"
#define GT_CHUNKS WIDE_CHUNKS
#define GT_TOTAL_ENTRIES ((uint32_t)WIDE_TOTAL_ENTRIES)
#define GT_LO 8192
#define GT_HI 8192
__host__ __device__ __forceinline__ unsigned gt_entries(int c){return wide_entries(c);}
__host__ __device__ __forceinline__ unsigned gt_offset(int c){return wide_offset(c);}
__host__ __device__ __forceinline__ int gt_shift(int c){return wide_shift(c);}

''' + s[end:]
    old = function(s, '__device__ __forceinline__ int32_t gt_recode_step(')
    s = s.replace(old, '''__device__ __forceinline__ int32_t gt_recode_step(uint64_t M[4],int sign,int c){
    return c<GT_CHUNKS-1?wide_step(M,sign,wide_bits(c)):sign*(int32_t)M[0];
}''')
    old = function(s, '__device__ __forceinline__ void gt_recode_signed(')
    s = s.replace(old, '''__device__ __forceinline__ void gt_recode_signed(const uint64_t k[4],int32_t e[GT_CHUNKS]){
    uint64_t M[4];int sign;gt_recode_setup(k,M,&sign);
    for(int c=0;c<GT_CHUNKS;++c)e[c]=gt_recode_step(M,sign,c);
}''')
    # Preserve the existing two-pointer ABI across ranked and generic kernels.
    # The unused second pointer is null; only the single interleaved table is read.
    old = function(s, '__device__ __forceinline__ void gt_load_signed(')
    new = old.replace('size_t off = ((size_t)c * GT_ENTRIES + idx) * 32;',
                      'size_t off = ((size_t)gt_offset(c) + idx) * 64;\n    (void)gTY;')
    new = new.replace('(const ulonglong2 *)(gTY+off)', '(const ulonglong2 *)(gTX+off+32)')
    assert new != old and 'gTY+off' not in new
    s = s.replace(old, new)
    start = s.index('/* Branchless windowed fixed-base multiply')
    end = s.index('__device__ __forceinline__ void gt_digit_idx', start)
    s = s[:start] + '/* Stream signed digits and retain the deferred-Y point chain. */\n' + s[end:]
    start = s.index('/* Accumulate sixteen points')
    end = s.index('__device__ void _FixedBaseSignedXYZZ(', start)
    s = s[:start] + '''/* Ten points: 3M+2S seed, seven 7M+2S deferred adds and 8M+2S
 * final add = 60M+18S, versus PR60's 102M+30S. Recovery is unchanged.
 * This trades the 32 MiB control table for 16 GiB; cache behavior is unmeasured. */
''' + s[end:]
    start = s.index('/* ============================================================\n * Fixed-base table construction')
    end = s.index('/* ============================================================\n * Host code', start)
    s = s[:start] + '#include "wide_table_kernels.cuh"\n\n' + s[end:]
    start = s.index('/* The two ladders the GPU builder needs:')
    end = s.index('/* Digest params loader */', start)
    host = '\n'.join(function(donor, signature) for signature in (
        'static void gt_require(', 'static void gt_build_ladders(', 'static int gt_spot_check('))
    s = s[:start] + host + '\n#include "wide_table_host.cuh"\n\n' + s[end:]
    start = s.index('    size_t gt_sz = (size_t)GT_CHUNKS*GT_ENTRIES*32;')
    end = s.index('    /* Upload params */', start)
    s = s[:start] + '''    // Bound the table plus ranked state, generic combo buffer and startup margin.
    size_t free_bytes=0,total_bytes=0;
    wide_cuda_require(cudaMemGetInfo(&free_bytes,&total_bytes),"query free GPU memory");
    const size_t gt_sz=(size_t)GT_TOTAL_ENTRIES*64;
    const size_t search_reserve=(2ull<<30);
    if(free_bytes<gt_sz+search_reserve){
        fprintf(stderr,"Insufficient free GPU memory for wide subset table and scratch\\n");return 2;
    }
    uint8_t *d_gtX=nullptr,*d_gtY=nullptr;
    wide_cuda_require(cudaMalloc(&d_gtX,gt_sz),"allocate wide subset table");
    wide_build_table(d_gtX,dp.neg_r_inv);

''' + s[end:]
    assert 'GT_ENTRIES' not in s and 'kernel_build_gtable' not in s
    tree_path.write_text(s)
    for name in ('wide_geometry.cuh','wide_table_kernels.cuh','wide_table_host.cuh'):
        shutil.copyfile(PINNING/name, tree_path.parent/name)
    record = {
        'status':'isolated full grinder prototype; not submitted',
        'control':identity,
        'candidate':source_identity(out),
        'donor_submission':'66c031c6-16e4-484e-ad4d-0e7e0ef3f38f',
        'donor_source_sha256':pin_hashes,
        'sources':['https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/60',
                   'https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/74'],
        'limits':'No GPU execution or throughput result. Production PR60 source remains unchanged.'
    }
    (HERE/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'control':identity['source_fingerprint'], 'candidate':record['candidate']['source_fingerprint']}))


if __name__ == '__main__':
    main()
