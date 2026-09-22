#!/usr/bin/env python3
"""Host-only differential test of the actual candidate SHA function.
No CUDA compilation, device execution, EC verification, or throughput claim.
SPDX-License-Identifier: GPL-3.0-only
"""
import ctypes
import hashlib
import json
from pathlib import Path
import random
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def extract_function(text, name):
    start = text.index('__device__ __forceinline__ void ' + name + '(')
    brace = text.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def main():
    src = (HERE / 'pinning.cu').read_text()
    fun = extract_function(src, '_SHA256TransformPubkey33')
    assert fun.count('QSB_SHA_INTERLEAVED_16(') == 2
    macros = (HERE / 'GPUHash.h').read_text().split('//Take the last 8 bytes')[0]
    header = (HERE / 'sha_schedule_interleaved.cuh').read_text()
    baseline = fun.replace('QSB_SHA_INTERLEAVED_16(32);', 'WMIX(); SHA256_RND(32);').replace(
        'QSB_SHA_INTERLEAVED_16(48);', 'WMIX(); SHA256_RND(48);').replace(
        '_SHA256TransformPubkey33', 'baseline_pubkey33')
    harness = '''#include <cstdint>
#define __device__
#define __constant__
#define __forceinline__ inline
''' + macros + header + fun + baseline + '''
extern "C" void candidate(uint32_t *o, const uint32_t *m) { _SHA256TransformPubkey33(o,m); }
extern "C" void baseline(uint32_t *o, const uint32_t *m) { baseline_pubkey33(o,m); }
'''
    rng = random.Random(0x515342)
    # Both compressed-key prefixes, constant patterns, every one-hot x bit,
    # and random 33-byte strings (including non-key prefixes).
    vectors = [bytes([p]) + bytes([b])*32 for p in (2, 3) for b in (0, 1, 127, 128, 255)]
    vectors += [bytes([p]) + (1 << bit).to_bytes(32, 'big') for p in (2, 3) for bit in range(256)]
    vectors += [bytes([p]) + rng.randbytes(32) for p in (2, 3) for _ in range(5000)]
    vectors += [rng.randbytes(33) for _ in range(1000)]
    u32 = ctypes.c_uint32
    with tempfile.TemporaryDirectory(prefix='qsb-sha-host-') as tmp:
        cpp, so = Path(tmp) / 'test.cpp', Path(tmp) / 'test.so'
        cpp.write_text(harness)
        cmd = ['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Wno-unknown-pragmas', '-shared', '-fPIC', str(cpp), '-o', str(so)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(proc.stderr)
        lib = ctypes.CDLL(str(so))
        for name in ('candidate', 'baseline'):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.POINTER(u32), ctypes.POINTER(u32)]
            fn.restype = None
        for index, message in enumerate(vectors):
            words = [int.from_bytes((message + b'\x80\x00\x00')[i:i+4], 'big') for i in range(0, 36, 4)]
            expected = hashlib.sha256(message).digest()
            for name in ('candidate', 'baseline'):
                inp, out = (u32 * 9)(*words), (u32 * 8)()
                getattr(lib, name)(out, inp)
                digest = b''.join(int(x).to_bytes(4, 'big') for x in out)
                assert digest == expected, (index, name)
            alias = (u32 * 9)(*words)
            lib.candidate(alias, alias)
            assert b''.join(int(x).to_bytes(4, 'big') for x in alias[:8]) == expected, (index, 'alias')
    print(json.dumps({
        'test': 'actual candidate function extracted and compiled as host C++',
        'vectors': len(vectors), 'digest_comparisons': len(vectors)*3,
        'candidate_vs_hashlib': 'pass', 'block_schedule_vs_hashlib': 'pass',
        'in_place_alias': 'pass', 'compiler': subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
        'compiler_stderr': proc.stderr,
        'function_sha256': hashlib.sha256(fun.encode()).hexdigest(),
        'cuda_compiled': False, 'gpu_executed': False,
        'official_score': None, 'speedup': None,
    }, indent=2))


if __name__ == '__main__':
    main()
