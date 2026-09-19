#!/usr/bin/env python3
"""Host-only differential test of _SHA256TransformDigest32 interleave.
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
    fun = extract_function(src, '_SHA256TransformDigest32')
    assert fun.count('QSB_SHA_INTERLEAVED_16(') == 2, fun.count('QSB_SHA_INTERLEAVED_16(')
    macros = (HERE / 'GPUHash.h').read_text().split('//Take the last 8 bytes')[0]
    header = (HERE / 'sha_schedule_interleaved.cuh').read_text()
    baseline = fun.replace('QSB_SHA_INTERLEAVED_16(32);', 'WMIX(); SHA256_RND(32);').replace(
        'QSB_SHA_INTERLEAVED_16(48);', 'WMIX(); SHA256_RND(48);').replace(
        '_SHA256TransformDigest32', 'baseline_digest32')
    harness = """#include <cstdint>
#define __device__
#define __constant__
#define __forceinline__ inline
""" + macros + header + fun + baseline + """
extern "C" void candidate(uint32_t *o, const uint32_t *m) { _SHA256TransformDigest32(o,m); }
extern "C" void baseline(uint32_t *o, const uint32_t *m) { baseline_digest32(o,m); }
"""
    rng = random.Random(0xD16357)
    vectors = []
    for b in (0, 1, 127, 128, 255):
        vectors.append(bytes([b])*32)
    for bit in range(256):
        vectors.append((1 << bit).to_bytes(32, 'big'))
    vectors += [rng.randbytes(32) for _ in range(8000)]
    u32 = ctypes.c_uint32
    with tempfile.TemporaryDirectory(prefix='qsb-sha-d32-') as tmp:
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
            words = [int.from_bytes(message[i:i+4], 'big') for i in range(0, 32, 4)]
            expected = hashlib.sha256(message).digest()
            for name in ('candidate', 'baseline'):
                inp, out = (u32 * 8)(*words), (u32 * 8)()
                getattr(lib, name)(out, inp)
                digest = b''.join(int(x).to_bytes(4, 'big') for x in out)
                assert digest == expected, (index, name, digest.hex(), expected.hex())
            alias = (u32 * 8)(*words)
            # Digest32 writes to out[8]; aliasing m into out needs 8 words only for input
            out2 = (u32 * 8)()
            lib.candidate(out2, alias)
            assert b''.join(int(x).to_bytes(4, 'big') for x in out2) == expected
    print(json.dumps({
        'test': 'Digest32 interleaved schedule vs hashlib',
        'vectors': len(vectors), 'digest_comparisons': len(vectors)*3,
        'candidate_vs_hashlib': 'pass', 'block_schedule_vs_hashlib': 'pass',
    }))

if __name__ == '__main__':
    main()
