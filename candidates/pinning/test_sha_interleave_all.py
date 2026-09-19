#!/usr/bin/env python3
"""Host-only differential test for interleaved SHA schedule in all three
sparse pinning transforms (extends test_sha_interleave.py / ff275e4).

Verifies without CUDA/GPU that replacing the two dense WMIX()+SHA256_RND
bursts with QSB_SHA_INTERLEAVED_16(32/48) preserves digests:
- _SHA256TransformPubkey33 vs hashlib.sha256 (33-byte compressed-key blocks)
- _SHA256TransformDigest32 vs hashlib.sha256 (32-byte second-SHA blocks)
- _SHA256TransformFastTail11 candidate vs burst-baseline (midstate continuation)

SPDX-License-Identifier: GPL-3.0-only
"""
import ctypes
import hashlib
import json
import random
import subprocess
import tempfile
from pathlib import Path

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


def baseline_of(fun, new_name, old_name):
    b = fun.replace('QSB_SHA_INTERLEAVED_16(32);', 'WMIX(); SHA256_RND(32);')
    b = b.replace('QSB_SHA_INTERLEAVED_16(48);', 'WMIX(); SHA256_RND(48);')
    return b.replace(old_name, new_name, 1)


def main():
    src = (HERE / 'pinning.cu').read_text()
    fun_tail = extract_function(src, '_SHA256TransformFastTail11')
    fun_dgst = extract_function(src, '_SHA256TransformDigest32')
    fun_pub = extract_function(src, '_SHA256TransformPubkey33')
    for name, fun in (('tail', fun_tail), ('digest', fun_dgst), ('pubkey', fun_pub)):
        assert fun.count('QSB_SHA_INTERLEAVED_16(32);') == 1, name
        assert fun.count('QSB_SHA_INTERLEAVED_16(48);') == 1, name
        assert 'WMIX();' not in fun, name
    macros = (HERE / 'GPUHash.h').read_text().split('//Take the last 8 bytes')[0]
    header = (HERE / 'sha_schedule_interleaved.cuh').read_text()
    base_tail = baseline_of(fun_tail, 'baseline_tail11', '_SHA256TransformFastTail11')
    base_dgst = baseline_of(fun_dgst, 'baseline_digest32', '_SHA256TransformDigest32')
    base_pub = baseline_of(fun_pub, 'baseline_pubkey33', '_SHA256TransformPubkey33')
    harness = '''#include <cstdint>
#define __device__
#define __constant__
#define __forceinline__ inline
''' + macros + header + fun_tail + base_tail + fun_dgst + base_dgst + fun_pub + base_pub + '''
extern "C" void cand_tail(uint32_t *s, uint32_t w0, uint32_t w1, uint32_t w2)
{ _SHA256TransformFastTail11(s, w0, w1, w2); }
extern "C" void base_tail(uint32_t *s, uint32_t w0, uint32_t w1, uint32_t w2)
{ baseline_tail11(s, w0, w1, w2); }
extern "C" void cand_digest(uint32_t *o, const uint32_t *m)
{ _SHA256TransformDigest32(o, m); }
extern "C" void base_digest(uint32_t *o, const uint32_t *m)
{ baseline_digest32(o, m); }
extern "C" void cand_pub(uint32_t *o, const uint32_t *m)
{ _SHA256TransformPubkey33(o, m); }
extern "C" void base_pub(uint32_t *o, const uint32_t *m)
{ baseline_pubkey33(o, m); }
'''
    rng = random.Random(0xC0FFEE)
    # 32-byte messages for Digest32 -> hashlib oracle
    digest_vecs = [bytes(32), bytes([255]) * 32]
    digest_vecs += [bytes([i % 256]) * 32 for i in range(8)]
    digest_vecs += [rng.randbytes(32) for _ in range(5000)]
    # 33-byte messages for Pubkey33 -> hashlib oracle
    pub_vecs = [bytes([p]) + bytes([b]) * 32 for p in (2, 3) for b in (0, 1, 255)]
    pub_vecs += [bytes([p]) + (1 << bit).to_bytes(32, 'big') for p in (2, 3) for bit in range(256)]
    pub_vecs += [bytes([p]) + rng.randbytes(32) for p in (2, 3) for _ in range(2000)]
    # (state, w0, w1, w2) for FastTail11 candidate-vs-baseline
    tail_vecs = []
    for _ in range(5000):
        st = [rng.getrandbits(32) for _ in range(8)]
        w = [rng.getrandbits(32) for _ in range(3)]
        tail_vecs.append((st, w))
    tail_vecs += [([0] * 8, [0, 0, 0]), ([0xFFFFFFFF] * 8, [0xFFFFFFFF] * 3)]
    u32 = ctypes.c_uint32
    with tempfile.TemporaryDirectory(prefix='qsb-sha-all-') as tmp:
        cpp, so = Path(tmp) / 'test.cpp', Path(tmp) / 'test.so'
        cpp.write_text(harness)
        cmd = ['g++', '-std=c++17', '-O2', '-Wall', '-Wextra',
               '-Wno-unknown-pragmas', '-shared', '-fPIC', str(cpp), '-o', str(so)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(proc.stderr)
        lib = ctypes.CDLL(str(so))
        lib.cand_tail.argtypes = [ctypes.POINTER(u32), u32, u32, u32]
        lib.cand_tail.restype = None
        lib.base_tail.argtypes = [ctypes.POINTER(u32), u32, u32, u32]
        lib.base_tail.restype = None
        for name in ('cand_digest', 'base_digest', 'cand_pub', 'base_pub'):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.POINTER(u32), ctypes.POINTER(u32)]
            fn.restype = None
        # Digest32 vs hashlib + baseline
        for msg in digest_vecs:
            words = [int.from_bytes(msg[i:i + 4], 'big') for i in range(0, 32, 4)]
            expected = hashlib.sha256(msg).digest()
            for fn_name in ('cand_digest', 'base_digest'):
                inp, out = (u32 * 8)(*words), (u32 * 8)()
                getattr(lib, fn_name)(out, inp)
                digest = b''.join(int(x).to_bytes(4, 'big') for x in out)
                assert digest == expected, (fn_name, msg.hex())
        # Pubkey33 vs hashlib + baseline
        for msg in pub_vecs:
            words = [int.from_bytes((msg + b'\x80\x00\x00')[i:i + 4], 'big') for i in range(0, 36, 4)]
            expected = hashlib.sha256(msg).digest()
            for fn_name in ('cand_pub', 'base_pub'):
                inp, out = (u32 * 9)(*words), (u32 * 8)()
                getattr(lib, fn_name)(out, inp)
                digest = b''.join(int(x).to_bytes(4, 'big') for x in out)
                assert digest == expected, (fn_name, msg.hex())
        # FastTail11 candidate vs baseline (midstate continuation; no hashlib oracle)
        for st, w in tail_vecs:
            s1, s2 = (u32 * 8)(*st), (u32 * 8)(*st)
            lib.cand_tail(s1, *w)
            lib.base_tail(s2, *w)
            assert list(s1) == list(s2), (st, w)
    print(json.dumps({
        'test': 'interleaved schedule in FastTail11/Digest32/Pubkey33, host C++',
        'digest_vectors': len(digest_vecs),
        'pubkey_vectors': len(pub_vecs),
        'tail_vectors': len(tail_vecs),
        'digest_vs_hashlib': 'pass',
        'pubkey_vs_hashlib': 'pass',
        'tail_candidate_vs_baseline': 'pass',
        'compiler': subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
        'cuda_compiled': False, 'gpu_executed': False,
    }, indent=2))


if __name__ == '__main__':
    main()
