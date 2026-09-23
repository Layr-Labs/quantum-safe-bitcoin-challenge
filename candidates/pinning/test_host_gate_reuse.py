#!/usr/bin/env python3
"""Execute the production OpenSSL gate, both switches, against the CPU verifier.

Low difficulty exercises preferred/alternate/both/neither outcomes. A wrapper
counts actual EC_POINT_mul calls; this is not a CUDA throughput measurement.
SPDX-License-Identifier: GPL-3.0-only
"""
import ctypes
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'harness'))
import crypto as C
import problem as PB


def function(source, name):
    start = source.index('static int ' + name + '(')
    brace = source.index('{', start)
    end, depth = brace + 1, 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


class Params(ctypes.Structure):
    _fields_ = [('midstate', ctypes.c_uint32 * 8),
               ('suffix_len', ctypes.c_uint32),
               ('suffix', ctypes.POINTER(ctypes.c_uint8)),
               ('total_preimage_len', ctypes.c_uint32),
               ('seq_offset', ctypes.c_uint32), ('lt_offset', ctypes.c_uint32),
               ('neg_r_inv', ctypes.c_uint8 * 32),
               ('u2r_x', ctypes.c_uint8 * 32), ('u2r_y', ctypes.c_uint8 * 32)]


CPP = r'''
#include <stdint.h>
#include <string.h>
#include <openssl/sha.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
struct pinning2_params_t {
    uint32_t midstate[8], suffix_len;
    uint8_t *suffix;
    uint32_t total_preimage_len, seq_offset, lt_offset;
    uint8_t neg_r_inv[32], u2r_x[32], u2r_y[32];
};
static int multiplication_count;
static int counted_mul(const EC_GROUP *g, EC_POINT *r, const BIGNUM *n,
                       const EC_POINT *q, const BIGNUM *m, BN_CTX *c) {
    ++multiplication_count;
    return EC_POINT_mul(g, r, n, q, m, c);
}
#define EC_POINT_mul counted_mul
'''

WRAPPER = r'''
#undef EC_POINT_mul
extern "C" int evaluate(const pinning2_params_t *pp, uint32_t seq, uint32_t lt,
                        int ri, int *calls) {
    EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx=BN_CTX_new();
    BIGNUM *order=BN_new(), *nri=BN_lebin2bn(pp->neg_r_inv,32,NULL);
    BIGNUM *rx=BN_lebin2bn(pp->u2r_x,32,NULL), *ry=BN_lebin2bn(pp->u2r_y,32,NULL);
    EC_POINT *r=EC_POINT_new(grp);
    EC_GROUP_get_order(grp,order,ctx);
    int result=-2;
    multiplication_count=0;
    if (EC_POINT_set_affine_coordinates(grp,r,rx,ry,ctx))
        result=qsb_gate_accept(pp,seq,lt,ri,grp,ctx,order,nri,r);
    *calls=multiplication_count;
    EC_POINT_free(r);BN_free(order);BN_free(nri);BN_free(rx);BN_free(ry);
    EC_GROUP_free(grp);BN_CTX_free(ctx);
    return result;
}
'''


def main():
    source = (HERE / 'pinning.cu').read_text()
    legacy = subprocess.run(['git', 'show',
        '1fe5a8e40008befcd917668ea9b1a23c6ee590c4:candidates/pinning/pinning.cu'],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout
    prob = PB.load_problem('pinning')
    suffix = (ctypes.c_uint8 * len(bytes.fromhex(prob['suffix']))).from_buffer_copy(
        bytes.fromhex(prob['suffix']))
    pp = Params()
    pp.midstate[:] = C.sha256_midstate(bytes.fromhex(prob['pin_prefix']))
    pp.suffix, pp.suffix_len = suffix, len(suffix)
    for name in ['total_preimage_len', 'seq_offset', 'lt_offset']:
        setattr(pp, name, prob[name])
    for name in ['neg_r_inv', 'u2r_x', 'u2r_y']:
        getattr(pp, name)[:] = int(prob[name], 16).to_bytes(32, 'little')
    rng = random.Random(0x47415445)
    outcomes = [0] * 4
    cases = calls_control = calls_reuse = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        variants = []
        for name, text, mode in [('legacy', legacy, 0), ('control', source, 0),
                                 ('reuse', source, 1)]:
            cpp = CPP + '\n'.join(function(text, f) for f in [
                'qsb_host_zeros', 'qsb_host_exact_hit', 'qsb_gate_accept']) + WRAPPER
            cpp_path = tmp / (name + '.cpp')
            cpp_path.write_text(cpp)
            lib_path = tmp / (name + '.so')
            subprocess.run(['g++', '-O2', '-shared', '-fPIC',
                '-Wno-deprecated-declarations', '-DQSB_ZEROS_N=2',
                f'-DQSB_HOST_GATE_REUSE={mode}', str(cpp_path), '-lcrypto',
                '-o', str(lib_path)], check=True)
            lib = ctypes.CDLL(str(lib_path))
            lib.evaluate.argtypes = [ctypes.POINTER(Params), ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
            lib.evaluate.restype = ctypes.c_int
            variants.append(lib)
        for _ in range(192):
            seq, lt = rng.getrandbits(32), rng.getrandbits(32)
            valid = [C.leading_zero_bits(PB.candidate_hash(
                prob, {'sequence': seq, 'locktime': lt}, ri)) >= 2 for ri in [0, 1]]
            outcomes[int(valid[0]) + 2 * int(valid[1])] += 1
            for ri in [0, 1]:
                expected = ri if valid[ri] else (1 - ri if valid[1 - ri] else -1)
                counts = []
                for lib in variants:
                    n = ctypes.c_int()
                    actual = lib.evaluate(ctypes.byref(pp), seq, lt, ri, ctypes.byref(n))
                    assert actual == expected, (seq, lt, ri, actual, expected)
                    counts.append(n.value)
                assert counts == [1 if valid[ri] else 2] * 2 + [1], counts
                calls_control += counts[1]
                calls_reuse += counts[2]
                cases += 1
        assert all(outcomes), outcomes
        # Bounds checks reject malformed offsets before hashing or dereferencing.
        for field, value in [('suffix_len', 0), ('suffix_len', 120),
                             ('seq_offset', 0xffffffff), ('lt_offset', 0xffffffff)]:
            bad = Params.from_buffer_copy(pp)
            setattr(bad, field, value)
            for lib in variants[1:]:
                n = ctypes.c_int()
                assert lib.evaluate(ctypes.byref(bad), 0, 0, 0, ctypes.byref(n)) == -1
                assert n.value == 0
        for ri in [-2147483648, -1, 2, 2147483647]:
            for lib in variants[1:]:
                n = ctypes.c_int()
                assert lib.evaluate(ctypes.byref(pp), 0, 0, ri, ctypes.byref(n)) == -1
                assert n.value == 0
    print(json.dumps({'passed': True, 'verifier_comparisons_per_variant': cases,
        'outcomes_neither_zero_one_both': outcomes,
        'generator_multiplications_control': calls_control,
        'generator_multiplications_reuse': calls_reuse, 'gpu_executed': False}, indent=2))


if __name__ == '__main__':
    main()
