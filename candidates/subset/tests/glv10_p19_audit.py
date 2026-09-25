#!/usr/bin/env python3
"""Compile the actual portable recoders and audit sparse OpenSSL point sums.

No GPU, large table, search, or publication is involved. The scalar oracle
uses independent Python integers. Sparse table points and the final expected
point use cryptography's OpenSSL secp256k1 implementation; point addition is
independently implemented below. CUDA point arithmetic is not tested here.
"""
import argparse
import ctypes
import os
from pathlib import Path
import random
import subprocess
import tempfile

N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
P = (1 << 256) - (1 << 32) - 977
BOUND = 0xA2A8918CA85BAFE22016D0B917E4DD77
A1 = 0x3086D221A7D46BCDE86C90E49284EB15
B1 = 0xE4437ED6010E88286F547FA90ABFE4C3
A2 = A1 + B1
LAMBDA = 0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72
BETA = 0x7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE
MASK64 = (1 << 64) - 1


def block(source, begin, end):
    return source.split(begin, 1)[1].split(end, 1)[0]


def function(source, name, start=0):
    position = source.index(name + "(", start)
    beginning = source.rfind("\n", 0, position) + 1
    brace = source.index("{", position)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[beginning:end]


def compile_source(directory, mode, compiler, scalar_source, tree_source):
    geometry = tree_source[tree_source.index("#if QSB_GLV10_P19\n/* Shared P19 geometry"):
                           tree_source.index("#elif ZLAB_T14", tree_source.index("#include \"../../GLVScalar.cuh\""))]
    portable = block(scalar_source, "// BEGIN QSB_BIGTBL_HOST_EXACT", "// END QSB_BIGTBL_HOST_EXACT")
    walker = block(tree_source, "// BEGIN QSB_S3_HOST_EXACT", "// END QSB_S3_HOST_EXACT")
    check_start = tree_source.index("/* P19 checks every field boundary") if mode else tree_source.index("/* Host self-check of the 11-term walker")
    check = function(tree_source, "qsb_s3_selfcheck", check_start)
    table_scalar = function(tree_source, "gt_table_scalar")
    # This small host builder scalar fits in 128 bits for every legal record.
    # Compiling its original source avoids a second manually copied bias.
    wrapper = f"""
#include <stdint.h>
#define __host__
#define __device__
#define __constant__
#define __forceinline__ inline
#define QSB_GLV10_P19 {mode}
#define QSB_GLV11 1
#define QSB_GLV11_P18 1
{portable}
{geometry}
{walker}
{check}
typedef unsigned __int128 BIGNUM;
typedef uint64_t BN_ULONG;
static void BN_set_word(BIGNUM *r,uint64_t x){{*r=x;}}
static void BN_one(BIGNUM *r){{*r=1;}}
static void BN_lshift(BIGNUM *r,const BIGNUM *a,int n){{*r=*a<<n;}}
static void BN_sub_word(BIGNUM *r,uint64_t a){{*r-=a;}}
static void BN_add_word(BIGNUM *r,uint64_t a){{*r+=a;}}
static void BN_mul_word(BIGNUM *r,uint64_t a){{*r*=a;}}
{table_scalar}
extern "C" int audit_selfcheck(){{return qsb_s3_selfcheck();}}
extern "C" unsigned audit_geometry(unsigned *out){{
 for(int c=0;c<GT_CHUNKS;c++){{out[3*c]=gt_entries(c);out[3*c+1]=gt_offset(c);out[3*c+2]=gt_shift(c);}}
 out[3*GT_CHUNKS]=GT_TOTAL_ENTRIES;out[3*GT_CHUNKS+1]=GT_DENSE_ENTRIES;
 return GT_CHUNKS;
}}
extern "C" void audit_codes(const uint64_t *mag,unsigned signs,uint32_t *out){{
 qsb_s3_walker w;qsb_s3_begin(w,mag,signs&1u,mag+2,signs>>1);
 for(int t=0;t<GT_GLV_TERMS;t++)out[t]=qsb_s3_code(w,t,QSB_S3_DESC[t]);
}}
extern "C" void audit_scalar(int bank,unsigned index,uint64_t *out){{
 BIGNUM scalar;gt_table_scalar(&scalar,bank,index);
 out[0]=(uint64_t)scalar;out[1]=(uint64_t)(scalar>>64);
}}
"""
    # Independently compile the actual host ladder bias construction too.
    bias_line = next(line.strip() for line in tree_source.splitlines()
                     if "BN_set_word(bias,170559770)" in line)
    wrapper += f"""
extern "C" void audit_ladder_bias(uint64_t *out){{
 BIGNUM value,*bias=&value;{bias_line}
 out[0]=(uint64_t)value;out[1]=(uint64_t)(value>>64);
}}
"""
    cpp, library = directory / f"p19_{mode}.cpp", directory / f"p19_{mode}.so"
    cpp.write_text(wrapper)
    subprocess.run([compiler, "-std=c++17", "-O2", "-shared", "-fPIC", str(cpp), "-o", str(library)], check=True)
    loaded = ctypes.CDLL(str(library))
    loaded.audit_geometry.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    loaded.audit_geometry.restype = ctypes.c_uint32
    loaded.audit_codes.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,
                                   ctypes.POINTER(ctypes.c_uint32)]
    loaded.audit_scalar.argtypes = [ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
    loaded.audit_ladder_bias.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    return loaded


def split(k):
    k %= N
    g1 = ((1 << 384) * A1 + N // 2) // N
    g2 = ((1 << 384) * B1 + N // 2) // N
    c1 = (k * g1 + (1 << 383)) >> 384
    c2 = (k * g2 + (1 << 383)) >> 384
    r1, r2 = k - c1 * A1 - c2 * A2, c1 * B1 - c2 * A1
    assert (r1 + LAMBDA * r2 - k) % N == 0
    assert abs(r1) < BOUND and abs(r2) < BOUND
    return r1, r2


def add(left, right):
    if left is None:
        return right
    if right is None:
        return left
    x1, y1 = left
    x2, y2 = right
    if x1 == x2:
        if (y1 + y2) % P == 0:
            return None
        slope = 3 * x1 * x1 * pow(2 * y1, -1, P) % P
    else:
        slope = (y2 - y1) * pow((x2 - x1) % P, -1, P) % P
    x3 = (slope * slope - x1 - x2) % P
    return x3, (slope * (x1 - x3) - y1) % P


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random", type=int, default=20000)
    parser.add_argument("--curve-cases", type=int, default=192)
    parser.add_argument("--compiler", default=os.environ.get("CXX", "c++"))
    args = parser.parse_args()
    if not 0 <= args.random <= 1000000 or not 1 <= args.curve_cases <= 10000:
        parser.error("invalid bounded audit size")
    from cryptography.hazmat.primitives.asymmetric import ec

    def openssl_point(scalar):
        scalar %= N
        if scalar == 0:
            return None
        numbers = ec.derive_private_key(scalar, ec.SECP256K1()).public_key().public_numbers()
        return numbers.x, numbers.y

    root = Path(__file__).resolve().parent.parent
    scalar_source = (root / "GLVScalar.cuh").read_text()
    tree_source = (root / "tests/gpu_epochs/tree.cu").read_text()
    assert "qsb_s3_load(gTable, code, true, x1, y1)" in tree_source
    assert "if(requested_persist>hot_bytes) requested_persist=hot_bytes;" in tree_source
    rng = random.Random(0x503139474C563130)
    with tempfile.TemporaryDirectory(prefix="qsb-p19-audit-") as temporary:
        libraries = [compile_source(Path(temporary), mode, args.compiler, scalar_source, tree_source)
                     for mode in (0, 1)]
        for library in libraries:
            assert library.audit_selfcheck() == 1
        lib = libraries[1]
        geometry = (ctypes.c_uint32 * 26)()
        assert lib.audit_geometry(geometry) == 5
        entries = [524288, 67108864, 67108864, 67108864, 85279885]
        offsets = [0, 524288, 67633152, 134742016, 201850880]
        shifts = [0, 19, 46, 73, 100]
        assert list(geometry[:15]) == [value for row in zip(entries, offsets, shifts) for value in row]
        assert list(geometry[15:17]) == [287130765, 524288]
        donor_geometry = (ctypes.c_uint32 * 26)()
        assert libraries[0].audit_geometry(donor_geometry) == 8
        assert list(donor_geometry[24:26]) == [354501773, 786432]
        words = (ctypes.c_uint64 * 2)()
        lib.audit_ladder_bias(words)
        bias = (170559770 << 99) - (1 << 18)
        assert words[0] + (words[1] << 64) == bias

        def decode(r1, r2):
            magnitudes = [abs(r1), abs(r2)]
            raw = (ctypes.c_uint64 * 4)(*[value for m in magnitudes for value in (m & MASK64, m >> 64)])
            codes = (ctypes.c_uint32 * 10)()
            lib.audit_codes(raw, int(r1 < 0) | (int(r2 < 0) << 1), codes)
            terms = []
            for t, code in enumerate(codes):
                bank = t % 5
                index = (code & 0x7FFFFFFF) - offsets[bank]
                assert 0 <= index < entries[bank]
                lib.audit_scalar(bank, index, words)
                scalar = words[0] + (words[1] << 64)
                expected = bias + index if bank == 0 else (2 * index + 1) << (shifts[bank] - 1)
                assert scalar == expected
                terms.append(-scalar if code >> 31 else scalar)
            assert sum(terms[:5]) == r2
            assert sum(terms[5:]) == r1
            return terms

        edges = {0, 1, BOUND - 2, BOUND - 1}
        for bit in range(128):
            for delta in (-1, 0, 1):
                value = (1 << bit) + delta
                if 0 <= value < BOUND:
                    edges.add(value)
        recodings = 0
        for m in sorted(edges):
            for signs in range(4):
                decode(-m if signs & 1 else m,
                       -(BOUND - 1 - m) if signs & 2 else BOUND - 1 - m)
                recodings += 1
        scalar_edges = [0, 1, N - 1, N, N + 1, (1 << 256) - 1]
        scalar_edges += [(1 << bit) + delta for bit in range(256) for delta in (-1, 0, 1)]
        scalars = scalar_edges + [rng.getrandbits(256) for _ in range(args.random)]
        for k in scalars:
            decode(*split(k))
            recodings += 1

        builder_records = 0
        for bank in range(5):
            indices = {0, 1, 2047, 2048, 4095, 4096, entries[bank] - 2, entries[bank] - 1}
            for boundary in (1 << 23, 1 << 24, 3 << 23):
                indices.update((boundary - 1, boundary, boundary + 1))
            for index in sorted(i for i in indices if 0 <= i < entries[bank]):
                lib.audit_scalar(bank, index, words)
                expected_scalar = words[0] + (words[1] << 64)
                multiplier = index if bank == 0 else 2 * index + 1
                lo, hi, high2 = multiplier & 4095, (multiplier >> 12) & 4095, multiplier >> 24
                assert high2 < 16
                step = 1 if bank == 0 else 1 << (shifts[bank] - 1)
                ladder_scalars = [(bias if bank == 0 else 0) + lo * step,
                                  hi * (1 << 12) * step, high2 * (1 << 24) * step]
                assert sum(ladder_scalars) == expected_scalar
                base = rng.randrange(1, N)
                built = None
                for ladder_scalar in ladder_scalars:
                    built = add(built, openssl_point(ladder_scalar * base))
                assert built == openssl_point(expected_scalar * base)
                builder_records += 1

        # Sparse points exercise each actual code and host table coefficient,
        # both runtime base and isomorphic coordinate scalings, including zero.
        curve_inputs = scalar_edges[:6] + [rng.getrandbits(256) for _ in range(max(0, args.curve_cases - 6))]
        for case, k in enumerate(curve_inputs[:args.curve_cases]):
            base = 1 if case % 3 == 0 else rng.randrange(1, N)
            u = 1 if case % 3 == 0 else rng.randrange(1, P)
            u2, u3 = u * u % P, pow(u, 3, P)
            terms = decode(*split(k))
            result = None
            for t, scalar in enumerate(terms):
                if t == 5 and result is not None:
                    result = (result[0] * BETA % P, result[1])
                point = openssl_point(scalar * base)
                if point is not None:
                    point = point[0] * u2 % P, point[1] * u3 % P
                result = add(result, point)
            expected = openssl_point((k % N) * base)
            if expected is not None:
                expected = expected[0] * u2 % P, expected[1] * u3 % P
            assert result == expected, f"sparse OpenSSL chain mismatch at {case}"
        print(f"PASS: donor/P19 compiled startup checks; {recodings} exact recodings; "
              f"{builder_records} sparse ladder records; "
              f"{min(args.curve_cases, len(curve_inputs))} sparse OpenSSL chains; "
              "table=18376368960 bytes; hot=33554432 bytes")


if __name__ == "__main__":
    main()
