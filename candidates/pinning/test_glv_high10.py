#!/usr/bin/env python3
"""Audit the production GLV coefficient screen against exact big-integer rounding.

The test extracts the compiled C++ device expressions and models CUDA integer
operations with their exact C++ equivalents. It does not execute CUDA or GPU
kernels.
"""
import ctypes
import json
import random
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

source = Path(__file__).with_name("GLVScalar.cuh").read_text()


def extract(name):
    start = source.index("__device__ __forceinline__", source.index(name) - 55)
    brace = source.index("{", start)
    end = brace + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end].replace("__device__", "").replace("__forceinline__", "inline")


constants = []
for name in ("g1", "g2"):
    words = re.search(r"const uint64_t " + name + r"\[4\]=\{([^}]+)\}", source).group(1)
    limbs = [int(value.removesuffix("ULL"), 16) for value in words.split(",")]
    constants.append(sum(value << (64 * index) for index, value in enumerate(limbs)))

B = 1 << 32
error_units = []
for which, reciprocal in enumerate(constants):
    words = [(reciprocal >> (32 * index)) & (B - 1) for index in range(8)]
    assert words[7] + words[6] < B
    omitted = sum(
        (B - 1) * words[j] * B ** (i + j)
        for i in range(8)
        for j in range(8)
        if i + j < 10
    )
    omitted += 5 * (B - 1) * B**10
    error_units.append((omitted + B**11 - 1) // B**11)
assert error_units == [9, 8]

cpp = r"""
#include <stdint.h>
struct ulonglong2 { uint64_t x,y; };
static uint64_t fallback_calls = 0;
static uint64_t gvalues[2][4];
inline void q9_high15_add(uint64_t *acc,uint32_t *overflow,uint64_t product) {
    uint64_t before=*acc;
    *acc=before+product;
    *overflow+=(uint32_t)(*acc<before);
}
template<int WHICH> inline ulonglong2 q9_coeff_fallback(
    uint64_t k0,uint64_t k1,uint64_t k2,uint64_t k3) {
    fallback_calls++;
    const uint64_t input[4]={k0,k1,k2,k3};
    uint32_t a[8],b[8],product[16]={};
    for(int i=0;i<8;i++) {
        a[i]=(uint32_t)(input[i/2]>>(32*(i%2)));
        b[i]=(uint32_t)(gvalues[WHICH-1][i/2]>>(32*(i%2)));
    }
    for(int i=0;i<8;i++) {
        uint64_t carry=0;
        for(int j=0;j<8;j++) {
            uint64_t value=(uint64_t)a[i]*b[j]+product[i+j]+carry;
            product[i+j]=(uint32_t)value;
            carry=value>>32;
        }
        product[i+8]=(uint32_t)carry;
    }
    unsigned __int128 low=(unsigned __int128)(
        (uint64_t)product[12]|((uint64_t)product[13]<<32))+(product[11]>>31);
    return {(uint64_t)low,
            ((uint64_t)product[14]|((uint64_t)product[15]<<32))+(uint64_t)(low>>64)};
}
#define QSB_GLV_COEFF_BOUNDS 1
"""
cpp += extract("q9_high15_begin(") + "\n"
cpp += extract("q9_mulhi32(") + "\n"
cpp += "template<int WHICH,uint32_t FALLBACK_WORD>\n" + extract("q9_coeff_high15(")
cpp += r"""
extern "C" uint64_t evaluate(
    int which,const uint64_t *input,const uint64_t *reciprocal,uint64_t *out) {
    for(int i=0;i<4;i++) gvalues[which][i]=reciprocal[i];
    fallback_calls=0;
    if(which==0) q9_coeff_high15<1,0x7ffffffcU>(out,input,gvalues[which]);
    else q9_coeff_high15<2,0x7ffffffdU>(out,input,gvalues[which]);
    return fallback_calls;
}
"""

rng = random.Random(0x350A17)
mode_names = ("control", "high10")
counts = defaultdict(int)
fallbacks = defaultdict(int)
with tempfile.TemporaryDirectory() as temporary:
    temporary = Path(temporary)
    (temporary / "probe.cpp").write_text(cpp)
    calls = []
    for high10, mode_name in enumerate(mode_names):
        output = temporary / f"{mode_name}.so"
        subprocess.run(
            [
                "c++", "-O3", "-shared", "-fPIC",
                f"-DQSB_GLV_HIGH10_HI={high10}",
                str(temporary / "probe.cpp"), "-o", str(output),
            ],
            check=True,
        )
        library = ctypes.CDLL(str(output))
        call = library.evaluate
        call.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        call.restype = ctypes.c_uint64
        calls.append(call)

    def check(which, value, category):
        if not 0 <= value < 1 << 256:
            return
        reciprocal = constants[which]
        expected = (value * reciprocal + (1 << 383)) >> 384
        input = (ctypes.c_uint64 * 4)(
            *[(value >> (64 * index)) & ((1 << 64) - 1) for index in range(4)]
        )
        reciprocal_words = (ctypes.c_uint64 * 4)(
            *[(reciprocal >> (64 * index)) & ((1 << 64) - 1) for index in range(4)]
        )
        for mode, call in enumerate(calls):
            output = (ctypes.c_uint64 * 2)()
            fell_back = call(which, input, reciprocal_words, output)
            actual = output[0] + (output[1] << 64)
            assert actual == expected, (mode_names[mode], which, category, hex(value))
            key = (mode_names[mode], which, category)
            counts[key] += 1
            fallbacks[key] += int(fell_back)

    for which, reciprocal in enumerate(constants):
        for value in (0, 1, (1 << 256) - 1):
            check(which, value, "edge")
        for bit in range(256):
            for delta in (-2, -1, 0, 1, 2):
                check(which, (1 << bit) + delta, "edge")
        for _ in range(100_000):
            check(which, rng.getrandbits(256), "uniform")
        for _ in range(12_000):
            quotient = rng.randrange(reciprocal >> 128)
            boundary = ((2 * quotient + 1) * (1 << 383)) // reciprocal
            for delta in (-3, -2, -1, 0, 1, 2, 3):
                check(which, boundary + delta, "rounding")
        unit = (1 << 352) // reciprocal
        for _ in range(1_000):
            quotient = rng.randrange(reciprocal >> 128)
            for origin in (
                ((2 * quotient + 1) * (1 << 383)) // reciprocal,
                (quotient * (1 << 384)) // reciprocal,
            ):
                for offset in range(-14, 5):
                    for delta in (-1, 0, 1):
                        check(which, origin + offset * unit + delta, "guard")
        for word in (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF):
            for mask in range(256):
                value = sum(
                    (word if mask & (1 << index) else 0) << (32 * index)
                    for index in range(8)
                )
                check(which, value, "patterns")

report = {
    "passed": True,
    "gpu_executed": False,
    "error_bound_units_2pow352": error_units,
    "cases_by_mode_which_category": {
        "|".join(map(str, key)): value for key, value in sorted(counts.items())
    },
    "fallback_calls_by_mode_which_category": {
        "|".join(map(str, key)): value for key, value in sorted(fallbacks.items())
    },
}
print(json.dumps(report, indent=2, sort_keys=True))
