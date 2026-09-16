#!/usr/bin/env python3
"""Compile and exercise the production host branches of the field primitives."""

from __future__ import annotations

import hashlib
import ctypes
from pathlib import Path
import random
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "_ModMultCore": "cb5ceb57087b8c1244e5c7e62394b6ac109ae56692fd17755753aebf65bd7a28",
    "qsb_square32": "5ed7666ab09b0b4a32a399a4b556a702ce735cdecd7c7b9a78217ee968f83a5a",
}


def extract_function(path: Path, marker: str) -> str:
    source = path.read_text()
    marker_at = source.index(marker)
    start = source.rfind("\n", 0, marker_at) + 1
    brace = source.index("{", marker_at)
    depth = 0
    for pos in range(brace, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[start : pos + 1] + "\n"
    raise RuntimeError(f"unterminated function: {marker}")


mult = extract_function(ROOT / "GPUMath.h", "__device__ __forceinline__ void _ModMultCore")
square = extract_function(ROOT / "square32.cuh", "__device__ __forceinline__ void qsb_square32")
for name, source in (("_ModMultCore", mult), ("qsb_square32", square)):
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != EXPECTED[name]:
        raise SystemExit(f"FAIL: unexpected {name} source digest {digest}")

# The selected !__CUDA_ARCH__ branches are the production routines' exact C++
# reference paths.  Compile the extracted functions rather than a handwritten
# translation so aliasing and reduction behavior are exercised directly.
unit = r'''#include <cstddef>
#include <cstdint>
#define __device__
#define __forceinline__ inline
''' + mult + "\n" + square + r'''
extern "C" void audit_mult(uint64_t *out, const uint64_t *a, const uint64_t *b, int alias) {
    uint64_t aa[4], bb[4];
    for (int i=0;i<4;i++) { aa[i]=a[i]; bb[i]=b[i]; }
    if (alias==1) { _ModMultCore(aa,aa,bb); for(int i=0;i<4;i++) out[i]=aa[i]; }
    else if (alias==2) { _ModMultCore(bb,aa,bb); for(int i=0;i<4;i++) out[i]=bb[i]; }
    else _ModMultCore(out,aa,bb);
}
extern "C" void audit_square(uint64_t *out, const uint64_t *a, int alias) {
    uint64_t aa[4]; for(int i=0;i<4;i++) aa[i]=a[i];
    if(alias) { qsb_square32(aa,aa); for(int i=0;i<4;i++) out[i]=aa[i]; }
    else qsb_square32(out,aa);
}
'''

with tempfile.TemporaryDirectory(prefix=".field-audit-", dir=ROOT) as temp:
    library_path = Path(temp) / "audit.so"
    build = subprocess.run(
        ["g++", "-std=c++17", "-O2", "-shared", "-fPIC", "-x", "c++", "-", "-o", str(library_path)],
        input=unit,
        text=True,
        capture_output=True,
    )
    if build.returncode:
        raise SystemExit("FAIL: host compile\n" + build.stderr)
    library = ctypes.CDLL(str(library_path))
    limbs = ctypes.c_uint64 * 4
    library.audit_mult.argtypes = (ctypes.POINTER(ctypes.c_uint64),) * 3 + (ctypes.c_int,)
    library.audit_square.argtypes = (ctypes.POINTER(ctypes.c_uint64),) * 2 + (ctypes.c_int,)

    def words(value: int) -> ctypes.Array[ctypes.c_uint64]:
        return limbs(*(value >> (64 * index) & ((1 << 64) - 1) for index in range(4)))

    def integer(value: ctypes.Array[ctypes.c_uint64]) -> int:
        return sum(int(value[index]) << (64 * index) for index in range(4))

    prime = (1 << 256) - (1 << 32) - 977
    edges = (0, 1, 2, 3, prime - 2, prime - 1, 1 << 128, 1 << 255)
    rng = random.Random(0x783BDBD5209EA7)
    pairs = [(a, b) for a in edges for b in edges]
    pairs.extend((rng.randrange(prime), rng.randrange(prime)) for _ in range(32768))
    out = limbs()
    for sample, (a, b) in enumerate(pairs):
        aa, bb = words(a), words(b)
        expected = a * b % prime
        for alias in range(3):
            library.audit_mult(out, aa, bb, alias)
            if integer(out) % prime != expected:
                raise SystemExit(f"FAIL: multiply sample={sample} alias={alias}")
        expected = a * a % prime
        for alias in range(2):
            library.audit_square(out, aa, alias)
            if integer(out) % prime != expected:
                raise SystemExit(
                    f"FAIL: square sample={sample} alias={alias} "
                    f"a={a:064x} got={integer(out):064x} expected={expected:064x}"
                )
    print(f"PASS: {len(pairs)} canonical-input pairs; field-equivalent multiply in 3 alias modes; square in 2 alias modes")
