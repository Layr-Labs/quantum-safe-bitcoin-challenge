#!/usr/bin/env python3
"""Exercise the production XYZZ add source with host field branches.

The default control is the immutable PR #219 Git object. Pass --control with
an unmodified worktree or GPUMath.h path to test a separately checked-out copy.
This is a host source audit, not CUDA compilation or a GPU benchmark.
"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "harness")]

import audit_deferred_chain as curve
import audit_stream_recode as recode
import problem

PR219 = "f7f588c7a9baacc261f5d6214c13e54130583f31"
P = curve.P

HOST_PRELUDE = r"""
#include <cstdint>
#include <cstring>
#define __device__
#define __forceinline__ inline
static uint64_t cc;
static uint64_t addw(uint64_t a,uint64_t b,bool use,bool set){__uint128_t t=(__uint128_t)a+b+(use?cc:0);if(set)cc=(uint64_t)(t>>64);return (uint64_t)t;}
static uint64_t subw(uint64_t a,uint64_t b,bool use,bool set){__uint128_t t=(__uint128_t)a-b-(use?cc:0);if(set)cc=(uint64_t)(t>>127);return (uint64_t)t;}
#define UADDO(c,a,b) c=addw(a,b,false,true)
#define UADDC(c,a,b) c=addw(a,b,true,true)
#define UADD(c,a,b) c=addw(a,b,true,false)
#define UADDO1(c,a) c=addw(c,a,false,true)
#define UADDC1(c,a) c=addw(c,a,true,true)
#define UADD1(c,a) c=addw(c,a,true,false)
#define USUBO(c,a,b) c=subw(a,b,false,true)
#define USUBC(c,a,b) c=subw(a,b,true,true)
#define USUB(c,a,b) c=subw(a,b,true,false)
#define USUBO1(c,a) c=subw(c,a,false,true)
#define USUBC1(c,a) c=subw(c,a,true,true)
#define USUB1(c,a) c=subw(c,a,true,false)
static void Load256(uint64_t *r,const uint64_t *a){memmove(r,a,32);}
"""

HOST_WRAPPERS = r"""
extern "C" void chain(uint64_t *points,uint64_t *out){
 uint64_t X[4],Y[4],U[4],V[4],anchor[4];
 _PointAddXYZZ_mm(X,Y,U,V,points,points+4,points+8,points+12);
 Load256(anchor,points+4);
 for(int step=0;step<14;step++){
  if(step){int i=step+1;
   if(i==14)_PointAddXYZZ<false>(X,Y,U,V,points+8*i,points+8*i+4,anchor);
   else _PointAddXYZZ<true>(X,Y,U,V,points+8*i,points+8*i+4,anchor);
   Load256(anchor,points+8*i+4);
  }
  Load256(out+step*16,X);Load256(out+step*16+4,Y);Load256(out+step*16+8,U);Load256(out+step*16+12,V);
 }
}
extern "C" void madd(uint64_t *input,uint64_t *out,int defer){
 uint64_t X[4],Y[4],U[4],V[4];Load256(X,input);Load256(Y,input+4);Load256(U,input+8);Load256(V,input+12);
 if(defer)_PointAddXYZZ<true>(X,Y,U,V,input+16,input+20,input+24);
 else _PointAddXYZZ<false>(X,Y,U,V,input+16,input+20,input+24);
 Load256(out,X);Load256(out+4,Y);Load256(out+8,U);Load256(out+12,V);
}
"""


def function(source, name):
    match = re.search(r"__device__[^\n]*\b" + re.escape(name) + r"\(", source)
    assert match, name
    start = source.index("{", match.start())
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


def overloads(source, name):
    matches = re.finditer(r"__device__[^\n]*\b" + re.escape(name) + r"\(", source)
    return "\n".join(function(source[match.start():], name) for match in matches)


def build(source, tag, mulsub, sqraddsub2, directory, cxx):
    parts = [HOST_PRELUDE,
             f"#define QSB_FUSE_MULSUB {mulsub}\n#define QSB_FUSE_SQRADDSUB2 {sqraddsub2}"]
    for name in ("_ModMultCore", "_ModMult", "_ModSqr", "_ModSub256",
                 "_ModAddLazy", "_ModX3Fused"):
        parts.append(overloads(source, name))
    if "_ModMulSubCore(" in source:
        parts.extend((function(source, "_ModMulSubCore"),
                      function(source, "_ModSqrAddSub2")))
    parts.extend(("template<bool DEFER_Y>\n" + function(source, "_PointAddXYZZ"),
                  function(source, "_PointAddXYZZ_mm"), HOST_WRAPPERS))
    cpp = directory / f"{tag}.cpp"
    lib = directory / f"{tag}.dylib"
    cpp.write_text("\n".join(parts))
    subprocess.run([cxx, "-std=c++17", "-O2", "-shared", "-fPIC",
                    str(cpp), "-o", str(lib)], check=True)
    loaded = ctypes.CDLL(str(lib))
    loaded.chain.argtypes = [ctypes.POINTER(ctypes.c_uint64)] * 2
    loaded.madd.argtypes = [ctypes.POINTER(ctypes.c_uint64)] * 2 + [ctypes.c_int]
    return loaded


def control_source(control):
    if control:
        path = Path(control)
        if path.is_dir():
            path = path / "candidates/pinning/GPUMath.h"
        return path.read_text()
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{PR219}:candidates/pinning/GPUMath.h"],
        text=True)


def words(values):
    return (ctypes.c_uint64 * (4 * len(values)))(
        *[(value >> (64 * i)) & ((1 << 64) - 1)
          for value in values for i in range(4)])


def nums(data):
    return [sum(data[4 * i + j] << (64 * j) for j in range(4))
            for i in range(len(data) // 4)]


def source_chain(lib, points):
    output = (ctypes.c_uint64 * (14 * 16))()
    lib.chain(words([value for point in points for value in point]), output)
    result = nums(output)
    return [result[i * 4:i * 4 + 4] for i in range(14)]


def normalize(state, anchor=0):
    x, y, u, v = state
    assert u % P and v % P
    return x * pow(u, -1, P) % P, (y - anchor * v) * pow(v, -1, P) % P


def audit(libs, candidate_count):
    challenge = json.loads((ROOT / "problems/pinning.json").read_text())
    nri = int(challenge["neg_r_inv"], 16)
    recovery = (int(challenge["u2r_x"], 16), int(challenge["u2r_y"], 16))
    half = curve.scalar_mult(nri * pow(2, -1, curve.N) % curve.N)
    bases = []
    position = 0
    for j in range(15):
        shift = recode.shift_for(j)
        for _ in range(shift - position):
            half = curve.affine_add(half, half)
        bases.append(half)
        position = shift

    hits = {name: [] for name in libs}
    negatives = 0
    compared = 0
    for offset in range(candidate_count):
        candidate = {"sequence": 0x80000000, "locktime": 500000000 + offset}
        preimage = problem.pin_preimage(challenge, **candidate)
        raw = hashlib.sha256(hashlib.sha256(preimage).digest()).digest()
        scalar = int.from_bytes(raw, "big")
        digits = recode.streamed(scalar)
        negatives += sum(digit < 0 for digit in digits)
        points = []
        for j, digit in enumerate(digits):
            point = curve.scalar_mult(abs(digit), bases[j])
            points.append(point if digit > 0 else (point[0], -point[1] % P))
        prefix = []
        point = points[0]
        for item in points[1:]:
            point = curve.affine_add(point, item)
            prefix.append(point)
        assert point == curve.scalar_mult(nri * scalar % curve.N)

        baseline = source_chain(libs["control"], points)
        expected = [problem.candidate_hash(challenge, candidate, rid) for rid in (0, 1)]
        for name, lib in libs.items():
            states = baseline if name == "control" else source_chain(lib, points)
            if name == "00":
                assert states == baseline, ("disabled path changed raw states", offset)
            for j, state in enumerate(states):
                anchor = points[0][1] if j == 0 else points[j + 1][1]
                got = normalize(state, 0 if j == 13 else anchor)
                assert got == prefix[j], (name, offset, j, got, prefix[j])
                compared += 1
            got = normalize(states[-1])
            for rid in (0, 1):
                recovered = curve.affine_add(
                    got, recovery if rid == 0 else (recovery[0], -recovery[1] % P))
                encoded = bytes([2 | (recovered[1] & 1)]) + recovered[0].to_bytes(32, "big")
                digest = hashlib.sha256(encoded).digest()
                assert digest == expected[rid], (name, offset, rid)
                if digest[0] >> 4 == 0:
                    hits[name].append((offset, rid, encoded.hex()))
    assert hits["control"] and all(value == hits["control"] for value in hits.values())

    # A nontrivial cube root gives distinct x, equal y, and a valid zero slope.
    beta = next(pow(i, (P - 1) // 3, P) for i in range(2, 20)
                if pow(i, (P - 1) // 3, P) != 1)
    generator = curve.G
    zero_slope = (beta * generator[0] % P, generator[1])
    reference = curve.affine_add(generator, zero_slope)
    special = 0
    for name, lib in libs.items():
        for defer in (0, 1):
            output = (ctypes.c_uint64 * 16)()
            lib.madd(words([generator[0], generator[1], 1, 1,
                            zero_slope[0], zero_slope[1], 0]), output, defer)
            got = normalize(nums(output), zero_slope[1] if defer else 0)
            assert got == reference, (name, "zero slope", defer)
            special += 1
        for other in (generator, (generator[0], -generator[1] % P)):
            output = (ctypes.c_uint64 * 16)()
            lib.madd(words([generator[0], generator[1], 1, 1,
                            other[0], other[1], 0]), output, 0)
            state = nums(output)
            assert state[2] % P == state[3] % P == 0, (name, "inherited singular")

    print("PASS: actual source point add: control + 00/10/01/11;"
          f" {candidate_count} candidate chains, {negatives} negative digits,"
          f" {compared} normalized intermediate comparisons,"
          f" both compressed keys, {len(hits['control'])} identical prefix hits,"
          f" {special} valid zero-slope cases; inherited singular scales preserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", help="unmodified PR #219 worktree or GPUMath.h")
    parser.add_argument("--cxx", default="clang++", help="host C++ compiler")
    parser.add_argument("--candidates", type=int, default=128)
    args = parser.parse_args()
    assert args.candidates > 0
    current = (HERE / "GPUMath.h").read_text()
    baseline = control_source(args.control)
    with tempfile.TemporaryDirectory(prefix="qsb-fused-point-source-") as temporary:
        directory = Path(temporary)
        libs = {"control": build(baseline, "control", 0, 0, directory, args.cxx)}
        for mulsub, square in ((0, 0), (1, 0), (0, 1), (1, 1)):
            tag = f"{mulsub}{square}"
            libs[tag] = build(current, tag, mulsub, square, directory, args.cxx)
        audit(libs, args.candidates)


if __name__ == "__main__":
    main()
