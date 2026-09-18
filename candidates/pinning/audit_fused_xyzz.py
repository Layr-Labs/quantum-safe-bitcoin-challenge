#!/usr/bin/env python3
"""Compile the production host branches of the two experimental XYZZ reducers.

This is an arithmetic audit, not a GPU benchmark. Both functions are extracted
from GPUMath.h so the checked C-reference schedule is the edited source.
"""

import ctypes
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

B = 1 << 256
K = (1 << 32) + 977
P = B - K
HERE = Path(__file__).resolve().parent


def source_functions():
    source = (HERE / "GPUMath.h").read_text()
    base_mul_start = source.index("__device__ __forceinline__ void _ModMultCore(")
    mul_start = source.index("__device__ __forceinline__ void _ModMulSubCore(")
    mul_end = source.index("// Compute a*b", mul_start)
    base_sq_start = source.index("__device__ __forceinline__ void _ModSqr(")
    sq_start = source.index("__device__ __forceinline__ void _ModSqrAddSub2(")
    sq_end = source.index("//Very efficient way of finding", sq_start)
    base_mul = source[base_mul_start:mul_start]
    base_sq = source[base_sq_start:sq_start]
    mul = source[mul_start:mul_end]
    sq = source[sq_start:sq_end]
    point = source[source.index("__device__ __forceinline__ void _PointAddXYZZ(") :]
    assert "_ModMultCore(" not in mul and "_ModSqr(" not in sq
    assert "Inject 2p-c" in mul and "Inject 3p+e-2q" in sq
    assert "#if QSB_FUSE_MULSUB" in point
    assert "_ModMulSubCore(R, S2, ZZZ1, Y1);" in point
    assert "#if QSB_FUSE_SQRADDSUB2" in point
    assert "_ModSqrAddSub2(T, R, PPP, Q);" in point
    return base_mul, mul, base_sq, sq


def build_host_reference(directory):
    base_mul, mul, base_sq, sq = source_functions()
    cpp = Path(directory) / "fused.cpp"
    library = Path(directory) / ("fused.dylib" if sys.platform == "darwin" else "fused.so")
    cpp.write_text(
        "#include <cstdint>\n#define __device__\n#define __forceinline__ inline\n"
        + base_mul + mul + base_sq + sq
        + '\nextern "C" void mul(uint64_t *out, const uint64_t *a, '
          'const uint64_t *b) { _ModMultCore(out,a,b); }\n'
        + 'extern "C" void sqr(uint64_t *out, const uint64_t *a) '
          '{ _ModSqr(out,a); }\n'
        + '\nextern "C" void mulsub(uint64_t *out, const uint64_t *a, '
          'const uint64_t *b, const uint64_t *c) { _ModMulSubCore(out,a,b,c); }\n'
        + 'extern "C" void sqraddsub2(uint64_t *out, const uint64_t *a, '
          'const uint64_t *e, const uint64_t *q) { _ModSqrAddSub2(out,a,e,q); }\n'
    )
    cmd = ["clang++", "-std=c++17", "-O2", "-fPIC"]
    cmd += ["-dynamiclib"] if sys.platform == "darwin" else ["-shared"]
    subprocess.run(cmd + [str(cpp), "-o", str(library)], check=True, capture_output=True)
    dll = ctypes.CDLL(str(library))
    wordp = ctypes.POINTER(ctypes.c_uint64)
    for name, arity in (("mul", 3), ("sqr", 2), ("mulsub", 4), ("sqraddsub2", 4)):
        getattr(dll, name).argtypes = [wordp] * arity
        getattr(dll, name).restype = None
    return dll


def compile_point_matrix(directory):
    """Compile both template instantiations under each independent switch."""
    source = (HERE / "GPUMath.h").read_text()
    switches = source[source.index("// Experimental XYZZ reductions.") :
                      source.index("// Assembly directives", source.index("// Experimental XYZZ reductions."))]
    point = source[source.index("template<bool DEFER_Y>\n__device__ __forceinline__ void _PointAddXYZZ(") :
                   source.index("__device__ void _PointAddXYZZ_mm(")]
    cpp = Path(directory) / "point_matrix.cpp"
    cpp.write_text(
        "#include <cstdint>\n#define __device__\n#define __forceinline__ inline\n"
        "#define Load256(r,a) do { for(int i=0;i<4;i++) (r)[i]=(a)[i]; } while(0)\n"
        + switches
        + "void _ModMult(uint64_t*,uint64_t*,uint64_t*);\n"
          "void _ModMult(uint64_t*,uint64_t*);\n"
          "void _ModSub256(uint64_t*,uint64_t*,uint64_t*);\n"
          "void _ModSub256(uint64_t*,uint64_t*);\n"
          "void _ModAddLazy(uint64_t*,const uint64_t*,const uint64_t*);\n"
          "void _ModSqr(uint64_t*,const uint64_t*);\n"
          "void _ModX3Fused(uint64_t*,const uint64_t*,const uint64_t*,const uint64_t*);\n"
          "void _ModMulSubCore(uint64_t*,const uint64_t*,const uint64_t*,const uint64_t*);\n"
          "void _ModSqrAddSub2(uint64_t*,const uint64_t*,const uint64_t*,const uint64_t*);\n"
        + point
        + "extern \"C\" void instantiate(uint64_t *x,uint64_t *y,uint64_t *u,"
          "uint64_t *v,const uint64_t *a,const uint64_t *b,const uint64_t *off) {\n"
          "_PointAddXYZZ<true>(x,y,u,v,a,b,off);\n"
          "_PointAddXYZZ<false>(x,y,u,v,a,b,off);\n}\n"
    )
    for mul in (0, 1):
        for square in (0, 1):
            obj = Path(directory) / f"point_{mul}{square}.o"
            subprocess.run([
                "clang++", "-std=c++17", "-O2", "-c", str(cpp), "-o", str(obj),
                f"-DQSB_FUSE_MULSUB={mul}", f"-DQSB_FUSE_SQRADDSUB2={square}",
            ], check=True, capture_output=True)
            symbols = subprocess.run(["nm", "-u", str(obj)], check=True,
                                     capture_output=True, text=True).stdout
            assert ("_ModMulSubCore" in symbols) == bool(mul), (mul, square, symbols)
            assert ("_ModSqrAddSub2" in symbols) == bool(square), (mul, square, symbols)


def words(value):
    return (ctypes.c_uint64 * 4)(*(value >> (64 * i) & ((1 << 64) - 1) for i in range(4)))


def integer(array):
    return sum(int(array[i]) << (64 * i) for i in range(4))


def call(fn, inputs, alias=None):
    arrays = [words(x) for x in inputs]
    output = arrays[alias] if alias is not None else words(0)
    fn(output, *arrays)
    return integer(output)


def first_fold_carry(raw, correction, bias):
    first = raw % B + (raw // B) * K + bias * P + correction
    assert 0 <= first < (K + 5) * B
    second = first % B + (first // B) * K
    assert second // B <= 1
    return second // B


def check_mul(fn, a, b, c, counters, alias=False):
    expected = (a * b - c) % P
    got = call(fn, (a, b, c))
    assert 0 <= got < B and got % P == expected, ("mulsub", a, b, c, got, expected)
    counters[0] += first_fold_carry(a * b, -c, 2)
    if alias:
        for index in range(3):
            got = call(fn, (a, b, c), index)
            assert got % P == expected, ("mulsub alias", index, a, b, c, got)


def check_square(fn, a, e, q, counters, alias=False):
    expected = (a * a + e - 2 * q) % P
    got = call(fn, (a, e, q))
    assert 0 <= got < B and got % P == expected, ("sqraddsub2", a, e, q, got, expected)
    counters[1] += first_fold_carry(a * a, e - 2 * q, 3)
    if alias:
        for index in range(3):
            got = call(fn, (a, e, q), index)
            assert got % P == expected, ("sqraddsub2 alias", index, a, e, q, got)


def check_point_chains(dll):
    """Exercise the fused calls in every step of the 15-point deferred chain."""
    import audit_deferred_chain as chain
    import audit_stream_recode as recode

    sys.path.insert(0, str(HERE.parents[1] / "harness"))
    import problem

    prob = json.loads((HERE.parents[1] / "problems" / "pinning.json").read_text())
    neg_r_inv = int(prob["neg_r_inv"], 16)
    recovery = int(prob["u2r_x"], 16), int(prob["u2r_y"], 16)
    half_base = chain.scalar_mult(neg_r_inv * pow(2, -1, chain.N) % chain.N)
    table_bases = []
    shift = 0
    for chunk in range(15):
        target = recode.shift_for(chunk)
        for _ in range(target - shift):
            half_base = chain.affine_add(half_base, half_base)
        table_bases.append(half_base)
        shift = target

    def run(points, mul_fused, square_fused):
        state = chain.mm_deferred(points[0], points[1])
        anchor = points[0][1]
        control = state
        for index, (xa, ya) in enumerate(points[2:], 2):
            x, yd, zz, zzz = state
            u = xa * zz % P
            s2 = (ya + anchor) % P
            p = (u - x) % P
            r = (call(dll.mulsub, (s2, zzz, yd)) if mul_fused else s2 * zzz - yd) % P
            pp = p * p % P
            ppp = pp * p % P
            q = u * pp % P
            xn = (call(dll.sqraddsub2, (r, ppp, q)) if square_fused
                  else r * r + ppp - 2 * q) % P
            zzn = zz * pp % P
            zzzn = zzz * ppp % P
            ycore = r * (q - xn) % P
            final = index == 14
            yn = (ycore - ya * zzzn) % P if final else ycore
            state = xn, yn, zzn, zzzn
            control = chain.madd_from_deferred(control, (xa, ya), anchor, not final)
            assert state == control, (mul_fused, square_fused, index)
            if not final:
                anchor = ya
        return chain.normalize(state)

    hits = {(False, False): [], (True, False): [],
            (False, True): [], (True, True): []}
    negative_digits = 0
    for offset in range(32):
        candidate = {"sequence": 0x80000000, "locktime": 500000000 + offset}
        digest = hashlib.sha256(hashlib.sha256(problem.pin_preimage(
            prob, candidate["sequence"], candidate["locktime"]
        )).digest()).digest()
        z = int.from_bytes(digest, "big")
        digits = recode.streamed(z)
        negative_digits += sum(digit < 0 for digit in digits)
        points = []
        for chunk, digit in enumerate(digits):
            point = chain.scalar_mult(abs(digit), table_bases[chunk])
            if digit < 0:
                point = point[0], -point[1] % P
            points.append(point)
        assert chain.denominators_nonzero(points)
        affine = None
        for point in points:
            affine = chain.affine_add(affine, point)
        assert affine == chain.scalar_mult(neg_r_inv * z % chain.N)

        for mode in hits:
            result = run(points, *mode)
            assert result == affine
            for recid in (0, 1):
                offset_point = recovery if recid == 0 else (recovery[0], -recovery[1] % P)
                recovered = chain.affine_add(result, offset_point)
                compressed = bytes([2 | (recovered[1] & 1)]) + recovered[0].to_bytes(32, "big")
                key_hash = hashlib.sha256(compressed).digest()
                assert key_hash == problem.candidate_hash(prob, candidate, recid)
                if key_hash[0] >> 4 == 0:  # Four-bit prefix gate.
                    hits[mode].append((offset, recid, compressed))
    assert negative_digits > 0
    assert hits[False, False]
    assert all(result == hits[False, False] for result in hits.values())
    return len(hits[False, False]), negative_digits


def check_zero_slope_and_singular(dll):
    """Feed a valid-curve zero slope, including a noncanonical p, downstream."""
    import audit_deferred_chain as chain

    beta = next(value for base in range(2, 20)
                if (value := pow(base, (P - 1) // 3, P)) != 1)
    assert pow(beta, 3, P) == 1
    left = chain.G
    same_y = beta * left[0] % P, left[1]
    assert same_y[0] != left[0]
    assert (same_y[1] * same_y[1] - same_y[0] ** 3 - 7) % P == 0

    def add_once(right, mul_fused, square_fused):
        x, y = left
        xa, ya = right
        u = call(dll.mul, (xa, 1))
        slope_numerator = (call(dll.mulsub, (ya, 1, y)) if mul_fused
                           else call(dll.mul, (ya, 1)) - y) % B
        delta = (u - x) % P
        pp = call(dll.sqr, (delta,))
        ppp = call(dll.mul, (pp, delta))
        q = call(dll.mul, (u, pp))
        xn = (call(dll.sqraddsub2, (slope_numerator, ppp, q)) if square_fused
              else call(dll.sqr, (slope_numerator,)) + ppp - 2 * q) % P
        zzzn = call(dll.mul, (1, ppp))
        ycore = call(dll.mul, ((q - xn) % P, slope_numerator))
        yn = (ycore - call(dll.mul, (ya, zzzn))) % P
        return xn, yn, pp % P, zzzn % P, slope_numerator

    noncanonical = 0
    for mul_fused in (False, True):
        for square_fused in (False, True):
            xn, yn, zz, zzz, numerator = add_once(same_y, mul_fused, square_fused)
            assert numerator % P == 0
            noncanonical += numerator == P
            want = chain.affine_add(left, same_y)
            assert (xn * pow(zz, -1, P) % P,
                    yn * pow(zzz, -1, P) % P) == want

            # The inherited XYZZ madd has no doubling or opposite-point path:
            # both produce zero projective scales and must not be normalized.
            for right in (left, (left[0], -left[1] % P)):
                _, _, zz, zzz, _ = add_once(right, mul_fused, square_fused)
                assert zz == zzz == 0
    assert noncanonical >= 2  # Both mul-fused modes passed a raw p downstream.
    return noncanonical


def main():
    rng = random.Random(0xF217219)
    boundary = [0, 1, 2, 976, 977, K - 1, K, K + 1,
                (1 << 32) - 1, 1 << 32, (1 << 64) - 1,
                1 << 128, B // 2, B // 2 + 1, B - K - 65537,
                P - 1, P, P + 1, B - 2, B - 1]
    with tempfile.TemporaryDirectory(prefix="qsb-fused-audit-") as directory:
        dll = build_host_reference(directory)
        compile_point_matrix(directory)
        counters = [0, 0]
        cases = 0
        for a in boundary:
            for b in boundary:
                for c in boundary:
                    check_mul(dll.mulsub, a, b, c, counters, alias=(cases % 97 == 0))
                    check_square(dll.sqraddsub2, a, b, c, counters, alias=(cases % 97 == 0))
                    cases += 1
        for i in range(100_000):
            a, b, c = (rng.randrange(B) for _ in range(3))
            check_mul(dll.mulsub, a, b, c, counters, alias=(i % 1009 == 0))
            check_square(dll.sqraddsub2, a, b, c, counters, alias=(i % 1009 == 0))
            cases += 1
        assert counters[0] > 0 and counters[1] > 0, counters
        hits, negative_digits = check_point_chains(dll)
        noncanonical = check_zero_slope_and_singular(dll)
    print(f"PASS: four host template switch builds; {cases} cases per fused reducer, "
          f"aliasing, full-width inputs; "
          f"final carries mul={counters[0]} square={counters[1]}; "
          f"32 exact signed-digit candidate chains, {negative_digits} negative digits, "
          f"both recovered keys, {hits} identical four-bit prefix hits across 00/10/01/11; "
          f"valid zero-slope raw-p paths={noncanonical}, singular scales=0")


if __name__ == "__main__":
    main()
