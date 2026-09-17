#!/usr/bin/env python3
"""Independent model of a full 15-point deferred-Y XYZZ chain."""

from __future__ import annotations

import random
import re
from pathlib import Path


P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
G = (
    0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
)


def affine_add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    x1, y1 = a
    x2, y2 = b
    if x1 == x2:
        if (y1 + y2) % P == 0:
            return None
        slope = 3 * x1 * x1 * pow(2 * y1, -1, P) % P
    else:
        slope = (y2 - y1) * pow(x2 - x1, -1, P) % P
    x3 = (slope * slope - x1 - x2) % P
    return x3, (slope * (x1 - x3) - y1) % P


def scalar_mult(k, point=G):
    out = None
    addend = point
    k %= N
    while k:
        if k & 1:
            out = affine_add(out, addend)
        addend = affine_add(addend, addend)
        k >>= 1
    return out


def mm(a, b):
    x1, y1 = a
    x2, y2 = b
    h = (x2 - x1) % P
    r = (y2 - y1) % P
    hh = h * h % P
    hhh = h * hh % P
    q = x1 * hh % P
    x = (r * r - hhh - 2 * q) % P
    y = (r * (q - x) - y1 * hhh) % P
    return x, y, hh, hhh


def mm_deferred(a, b):
    """D(P1+P2; y1): stored Y is actual Y + y1*ZZZ."""
    x1, y1 = a
    x2, y2 = b
    h = (x2 - x1) % P
    r = (y2 - y1) % P
    hh = h * h % P
    hhh = h * hh % P
    q = x1 * hh % P
    x = (r * r - hhh - 2 * q) % P
    yd = r * (q - x) % P
    return x, yd, hh, hhh


def madd_old(s, a):
    x, y, zz, zzz = s
    xa, ya = a
    u = xa * zz % P
    ss = ya * zzz % P
    h = (u - x) % P
    r = (ss - y) % P
    hh = h * h % P
    hhh = h * hh % P
    q = x * hh % P
    xn = (r * r - hhh - 2 * q) % P
    yn = (r * (q - xn) - y * hhh) % P
    return xn, yn, zz * hh % P, zzz * hhh % P


def madd_from_deferred(s, a, old_anchor_y, defer_output):
    """Add affine a to D(P; old_anchor_y), returning D(P+a; a.y) or exact."""
    x, yd, zz, zzz = s
    xa, ya = a
    u = xa * zz % P
    ss = (ya + old_anchor_y) * zzz % P
    h = (u - x) % P
    r = (ss - yd) % P
    hh = h * h % P
    hhh = h * hh % P
    v = u * hh % P
    xn = (r * r + hhh - 2 * v) % P
    zzn = zz * hh % P
    zzzn = zzz * hhh % P
    ycore = r * (v - xn) % P
    yn = ycore if defer_output else (ycore - ya * zzzn) % P
    return xn, yn, zzn, zzzn


def normalize(s):
    x, y, zz, zzz = s
    assert zz and zzz
    return x * pow(zz, -1, P) % P, y * pow(zzz, -1, P) % P


def denominators_nonzero(points):
    s = mm(points[0], points[1])
    if s[2] == 0:
        return False
    for a in points[2:]:
        if (a[0] * s[2] - s[0]) % P == 0:
            return False
        s = madd_old(s, a)
    return True


def check(points, check_curve):
    assert len(points) == 15
    exact = mm(points[0], points[1])
    chained = mm_deferred(points[0], points[1])
    anchor_y = points[0][1]
    assert chained[0] == exact[0] and chained[2:] == exact[2:]
    assert chained[1] == (exact[1] + anchor_y * exact[3]) % P
    for i, point in enumerate(points[2:], start=2):
        exact = madd_old(exact, point)
        final = i == len(points) - 1
        chained = madd_from_deferred(chained, point, anchor_y, not final)
        if final:
            assert chained == exact, (i, points, exact, chained)
        else:
            anchor_y = point[1]
            assert chained[0] == exact[0] and chained[2:] == exact[2:]
            assert chained[1] == (exact[1] + anchor_y * exact[3]) % P
    if check_curve:
        want = None
        for point in points:
            want = affine_add(want, point)
        assert want is not None
        assert normalize(chained) == want


def arbitrary_field_audit():
    edge = [0, 1, 2, 3, P - 3, P - 2, P - 1]
    count = 0
    for i in range(1024):
        pts = tuple(
            (edge[(i + 3 * j) % 7], edge[(5 * i + 2 * j + 1) % 7])
            for j in range(15)
        )
        if denominators_nonzero(pts):
            check(pts, False)
            count += 1
    rng = random.Random(0xD3F3_15A1)
    while count < 20_000:
        pts = tuple((rng.randrange(P), rng.randrange(P)) for _ in range(15))
        if denominators_nonzero(pts):
            check(pts, False)
            count += 1
    return count


def curve_audit():
    rng = random.Random(0xD3F3_C0DE)
    scalars = [1, 2, 3, 4, N - 1, N - 2, N // 2, N // 2 + 1]
    scalars += [rng.randrange(1, N) for _ in range(64)]
    pool = [scalar_mult(k) for k in scalars]
    assert all(p is not None for p in pool)
    count = 0
    while count < 1_000:
        pts = tuple(pool[rng.randrange(len(pool))] for _ in range(15))
        if denominators_nonzero(pts):
            check(pts, True)
            count += 1
    return count


def mixed_table_audit():
    # Production point i is ei*2^shift[i]*(G/2); the common factor 1/2 is
    # immaterial. Exercise the exact 15-point mixed shift/digit boundaries.
    shifts = [0] + [17 * i + 1 for i in range(1, 15)]
    digits0 = [-(2**18 - 1), -3, -1, 1, 3, 2**18 - 1]
    digitsn = [-(2**17 - 1), -3, -1, 1, 3, 2**17 - 1]
    rng = random.Random(0xF04_7AB1E)
    choices = [digits0] + [digitsn] * 14
    cache = {}
    count = 0
    while count < 1_000:
        ds = [choices[i][rng.randrange(len(choices[i]))] for i in range(15)]
        pts = []
        for i, digit in enumerate(ds):
            key = i, digit
            if key not in cache:
                cache[key] = scalar_mult(digit * pow(2, shifts[i], N))
            pts.append(cache[key])
        assert all(p is not None for p in pts)
        assert denominators_nonzero(pts), ds
        check(tuple(pts), True)
        count += 1
    return count


def function_body(text, marker):
    start = text.index(marker)
    brace = text.index("{", start)
    depth = 0
    for pos in range(brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    raise AssertionError(f"unterminated function {marker}")


def source_audit():
    root = Path(__file__).resolve().parent
    math = (root / "GPUMath.h").read_text()
    pinning = (root / "pinning.cu").read_text()
    mixed = function_body(math, "__device__ __forceinline__ void _PointAddXYZZ(")
    assert len(re.findall(r"\b_ModMult\(", mixed)) == 8
    assert len(re.findall(r"\b_ModSqr\(", mixed)) == 2
    assert "const uint64_t *__restrict__ Yoff)" in mixed
    assert "if (DEFER_Y)" in mixed
    assert "Load256(Y1, Q);" in mixed
    assert "_ModMult(S2, (uint64_t *)Y2, ZZZ1);" in mixed
    # Merged truth: the two production loops use the template split (chunks
    # 2..GT_CHUNKS-2 defer the anchor; the final chunk resolves the ordinate
    # outside the loop and its dead anchor copy is gone). The two disabled
    # experiment branches retained verbatim from exact 1a (QSB_PREFETCH and
    # QSB_S0_SHM) keep the runtime-defer form, the shared-memory one through
    # its local `ya` anchor.
    assert pinning.count("_PointAddXYZZ<true>(X,Y,ZZ,ZZZ, cx,cy, y0);") == 2
    assert pinning.count("_PointAddXYZZ<false>(X,Y,ZZ,ZZZ, cx,cy, y0);") == 2
    assert pinning.count("_PointAddXYZZ(X,Y,ZZ,ZZZ, cx,cy, y0, c != GT_CHUNKS-1);") == 1
    assert pinning.count("_PointAddXYZZ(X,Y,ZZ,ZZZ, cx,cy, ya, c != GT_CHUNKS-1);") == 1
    assert pinning.count("for (int c=2;c<GT_CHUNKS-1;c++){") == 2
    assert pinning.count("Load256(y0, cy);") == 3


if __name__ == "__main__":
    source_audit()
    af = arbitrary_field_audit()
    cv = curve_audit()
    mt = mixed_table_audit()
    print(
        "PASS: full 15-point deferred-Y chain; "
        f"{af} arbitrary-field accumulations; {cv} curve accumulations; "
        f"{mt} mixed-window accumulations"
    )
