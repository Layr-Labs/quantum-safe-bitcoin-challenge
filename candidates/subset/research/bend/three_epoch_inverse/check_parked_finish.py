#!/usr/bin/env python3
"""Audit the K3 parked-numerator cut against affine secp256k1 addition."""

import random

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
G = (
    0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
)


def add(p, q):
    if p is None:
        return q
    if q is None:
        return p
    x1, y1 = p
    x2, y2 = q
    if x1 == x2:
        if (y1 + y2) % P == 0:
            return None
        slope = 3 * x1 * x1 * pow(2 * y1, -1, P) % P
    else:
        slope = (y2 - y1) * pow(x2 - x1, -1, P) % P
    x3 = (slope * slope - x1 - x2) % P
    return x3, (slope * (x1 - x3) - y1) % P


def mul(k, p=G):
    out = None
    while k:
        if k & 1:
            out = add(out, p)
        p = add(p, p)
        k >>= 1
    return out


def parked_finish(point, z, r):
    """Use the exact promoted W=ZZZ*d cut with pre-inverse parked numerators."""
    xp, yp = point
    xr, yr = r
    zz = z * z % P
    zzz = zz * z % P
    X = xp * zz % P
    Y = yp * zzz % P
    d = (xr * zz - X) % P
    W = zzz * d % P
    if W == 0:
        return None
    # These are the two 256-bit values parked for an earlier epoch.
    p1 = (yr * zzz - Y) * zz % P
    p2 = (yr * zzz + Y) * zz % P
    inv = pow(W, -1, P)
    m1 = p1 * inv % P
    m2 = p2 * inv % P
    c = 3 * xr * xr * pow(2 * yr, -1, P) % P
    s = (m1 + m2) % P
    x1 = (s * (m1 - c) + xr) % P
    y1 = (m1 * (xr - x1) - yr) % P
    x2 = (s * (m2 - c) + xr) % P
    y2 = (yr - m2 * (xr - x2)) % P
    return (x1, y1), (x2, y2)


def main():
    rng = random.Random(0x4B335041524B4544)
    r = mul(1_984_321)
    cases = 0
    mutation_rejections = 0
    boundary = [1, 2, 3, P - 2, P - 1]
    scalars = boundary + [rng.randrange(1, N) for _ in range(4096)]
    zs = boundary + [rng.randrange(1, P) for _ in range(4096)]
    for scalar, z in zip(scalars, zs):
        point = mul(scalar)
        if point == r or point == (r[0], (-r[1]) % P):
            continue
        got = parked_finish(point, z, r)
        want = (add(point, r), add(point, (r[0], (-r[1]) % P)))
        assert got == want, (scalar, z, got, want)
        # Deliberately omit ZZ from the parked value: this must almost always fail.
        xp, yp = point
        xr, yr = r
        zz, zzz = z * z % P, z * z % P * z % P
        X, Y = xp * zz % P, yp * zzz % P
        inv = pow(zzz * (xr * zz - X) % P, -1, P)
        bad_m1 = (yr * zzz - Y) * inv % P
        good_m1 = (yr * zzz - Y) * zz % P * inv % P
        mutation_rejections += bad_m1 != good_m1
        cases += 1
    assert mutation_rejections >= cases - len(boundary)
    print(f"PASS: {cases} parked-finish recoveries; {mutation_rejections} omitted-ZZ mutations rejected")


if __name__ == "__main__":
    main()
