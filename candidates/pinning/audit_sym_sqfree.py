#!/usr/bin/env python3
"""CPU identity for stacking the subset squaring-free finish onto d93b4cd SYM.

Crown `d93b4cd` (commit f0f4256) uses xlib's 6-plane symmetric recovery:

    t = 1/(xR-xP), u = yR*t, v = yP*t, K = 3*xR^2
    F = 2*u^2 - K*t + xR, H = 2*u*v
    x(P+R) = F-H, x(P-R) = F+H

The remaining square is `_ModSqr` of u (plus a normalize/neg workaround
because GPUMath's square drops a near-p carry). The subset-promoted
on-curve identity (`e00f556`, also `db5767e` on the old 8-plane tree)
uses c = 3*xR^2/(2*yR) and no squares. Algebra:

    2*u*c = K*t, so F = 2*u*(u-c) + xR

This keeps the 6-plane Y/ZZZ/W layout (do not restore C or ZZ). It is
not a rematch of `db5767e`'s 8-plane W=ZZZ*d port, and it is not
QSB_FINAL_TEMPLATE (already -0.9% on this lineage).
"""

from __future__ import annotations

import random
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


def on_curve(pt):
    x, y = pt
    return (y * y - (x * x * x + 7)) % P == 0


def sym_sqfree_pair(Pxy, Rxy):
    """Return (x_plus, x_minus, y1_parity, y2_parity) two ways plus affine."""
    xP, yP = Pxy
    xR, yR = Rxy
    dx = (xR - xP) % P
    if dx == 0:
        return None
    t = pow(dx, -1, P)
    u = yR * t % P
    v = yP * t % P
    K = 3 * xR * xR % P
    c = K * pow(2 * yR, -1, P) % P

    F_sqr = (2 * u * u - K * t + xR) % P
    F_id = (2 * u * ((u - c) % P) + xR) % P
    if F_sqr != F_id:
        raise AssertionError("F identity failed")

    H = (2 * u * v) % P
    x_plus = (F_id - H) % P
    x_minus = (F_id + H) % P

    # subset form: lambda1=u-v, m2=u+v
    lam1 = (u - v) % P
    m2 = (u + v) % P
    x1 = ((lam1 + m2) % P) * ((lam1 - c) % P) % P
    x1 = (x1 + xR) % P
    x2 = ((lam1 + m2) % P) * ((m2 - c) % P) % P
    x2 = (x2 + xR) % P
    if x1 != x_plus or x2 != x_minus:
        raise AssertionError("subset vs SYM x mismatch")

    y1 = (lam1 * ((xR - x_plus) % P) - yR) % P
    y2 = (-(m2 * ((xR - x_minus) % P) - yR)) % P

    plus = affine_add(Pxy, Rxy)
    minus = affine_add(Pxy, (xR, (-yR) % P))
    if plus is None or minus is None:
        return None
    if plus[0] != x_plus or minus[0] != x_minus:
        raise AssertionError("affine x mismatch")
    if (plus[1] & 1) != (y1 & 1) or (minus[1] & 1) != (y2 & 1):
        raise AssertionError("y parity mismatch")
    return x_plus, x_minus, y1 & 1, y2 & 1


def audit_source():
    src = Path(__file__).with_name("pinning.cu").read_text(encoding="utf-8")
    assert "pin_u2rc_words" in src
    assert "pin_u2rk_words" not in src
    assert "_ModSqr(f, h)" not in src
    assert "F = 2*u*(u-c)+xR" in src
    assert "qsb_xyzz_finish_symmetric(" in src
    assert "QSB_STATE_PLANES (QSB_SYM_FINISH ? 6u : 8u)" in src


def main():
    audit_source()
    rng = random.Random(20260917)
    # Four fixed R points (problem-like) plus random R.
    Rs = [scalar_mult(1), scalar_mult(2), scalar_mult(N - 1), scalar_mult(7)]
    cases = 0
    skipped = 0
    for r_i, R in enumerate(Rs):
        assert on_curve(R)
        for k in (1, 2, 3, N - 1, N - 2, N - 3):
            Pxy = scalar_mult(k)
            if Pxy[0] == R[0]:
                skipped += 1
                continue
            sym_sqfree_pair(Pxy, R)
            cases += 1
        for _ in range(2500):
            Pxy = scalar_mult(rng.randrange(1, N))
            if Pxy[0] == R[0]:
                skipped += 1
                continue
            sym_sqfree_pair(Pxy, R)
            cases += 1
    for _ in range(2500):
        R = scalar_mult(rng.randrange(1, N))
        Pxy = scalar_mult(rng.randrange(1, N))
        if Pxy[0] == R[0]:
            skipped += 1
            continue
        sym_sqfree_pair(Pxy, R)
        cases += 1
    print(f"audit_sym_sqfree: {cases} exact matches, {skipped} singular skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
