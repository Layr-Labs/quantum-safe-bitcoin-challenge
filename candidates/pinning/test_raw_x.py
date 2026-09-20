#!/usr/bin/env python3
"""Host identities for QSB_RAW_X finish products.

qsb_recovery_mul = field mul then normalize to [0,p).
qsb_packed_raw_mul = field mul leaving a [0,2^256) representative.
qsb_add_boundary normalizes that representative only when a[3]==2^64-1.
Otherwise raw+a < 2p, so one conditional subtract of p is canonical x.

No GPU. SPDX-License-Identifier: GPL-3.0-only
"""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
K = 0x1000003D1
P = (1 << 256) - K
MASK256 = (1 << 256) - 1
LIMB = (1 << 64) - 1


def to_limbs(x: int) -> list[int]:
    return [(x >> (64 * i)) & LIMB for i in range(4)]


def from_limbs(ls: list[int]) -> int:
    return sum((ls[i] & LIMB) << (64 * i) for i in range(4))


def mod_add256(raw: int, a: int) -> int:
    """Device _ModAdd256 on two [0,2^256) inputs: add, then subtract p on wrap/overflow."""
    total = raw + a
    t = total & MASK256
    carry = total >> 256
    # _ModAdd256: add, then if carry or t>=p subtract p (one conditional).
    if carry or t >= P:
        t = (t - P) & MASK256
    return t


def add_boundary(raw: int, a: int) -> int:
    a3 = (a >> 192) & LIMB
    if a3 == LIMB:
        return raw % P
    return raw & MASK256


def audit(n: int = 4000):
    rng = random.Random(0x12A00001)
    bad = {"add": 0, "boundary_max": 0, "x_canon": 0}
    for _ in range(n):
        raw = rng.randrange(1 << 256)
        a = rng.randrange(P)
        s = add_boundary(raw, a)
        got = mod_add256(s, a)
        want = (raw + a) % P
        if got != want:
            bad["add"] += 1
        a_hi = from_limbs([rng.randrange(LIMB + 1) for _ in range(3)] + [LIMB])
        s_hi = add_boundary(raw, a_hi)
        if s_hi != raw % P:
            bad["boundary_max"] += 1
        if got >= P:
            bad["x_canon"] += 1
    return n, bad


def audit_source() -> dict[str, bool]:
    cu = (HERE / "pinning.cu").read_text()
    pr = (HERE / "PackedRecovery.cuh").read_text()
    return {
        "raw_x_on": "#define QSB_RAW_X 1" in cu,
        "fkiene_off": "#define QSB_FKIENE 0" in cu,
        "rp_sqr_off": "#define QSB_RP_SQR 0" in cu,
        "parity_raw": "qsb_packed_raw_mul(s,sum,t); qsb_add_boundary(s,a);" in pr,
        "no_dead_overwrite": pr.count("qsb_recovery_mul(s,sum,t)") == 2,
    }


def main():
    n, bad = audit()
    src = audit_source()
    assert all(v == 0 for v in bad.values()), bad
    assert all(src.values()), src
    print(json.dumps({"test": "QSB_RAW_X finish identities", "samples": n, "bad": bad, "source": src}))


if __name__ == "__main__":
    main()
