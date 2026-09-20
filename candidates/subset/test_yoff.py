#!/usr/bin/env python3
"""Host identities for subset QSB_YOFF (pinning P9 port)."""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
K = 0x1000003D1
P = (1 << 256) - K
C = (K - 1) // 2  # 0x800001E8
MASK256 = (1 << 256) - 1


def to_limbs(x: int) -> list[int]:
    return [(x >> (64 * i)) & ((1 << 64) - 1) for i in range(4)]


def from_limbs(ls: list[int]) -> int:
    return sum((ls[i] & ((1 << 64) - 1)) << (64 * i) for i in range(4))


def lazy_off(a: int, b: int) -> int:
    """Device _ModAddLazyOff: r = a+b-(K-1) with short carry through limb 1."""
    total = a + b
    t = total & MASK256
    k = total >> 256
    mk = (k - 1) & ((1 << 64) - 1)
    c0 = (mk & 0xFFFFFFFEFFFFFC2F) + 1
    ls = to_limbs(t)
    t0n = (ls[0] + c0) & ((1 << 64) - 1)
    cy = int((ls[0] + c0) >> 64)
    ls[0] = t0n
    ls[1] = (ls[1] + mk + cy) & ((1 << 64) - 1)
    return from_limbs(ls)


def yoff_to_y(yp: int) -> int:
    ls = to_limbs(yp)
    r0 = (ls[0] - C) & ((1 << 64) - 1)
    br = int(ls[0] < C)
    r1 = (ls[1] - br) & ((1 << 64) - 1)
    ls[0], ls[1] = r0, r1
    return from_limbs(ls)


def xor_neg(yp: int) -> int:
    return yp ^ MASK256


def audit(n: int = 4000):
    rng = random.Random(0x10FF0001)
    bad = {"xor": 0, "lazy": 0, "convert": 0, "sum": 0}
    for _ in range(n):
        y = rng.randrange(P)
        y2 = rng.randrange(P)
        yp = y + C
        y2p = y2 + C
        if xor_neg(yp) != ((P - y + C) & MASK256):
            bad["xor"] += 1
        if yoff_to_y(yp) != y:
            bad["convert"] += 1
        got = lazy_off(yp, y2p)
        want = (y + y2) % P
        if got % P != want:
            bad["lazy"] += 1
        if (yp + y2p - (K - 1)) % P != want:
            bad["sum"] += 1
    return n, bad


def audit_source() -> dict[str, bool]:
    tree = (HERE / "tests" / "gpu_epochs" / "tree.cu").read_text()
    math = (HERE / "GPUMath.h").read_text()
    sc = (HERE / "hit_filter_field_sc.cuh").read_text()
    replay = (HERE / "chain_replay_field.cuh").read_text()
    return {
        "tree_yoff_on": "#define QSB_YOFF 1" in tree,
        "table_kernel": "qsb_table_offset_y" in tree and "0x800001E8" in tree,
        "xor_load": "#if !QSB_YOFF" in tree and "0xFFFFFFFEFFFFFC30ULL&m" in tree,
        "lazy_off": "_ModAddLazyOff" in math and "0xFFFFFFFEFFFFFC2F" in math,
        "filter_ptx": "0xFFFFFFFEFFFFFC2F" in sc and "addc.u64 S1,S1,f1_h" in sc,
        "replay_ptx": "0xFFFFFFFEFFFFFC2F" in replay,
        "last_convert": "qsb_yoff_to_y" in tree,
    }


def main():
    n, bad = audit()
    src = audit_source()
    assert all(v == 0 for v in bad.values()), bad
    assert all(src.values()), src
    print(json.dumps({"test": "QSB_YOFF identities", "samples": n, "bad": bad, "source": src}))


if __name__ == "__main__":
    main()
