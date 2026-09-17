#!/usr/bin/env python3
"""Audit the subset _ModSub256 borrow fold.

The current source subtracts K = 2^32 + 977 after a 256-bit subtraction
borrows. The previous formulation added p = 2^256 - K on borrow. This script
checks the actual source contains the intended PTX-macro sequence, then compares
the old and new bit patterns over boundary and deterministic random cases.
"""

from __future__ import annotations

import argparse
import pathlib
import random
import re


MASK256 = (1 << 256) - 1
MASK64 = (1 << 64) - 1
K = (1 << 32) + 977
P = (1 << 256) - K


def limbs(x: int) -> list[int]:
    return [(x >> (64 * i)) & MASK64 for i in range(4)]


def from_limbs(values: list[int]) -> int:
    return sum(v << (64 * i) for i, v in enumerate(values))


def old_masked_p_add(a: int, b: int) -> int:
    r = (a - b) & MASK256
    if a < b:
        r = (r + P) & MASK256
    return r


def new_ptx_style_fold(a: int, b: int) -> int:
    al = limbs(a)
    bl = limbs(b)
    r: list[int] = []
    borrow = 0
    for i in range(4):
        value = al[i] - bl[i] - borrow
        if value < 0:
            value += 1 << 64
            borrow = 1
        else:
            borrow = 0
        r.append(value)

    # PTX subc.u64 t, 0, 0 after the subtract chain yields all ones on borrow,
    # else zero. The source masks that into either K or 0.
    t = ((0 - 0 - borrow) & MASK64) & K
    borrow = 0
    for i, subtrahend in enumerate((t, 0, 0, 0)):
        value = r[i] - subtrahend - borrow
        if value < 0:
            value += 1 << 64
            borrow = 1
        else:
            borrow = 0
        r[i] = value
    return from_limbs(r)


def extract_modsub_bodies(source: str) -> list[str]:
    pattern = re.compile(
        r"__device__\s+void\s+_ModSub256\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
        re.DOTALL,
    )
    return [match.group("body") for match in pattern.finditer(source)]


def check_source_shape(path: pathlib.Path) -> None:
    source = path.read_text()
    bodies = extract_modsub_bodies(source)
    if len(bodies) < 2:
        raise AssertionError("expected both _ModSub256 overloads in GPUMath.h")
    for body in bodies[:2]:
        required = [
            "USUB(t, 0ULL, 0ULL);",
            "t &= 0x1000003D1ULL;",
            "USUBO1(r[0], t);",
            "USUBC1(r[1], 0ULL);",
            "USUBC1(r[2], 0ULL);",
            "USUB1(r[3], 0ULL);",
        ]
        for needle in required:
            if needle not in body:
                raise AssertionError(f"missing expected source line: {needle}")
        forbidden = [
            "uint64_t T[4]",
            "0xFFFFFFFEFFFFFC2FULL & t",
            "UADDO1(r[0], T[0])",
            "UADDC1(r[1], T[1])",
            "UADDC1(r[2], T[2])",
            "UADD1(r[3], T[3])",
        ]
        for needle in forbidden:
            if needle in body:
                raise AssertionError(f"old masked-p add remains in body: {needle}")


def test_cases(random_cases: int) -> list[tuple[int, int]]:
    edges = [
        0,
        1,
        2,
        K - 1,
        K,
        K + 1,
        P - 2,
        P - 1,
        P,
        (P + 1) & MASK256,
        MASK256 - 1,
        MASK256,
    ]
    cases = [(a & MASK256, b & MASK256) for a in edges for b in edges]
    rng = random.Random(0x51B50003)
    cases.extend((rng.getrandbits(256), rng.getrandbits(256)) for _ in range(random_cases))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=200_000, help="deterministic random cases")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parent
    check_source_shape(root / "GPUMath.h")
    cases = test_cases(args.cases)
    for index, (a, b) in enumerate(cases):
        old = old_masked_p_add(a, b)
        new = new_ptx_style_fold(a, b)
        if old != new:
            raise AssertionError(
                f"case {index} mismatch: a={a:064x} b={b:064x} old={old:064x} new={new:064x}"
            )
    print(f"PASS: _ModSub256 borrow fold source+model cases={len(cases)}")


if __name__ == "__main__":
    main()
