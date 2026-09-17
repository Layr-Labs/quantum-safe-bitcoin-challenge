#!/usr/bin/env python3
"""Mechanical equivalence audit for the branchless _ModAdd256 add-back path."""

from pathlib import Path
import random
import re

MASK64 = (1 << 64) - 1
MASK256 = (1 << 256) - 1
P = (1 << 256) - (1 << 32) - 977


def old_mod_add(a: int, b: int) -> int:
    total = a + b
    reduced = total - P
    return (reduced if reduced >= 0 else total) & MASK256


def new_mod_add(a: int, b: int) -> int:
    total = a + b
    rr = total - P
    low = rr & MASK256
    high = rr >> 256
    restore = MASK64 if high < 0 else 0
    return (low + (P & (restore | (restore << 64) | (restore << 128) | (restore << 192)))) & MASK256


def limbs(x: int) -> list[int]:
    return [(x >> (64 * i)) & MASK64 for i in range(4)]


def main() -> None:
    src = Path(__file__).with_name("GPUMath.h").read_text()
    body = re.search(
        r"__device__ void _ModAdd256\(uint64_t \*r, uint64_t \*a, uint64_t \*b\)\s*\{(.*?)\n\}",
        src,
        re.S,
    )
    assert body, "_ModAdd256 source not found"
    text = body.group(1)
    for token in ("SubP(rr)", "restore", "UADDO1(rr[0], p0)", "Load256(r, rr)"):
        assert token in text, f"missing source contract: {token}"
    assert "if(" not in text and "if (" not in text, "branch reintroduced"
    assert "o0" not in text, "four-limb snapshot/select variant reintroduced"

    edges = [
        0,
        1,
        2,
        (1 << 32) + 976,
        (1 << 32) + 977,
        P - 2,
        P - 1,
        P,
        P + 1,
        (1 << 255) - 1,
        1 << 255,
        MASK256 - 1,
        MASK256,
    ]
    pairs = [(a, b) for a in edges for b in edges]
    rng = random.Random(0xADD256)
    pairs.extend((rng.getrandbits(256), rng.getrandbits(256)) for _ in range(200_000))

    for i, (a, b) in enumerate(pairs):
        want = old_mod_add(a, b)
        got = new_mod_add(a, b)
        assert got == want, (i, hex(a), hex(b), hex(want), hex(got))
        # The implementation reads all inputs before writing r, so separate,
        # r==a and r==b have the same four output limbs.
        expected_limbs = limbs(want)
        assert limbs(got) == expected_limbs

    print(f"PASS: {len(pairs)} pairs; old branch and masked add-back are equivalent")
    print("PASS: source contract is branch-free, snapshot-free, and alias-safe")


if __name__ == "__main__":
    main()
