#!/usr/bin/env python3
"""Algebra and source audit for the merged TOP16 cofactor traversal."""

from __future__ import annotations

import json
import random
from pathlib import Path


HERE = Path(__file__).resolve().parent
P = (1 << 256) - (1 << 32) - 977


def merged_top16(xs: list[int]) -> tuple[int, list[int]]:
    p2 = [xs[j] * xs[j + 8] % P for j in range(8)]
    e4 = [xs[i ^ 8] * p2[(i & 7) ^ 4] % P for i in range(16)]
    p4 = [p2[j] * p2[j + 4] % P for j in range(4)]
    e8 = [e4[i] * p4[(i & 3) ^ 2] % P for i in range(16)]
    p8 = [p4[j] * p4[j + 2] % P for j in range(2)]
    e16 = [e8[i] * p8[(i & 1) ^ 1] % P for i in range(16)]
    return p8[0] * p8[1] % P, e16


def direct(xs: list[int]) -> tuple[int, list[int]]:
    root = 1
    for x in xs:
        root = root * x % P
    excluded = []
    for i in range(16):
        v = 1
        for j, x in enumerate(xs):
            if i != j:
                v = v * x % P
        excluded.append(v)
    return root, excluded


def source_audit() -> dict[str, bool]:
    src = (HERE / "cofactor_checkpoint.h").read_text()
    return {
        "default_on": "#define QSB_TOP16 1" in src,
        "kill_switch": "#if QSB_TOP16" in src and "#elif QSB_TREE_TOP2" in src,
        "short_carry": "#define QSB_TOP16_MUL qsb_field_mul_sc" in src,
        "four_waves": all(f"if(tid<{n})" in src for n in (8, 20, 18, 17)),
        "stop_at_16": "for(int count=N;count>16;count>>=1)" in src,
        "resume_count_32": "for(int count=32;count<N;count<<=1)" in src,
        "resume_offset": "offset=2*N-64" in src,
    }


def main() -> None:
    rng = random.Random(0x7016C0FA)
    samples = 5000
    for _ in range(samples):
        xs = [rng.randrange(1, P) for _ in range(16)]
        assert merged_top16(xs) == direct(xs)
    src = source_audit()
    assert all(src.values()), src
    print(json.dumps({"test": "QSB_TOP16 algebra", "samples": samples, "source": src}))


if __name__ == "__main__":
    main()
