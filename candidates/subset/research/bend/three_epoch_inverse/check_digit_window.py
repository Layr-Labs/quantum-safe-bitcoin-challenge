#!/usr/bin/env python3
"""Check the 17-bit carried digit window against direct bit extraction."""

import hashlib
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
TREE = HERE / "k3_candidate/tests/gpu_epochs/tree.cu"
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


def setup(k):
    m = (2 * (k % N)) % N
    return (m if m & 1 else N - m), (1 if m & 1 else -1)


def digit(f, sign, last):
    t = f >> 16
    idx = (f & 0xffff) if last else ((f ^ (t - 1)) & 0xffff)
    neg = (0 if last else t ^ 1) ^ (sign < 0)
    return idx, neg


def main():
    source = TREE.read_text()
    for token in ("#define ZLAB_DGW 1", "gt_window_digit(W[0]", "gt_window_shift17(W)",
                  "W[0]=(W[0]>>36)|(W[1]<<28)"):
        assert token in source, token
    rng = random.Random(260917622)
    scalars = [0, 1, N - 1, N, (1 << 256) - 1]
    scalars += [rng.randrange(1 << 256) for _ in range(100_000)]
    mutations = 0
    for k in scalars:
        m, sign = setup(k)
        window = m >> 36
        wrong = m >> 35
        for c in range(2, 15):
            last = c == 14
            direct = digit((m >> (17 * c + 2)) & 0x1ffff, sign, last)
            carried = digit(window & 0x1ffff, sign, last)
            assert carried == direct, (k, c)
            mutations += digit(wrong & 0x1ffff, sign, last) != direct
            window >>= 17
            wrong >>= 17
    assert mutations > len(scalars) * 10
    print(f"PASS: {len(scalars)} scalars, {len(scalars)*13} carried/direct digits, "
          f"{mutations} shifted-window mutations rejected; tree_sha256={hashlib.sha256(TREE.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
