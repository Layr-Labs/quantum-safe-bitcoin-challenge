#!/usr/bin/env python3
"""Host audit for the exact signed-digit/chain schedule composition."""

from __future__ import annotations

import json
import random
from pathlib import Path


HERE = Path(__file__).resolve().parent
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1


def old_extract(m: list[int], c: int) -> int:
    pos = 1 if c == 0 else 17 * c + 2
    bits = 18 if c == 0 else 17
    j, sh = divmod(pos, 64)
    value = m[j] >> sh
    if j < 3 and sh > 46:
        value |= (m[j + 1] << (64 - sh)) & MASK64
    return value & ((1 << bits) - 1)


def funnel_extract(m: list[int], c: int) -> int:
    pos = 1 if c == 0 else 17 * c + 2
    bits = 18 if c == 0 else 17
    words = []
    for limb in m:
        words.extend((limb & MASK32, limb >> 32))
    wi, sh = divmod(pos, 32)
    if wi < 7:
        f = words[wi] if sh == 0 else ((words[wi] >> sh) | (words[wi + 1] << (32 - sh)))
    else:
        f = words[7] >> sh
    return f & ((1 << bits) - 1)


def old_code(f: int, c: int, negative: int) -> int:
    bits = 18 if c == 0 else 17
    tm = -negative if c == 14 else (f >> (bits - 1)) - 1
    idx = (f ^ (tm & MASK32)) & ((1 << (bits - 1)) - 1)
    return idx | ((tm < 0) << 31)


def folded_code(f: int, c: int, negative: int) -> int:
    bits = 18 if c == 0 else 17
    ftop = (f << (32 - bits)) & MASK32
    signed_ftop = ftop if ftop < (1 << 31) else ftop - (1 << 32)
    tm = -negative if c == 14 else (~signed_ftop) >> 31
    sign_bit = ((tm < 0) << 31) if c == 14 else ((~ftop) & (1 << 31))
    return ((f ^ (tm & MASK32)) & ((1 << (bits - 1)) - 1)) | sign_bit


def slot(c: int, lane: int) -> int:
    if c < 3:
        return 1536 + (c << 7) + lane
    return (((c - 3) >> 1) << 8) + (lane << 1) + ((c - 3) & 1)


def main() -> None:
    rng = random.Random(0xC1A17)
    comparisons = 0
    for _ in range(200_000):
        m = [rng.getrandbits(64) for _ in range(4)]
        negative = rng.getrandbits(1)
        for c in range(15):
            a = old_extract(m, c)
            b = funnel_extract(m, c)
            assert a == b, (c, m, a, b)
            assert old_code(a, c, negative) == folded_code(b, c, negative)
            comparisons += 1

    used = set()
    for c in range(15):
        for lane in range(128):
            s = slot(c, lane)
            assert 0 <= s < 3072
            assert s not in used
            used.add(s)
    for c in range(3, 15, 2):
        for lane in range(128):
            assert slot(c + 1, lane) == slot(c, lane) + 1
            assert slot(c, lane) % 2 == 0

    source = (HERE / "pinning.cu").read_text()
    required = {
        "window32": "#define QSB_DIGIT_WINDOW32 1",
        "sign_fold": "#define QSB_DIGIT_SIGN_FOLD 1",
        "seed_registers": "#define QSB_DIGIT_SEED_REG 1",
        "rotating_chain": "#define QSB_CHAIN_ROT2 1",
        "paired_shared": "#define QSB_DIGIT_PAIRLDS 1",
        "no_pointer_spill": "#define QSB_CHAIN_PTR 0",
        "pointer_loader": "gt_load_signed_flat_p",
    }
    result = {name: token in source for name, token in required.items()}
    assert all(result.values()), result
    print(json.dumps({
        "test": "exact signed-digit and chain schedule",
        "digit_comparisons": comparisons,
        "arena_words_used": len(used),
        "source": result,
    }))


if __name__ == "__main__":
    main()
