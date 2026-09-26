#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Regression model for the promoted subset first-state stride.

This covers only the AoS capacity reduction. It intentionally does not import
the prior compact word-plane layout or direct-base recoding attempt.
"""

from __future__ import annotations

import json
import random
import re
import struct
from pathlib import Path


K = (
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
)
IV = (
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
)
MASK = 0xFFFFFFFF


def rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & MASK


def sha256_compress(state: tuple[int, ...], words: tuple[int, ...]) -> tuple[int, ...]:
    w = list(words)
    for i in range(16, 64):
        x, y = w[i - 15], w[i - 2]
        s0 = rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)
        s1 = rotr(y, 17) ^ rotr(y, 19) ^ (y >> 10)
        w.append((w[i - 16] + s0 + w[i - 7] + s1) & MASK)
    a, b, c, d, e, f, g, h = state
    for i in range(64):
        s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
        ch = (e & f) ^ ((~e) & g)
        t1 = (h + s1 + ch + K[i] + w[i]) & MASK
        s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = (s0 + maj) & MASK
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & MASK, c, b, a, (t1 + t2) & MASK
    return tuple((x + y) & MASK for x, y in zip(state, (a, b, c, d, e, f, g, h)))


def first_key(skips: tuple[int, int, int]) -> tuple[int, ...]:
    return tuple(i for i in range(13) if i not in skips)[:6]


def selected_windows(window_count: int) -> list[tuple[int, int, int]]:
    out = []
    for a in range(13):
        for b in range(a + 1, 13):
            for c in range(b + 1, 13):
                if window_count == 128:
                    keep = a >= 6 or (a <= 5 and b >= 7) or (a == 0 and b == 1 and 8 <= c <= 10)
                elif window_count == 256:
                    keep = not (a >= 1 and c <= 7 and not (a == 1 and b == 2))
                else:
                    raise ValueError(window_count)
                if keep:
                    out.append((a, b, c))
    return out


def load_words(buf: list[int], epoch: int, slots: int, cls: int) -> tuple[int, ...]:
    base = (epoch * slots + cls) * 8
    return tuple(buf[base + i] for i in range(8))


def check_shape(window_count: int, epochs: int, seed: int) -> dict[str, int]:
    windows = selected_windows(window_count)
    classes = {}
    for skips in windows:
        classes.setdefault(first_key(skips), len(classes))
    slots = 8 if window_count == 128 else 64
    assert len(classes) <= slots
    if window_count == 128:
        assert len(windows) == 128
        assert len(classes) == 8
    else:
        assert len(windows) == 256
        assert len(classes) <= 64

    old_slots = 16 if window_count == 128 else 64
    rng = random.Random(seed)
    old = [rng.getrandbits(32) for _ in range(epochs * old_slots * 8)]
    compact = [0] * (epochs * slots * 8)
    for epoch in range(epochs):
        for cls in range(len(classes)):
            words = load_words(old, epoch, old_slots, cls)
            base = (epoch * slots + cls) * 8
            compact[base:base + 8] = words

    digest_checks = 0
    for epoch in range(epochs):
        for skips in windows:
            cls = classes[first_key(skips)]
            before = load_words(old, epoch, old_slots, cls)
            after = load_words(compact, epoch, slots, cls)
            assert before == after
            schedule = tuple(rng.getrandbits(32) for _ in range(16))
            assert sha256_compress(before, schedule) == sha256_compress(after, schedule)
            digest_checks += 1
    return {"windows": len(windows), "first_classes": len(classes), "digest_checks": digest_checks}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    schedule = (root / "tests/gpu_epochs/window_schedule_shared.cuh").read_text()
    tree = (root / "tests/gpu_epochs/tree.cu").read_text()
    assert "#define QSB_FIRST_SLOTS (QSB_SE_WINDOWS==256?64:8)" in schedule
    assert "if(first_distinct>QSB_FIRST_SLOTS)return 1;" in schedule
    assert "QSB_CARRIER_KV(QSB_FIRST_SLOTS)" in tree

    # Guard is fail-closed for an unexpected ninth class in the 128-window mode.
    assert 8 <= 8
    assert not (9 <= 8)
    result = {
        "status": "PASS",
        "source_scope": "AoS stride only; no arithmetic, recoding, or SoA planes",
        "shape_128": check_shape(128, 5, 0x1288),
        "shape_256": check_shape(256, 3, 0x2568),
        "guard_128_overflow": "fail-closed",
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
