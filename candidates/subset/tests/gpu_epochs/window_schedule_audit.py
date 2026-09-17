#!/usr/bin/env python3
"""Host audit for the short-epoch window SHA scheduler.

This mirrors window_schedule_shared.cuh with deterministic synthetic rows. It
checks that the compact first-state cache is large enough for the retained
256-window schedule and that the optimized first/second/constant schedule path
matches ordinary SHA-256 compression for every lane.
"""

from __future__ import annotations

import hashlib
import random
import struct


K = [
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
]

FIRST_STATE_CAP = 64
CUT = 137


def rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF


def s0(x: int) -> int:
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)


def s1(x: int) -> int:
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)


def big_s0(x: int) -> int:
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)


def big_s1(x: int) -> int:
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)


def ch(x: int, y: int, z: int) -> int:
    return z ^ (x & (y ^ z))


def maj(x: int, y: int, z: int) -> int:
    return (x & y) | (z & (x | y))


def expand(words: list[int]) -> list[int]:
    w = words[:]
    for i in range(16, 64):
        w.append((w[i - 16] + s0(w[i - 15]) + w[i - 7] + s1(w[i - 2])) & 0xFFFFFFFF)
    return w


def compress(state: tuple[int, ...], words: list[int]) -> tuple[int, ...]:
    sched = expand(words)
    a, b, c, d, e, f, g, h = state
    for i in range(64):
        t1 = (h + big_s1(e) + ch(e, f, g) + K[i] + sched[i]) & 0xFFFFFFFF
        t2 = (big_s0(a) + maj(a, b, c)) & 0xFFFFFFFF
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & 0xFFFFFFFF, c, b, a, (t1 + t2) & 0xFFFFFFFF
    return tuple((state[i] + x) & 0xFFFFFFFF for i, x in enumerate((a, b, c, d, e, f, g, h)))


def compress_preadded(state: tuple[int, ...], schedule_plus_k: list[int]) -> tuple[int, ...]:
    a, b, c, d, e, f, g, h = state
    for i in range(64):
        t1 = (h + big_s1(e) + ch(e, f, g) + schedule_plus_k[i]) & 0xFFFFFFFF
        t2 = (big_s0(a) + maj(a, b, c)) & 0xFFFFFFFF
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & 0xFFFFFFFF, c, b, a, (t1 + t2) & 0xFFFFFFFF
    return tuple((state[i] + x) & 0xFFFFFFFF for i, x in enumerate((a, b, c, d, e, f, g, h)))


def window_second_key(w: tuple[int, int, int]) -> int:
    key = 0
    n = 0
    omitted = {x - CUT for x in w}
    for i in range(12, -1, -1):
        if i not in omitted:
            key = (key << 4) | i
            n += 1
            if n == 5:
                break
    return key


def window_first_key(w: tuple[int, int, int]) -> int:
    key = 0
    n = 0
    omitted = {x - CUT for x in w}
    for i in range(13):
        if i not in omitted:
            key = (key << 4) | i
            n += 1
            if n == 6:
                break
    return key


def retained_windows() -> list[tuple[int, int, int]]:
    windows = []
    for a in range(13):
        for b in range(a + 1, 13):
            for c in range(b + 1, 13):
                if a >= 1 and c <= 7 and not (a == 1 and b == 2):
                    continue
                windows.append((CUT + a, CUT + b, CUT + c))
    assert len(windows) == 256
    return sorted(windows, key=lambda w: (window_second_key(w), window_first_key(w)))


def prepare(rows: bytes, windows: list[tuple[int, int, int]], constant: list[int]):
    first_unique: list[list[int]] = []
    first_classes = []
    second_unique: list[list[int]] = []
    second_classes = []
    second_sched: list[list[int]] = []
    lane_words: list[list[int]] = []
    for w in windows:
        block = bytearray(128)
        pos = 8
        omitted = set(w)
        for i in range(CUT, 150):
            if i in omitted:
                continue
            block[pos:pos + 10] = rows[i * 10:(i + 1) * 10]
            pos += 10
        for word in constant[:5]:
            block[pos:pos + 4] = word.to_bytes(4, "big")
            pos += 4
        assert pos == 128
        words = list(struct.unpack(">32I", block))
        lane_words.append(words)
        first = words[2:16]
        if first not in first_unique:
            if len(first_unique) >= FIRST_STATE_CAP:
                raise AssertionError("first schedule cap exceeded")
            first_unique.append(first)
        first_classes.append(first_unique.index(first))
        second = words[16:32]
        if second not in second_unique:
            second_unique.append(second)
            second_sched.append([(x + k) & 0xFFFFFFFF for x, k in zip(expand(second), K)])
        second_classes.append(second_unique.index(second))
    return first_unique, first_classes, second_sched, second_classes, lane_words


def main() -> None:
    rows = b"".join(hashlib.sha256(f"row:{i}".encode()).digest()[:10] for i in range(150))
    constant = [int.from_bytes(hashlib.sha256(f"const:{i}".encode()).digest()[:4], "big") for i in range(69)]
    windows = retained_windows()
    first_unique, first_classes, second_sched, second_classes, lane_words = prepare(rows, windows, constant)
    print(f"first_classes={len(first_unique)} second_classes={len(second_sched)} cap={FIRST_STATE_CAP}")
    assert len(first_unique) == 54
    rng = random.Random(0x51B5)
    const_blocks = [constant[5 + 16 * block:5 + 16 * (block + 1)] for block in range(4)]
    for trial in range(16):
        state0 = tuple(rng.getrandbits(32) for _ in range(8))
        rem = [rng.getrandbits(32), rng.getrandbits(32)]
        first_states = [compress(state0, rem + first) for first in first_unique]
        for lane, words in enumerate(lane_words):
            generic = compress(state0, rem + words[2:16])
            generic = compress(generic, words[16:32])
            optimized = first_states[first_classes[lane]]
            optimized = compress_preadded(optimized, second_sched[second_classes[lane]])
            for block in const_blocks:
                generic = compress(generic, block)
                optimized = compress(optimized, block)
            assert optimized == generic, (trial, lane, windows[lane])
    print("window schedule audit passed")


if __name__ == "__main__":
    main()
