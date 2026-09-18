#!/usr/bin/env python3
"""Portable audit for the short-epoch window schedule cache.

Mirrors window_schedule_shared.cuh and the retained-window construction in
tree.cu. The audit is intentionally source-local so the compact first-state
cache, default broadcast descriptor loads, and macro-on direct-load route can
be checked without CUDA hardware.
"""

from __future__ import annotations

import hashlib
import random
import struct


MASK32 = 0xFFFFFFFF
FIRST_STATE_CAP = 64
FIRST_COUNT_EXPECTED = 54
SECOND_COUNT_EXPECTED = 56
LANES_EXPECTED = 256
QSB_SE_CUT = 137


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


def u32(x: int) -> int:
    return x & MASK32


def rotr(x: int, n: int) -> int:
    return u32((x >> n) | (x << (32 - n)))


def sha256_compress(state: tuple[int, ...], words16: tuple[int, ...]) -> tuple[int, ...]:
    assert len(state) == 8
    assert len(words16) == 16
    w = list(words16)
    for i in range(16, 64):
        s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)
        s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)
        w.append(u32(w[i - 16] + s0 + w[i - 7] + s1))
    a, b, c, d, e, f, g, h = state
    for i in range(64):
        s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
        ch = (e & f) ^ ((~e) & g)
        t1 = u32(h + s1 + ch + K[i] + w[i])
        s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = u32(s0 + maj)
        h, g, f, e, d, c, b, a = g, f, e, u32(d + t1), c, b, a, u32(t1 + t2)
    return tuple(u32(x + y) for x, y in zip(state, (a, b, c, d, e, f, g, h)))


def digest_words(seed: int, label: str, count: int) -> tuple[int, ...]:
    out = bytearray()
    ctr = 0
    while len(out) < count * 4:
        out += hashlib.sha256(f"{label}:{seed}:{ctr}".encode("ascii")).digest()
        ctr += 1
    return struct.unpack(f">{count}I", out[: count * 4])


def rows_for_seed(seed: int) -> bytes:
    out = bytearray()
    for i in range(150):
        out += hashlib.sha256(f"row:{seed}:{i}".encode("ascii")).digest()[:10]
    return bytes(out)


def window_second_key(window: tuple[int, int, int]) -> int:
    key = 0
    kept = 0
    omitted = {x - QSB_SE_CUT for x in window}
    for i in range(12, -1, -1):
        if i in omitted:
            continue
        key = (key << 4) | i
        kept += 1
        if kept == 5:
            return key
    raise AssertionError("short second key")


def window_first_key(window: tuple[int, int, int]) -> int:
    key = 0
    kept = 0
    omitted = {x - QSB_SE_CUT for x in window}
    for i in range(13):
        if i in omitted:
            continue
        key = (key << 4) | i
        kept += 1
        if kept == 6:
            return key
    raise AssertionError("short first key")


def retained_windows() -> list[tuple[int, int, int]]:
    windows = []
    for a in range(13):
        for b in range(a + 1, 13):
            for c in range(b + 1, 13):
                if a >= 1 and c <= 7 and not (a == 1 and b == 2):
                    continue
                windows.append((QSB_SE_CUT + a, QSB_SE_CUT + b, QSB_SE_CUT + c))
    assert len(windows) == LANES_EXPECTED
    windows.sort(key=lambda w: (window_second_key(w), window_first_key(w)))
    return windows


def words_for_lane(rows: bytes, window: tuple[int, int, int], constant: tuple[int, ...]) -> tuple[int, ...]:
    msg = bytearray(128)
    pos = 8
    selected = 0
    for i in range(QSB_SE_CUT, 150):
        if selected < 3 and window[selected] == i:
            selected += 1
            continue
        msg[pos : pos + 10] = rows[i * 10 : i * 10 + 10]
        pos += 10
    assert pos == 108 and selected == 3
    for word in constant[:5]:
        msg[pos : pos + 4] = word.to_bytes(4, "big")
        pos += 4
    assert pos == 128
    return struct.unpack(">32I", msg)


def descriptor_direct(mid: tuple[int, ...], rem: tuple[int, int], lane: int, first_count: int) -> tuple[tuple[int, ...], tuple[int, int]]:
    assert 0 <= lane < first_count
    return mid, rem


def descriptor_broadcast(mid: tuple[int, ...], rem: tuple[int, int], lane: int, first_count: int) -> tuple[tuple[int, ...], tuple[int, int]]:
    assert 0 <= lane < first_count
    source_lane = lane & ~31
    assert source_lane < first_count
    return mid, rem


def descriptor_broadcast_source(lane: int, first_count: int) -> int:
    assert 0 <= lane < first_count
    source_lane = lane & ~31
    assert source_lane < first_count
    return source_lane


def audit_seed(seed: int, windows: list[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    rows = rows_for_seed(seed)
    constant = digest_words(seed, "constant", 5)
    mid = digest_words(seed, "midstate", 8)
    rem = digest_words(seed, "remainder", 2)

    first_unique: list[tuple[int, ...]] = []
    second_unique: list[tuple[int, ...]] = []
    first_class: list[int] = []
    second_class: list[int] = []
    lane_words: list[tuple[int, ...]] = []
    for window in windows:
        words = words_for_lane(rows, window, constant)
        lane_words.append(words)
        first = words[2:16]
        second = words[16:32]
        if first not in first_unique:
            if len(first_unique) >= FIRST_STATE_CAP:
                raise AssertionError("first-state cap exceeded")
            first_unique.append(first)
        if second not in second_unique:
            second_unique.append(second)
        first_class.append(first_unique.index(first))
        second_class.append(second_unique.index(second))

    assert len(first_unique) == FIRST_COUNT_EXPECTED
    assert len(second_unique) == SECOND_COUNT_EXPECTED
    assert max(first_class) < FIRST_STATE_CAP
    assert max(second_class) == SECOND_COUNT_EXPECTED - 1

    first_states = []
    descriptor_batches = set()
    for lane in range(len(first_unique)):
        source_lane = descriptor_broadcast_source(lane, len(first_unique))
        descriptor_batches.add(source_lane // 32)
        direct_mid, direct_rem = descriptor_direct(mid, rem, lane, len(first_unique))
        bcast_mid, bcast_rem = descriptor_broadcast(mid, rem, lane, len(first_unique))
        assert direct_mid == bcast_mid and direct_rem == bcast_rem
        direct_first = sha256_compress(direct_mid, direct_rem + first_unique[lane])
        bcast_first = sha256_compress(bcast_mid, bcast_rem + first_unique[lane])
        assert direct_first == bcast_first
        first_states.append(direct_first)

    for lane, words in enumerate(lane_words):
        compact = sha256_compress(first_states[first_class[lane]], words[16:32])
        direct_first = sha256_compress(mid, rem + words[2:16])
        direct_second = sha256_compress(direct_first, words[16:32])
        if compact != direct_second:
            raise AssertionError(f"compact SHA mismatch seed={seed} lane={lane}")
        if tuple(words[2:16]) != first_unique[first_class[lane]]:
            raise AssertionError(f"first-class mismatch seed={seed} lane={lane}")

    return len(first_unique), len(second_unique), len(descriptor_batches), len(windows)


def main() -> None:
    windows = retained_windows()
    rng = random.Random(0xC0640196)
    seeds = [0, 1, 2, 7, 19, 31, 32, 53, 54, 255, 256, 257, 12345, 0xFFFFFFFF]
    seeds.extend(rng.getrandbits(64) for _ in range(64))
    counts = {audit_seed(seed, windows) for seed in seeds}
    assert counts == {(FIRST_COUNT_EXPECTED, SECOND_COUNT_EXPECTED, 2, LANES_EXPECTED)}
    print(
        "PASS window schedule audit: "
        f"lanes={len(windows)} first={FIRST_COUNT_EXPECTED}/{FIRST_STATE_CAP} "
        f"second={SECOND_COUNT_EXPECTED} descriptor_batches=2 seeds={len(seeds)}"
    )


if __name__ == "__main__":
    main()
