#!/usr/bin/env python3
"""Host differential for stream signed-odd window peeling.

qsb_decode_to_shared extracts fifteen mixed windows from the signed 2k-n
residue M by independent limb/shift addressing and stores packed (idx,neg)
codes. The production chain now peels those same windows from a sliding
register window at each table load. This script is the extracted-function
identity: sliding peeks equal the independent extracts, including the last
window's explicit sign of D.
"""
import random

GT_CHUNKS = 15


def extract_independent(M, pos, bits):
    j, sh = divmod(pos, 64)
    value = M[j] >> sh
    if j < 3 and sh > 46:
        value |= M[j + 1] << (64 - sh)
    return value & ((1 << bits) - 1)


class DigitWindow:
    def __init__(self, M, pos):
        w0, w1, w2, w3 = M
        sh = pos
        while sh >= 64:
            w0, w1, w2, w3 = w1, w2, w3, 0
            sh -= 64
        self.w = [w0, w1, w2, w3]
        self.sh = sh

    def peek(self):
        w0, w1 = self.w[0], self.w[1]
        sh = self.sh
        return (w0 >> sh | ((w1 << 1) << (63 - sh))) & 0xFFFFFFFF

    def advance(self, bits):
        self.sh += bits
        if self.sh >= 64:
            self.w = self.w[1:] + [0]
            self.sh -= 64


def signed_odd_code(f, bits, tm):
    idx = (f ^ (tm & 0xFFFFFFFF)) & ((1 << (bits - 1)) - 1)
    negmask = 0xFFFFFFFFFFFFFFFF if tm < 0 else 0
    return idx, negmask


def decode_shared(M, negative):
    codes = []
    for c in range(GT_CHUNKS):
        pos = 1 if c == 0 else 17 * c + 2
        bits = 18 if c == 0 else 17
        f = extract_independent(M, pos, bits)
        tm = -negative if c == GT_CHUNKS - 1 else (f >> (bits - 1)) - 1
        codes.append(signed_odd_code(f, bits, tm))
    return codes


def decode_stream(M, negative):
    win = DigitWindow(M, 1)
    out = []
    f0 = win.peek() & ((1 << 18) - 1)
    win.advance(18)
    out.append(signed_odd_code(f0, 18, (f0 >> 17) - 1))
    f1 = win.peek() & ((1 << 17) - 1)
    win.advance(17)
    out.append(signed_odd_code(f1, 17, (f1 >> 16) - 1))
    for c in range(2, GT_CHUNKS):
        f = win.peek() & ((1 << 17) - 1)
        win.advance(17)
        tm = -negative if c == GT_CHUNKS - 1 else (f >> 16) - 1
        out.append(signed_odd_code(f, 17, tm))
    return out


def main():
    rng = random.Random(20260919)
    n = 0
    fixtures = [
        [1, 0, 0, 0],
        [2**64 - 1] * 4,
        [1, 0, 0, 1 << 63],
        [0x5555555555555555] * 4,
        [0xAAAAAAAAAAAAAAAA] * 4,
        [1, 0, 0, 0xFFFF],
        [1, (1 << 53) - 1, 0, 0],
        [0, 0, 0, 1 << 48],
    ]
    for M in fixtures:
        M = list(M)
        M[0] |= 1
        for negative in (0, 1):
            a, b = decode_shared(M, negative), decode_stream(M, negative)
            if a != b:
                raise SystemExit(f"fixture mismatch {M=} {negative=}")
            n += 1
    for _ in range(20000):
        M = [rng.getrandbits(64) for _ in range(4)]
        M[0] |= 1
        for negative in (0, 1):
            a, b = decode_shared(M, negative), decode_stream(M, negative)
            if a != b:
                raise SystemExit(f"random mismatch {M=} {negative=}")
            n += 1
    print(f"ok {n} recode pairs")


if __name__ == "__main__":
    main()
