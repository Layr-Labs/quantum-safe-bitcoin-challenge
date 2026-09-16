#!/usr/bin/env python3
"""Audit the dual-chain compressed-key SHA mapping and source contract."""

import hashlib
import random
from pathlib import Path

from audit_sparse_pk33 import (
    generic_first_mix,
    sha256_block,
    small0,
    small1,
    sparse_first_mix,
)


def compressed_words(x_coordinate: bytes, parity: int) -> list[int]:
    message = bytes((2 + parity,)) + x_coordinate
    padded = message + b"\x80" + b"\0" * 22 + (264).to_bytes(8, "big")
    return [int.from_bytes(padded[offset:offset + 4], "big")
            for offset in range(0, 64, 4)]


def interleaved_first_mix(left: list[int], right: list[int]):
    """Execute each dependency-preserving ring update left then right."""
    outputs = [list(left), list(right)]
    for index in range(16):
        for words in outputs:
            words[index] = (
                words[index]
                + small1(words[(index + 14) & 15])
                + words[(index + 9) & 15]
                + small0(words[(index + 1) & 15])
            ) & 0xFFFFFFFF
    return outputs


def audit_source():
    source = Path(__file__).with_name("pinning.cu").read_text()
    assert "_SHA256TransformPk33Dual(hs0,hs1,q1x,q2x,y_parities)" in source
    assert "QSB_SHA256_DUAL_STEP" in source
    assert "QSB_SHA256_DUAL_WMIX" in source
    assert "if (FAST_TAIL) {" in source
    assert "if(gpu_bench_valid_words(hs0))ri=0;" in source
    assert "else if(gpu_bench_valid_words(hs1))ri=1;" in source
    assert source.index("_SHA256TransformPk33Dual(hs0,hs1,q1x,q2x,y_parities)") < source.index("/* Check both pubkeys")


def main():
    audit_source()
    rng = random.Random(0x4455414C534841)
    for _ in range(50_000):
        x0 = rng.randbytes(32)
        x1 = rng.randbytes(32)
        parity0 = rng.getrandbits(1)
        parity1 = rng.getrandbits(1)
        words0 = compressed_words(x0, parity0)
        words1 = compressed_words(x1, parity1)
        assert words0[9:15] == [0] * 6 and words0[15] == 0x108
        assert words1[9:15] == [0] * 6 and words1[15] == 0x108
        mixed0, mixed1 = interleaved_first_mix(words0, words1)
        assert mixed0 == generic_first_mix(words0)
        assert mixed1 == generic_first_mix(words1)
        assert mixed0 == sparse_first_mix(words0[:9])
        assert mixed1 == sparse_first_mix(words1[:9])
        message0 = bytes((2 + parity0,)) + x0
        message1 = bytes((2 + parity1,)) + x1
        assert sha256_block(message0) == hashlib.sha256(message0).digest()
        assert sha256_block(message1) == hashlib.sha256(message1).digest()
    print("PASS: 50000 dual compressed-key mappings and interleaved first mixes")


if __name__ == "__main__":
    main()
