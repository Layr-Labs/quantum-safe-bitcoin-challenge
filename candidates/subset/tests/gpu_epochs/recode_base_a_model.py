#!/usr/bin/env python3
"""CPU proof/model and scalar_audit fixture generator for QSB_RECODE_BASE_A.

The curve generator has order N, so equality of the reconstructed scalars
modulo N is exactly equality of the resulting points.
"""

import argparse
import random
import struct
from pathlib import Path

N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
MASK256 = (1 << 256) - 1
INV2 = pow(2, -1, N)
SHIFTS = [0] + [17 * c + 1 for c in range(1, 15)]
WIDTHS = [18] + [17] * 14


def reduced(k: int) -> int:
    # Production inputs are 256-bit, and 2*N > 2^256, so one subtraction is exact.
    return k - N if k >= N else k


def setup(k: int, base_a: bool) -> tuple[int, int]:
    k = reduced(k)
    m = k if base_a else (2 * k) % N
    if m & 1:
        return m, 1
    return N - m, -1


def peel(m: int, sign: int) -> list[int]:
    out = []
    for width in WIDTHS[:-1]:
        digit = (m & ((1 << (width + 1)) - 1)) - (1 << width)
        out.append(sign * digit)
        m = 2 * (m >> (width + 1)) + 1
    out.append(sign * m)
    return out


def reconstruct(digits: list[int]) -> int:
    return sum(digit << shift for digit, shift in zip(digits, SHIFTS))


def direct_codes(m: int, sign: int) -> list[int]:
    codes = []
    pos = 1
    for c, width in enumerate(WIDTHS):
        field = (m >> pos) & ((1 << width) - 1)
        if c == len(WIDTHS) - 1:
            idx = field & ((1 << (width - 1)) - 1)
            neg = int(sign < 0)
        else:
            top = field >> (width - 1)
            idx = (field ^ (top - 1)) & ((1 << (width - 1)) - 1)
            neg = (top ^ 1) ^ int(sign < 0)
        codes.append(idx | (neg << 31))
        pos += width
    return codes


def recurrence_codes(m: int, sign: int) -> list[int]:
    codes = []
    for digit in peel(m, sign):
        absolute = abs(digit)
        assert absolute & 1
        codes.append(((absolute - 1) >> 1) | (int(digit < 0) << 31))
    return codes


def audit_scalar(k: int) -> None:
    old_m, old_sign = setup(k, False)
    new_m, new_sign = setup(k, True)
    old_digits = peel(old_m, old_sign)
    new_digits = peel(new_m, new_sign)
    old_sum = reconstruct(old_digits)
    new_sum = reconstruct(new_digits)
    kr = reduced(k)
    assert old_sum == old_sign * old_m
    assert new_sum == new_sign * new_m
    assert old_sum % N == (2 * kr) % N
    assert new_sum % N == kr
    assert (old_sum * INV2) % N == new_sum % N
    assert direct_codes(old_m, old_sign) == recurrence_codes(old_m, old_sign)
    assert direct_codes(new_m, new_sign) == recurrence_codes(new_m, new_sign)


def fixture_cases() -> list[int]:
    boundaries = [
        0, 1, 2, N - 2, N - 1, N,
        1 << 255, (1 << 255) + 1, (3 << 254) - 1,
        MASK256, 0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,
        0x5555555555555555555555555555555555555555555555555555555555555555,
    ]
    rng = random.Random(0x12B_A11D17)
    return boundaries + [rng.getrandbits(256) for _ in range(4096)]


def write_fixture(path: Path) -> int:
    cases = fixture_cases()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(struct.pack("<I", len(cases)))
        for k in cases:
            m, sign = setup(k, True)
            out.write(struct.pack("<4Q", *(k >> (64 * i) & ((1 << 64) - 1) for i in range(4))))
            out.write(struct.pack("<4Q", *(m >> (64 * i) & ((1 << 64) - 1) for i in range(4))))
            out.write(struct.pack("<i", sign))
            out.write(struct.pack("<15I", *direct_codes(m, sign)))
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=int, default=1_000_000)
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()

    boundaries = fixture_cases()[:12]
    for k in boundaries:
        audit_scalar(k)

    # Explicitly cover all-zero and all-one direct digit fields. M remains odd,
    # as required by the regular recoder; these are valid extrema of its decoder.
    for synthetic_m in (1, MASK256):
        assert direct_codes(synthetic_m, 1) == recurrence_codes(synthetic_m, 1)
        assert direct_codes(synthetic_m, -1) == recurrence_codes(synthetic_m, -1)

    rng = random.Random(0xC7_12B_5EED)
    for _ in range(args.random):
        audit_scalar(rng.getrandbits(256))

    table_entries = 0
    for shift, width in zip(SHIFTS, WIDTHS):
        for digit_index in range(1 << (width - 1)):
            odd = 2 * digit_index + 1
            old_entry = odd * (1 << shift) * INV2 % N
            new_entry = odd * (1 << shift) % N
            assert new_entry == 2 * old_entry % N
            table_entries += 1

    fixture_count = write_fixture(args.fixture) if args.fixture else 0
    print(f"scalar equivalence: PASS ({args.random:,} random + {len(boundaries)} boundary)")
    print("digit-field extrema: PASS (all-zero/all-one, both signs)")
    print(f"table rescaling: PASS ({table_entries:,} entries, new == 2*old mod n)")
    if args.fixture:
        print(f"base-A scalar_audit fixture: {args.fixture} ({fixture_count:,} cases)")


if __name__ == "__main__":
    main()
