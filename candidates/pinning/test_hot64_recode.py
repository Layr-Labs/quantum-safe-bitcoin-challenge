# SPDX-License-Identifier: GPL-3.0-only
"""Independent integer oracle for the bounded HOT64 signed GLV recoder.
This checks algebra and record bounds, not CUDA throughput or the full split.
"""
import json
import random

SHIFTS = (0, 18, 37, 56, 75, 101)
WIDTHS = (18, 19, 19, 19, 26)
COUNTS = (262144, 262144, 262144, 262144, 33554432, 42639943)
CENTER = 85279885
BOUND = int('a2a8918ca85bafe22016d0b917e4dd77', 16)
BIAS = ((CENTER + 1) << 100) - (1 << 17)

def reconstruct(magnitude: int, negative: int) -> int:
    result = 0
    for chunk in range(6):
        field = magnitude >> SHIFTS[chunk]
        if chunk == 0:
            index = field & ((1 << 18) - 1)
            local_negative = 0
            coefficient = BIAS + index
        elif chunk == 5:
            digit = 2 * field - CENTER
            assert digit & 1 and abs(digit) <= CENTER
            local_negative = int(digit < 0)
            index = (abs(digit) - 1) // 2
            coefficient = (2 * index + 1) << 100
        else:
            width = WIDTHS[chunk]
            field &= (1 << width) - 1
            local_negative = 1 - (field >> (width - 1))
            index = (field ^ -local_negative) & ((1 << (width - 1)) - 1)
            coefficient = (2 * index + 1) << (SHIFTS[chunk] - 1)
        assert 0 <= index < COUNTS[chunk], (chunk, index)
        result += -coefficient if local_negative ^ negative else coefficient
    return result

def main() -> None:
    rng = random.Random(77)
    values = [0, 1, BOUND, BOUND - 1]
    values.extend(rng.randrange(BOUND + 1) for _ in range(100000))
    for shift in SHIFTS[1:]:
        values.extend(max(0, min(BOUND, (1 << shift) + delta)) for delta in (-1, 0, 1))
    for magnitude in values:
        for negative in (0, 1):
            expected = -magnitude if negative else magnitude
            assert reconstruct(magnitude, negative) == expected, (magnitude, negative)
    assert sum(COUNTS) == 77242951
    assert sum(COUNTS) * 64 == 4943548864
    assert sum(COUNTS[:4]) * 64 == 64 * 1024 * 1024
    print(json.dumps({'signed_recode_cases': len(values) * 2, 'mismatches': 0,
                      'records': sum(COUNTS), 'table_bytes': sum(COUNTS) * 64}))

if __name__ == '__main__':
    main()
