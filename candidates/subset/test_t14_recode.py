import random
import re
import unittest
from pathlib import Path


ORDER_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
SOURCE = Path(__file__).parent / "tests" / "gpu_epochs" / "tree.cu"


def recode_t14(value: int) -> list[int]:
    value %= ORDER_N
    doubled = (2 * value) % ORDER_N
    if doubled & 1:
        state, sign = doubled, 1
    else:
        state, sign = ORDER_N - doubled, -1

    digits: list[int] = []
    for bits in [19] * 4 + [18] * 9:
        digit = (state & ((1 << (bits + 1)) - 1)) - (1 << bits)
        digits.append(sign * digit)
        state = 2 * (state >> (bits + 1)) + 1
    digits.append(sign * state)
    return digits


class T14RecodeTests(unittest.TestCase):
    def test_t14_is_the_ranked_default(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        default = re.search(r"#ifndef ZLAB_T14\s+#define ZLAB_T14 (\d+)", text)
        self.assertIsNotNone(default)
        self.assertEqual(default.group(1), "1")

    def test_geometry_tiles_the_table_without_overlap(self) -> None:
        lengths = [1 << 18] * 4 + [1 << 17] * 10
        offsets = [c << 18 if c <= 4 else (4 << 18) + ((c - 4) << 17) for c in range(14)]
        self.assertEqual(offsets[0], 0)
        for previous, current, length in zip(offsets, offsets[1:], lengths):
            self.assertEqual(previous + length, current)
        self.assertEqual(offsets[-1] + lengths[-1], 2_359_296)

    def test_signed_digits_reconstruct_twice_the_scalar(self) -> None:
        rng = random.Random(0x514253543134)
        values = [
            0,
            1,
            2,
            ORDER_N - 2,
            ORDER_N - 1,
            ORDER_N,
            (1 << 256) - 1,
        ]
        values.extend(rng.getrandbits(256) for _ in range(20_000))
        shifts = [19 * c for c in range(5)] + [76 + 18 * (c - 4) for c in range(5, 14)]

        for value in values:
            digits = recode_t14(value)
            self.assertEqual(len(digits), 14)
            self.assertTrue(all(digit & 1 for digit in digits))
            self.assertTrue(all(abs(digit) < (1 << 19) for digit in digits[4:]))
            reconstructed = sum(digit << shift for digit, shift in zip(digits, shifts))
            self.assertEqual(reconstructed % ORDER_N, (2 * value) % ORDER_N)


if __name__ == "__main__":
    unittest.main()
