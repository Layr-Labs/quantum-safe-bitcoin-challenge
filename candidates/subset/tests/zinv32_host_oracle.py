#!/usr/bin/env python3
"""Portable source-bound tests for the cooperative zinv32 inverse."""
from __future__ import annotations

import hashlib
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

P = (1 << 256) - (1 << 32) - 977
ZI_B = 30
MASK32 = (1 << 32) - 1
MASK30 = (1 << 30) - 1
MM32 = 0xD2253531

ROOT = Path(__file__).resolve().parents[3]
TEST_DIR = ROOT / "candidates" / "subset" / "tests"
HEADER = TEST_DIR / "gpu_epochs" / "zinv32.cuh"
CPP = TEST_DIR / "host_zinv32_oracle.cpp"
BIN = Path(tempfile.gettempdir()) / ("qsb_host_zinv32_oracle_" + hashlib.sha256(HEADER.read_bytes()).hexdigest()[:16])


def limbs_signed_to_int(words: list[int]) -> int:
    value = sum((w & MASK32) << (32 * i) for i, w in enumerate(words))
    if words[8] & (1 << 31):
        value -= 1 << 288
    return value


def int_to_signed_limbs(value: int) -> list[int]:
    value %= 1 << 288
    return [(value >> (32 * i)) & MASK32 for i in range(9)]


def hex288(value: int) -> str:
    return f"{value % (1 << 288):072x}"


def s32(x: int) -> int:
    x &= MASK32
    return x - (1 << 32) if x & (1 << 31) else x


def trunc_row_orig(x: list[int], y: list[int], a: int, b: int, modp: int) -> list[int]:
    x = x[:]
    acc = a * x[0] + b * y[0]
    m = ((acc & MASK32) * MM32) & MASK30
    if not modp:
        m = 0
    acc -= 977 * m
    x[0] = acc & MASK32
    acc >>= 32
    acc += a * x[1] + b * y[1] - m
    x[1] = acc & MASK32
    acc >>= 32
    for i in range(2, 8):
        acc += a * x[i] + b * y[i]
        x[i] = acc & MASK32
        acc >>= 32
    acc += a * s32(x[8]) + b * s32(y[8]) + m
    x[8] = acc & MASK32
    for i in range(8):
        x[i] = ((x[i] >> ZI_B) | ((x[i + 1] << (32 - ZI_B)) & MASK32)) & MASK32
    x[8] = (s32(x[8]) >> ZI_B) & MASK32
    return x


def trunc_row_fixed(x: list[int], y: list[int], a: int, b: int, modp: int) -> list[int]:
    x = x[:]
    acc = a * x[0] + b * y[0]
    m = ((acc & MASK32) * MM32) & MASK30
    if not modp:
        m = 0
    acc -= 977 * m
    x[0] = acc & MASK32
    acc >>= 32
    acc += a * x[1] + b * y[1] - m
    x[1] = acc & MASK32
    acc >>= 32
    for i in range(2, 8):
        acc += a * x[i] + b * y[i]
        x[i] = acc & MASK32
        acc >>= 32
    acc += a * s32(x[8]) + b * s32(y[8]) + m
    full_acc = acc
    x[8] = acc & MASK32
    for i in range(8):
        x[i] = ((x[i] >> ZI_B) | ((x[i + 1] << (32 - ZI_B)) & MASK32)) & MASK32
    x[8] = (full_acc >> ZI_B) & MASK32
    return x


def exact_row(x: list[int], y: list[int], a: int, b: int, modp: int) -> list[int]:
    acc0 = a * (x[0] & MASK32) + b * (y[0] & MASK32)
    m = ((acc0 & MASK32) * MM32) & MASK30
    if not modp:
        m = 0
    wide = a * limbs_signed_to_int(x) + b * limbs_signed_to_int(y)
    if modp:
        wide += m * P
    return int_to_signed_limbs(wide >> ZI_B)


def compile_oracle() -> None:
    cmd = [
        "c++",
        "-std=c++17",
        "-O2",
        "-fsanitize=undefined",
        "-I",
        str(TEST_DIR),
        str(CPP),
        "-pthread",
        "-o",
        str(BIN),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_oracle(cases: list[int]) -> list[int]:
    proc = subprocess.run(
        [str(BIN)],
        cwd=ROOT,
        input="\n".join(f"{x:064x}" for x in cases) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    return [int(line, 16) for line in proc.stdout.splitlines() if line.strip()]


def run_canon(cases: list[int]) -> list[int]:
    proc = subprocess.run(
        [str(BIN), "--canon"],
        cwd=ROOT,
        input="\n".join(hex288(x) for x in cases) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    return [int(line, 16) for line in proc.stdout.splitlines() if line.strip()]


def canonical_cases() -> list[int]:
    vals = {0, 1, 2, P - 1, P - 2, P // 2, P - (1 << 27) - 1, P - (1 << 51) - 1}
    for k in range(256):
        vals.add((1 << k) % P)
        vals.add(((1 << k) - 1) % P)
        vals.add(((1 << k) + 1) % P)
        vals.add((P - (1 << k)) % P)
        vals.add((P - (1 << k) - 1) % P)
    rng = random.Random(0x21230)
    for _ in range(20000):
        vals.add(rng.randrange(P))
    return sorted(vals)


def canon_cases() -> list[int]:
    vals = {
        0, 1, -1, P - 1, P, P + 1, -P, -P + 1,
        (1 << 260) - 1, -((1 << 260) - 1),
    }
    lows = [0, 1, (1 << 32) - 1, (1 << 64) - 1, P & ((1 << 256) - 1)]
    for hi in range(-16, 17):
        for low in lows:
            vals.add(hi * (1 << 256) + low)
    rng = random.Random(0xC0A0212)
    for _ in range(2048):
        vals.add(rng.randrange(-(1 << 261) + 1, (1 << 261) - 1))
    return sorted(vals)


def row_cases() -> tuple[int, int, int, int, int]:
    mismatches = 0
    orig_mismatches = 0
    high_mismatches = 0
    orig_high_mismatches = 0
    max_abs_acc = 0
    rng = random.Random(0x5EED212)
    coeffs = [
        (1 << 30) - 1,
        -(1 << 30),
        (1 << 29) + 12345,
        -((1 << 29) + 54321),
        37,
        -91,
    ]
    vectors: list[list[int]] = []
    for hi in [-16, -2, -1, 0, 1, 2, 15]:
        vectors.append([MASK32] * 8 + [hi & MASK32])
        vectors.append([0] * 8 + [hi & MASK32])
    for _ in range(2000):
        vectors.append([rng.getrandbits(32) for _ in range(8)] + [rng.choice([-8, -1, 0, 1, 7]) & MASK32])
    for x in vectors:
        for y in vectors[:16]:
            for a in coeffs:
                for b in coeffs:
                    for modp in [0, 1]:
                        exact = exact_row(x, y, a, b, modp)
                        fixed = trunc_row_fixed(x, y, a, b, modp)
                        orig = trunc_row_orig(x, y, a, b, modp)
                        wide0 = a * limbs_signed_to_int(x) + b * limbs_signed_to_int(y)
                        max_abs_acc = max(max_abs_acc, abs(wide0 >> 256))
                        if fixed != exact:
                            mismatches += 1
                        if s32(fixed[8]) != s32(exact[8]):
                            high_mismatches += 1
                        if orig != exact:
                            orig_mismatches += 1
                        if s32(orig[8]) != s32(exact[8]):
                            orig_high_mismatches += 1
    return mismatches, high_mismatches, orig_mismatches, orig_high_mismatches, max_abs_acc


def changed_source_negative() -> bool:
    with tempfile.TemporaryDirectory() as td:
        mutated = Path(td) / "zinv32.cuh"
        data = HEADER.read_text()
        mutated.write_text(data + "\n// fingerprint negative control\n")
        env = os.environ.copy()
        env["ZINV32_HEADER_PATH"] = str(mutated)
        proc = subprocess.run(
            [str(BIN), "--fingerprint-only"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.returncode == 3 and "fingerprint mismatch" in proc.stderr


def main() -> int:
    print(f"header sha256: {hashlib.sha256(HEADER.read_bytes()).hexdigest()}")
    compile_oracle()
    cases = canonical_cases()
    outs = run_oracle(cases)
    failures = []
    for x, got in zip(cases, outs):
        want = 0 if x == 0 else pow(x, P - 2, P)
        if got != want:
            failures.append((x, got, want))
    print(f"inverse cases: {len(cases) - len(failures)}/{len(cases)} passed")
    for x, got, want in failures[:10]:
        print(f"FAIL x={x:064x} got={got:064x} want={want:064x}")
    zc = canon_cases()
    zc_out = run_canon(zc)
    zc_fail = [(x, got, x % P) for x, got in zip(zc, zc_out) if got != x % P]
    print(f"zi_canon signed-bound cases: {len(zc) - len(zc_fail)}/{len(zc)} passed")
    for x, got, want in zc_fail[:10]:
        print(f"CANON_FAIL x={x} got={got:064x} want={want:064x}")
    row_bad, row_high_bad, row_orig_bad, row_orig_high_bad, max_hi = row_cases()
    print(f"row exactness fixed mismatches: {row_bad}")
    print(f"row signed-high fixed mismatches: {row_high_bad}")
    print(f"row exactness original-style mismatches: {row_orig_bad}")
    print(f"row signed-high original-style mismatches: {row_orig_high_bad}")
    old_shift_ok = row_orig_bad > 0 and row_orig_high_bad > 0
    print(f"old-shift negative control: {'PASS' if old_shift_ok else 'FAIL'}")
    print(f"row sampled max |pre-shift high beyond 256b|: {max_hi}")
    neg_ok = changed_source_negative()
    print(f"changed-source negative control: {'PASS' if neg_ok else 'FAIL'}")
    return 0 if not failures and not zc_fail and row_bad == 0 and old_shift_ok and neg_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
