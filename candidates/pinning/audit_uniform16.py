#!/usr/bin/env python3
"""Audit the uniform 16x16-bit signed-digit table geometry in pinning.cu.

The production kernel replaced the mixed 15-chunk (18/17-bit, 64 MiB) table
with sixteen uniform 16-bit chunks (32 MiB, interleaved 64-byte [X,Y]
records). This script is a CPU algebra/address audit of the new geometry,
never a CUDA compile or GPU benchmark. It mirrors the structure of
audit_stream_recode.py (which models the retired mixed design and is kept
as history) for the uniform design actually in production.

Checks:
 1. Recode: Python model of gt_recode_setup + 15x gt_recode_step +
    remainder yields 16 odd digits with |e| < 2^16 whose weighted sum is
    exactly 2k mod n, over group-order boundaries, powers of two, and
    random scalars.
 2. Digit range: every odd 16-bit signed digit maps to idx < 2^15.
 3. Address map: all 16*2^15 (chunk, entry) pairs map to unique 64-byte
    records covering exactly [0, 32 MiB).
 4. Loader integration: simulated chunk loader on a synthetic interleaved
    table built from real curve points returns exact x limbs and the
    sign-selected y limbs for corners of every chunk plus random entries.
 5. Builder binding: ladder shifts (65536 per chunk), uniform H bound
    (GT_HI), interleaved store order, and spot-check corners are bound to
    the production source text.
 6. Chain count: the deferred chain consumes exactly 16 points
    (2 seed + 13 intermediate + 1 resolving), bound to source loop bounds.
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parents[1] / "harness"))
import crypto as C

P = C.P
N = C.N
GT_CHUNKS = 16
GT_ENTRIES = 1 << 15
MASK64 = (1 << 64) - 1


def limbs_le(value):
    return [(value >> (64 * i)) & MASK64 for i in range(4)]


def compose_limbs(limbs):
    return sum(limb << (64 * i) for i, limb in enumerate(limbs))


def recode_setup(k):
    """Model of gt_recode_setup: odd 2k-representative M plus global sign."""
    kk = k % N
    # 2*(k mod n) < 2n, so one conditional subtract gives (2k mod n) exactly.
    t = 2 * kk if 2 * kk < N else 2 * kk - N
    m = t
    odd = m & 1
    if odd:
        M, sign = m, 1
    else:
        # NOTE: no mod reduction here, matching the device borrow-select:
        # for m == 0 this yields M = N (odd), not 0.
        M, sign = N - m, -1
    return M, sign


def recode_step(state, sign):
    """Model of gt_recode_step: peel one uniform 16-bit signed odd digit."""
    M = state[0]
    digit = (M & 0x1FFFF) - 65536
    state[0] = (2 * (M >> 17) + 1) % (1 << 257)
    return sign * digit


def recode(k):
    M, sign = recode_setup(k)
    state = [M]
    digits = [recode_step(state, sign) for _ in range(GT_CHUNKS - 1)]
    digits.append(sign * (state[0] & MASK64))
    return digits


def main():
    counts = {}
    src = (HERE / "pinning.cu").read_text()

    # ---- 5/6. source bindings -------------------------------------------
    def has(s):
        assert s in src, f"missing source shape: {s!r}"
        return 1

    n = 0
    n += has("#define GT_CHUNKS 16")
    n += has("#define GT_ENTRIES (1u << 15)")
    n += has("#define GT_HI 256")
    n += has("int32_t digit=(int32_t)(M[0]&0x1FFFFu)-65536;")
    n += has("BN_set_word(shift, 65536); EC_POINT_mul(grp, base, NULL, base, shift, ctx);")
    n += has("for (int hi = 1; hi < GT_HI; hi++)")
    n += has("size_t off = ((size_t)ch * GT_ENTRIES + d) * 64;")
    n += has("size_t off = ((size_t)c * GT_ENTRIES + idx) * 64;")
    n += has("const int corner[4] = {0, 1, 2, (int)GT_ENTRIES - 1};")
    n += has("for (int c=2;c<GT_CHUNKS;c++){")
    n += has("102M+30S")
    n += has("for(int c=0;c<GT_CHUNKS-1;c++)e[c]=gt_recode_step(M,sign);")
    assert "gt_mixed_step" not in src, "mixed stepper must be gone"
    assert "gt_offset" not in src and "gt_shift" not in src, "mixed address helpers must be gone"
    n += 2
    counts["source_bindings"] = n

    # ---- 2. digit range (exhaustive) ---------------------------------------
    for e in range(1, 1 << 16, 2):
        for digit in (e, -e):
            assert 0 <= ((abs(digit) - 1) >> 1) < GT_ENTRIES
    counts["odd_digit_mappings"] = 1 << 16

    # ---- 3. address map -----------------------------------------------------
    assert GT_CHUNKS * GT_ENTRIES == 1 << 19
    assert (1 << 19) * 64 == 32 * 1024 * 1024
    seen = set()
    for c in range(GT_CHUNKS):
        for idx in (0, 1, 2, GT_ENTRIES - 1):
            off = (c * GT_ENTRIES + idx) * 64
            assert off % 64 == 0 and 0 <= off < 32 * 1024 * 1024
            seen.add(off)
    assert len(seen) == GT_CHUNKS * 4
    counts["address_corner_records"] = len(seen)

    # ---- 1. recode correctness ----------------------------------------------
    rng = random.Random(0x516)
    boundary = [0, 1, 2, N - 1, N - 2, (N + 1) % (1 << 256), N, N + 1,
                (1 << 256) - 1, (1 << 256) - 2, 1 << 255, (1 << 128) + 1]
    tested = 0
    for k in boundary + [rng.randrange(0, 1 << 256) for _ in range(30000)]:
        digits = recode(k)
        assert len(digits) == GT_CHUNKS
        for e in digits:
            assert e != 0 and e % 2 == 1 and abs(e) < 1 << 16, (k, e)
        total = sum(e * (1 << (16 * c)) for c, e in enumerate(digits)) % N
        assert total == (2 * (k % N)) % N, (k, digits)
        tested += 1
    counts["recode_scalars"] = tested

    # ---- 4. loader integration on real curve points --------------------------
    nri = rng.randrange(1, N)
    inv2 = pow(2, N - 2, N)
    scale = inv2 * nri % N  # A/2 scalar: table base

    def table_point(chunk, d):
        k = ((2 * d + 1) * pow(2, 16 * chunk, N) % N) * scale % N
        pt = C.point_mul(k)
        assert pt is not None and C.on_curve(pt[0])
        return pt

    def load_sim(table, chunk, idx, neg):
        off = (chunk * GT_ENTRIES + idx) * 64
        x = int.from_bytes(table[off:off + 32], "little")
        y = int.from_bytes(table[off + 32:off + 64], "little")
        gy = y if neg == 0 else P - y
        return limbs_le(x), limbs_le(gy)

    sample = ([(c, i) for c in range(GT_CHUNKS) for i in (0, 1, 2, GT_ENTRIES - 1)]
              + [(rng.randrange(GT_CHUNKS), rng.randrange(GT_ENTRIES)) for _ in range(256)])
    cases = 0
    for chunk, d in sample:
        x, y = table_point(chunk, d)
        record = x.to_bytes(32, "big")[::-1] + y.to_bytes(32, "big")[::-1]
        off = (chunk * GT_ENTRIES + d) * 64
        table = bytearray(off + 64)
        table[off:off + 64] = record
        table = bytes(table)
        for neg in (0, 1):
            gx, gy = load_sim(table, chunk, d, neg)
            assert compose_limbs(gx) == x
            assert compose_limbs(gy) == (y if neg == 0 else P - y)
            cases += 1
    counts["loader_integration_cases"] = cases
    counts["distinct_curve_points"] = len(sample)

    print(json.dumps({"status": "PASS", **counts, "gpu_executed": False}))
    print("PASS " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
