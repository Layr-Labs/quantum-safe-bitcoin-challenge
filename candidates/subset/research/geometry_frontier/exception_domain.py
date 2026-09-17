#!/usr/bin/env python3
"""Integer-domain proof checks for the frozen thirteen-window proposal.

No compiler or GPU is used. Python limb translations mirror the source-bound
setup/step/direct-digit helpers; independent bigint identities are the oracle.
The all-input proof is in exception_domain.md, not inferred from sampling.
"""
import hashlib
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'thirteen_candidate'
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
U = (1 << 64) - 1
U256 = (1 << 256) - 1
WIDTHS = [18] * 3 + [17] * 6 + [25] * 4
SHIFTS = [sum(WIDTHS[:i]) for i in range(13)]
ORDER = [9, 10, 11, 12] + list(range(9))
B, H = 156, 139


def limbs(x):
    return [(x >> (64 * i)) & U for i in range(4)]


def integer(x):
    return sum(v << (64 * i) for i, v in enumerate(x))


def sub4(a, b):
    out, borrow = [], 0
    for x, y in zip(a, b):
        s = (x - y - borrow) % (1 << 128)
        out.append(s & U)
        borrow = (s >> 64) & 1
    return out, borrow


def setup(k):
    """Source gt_recode_setup arithmetic, retaining limb wrap and borrow."""
    a, n = limbs(k), limbs(N)
    kd, kb = sub4(a, n)
    km = (-(1 - kb)) & U
    a = [(x & (~km & U)) | (y & km) for x, y in zip(a, kd)]
    t = [(a[0] << 1) & U] + [((a[i] << 1) | (a[i-1] >> 63)) & U for i in range(1, 4)]
    d, br = sub4(t, n)
    gm = (-((a[3] >> 63) | (1 - br))) & U
    m = [(x & (~gm & U)) | (y & gm) for x, y in zip(t, d)]
    odd = m[0] & 1
    p, _ = sub4(n, m)
    om = (-odd) & U
    return [(x & om) | (y & (~om & U)) for x, y in zip(m, p)], 2 * odd - 1


def step(m, sign, bits):
    """Source mixed_step: every uint64 assignment wraps before its next use."""
    e = (m[0] & ((1 << (bits + 1)) - 1)) - (1 << bits)
    r = [((m[i] >> (bits + 1)) | (m[i+1] << (63 - bits))) & U for i in range(3)]
    r.append(m[3] >> (bits + 1))
    return [((r[0] << 1) | 1) & U] + [((r[i] << 1) | (r[i-1] >> 63)) & U for i in range(1, 4)], sign * e


def direct(m, sign, pos, width, last):
    """Source gt_field_bits_v + gt_direct_digit with uint64/uint32 wrapping."""
    li, sh = pos >> 6, pos & 63
    lo, hi = m[li], m[li+1] if li < 3 else 0
    f = ((lo >> sh) | ((((hi << 1) & U) << (63 - sh)) & U)) & ((1 << 32) - 1)
    f &= (1 << width) - 1
    t = f >> (width - 1)
    idx = (f if last else f ^ ((t - 1) & ((1 << 32) - 1))) & ((1 << (width - 1)) - 1)
    neg = (0 if last else t ^ 1) ^ int(sign < 0)
    return (2 * idx + 1) * (-1 if neg else 1), idx, neg


def extract(text, marker):
    start = text.index(marker)
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def main():
    receipt = json.loads((HERE / 'thirteen-prepared.json').read_text())['candidate']
    assert receipt['source_fingerprint'] == '5d17e5fad6cb9d632d7b8b4977751da574242488d428cfedfb455accaa21ca5b'
    for path, digest in receipt['source_sha256'].items():
        assert hashlib.sha256((SOURCE / path).read_bytes()).hexdigest() == digest, path
    geom = (SOURCE / 'tests/gpu_epochs/compact_geometry.cuh').read_text()
    dev = (SOURCE / 'tests/gpu_epochs/compact_table_device.cuh').read_text()
    tree = (SOURCE / 'tests/gpu_epochs/tree.cu').read_text()
    assert 'return c<3?18:(c<9?17:25);' in geom
    assert 'return c<3?18*c:(c<9?54+17*(c-3):156+25*(c-9));' in geom
    assert 'int c=i<2?i+11:i-2;' in dev and 'int next=i==1?0:c+1;' in dev
    assert 'next==12,&idx,&neg)' in dev
    assert 'compact_load_signed(gTX,gTY,8,idx,neg,cx,cy);\n    qsb_asym_last_add' in dev
    bodies = {
        'gt_recode_setup': extract(tree, '__device__ __forceinline__ void gt_recode_setup('),
        'mixed_step': extract(geom, '__host__ __device__ __forceinline__ int32_t mixed_step('),
        'gt_field_bits_v': extract(dev, '__device__ __forceinline__ uint32_t gt_field_bits_v('),
        'gt_direct_digit': extract(dev, '__device__ __forceinline__ void gt_direct_digit('),
        'compact_fixed_xyzz': extract(dev, '__device__ void compact_fixed_xyzz('),
        'qsb_asym_last_add': extract(dev, '__device__ __forceinline__ void qsb_asym_last_add('),
    }
    deficit = (1 << 256) - N
    assert deficit.bit_length() == 129 and deficit < (1 << B) - (1 << H)
    cold_bounds = []
    for count in [2, 3, 4]:
        bound = (1 << (B + 25 * count)) - (1 << B)
        assert bound < N
        cold_bounds.append({'cold_terms': count, 'absolute_sum_or_difference_upper': str(bound)})
    hot_bounds = []
    for c in range(8):
        s = SHIFTS[c] + WIDTHS[c]
        lo = (1 << B) - ((1 << s) - 1)
        hi = (1 << 256) - (1 << B) + ((1 << s) - 1)
        assert 0 < lo <= hi < N
        hot_bounds.append({'window': c, 'end_bit': s, 'lower': str(lo), 'upper': str(hi), 'n_minus_upper': str(N-hi)})

    eq = N - ((1 << 18) - 2) * (1 << H)
    assert 1 <= eq <= N and eq & 1
    inv2 = pow(2, -1, N)
    witnesses = []
    scalar_cases = {0, 1, 2, N-2, N-1, N, N+1, N+2, U256}
    # Boundary neighborhoods cover limb edges, width edges, the n reduction,
    # doubling carry, parity, and both constructed final-guard conditions.
    pivots = [0, N, N//2, 1 << 255, U256]
    pivots += [1 << b for b in range(256)]
    odd_cases = {1, 3, N, eq}
    for s in SHIFTS + [64, 128, 192, 256]:
        for w in WIDTHS:
            for q in [0, 1, (1 << (w-1))-1, 1 << (w-1), (1 << w)-1]:
                for delta in [-3, -1, 1, 3]:
                    m = (q << s) + delta
                    if 1 <= m <= N and m & 1:
                        odd_cases.add(m)
    for x in [eq, N, 1]:
        for delta in [-2, 0, 2, -(1 << H), 1 << H]:
            m = x + delta
            if 1 <= m <= N and m & 1:
                odd_cases.add(m)
    for p in pivots:
        for delta in range(-3, 4):
            if 0 <= p + delta <= U256:
                scalar_cases.add(p + delta)
    for m in odd_cases:
        k = m * inv2 % N
        scalar_cases.update([k, (-k) % N])
        if k + N <= U256:
            scalar_cases.add(k + N)
    rng = random.Random(20260917156)
    scalar_cases.update(rng.getrandbits(256) for _ in range(20000))
    counters = {'scalars': 0, 'digits': 0, 'cold_tests': 0, 'hot_tests': 0, 'equal_final': 0, 'opposite_final': 0,
                'negative_control_wrong_sign': 0, 'negative_control_wrong_pos': 0, 'negative_control_wrong_last': 0}
    for k in sorted(scalar_cases):
        m, sign = setup(k)
        M = integer(m)
        residue = (2 * (k % N)) % N
        assert (M, sign) == ((residue, 1) if residue & 1 else (N-residue, -1))
        assert 1 <= M <= N and M & 1 and (sign * M - 2*k) % N == 0
        state, digits = m[:], []
        for c, w in enumerate(WIDTHS):
            before = integer(state)
            if c < 12:
                state, digit = step(state, sign, w)
                assert integer(state) == (before >> w) | 1
            else:
                digit = sign * state[0]
            expected = sign * (((M >> SHIFTS[c]) | 1) if c == 12 else ((((M >> SHIFTS[c]) & ((1 << (w+1))-1)) | 1) - (1 << w)))
            got, idx, neg = direct(m, sign, SHIFTS[c]+1, w, c == 12)
            assert digit == expected == got
            assert abs(digit) & 1 and 1 <= abs(digit) <= (1 << w)-1 and idx < 1 << (w-1)
            assert bool(neg) == (digit < 0)
            digits.append(digit)
            counters['digits'] += 1
        terms = [d << s for d, s in zip(digits, SHIFTS)]
        assert sum(terms) == sign * M
        cold = sum(terms[c] for c in range(9, 13))
        assert cold == sign * (((M >> B) | 1) << B)
        acc = terms[9]
        for c in [10, 11, 12]:
            for v in [acc + terms[c], acc - terms[c]]:
                assert v % (1 << B) == 0 and (v // (1 << B)) & 1
                assert 0 < abs(v) < N and v % N
                counters['cold_tests'] += 1
            acc += terms[c]
        for c in range(8):
            for v in [acc + terms[c], acc - terms[c]]:
                assert 0 < sign * v < N and v % N
                counters['hot_tests'] += 1
            acc += terms[c]
        final = terms[8]
        assert 0 < sign * (acc + final) < 2*N and 0 < sign * (acc-final) < 2*N
        equal, opposite = (acc-final) % N == 0, (acc+final) % N == 0
        assert not (equal and opposite)
        assert opposite == (k % N == 0)
        counters['equal_final'] += equal
        counters['opposite_final'] += opposite
        if M in [eq, N] and (equal or opposite):
            witnesses.append({'k_hex': hex(k), 'odd_M_hex': hex(M), 'global_sign': sign,
                              'last_digit': digits[8], 'accumulator_integer': str(acc), 'last_term_integer': str(final),
                              'equal': equal, 'opposite': opposite, 'signed_exception_multiple': (acc-final if equal else acc+final)//N})
        counters['negative_control_wrong_sign'] += direct(m, -sign, 1, 18, False)[0] != digits[0]
        counters['negative_control_wrong_pos'] += direct(m, sign, 0, 18, False)[0] != digits[0]
        counters['negative_control_wrong_last'] += direct(m, sign, 232, 25, False)[0] != digits[12]
        counters['scalars'] += 1
    assert counters['equal_final'] and counters['opposite_final']
    assert all(counters[k] for k in counters if k.startswith('negative_control'))
    eqd = (((eq >> H) & ((1 << 18)-1)) | 1) - (1 << 17)
    assert eqd == -(1 << 17)+1 and eq - 2*(eqd << H) == N
    # Enumerate the entire possible negative final-digit domain independently.
    # Equality requires M=n+2*d*2^H; verify the digit equation admits only eq.
    equal_domain = []
    for d in range(-(1 << 17)+1, 0, 2):
        m = N + 2*(d << H)
        if 1 <= m <= N and ((((m >> H) & ((1 << 18)-1)) | 1) - (1 << 17)) == d:
            equal_domain.append(m)
    assert equal_domain == [eq]
    assert any(w['equal'] and w['global_sign'] == 1 for w in witnesses)
    assert any(w['equal'] and w['global_sign'] == -1 for w in witnesses)
    # Both seed equality and opposite conditions are rejected by parity/range;
    # removing the final guard demonstrably leaves the equality witness exposed.
    hot = sum(1 << (w-1) for w in WIDTHS[:9]) * 64
    cold_bytes = sum(1 << (w-1) for w in WIDTHS[9:]) * 64
    assert hot == 48 << 20 and cold_bytes == 4 << 30
    report = {
        'status': 'PASS_INTEGER_DOMAIN_PROOF_AND_SOURCE_MATCHED_MODELS',
        'source': str(SOURCE), **receipt,
        'helper_sha256': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in bodies.items()},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'widths': WIDTHS, 'shifts': SHIFTS, 'order': ORDER,
        'hot_bytes': hot, 'cold_bytes': cold_bytes, 'total_bytes': hot+cold_bytes,
        'order_deficit': str(deficit), 'order_deficit_bit_length': deficit.bit_length(),
        'cold_bounds': cold_bounds, 'unguarded_hot_bounds': hot_bounds,
        'counts': counters, 'constructed_witnesses': witnesses,
        'exhaustive_negative_final_digits_checked': 1 << 16,
        'all_equal_odd_representatives': [hex(m) for m in equal_domain],
        'full_domain_proof': 'exception_domain.md', 'hidden_seed_or_intermediate_exception': False,
        'scope': 'Source-hash-bound Python integer and limb models plus all-domain inequalities; no extracted C++ execution, field/curve execution, CUDA compilation or GPU execution.',
        'preconditions': ['Runtime table base is a valid nonzero order-n secp256k1 point.', 'Source digit selection/order and final guard match the reviewed closure.', 'Field primitives/table builder satisfy their independent contracts.'],
    }
    (HERE / 'exception_domain.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'source_fingerprint': report['source_fingerprint'], 'counts': counters, 'witnesses': len(witnesses)}, indent=2))


if __name__ == '__main__':
    main()
