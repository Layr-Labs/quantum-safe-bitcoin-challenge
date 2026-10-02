#!/usr/bin/env python3
"""CPU feasibility screen for two integer MMA tiles per 256-bit product.

This evaluates matrix values, not PTX fragments or GPU execution. The mapping
uses one entire warp per independent product and is deliberately not described
as an optimal mapping. Byte packing, cross-lane routing, carries and reduction
still need a concrete CUDA schedule before any performance claim is possible.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
TRACK = HERE.parents[1]
P = (1 << 256) - (1 << 32) - 977
MASK = (1 << 256) - 1
C = (1 << 32) + 977


def byte_at(value, index):
    return (value >> (8 * index)) & 255 if 0 <= index < 32 else 0


def tile(a, b, omit_upper=False, signed=False):
    """A[r,k]=a[k-r], B[k,c]=b[7+8*c-k], with zero extension.

    Sum k=0..63 as two m16n8k32 tiles. D[r,c] is coefficient
    t=7+8*c-r in a*b; substitution i=k-r proves this identity.
    For a contributing term 0<=i,j<32 and 0<=r<16, k=i+r<47,
    so the two tiles include every term. There is no shared-b assumption.
    """
    aa = [[byte_at(a, k - r) for k in range(64)] for r in range(16)]
    bb = [[byte_at(b, 7 + 8 * c - k) for c in range(8)] for k in range(64)]
    if signed:
        aa = [[v - 256 if v >= 128 else v for v in row] for row in aa]
        bb = [[v - 256 if v >= 128 else v for v in row] for row in bb]
    d = [[0] * 8 for _ in range(16)]
    for begin in (0,) if omit_upper else (0, 32):
        for r in range(16):
            for c in range(8):
                d[r][c] += sum(aa[r][k] * bb[k][c] for k in range(begin, begin + 32))
    return d


def extract_coefficients(d):
    # r=0..7 covers each t=0..63 exactly once, including the zero t=63.
    return [d[7 - (t % 8)][t // 8] for t in range(64)]


def carry_bytes(coefficients):
    carry, answer, peak = 0, 0, 0
    for i, coefficient in enumerate(coefficients):
        total = coefficient + carry
        answer |= (total & 255) << (8 * i)
        carry = total >> 8
        peak = max(peak, carry)
    assert carry == 0, "512-bit product overflow"
    return answer, peak


def reduce_reference(product):
    # Arbitrary-precision algebraic folds only; no claim about CUDA cost.
    folds = 0
    while product > MASK:
        product = (product & MASK) + (product >> 256) * C
        folds += 1
    if product >= P:
        product -= P
    return product, folds


def reference_coefficients(a, b):
    answer = [0] * 64
    for i in range(32):
        for j in range(32):
            answer[i + j] += byte_at(a, i) * byte_at(b, j)
    return answer


def cases():
    boundary = [0, 1, 255, 256, 257, (1 << 128) - 1, 1 << 128,
                P - 1, P, P + 1, MASK - 1, MASK]
    for a in boundary:
        for b in boundary:
            yield a, b
    # Every byte-product position, including maximum-valued top bytes.
    for i in range(32):
        for j in range(32):
            yield 255 << (8 * i), 255 << (8 * j)
    rng = random.Random(0x25616832)
    for _ in range(512):
        a, b = rng.getrandbits(256), rng.getrandbits(256)
        yield a, b
        yield a, a


def main():
    receipt = json.loads((TRACK / 'research/wide_windows/submitted-source.json').read_text())
    actual = {name: hashlib.sha256((TRACK / name).read_bytes()).hexdigest()
              for name in receipt['production_sha256']}
    assert actual == receipt['production_sha256'], 'Submitted source changed; rebase the screen'
    counts, max_coefficient, max_carry, max_folds = 0, 0, 0, 0
    for a, b in cases():
        d = tile(a, b)
        coeff = extract_coefficients(d)
        assert coeff == reference_coefficients(a, b)
        # Check redundant matrix outputs too: an extraction-only check could
        # hide a shifted tile or a zero-extension mistake in its unused half.
        for r in range(16):
            for c in range(8):
                t = 7 + 8 * c - r
                assert d[r][c] == (coeff[t] if 0 <= t < 64 else 0)
        product, peak_carry = carry_bytes(coeff)
        assert product == a * b
        reduced, folds = reduce_reference(product)
        assert reduced == (a * b) % P
        max_coefficient = max(max_coefficient, max(coeff))
        max_carry = max(max_carry, peak_carry)
        max_folds = max(max_folds, folds)
        counts += 1
    # All coefficients are <=32*255^2, and carry<=8160 is inductive:
    # floor((32*255^2+8160)/256) = 8160. Both fit signed32 exactly.
    bound, carry_bound = 32 * 255**2, 32 * 255
    assert (bound + carry_bound) // 256 == carry_bound
    assert max_coefficient == bound and max_carry <= carry_bound
    assert bound + carry_bound < (1 << 31)
    rejected = []
    for name, kwargs in [('omit_second_mma', {'omit_upper': True}),
                         ('wrong_signed_int8', {'signed': True})]:
        try:
            assert extract_coefficients(tile(MASK, MASK, **kwargs)) == reference_coefficients(MASK, MASK)
        except AssertionError:
            rejected.append(name)
    assert len(rejected) == 2
    bound_terms = Counter()
    for r in range(16):
        for c in range(8):
            t = 7 + 8 * c - r
            bound_terms[t] += sum(0 <= k-r < 32 and 0 <= 7+8*c-k < 32 for k in range(64))
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'screen_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256': actual, 'submitted_fingerprint': receipt['fingerprint'],
        'status': 'PASS mathematical mapping; CUDA prototype not selected',
        'checks': {'full_products_and_residues': counts, 'mutations_rejected': rejected,
                   'max_coefficient_observed': max_coefficient, 'coefficient_bound': bound,
                   'max_carry_observed': max_carry, 'inductive_carry_bound': carry_bound,
                   'max_reference_folds': max_folds, 'all_128_outputs_checked': True},
        'mapping': {'shape': '16x8 output, K64 in two dense m16n8k32 u8/u8/s32 operations',
                    'A': 'A[r,k]=a[k-r], zero outside byte indices0..31',
                    'B': 'B[k,c]=b[7+8*c-k], zero outside byte indices0..31',
                    'D': 'D[r,c]=convolution_coefficient[7+8*c-r]',
                    'independent_products_per_warp': 1, 'current_candidates_per_warp': 32},
        'logical_budget_per_product': {
            'mma_warp_operations': 2, 'dense_byte_macs': 2*16*8*32,
            'distinct_useful_byte_products': 32*32,
            'nonzero_slot_macs_including_duplicate_outputs': sum(bound_terms.values()),
            'distinct_useful_fraction': (32*32)/(2*16*8*32),
            'input_operand_bytes': 64, 'packed_A_B_fragment_bytes_across_two_calls': 2*(16*32+32*8),
            'accumulator_words32_per_warp': 128, 'retained_coefficient_words32': 64,
            'carry_steps_in_this_serial_reference': 64,
            'note': 'Logical slots/bytes, not memory transactions, emitted instructions, registers allocated or cycles.'},
        'missing_for_selection': [
            'Exact per-lane fragment packing and routing from a persistent field/point representation.',
            'Cross-lane carry/reduction and repacking schedule; scalar Python folds do not supply this.',
            'Full EC schedule, table lookup, checkpoints and external inverse interoperability.',
            'An unchanged-source native resource comparison and an end-to-end gain case; later GPU timing.'
        ],
        'decision': 'Retain as a precise open research lead. Two MMA calls do not establish a speedup: this mapping consumes a warp per product, expands byte operands into fragments and retains carry/reduction work. Not a general rejection or an optimality bound.',
        'gpu_executed': False, 'cuda_compiled': False, 'production_changed': False,
    }
    (HERE / 'screen-results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'checks', 'logical_budget_per_product', 'decision']}, indent=2))


if __name__ == '__main__':
    main()
