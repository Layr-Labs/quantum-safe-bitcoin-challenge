#!/usr/bin/env python3
"""Bounded CPU model of 48-product multiplication; not selected production.

Explicit 32-bit limbs retain the signed middle term and the 257th cross bit.
The oracle is Python integer multiplication. No GPU performance is inferred.
"""
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preflight import source_identity

W = 1 << 32
MASK = W - 1
B = 1 << 256
C = (1 << 32) + 977
P = B - C


def limbs(x, n):
    assert 0 <= x < 1 << (32*n)
    return [(x >> (32*i)) & MASK for i in range(n)]


def value(a):
    return sum(x << (32*i) for i, x in enumerate(a))


def add(a, b, n):
    out, carry = [], 0
    for i in range(n):
        t = (a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) + carry
        out.append(t & MASK)
        carry = t >> 32
    assert carry == 0, 'lost high carry'
    return out


def sub(a, b, n):
    out, borrow = [], 0
    for i in range(n):
        t = (a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0) - borrow
        out.append(t & MASK)
        borrow = int(t < 0)
    assert borrow == 0, 'negative unsigned result'
    return out


def abs_difference(a, b):
    # Lexicographic comparison of high-to-low limbs, not a 129-bit sum.
    negative = a[::-1] < b[::-1]
    return (sub(b, a, 4) if negative else sub(a, b, 4)), negative


def mul128(a, b):
    assert len(a) == len(b) == 4
    out, carry, products, max_column_bits = [], 0, 0, 0
    for column in range(8):
        accum = carry
        for i in range(4):
            j = column-i
            if 0 <= j < 4:
                accum += a[i]*b[j]
                products += 1
        max_column_bits = max(max_column_bits, accum.bit_length())
        out.append(accum & MASK)
        carry = accum >> 32
    assert carry == 0 and products == 16
    # A real implementation needs enough accumulator carry bits, not uint64_t.
    assert max_column_bits <= 66
    return out, max_column_bits


def multiply(a, b):
    a0, a1 = limbs(a, 8)[:4], limbs(a, 8)[4:]
    b0, b1 = limbs(b, 8)[:4], limbs(b, 8)[4:]
    da, sa = abs_difference(a0, a1)
    db, sb = abs_difference(b0, b1)
    z0, bits0 = mul128(a0, b0)
    z2, bits2 = mul128(a1, b1)
    d, bitsd = mul128(da, db)
    middle = add(z0, z2, 9)
    middle = add(middle, d, 9) if sa != sb else sub(middle, d, 9)
    out = add(z0, [0]*4 + middle, 16)
    out = add(out, [0]*8 + z2, 16)
    return value(out), {'difference_sign': int(sa)*2+int(sb),
                        'middle_high_bit': middle[8],
                        'max_column_bits': max(bits0, bits2, bitsd)}


def reduce_field(product):
    first = (product & (B-1)) + (product >> 256)*C
    second = (first & (B-1)) + (first >> 256)*C
    carry = second >> 256
    assert carry in (0, 1)
    third = (second & (B-1)) + carry*C
    assert third < B
    return third-P if third >= P else third, carry


def balanced_windows(bits, count):
    # Minimum sum(2**(width-1)) at fixed total bits/count uses balanced widths.
    q, r = divmod(bits, count)
    widths = [q]*(count-r) + [q+1]*r
    return {'widths': widths, 'table_mib': sum(1 << (w-1) for w in widths)*64/2**20}


def main():
    rng = random.Random(26091648)
    edges = [0, 1, 2, (1 << 128)-1, 1 << 128, (1 << 128)+1,
             P-65537, P-2, P-1, P, P+1, B-2, B-1]
    cases = list(itertools.product(edges, repeat=2))
    # Force all four difference signs, zero differences, and long carry chains.
    halves = [0, 1, (1 << 64)-1, 1 << 64, (1 << 128)-2, (1 << 128)-1]
    structured = [lo+(hi << 128) for lo, hi in itertools.product(halves, repeat=2)]
    cases += list(itertools.product(structured, repeat=2))
    cases += [(rng.getrandbits(256), rng.getrandbits(256)) for _ in range(20000)]
    signs, high_middle, reduction_carries, max_bits = set(), 0, 0, 0
    for a, b in cases:
        product, info = multiply(a, b)
        assert product == a*b, (hex(a), hex(b))
        reduced, carry = reduce_field(product)
        assert reduced == (a*b) % P
        signs.add(info['difference_sign'])
        high_middle += bool(info['middle_high_bit'])
        reduction_carries += carry
        max_bits = max(max_bits, info['max_column_bits'])
    assert signs == {0, 1, 2, 3} and high_middle and reduction_carries
    # A concrete canonical-input case requiring the final 2^256 carry fold.
    edge_a = P-65537
    t = edge_a*edge_a
    first = (t & (B-1)) + (t >> 256)*C
    second = (first & (B-1)) + (first >> 256)*C
    assert second >= B
    report = {
        'status': 'PASS', 'validation_level': 'python_limb_model',
        'exact_products_and_reductions': len(cases), 'word_products': 48,
        'schoolbook_word_products': 64, 'difference_signs': sorted(signs),
        'cases_with_257_bit_middle': high_middle, 'final_fold_carry_cases': reduction_carries,
        'max_128bit_product_column_bits': max_bits,
        'carry_regression_case': {'a': hex(edge_a), 'b': hex(edge_a),
                                 'correct_residue': (edge_a*edge_a) % P,
                                 'incorrect_dropped_carry_residue': second & (B-1)},
        'ordinary_signed_table_screens': {str(n): balanced_windows(256, n) for n in (16, 15, 14, 13)},
        'idealized_glv_shared_table_screen': {
            'assumed_component_bits': 128, 'windows_each': 7,
            **balanced_windows(128, 7), 'total_point_lookups': 14,
            'additional_beta_multiplications_if_transformed_on_load': 7,
            'limitations': 'No GLV decomposition or signed recoder implemented; table is a lower-bound size model, not a complete algorithm.'},
        'source_identity': source_identity(),
        'model_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'selected_for_production': False, 'gpu_executed': False,
        'limitations': '48 products excludes reduction, sign, carry and merge overhead. No CUDA/PTX implementation, resource count or performance result.'}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
