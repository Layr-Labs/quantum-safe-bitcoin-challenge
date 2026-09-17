#!/usr/bin/env python3
"""Independent integer algebra screen; not extracted CUDA or a performance test."""
import hashlib
import json
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from preflight import source_identity

SOURCE = ROOT / 'research/asymmetric_windows/small48/frontier_fused/candidate'
EXPECTED = 'b832d7ba4ec2b2554aa5279e46858a5163e83e3ad7e714e9b108551165c95f7b'
P = 2**256 - 2**32 - 977
C = 2**32 + 977
G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
     0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)


def add(p, q):
    if p is None: return q
    if q is None: return p
    x, y = p; u, v = q
    if x == u:
        if (y + v) % P == 0: return None
        slope = 3*x*x*pow(2*y, -1, P) % P
    else:
        slope = (v-y)*pow(u-x, -1, P) % P
    xx = (slope*slope-x-u) % P
    return xx, (slope*(x-xx)-y) % P


class Circuit:
    def __init__(self): self.m = self.s = 0
    def mul(self, a, b): self.m += 1; return a*b % P
    def square(self, a): self.s += 1; return a*a % P
    def counts(self): return {'M': self.m, 'S': self.s}


def current(x, y, a, b, r):
    f = Circuit(); xr, yr = r
    d = (f.mul(xr, a)-x) % P
    w = f.mul(f.square(a), d)
    if w == 0: return None, None
    c = f.mul(f.square(d), a)
    inv = pow(w, -1, P)  # Same one collective inversion boundary; excluded from counts.
    yb = f.mul(yr, b); h = f.mul(b, inv)
    delta = f.mul(c, inv); xs = (2*xr-delta) % P
    m1 = f.mul((yb-y) % P, h); x1 = (f.square(m1)-xs) % P
    y1 = (f.mul(m1, (xr-x1) % P)-yr) % P
    m2 = f.mul((yb+y) % P, h); x2 = (f.square(m2)-xs) % P
    y2 = (yr-f.mul(m2, (xr-x2) % P)) % P
    return ((x1, y1), (x2, y2)), f.counts()


def cubic_recovery(x, y, a, b, r, mutate_constant=False):
    f = Circuit(); xr, yr = r
    d = (f.mul(xr, a)-x) % P
    w = f.mul(b, d)
    if w == 0: return None, None
    inv = pow(w, -1, P)
    h = f.mul(a, inv)
    k = f.mul(b, h)
    beta = f.mul(y, h)
    alpha = f.mul(yr, k)
    # Runtime-problem constant, computed once on the host, not per candidate.
    constant = ((2 if mutate_constant else 3)*xr*xr) % P
    center = (2*f.square(alpha)-f.mul(constant, k)+xr) % P
    difference = 2*f.mul(alpha, beta) % P
    x1 = (center-difference) % P; x2 = (center+difference) % P
    m1 = (alpha-beta) % P; m2 = (alpha+beta) % P
    y1 = (f.mul(m1, (xr-x1) % P)-yr) % P
    y2 = (yr-f.mul(m2, (xr-x2) % P)) % P
    return ((x1, y1), (x2, y2)), f.counts()


def deferred_step(x, y, a, b, anchor, point):
    u, v = point
    scaled_x = u*a % P
    p = (scaled_x-x) % P
    r = ((v+anchor)*b-y) % P
    assert p != 0
    pp = p*p % P; ppp = pp*p % P; q = scaled_x*pp % P
    xx = (r*r+ppp-2*q) % P
    aa = a*pp % P; bb = b*ppp % P
    factor = (q-xx) % P
    return (xx, r*factor % P, aa, bb), (r, factor)


def main():
    identity = source_identity(SOURCE)
    assert identity['source_fingerprint'] == EXPECTED
    rng = random.Random(2026091703)
    points = [G]
    for _ in range(255): points.append(add(points[-1], G))
    counts = {'recovery_cases': 0, 'recovered_affine_points': 0,
              'singular_recovery_cases': 0, 'wrong_constant_mutations_caught': 0,
              'deferred_transition_cases': 0, 'raw_product_difference_cases': 0,
              'lost_borrow_mutations_caught': 0}
    baseline_counts = proposal_counts = None
    for i in range(4096):
        p = rng.choice(points); r = rng.choice(points)
        z = [1, P-1, rng.randrange(1, P)][i % 3]
        a = z*z % P; b = a*z % P; x = p[0]*a % P; y = p[1]*b % P
        before, bc = current(x, y, a, b, r)
        after, ac = cubic_recovery(x, y, a, b, r)
        assert before == after
        if before is None:
            counts['singular_recovery_cases'] += 1
            continue
        want = (add(p, r), add(p, (r[0], -r[1] % P)))
        assert before == after == want
        assert bc == {'M': 10, 'S': 4} and ac == {'M': 10, 'S': 1}
        baseline_counts, proposal_counts = bc, ac
        wrong, _ = cubic_recovery(x, y, a, b, r, True)
        assert wrong != want
        counts['wrong_constant_mutations_caught'] += 1
        counts['recovery_cases'] += 1
        counts['recovered_affine_points'] += 2

    # Explicit P=+R, P=-R and infinity states keep the same inverse-zero gate.
    for r in points[:16]:
        for p in [r, (r[0], -r[1] % P), None]:
            x, y, a, b = (*p, 1, 1) if p else (0, 0, 0, 0)
            assert current(x, y, a, b, r)[0] is None
            assert cubic_recovery(x, y, a, b, r)[0] is None
            counts['singular_recovery_cases'] += 1

    # Deferred Y=left*right can be carried without that product's reduction.
    # The next slope numerator then consumes one raw product difference.
    for _ in range(4096):
        p, q, next_q = (rng.choice(points) for _ in range(3))
        if p[0] == q[0]: continue
        z = rng.randrange(1, P); a = z*z % P; b = a*z % P
        anchor = rng.randrange(P)
        x, y = p[0]*a % P, (p[1]+anchor)*b % P
        state, factors = deferred_step(x, y, a, b, anchor, q)
        xx, yy, aa, bb = state; left, right = factors
        coefficient = (next_q[1]+q[1]) % P
        direct = (coefficient*bb-yy) % P
        fused = (coefficient*bb-left*right) % P
        assert direct == fused
        affine = add(p, q)
        assert xx*pow(aa, -1, P) % P == affine[0]
        assert (yy*pow(bb, -1, P)-q[1]) % P == affine[1]
        counts['deferred_transition_cases'] += 1

    bounds = [0, 1, 2, C-1, C, C+1, P-3, P-2, P-1]
    values = [(a,b,c,d) for a in bounds for b in bounds for c,d in [(0,0),(P-1,P-1),(a,b)]]
    values += [tuple(rng.randrange(P) for _ in range(4)) for _ in range(16384)]
    for a,b,c,d in values:
        difference = a*b-c*d
        assert -(P-1)**2 <= difference <= (P-1)**2
        raw = difference % (1 << 512); borrow = int(difference < 0)
        want = difference % P
        # p=2^256-C implies 2^512=C^2 mod p; the borrow is not discardable.
        assert (raw-borrow*C*C) % P == want
        positive = difference + P*P
        assert 0 < positive < 2*P*P < 1 << 513
        assert positive % P == want
        if borrow:
            assert raw % P != want
            counts['lost_borrow_mutations_caught'] += 1
        counts['raw_product_difference_cases'] += 1

    assert source_identity(SOURCE) == identity
    report = {
        'status': 'PASS', 'source_fingerprint': EXPECTED,
        'validation_level': 'Independent Python integer identities and affine-law oracle; no extracted CUDA execution',
        **counts,
        'current_recovery_operations_excluding_collective': baseline_counts,
        'cubic_recovery_operations_excluding_collective': proposal_counts,
        'full_chain_plus_recovery_current': {'M': 98, 'S': 30},
        'full_chain_plus_recovery_cubic': {'M': 98, 'S': 27},
        'deferred_raw_difference_opportunity': {
            'point_chain_transitions': 12,
            'unchanged_field_product_count': True,
            'separate_512_to_field_reductions_removed': 12,
            'additional_logical_state_bits': 256,
            'signed_difference_magnitude_bound': '(p-1)^2',
            'raw_unsigned_representation': '512-bit subtraction result plus one borrow bit',
            'borrow_correction_mod_p': '-borrow*(2^32+977)^2',
            'terminal_exact_Y_fusion': 'One further two-product difference; separate optional region',
        },
        'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'gpu_executed': False, 'candidate_modified': False,
        'limits': 'No optimality claim, PTX implementation, register allocation, carry-chain instruction model or speed prediction.',
    }
    (HERE/'algebra-results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
