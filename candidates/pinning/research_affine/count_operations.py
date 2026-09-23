#!/usr/bin/env python3
"""Operation ledger for the active baseline and the local affine design.

Counts field operations, not latency, GPU instructions or complete solver work.
The baseline constants below trace the active TOP16 and 256-root hierarchy.
"""
from fractions import Fraction
from pathlib import Path
import json

N, B, GROUP = 128, 1 << 23, 256
roots = B // N
groups = roots // GROUP
assert roots == GROUP * GROUP
local_up = sum(N >> j for j in range(1, N.bit_length() - 4))
local_top = 8 + 20 + 18 + 17
local_down = sum(1 << j for j in range(5, N.bit_length() - 1))
local_leaf = N
local_total = local_up + local_top + local_down + local_leaf
assert local_total == 3 * N + 15 == 399
root_group = groups * (3 * GROUP - 3)
outer_tree_and_iso = 3 * GROUP - 3 + 1
root_ordinate_weights = roots
collective = roots * local_total + root_group + outer_tree_and_iso + root_ordinate_weights
assert collective == 3 * B + 19 * roots - 2

baseline_m = Fraction(3 + 13 * 7 + 1 + 8) + Fraction(collective, B)
affine_m = 13 * (5 - Fraction(3, N)) + 8 - Fraction(3, N)
assert affine_m == 73 - Fraction(42, N)
result = {
    'batch_candidates': B,
    'local_lanes': N,
    'baseline': {
        'scalar_M': 95, 'scalar_S': 28, 'noncollective_recovery_M': 8,
        'top16_local_tree_M': local_total, 'all_collectives_M': collective,
        'per_candidate_M': float(baseline_m), 'per_candidate_S': 28,
        'per_candidate_W': 2, 'per_candidate_I': 1 / B,
    },
    'affine_target': {
        'ordinary_add_waves': 13, 'paired_tail_waves': 1,
        'per_candidate_M': float(affine_m), 'per_candidate_S': 15,
        'per_candidate_W': 2, 'per_candidate_I': 14 / N,
    },
    'reference_probe_full_products': {
        'per_candidate_M': float(affine_m + 15 + 2),
        'per_candidate_S': 0, 'per_candidate_W': 0,
        'per_candidate_I': 14 / N,
    },
    'meaning': 'Algebraic field-operation ledger. Inversion internals, additions, normalization, memory, synchronization, SHA256 and setup are excluded. No GPU speed result.',
}
out = Path(__file__).with_name('operation_counts.json')
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
