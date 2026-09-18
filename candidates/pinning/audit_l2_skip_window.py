#!/usr/bin/env python3
"""Source-shape binder: QSB_L2_SKIP must be on and drive the access-policy skip."""
from pathlib import Path
src = Path(__file__).with_name('pinning.cu').read_text()
assert '#define QSB_L2_SKIP 1' in src, 'expected QSB_L2_SKIP enabled'
assert 'QSB_L2_SKIP ? (size_t)gt_entries(0) * 64u : 0u' in src, 'expected skip expression'
assert 'accessPolicyWindow' in src or 'AccessPolicyWindow' in src or 'cudaAccessPolicyWindow' in src
print('audit_l2_skip_window: PASS')
