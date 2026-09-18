#!/usr/bin/env python3
"""Binder: tip L2_SKIP window remains on; not re-derived as a new lever."""
from pathlib import Path
cu = (Path(__file__).resolve().parent / 'pinning.cu').read_text()
assert '#define QSB_L2_SKIP 1' in cu
assert 'QSB_L2_SKIP ? (size_t)gt_entries(0) * 64u' in cu
print('audit_l2_skip_window: PASS')
