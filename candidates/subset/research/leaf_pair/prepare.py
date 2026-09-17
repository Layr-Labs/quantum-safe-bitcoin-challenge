#!/usr/bin/env python3
"""Freeze a PR138 leaf-pair variant plus its missing n=64 block join."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from preflight import source_identity

base = ROOT/'research/asymmetric_windows/small48/frontier_fused/candidate'
identity = source_identity(base)
assert identity['source_fingerprint'] == 'b832d7ba4ec2b2554aa5279e46858a5163e83e3ad7e714e9b108551165c95f7b'
dest = HERE/'candidate'
assert not dest.exists(), 'Preserve frozen candidate'
for name in [*identity['source_sha256'], 'COPYING']:
    target = dest/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base/name, target)
public = (HERE/'pr138-tree_inverse.cuh').read_text()
marker = '    // tree[n/2 + tid/2] is the inverse of this lane pair.'
assert public.count(marker) == 1
guard = '    // At n=64, warp 0 produced pair inverses consumed by warp 1.\n    if(n==64)__syncthreads();\n\n'
fixed = public.replace(marker, guard+marker)
(dest/'tests/gpu_epochs/tree_inverse.cuh').write_text(fixed)
report = {
    'base_source': identity, 'candidate_source': source_identity(dest),
    'public_donor': {'pr': 138, 'commit': '2dc49dc4e4083050ffc34b9be61eb027eb8c291f',
                     'solver': 'AbdelStark', 'coauthor_if_used_before_promotion': 'AbdelStark',
                     'path': 'candidates/subset/tests/gpu_epochs/tree_inverse.cuh',
                     'sha256': hashlib.sha256(public.encode()).hexdigest()},
    'only_changed_source': 'tests/gpu_epochs/tree_inverse.cuh',
    'correction': 'Uniform if(n==64)__syncthreads() before all-lane final leaf expansion.',
    'scope': 'Supporting inverse component, not standalone submission or demonstrated speedup.',
    'license': 'Existing GPL notices and COPYING preserved; public donor snapshot retained.',
    'gpu_executed': False,
}
(HERE/'prepared-source.json').write_text(json.dumps(report, indent=2)+'\n')
print(report['candidate_source']['source_fingerprint'])
