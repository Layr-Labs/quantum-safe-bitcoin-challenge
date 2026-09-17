#!/usr/bin/env python3
"""Execute the actual shared13 chain against the independent existing oracle.

Adds an explicit source/output interface for corrected snapshots and upload
stages. Only input/output paths and the expected complete fingerprint change;
the geometry oracle, actual helper extraction and negative controls do not.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from preflight import source_identity

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=HERE/'candidate')
parser.add_argument('--output', type=Path, default=HERE)
args = parser.parse_args()
source, output = args.source.resolve(), args.output.resolve()
identity = source_identity(source)
assert identity['source_fingerprint']=='a03c3c1ca5f07be955c32af76f34e5a58cc81237d7b9e4c685ee56cfe5353bda'
output.mkdir(parents=True, exist_ok=True)
legacy = HERE.parent/'check.py'
code = legacy.read_text()
replacements = [
    ("source = HERE/'candidate'", f"source = Path({str(source)!r})"),
    ('e80c8d9124fdd85ab38a6079ead00969d1e60d612a5f28c72f28523d28177908', identity['source_fingerprint']),
    ("sys.argv = [str(legacy), '--source', str(source), '--output', str(HERE)]",
     f"sys.argv = [str(legacy), '--source', str(source), '--output', {str(output)!r}]"),
    ("p = HERE/'check-results.json'", f"p = Path({str(output/'check-results.json')!r})"),
]
for old, new in replacements:
    assert code.count(old)==1, old
    code=code.replace(old,new,1)
exec(compile(code,str(legacy)+'[corrected-policy-source]','exec'),
     {'__name__':'__main__','__file__':str(legacy)})
assert source_identity(source)==identity
p=output/'check-results.json'
report=json.loads(p.read_text())
assert report['status']=='PASS' and report['source_fingerprint']==identity['source_fingerprint']
report['policy_corrected_adapter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
report['adapted_shared_checker_sha256']=hashlib.sha256(code.encode()).hexdigest()
p.write_text(json.dumps(report,indent=2)+'\n')
