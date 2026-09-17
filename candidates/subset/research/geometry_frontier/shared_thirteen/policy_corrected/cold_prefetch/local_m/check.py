#!/usr/bin/env python3
"""Check actual shared13 code with an independent all-window prefetch oracle.

The four new cold hints must address both32-byte halves of exactly the later
loaded point, including sign/top-digit boundaries. No GPU prefetching is modeled.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
sys.path.insert(0,str(ROOT))
from preflight import source_identity
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,default=HERE/'candidate')
parser.add_argument('--output',type=Path,default=HERE)
args=parser.parse_args()
source,output=args.source.resolve(),args.output.resolve()
identity=source_identity(source)
assert identity['source_fingerprint']=='b2fd8e8896270638268058a41150409d42a470ce228ee1962c8dcc76f7ef1170'
output.mkdir(parents=True,exist_ok=True)
legacy=HERE.parents[2]/'check.py'
code=legacy.read_text()
replacements=[
 ("source = HERE/'candidate'",f"source = Path({str(source)!r})"),
 ('e80c8d9124fdd85ab38a6079ead00969d1e60d612a5f28c72f28523d28177908',identity['source_fingerprint']),
 ("\"'--prefetch-start','0','--prefetch-chunks','9'\"", "\"'--prefetch-start','0','--prefetch-chunks','13'\""),
 ("sys.argv = [str(legacy), '--source', str(source), '--output', str(HERE)]",f"sys.argv = [str(legacy), '--source', str(source), '--output', {str(output)!r}]"),
 ("p = HERE/'check-results.json'",f"p = Path({str(output/'check-results.json')!r})"),
]
for old,new in replacements:
 assert code.count(old)==1,old
 code=code.replace(old,new,1)
exec(compile(code,str(legacy)+'[all-thirteen-prefetch]','exec'),
     {'__name__':'__main__','__file__':str(legacy)})
assert source_identity(source)==identity
p=output/'check-results.json'
d=json.loads(p.read_text());assert d['status']=='PASS' and d['source_fingerprint']==identity['source_fingerprint']
d['cold_prefetch_adapter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
d['adapted_checker_sha256']=hashlib.sha256(code.encode()).hexdigest()
d['expected_prefetched_windows']=list(range(13))
d['prefetch_scope']='Actual addresses/sectors checked before every later load; no device cache/timing simulation.'
p.write_text(json.dumps(d,indent=2)+'\n')
