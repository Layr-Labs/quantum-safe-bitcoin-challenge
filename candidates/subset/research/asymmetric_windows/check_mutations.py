#!/usr/bin/env python3
"""Verify independent geometry oracles catch new cold-window boundary errors."""
import json,shutil,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from preflight import source_identity
base=HERE/'candidate';identity=source_identity(base)
widths='18,17,17,17,17,17,17,17,17,17,17,17,26,25'
cases=[
 ('last_window_shift','compact_geometry.cuh','17*c+1:231','17*c+1:230','check_wide.py',[]),
 ('cold_window_decode_boundary','compact_table_kernels.cuh','+(1u<<25))?12:13','+(1u<<25)+1u)?12:13','check_builder.py',['--low-bits','13']),
]
results=[]
for name,header,old,new,script,extra in cases:
 with tempfile.TemporaryDirectory(prefix='qsb-asymmetric-negative-') as td:
  p=Path(td);source=p/'candidate';shutil.copytree(base,source)
  h=source/'tests/gpu_epochs'/header;s=h.read_text();assert s.count(old)==1;h.write_text(s.replace(old,new))
  command=[sys.executable,'-B',str(HERE.parent/'wide_windows'/script),'--source',str(source),'--compact','--widths',widths,'--report',str(p/'result.json'),*extra]
  r=subprocess.run(command,capture_output=True,text=True,timeout=90)
  assert r.returncode!=0 and 'AssertionError' in r.stderr,(name,r.returncode,r.stderr[-2000:])
  assert not (p/'result.json').exists(),'Mutation unexpectedly generated a passing result'
  results.append({'mutation':name,'status':'DETECTED','oracle':script,'diagnostic_tail':r.stderr.replace(str(ROOT),'candidates/subset').splitlines()[-4:]})
assert source_identity(base)==identity
report={'status':'PASS','source_fingerprint':identity['source_fingerprint'],'mutations':results,'gpu_executed':False,'scope':'Wrong final shift and off-by-one cold-window decoder were applied only in temporary source copies; independent CPU geometry oracles rejected both.'}
(HERE/'mutation-results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
