#!/usr/bin/env python3
"""Observe the actual inherited full self-check, then request normal shutdown.
No timeout or clock verdict; success requires the requested explicit outcome.
The separate unchanged benchmark runs measure and verify hit throughput.
"""
import subprocess,signal,sys,json,os
from pathlib import Path
binary=Path(sys.argv[1]).resolve();corrupt=int(sys.argv[2]);prob=Path('benchmark-results/problem/subset.bin').resolve()
env=os.environ.copy();env['QSB_HP_CORRUPT']=str(corrupt)
argv=[str(binary),str(prob),'0','4081224','970060878','1','0','single_hash']
name='corrupt' if corrupt else 'clean';out=Path(f'candidates/subset/lab/iter5-alias-{name}-actual-selfcheck.log')
proc=subprocess.Popen(argv,cwd='benchmark-results',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
seen=False
with out.open('w') as f:
 for line in proc.stdout:
  f.write(line);f.flush()
  if 'self-check passed' in line:
   assert not corrupt,line;seen=True;proc.send_signal(signal.SIGTERM)
  if 'SELF-CHECK FAILED' in line:
   assert corrupt,line
  if 'self-check mismatch' in line:
   assert corrupt,line;seen=True;proc.send_signal(signal.SIGTERM)
code=proc.wait()
assert seen,f'missing actual check outcome; rc={code}; see {out}'
result={'selfcheck_observed':True,'corrupt':bool(corrupt),'normal_shutdown_rc':code,'log':str(out)}
print('ALIAS_ACTUAL_SELF_CHECK: PASS',json.dumps(result));Path(f'candidates/subset/lab/iter5-alias-{name}-actual-selfcheck.json').write_text(json.dumps(result,indent=2))
