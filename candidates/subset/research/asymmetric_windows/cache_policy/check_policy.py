#!/usr/bin/env python3
"""Bind emitted cache hints, address checks and a cold-prefetch negative control."""
import hashlib,json,re,shutil,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT))
from preflight import source_identity
base=HERE/'candidate';identity=source_identity(base)
native=json.loads((HERE/'native-results.json').read_text())
control=json.loads((HERE.parent/'native-results.json').read_text())
for name,sha in native['source_sha256'].items():assert identity['source_sha256'][name]==sha,name
ops=(base/'tests/gpu_epochs/cache_ops.cuh').read_text()
for instruction in ['ld.global.cs.u64 %0, [%1];','st.global.cs.u64 [%0], %1;',
 'ld.global.cs.v2.u64 {%0,%1}, [%2];','st.global.cs.v2.u64 [%0], {%1,%2};']:
 assert ops.count(instruction)==1,instruction
assert ops.count('"memory"')==4
prepare=next(v for k,v in native['kernels'].items() if 'prepare_compact' in k)
finish=next(v for k,v in native['kernels'].items() if 'ranked_finish' in k)
for kernel in [prepare,finish]:assert kernel['spill_store_bytes']==kernel['spill_load_bytes']==0
assert prepare['opcodes']['LDG.E.EF.128']==8
assert prepare['opcodes']['STG.E.EF.128']==8
assert prepare['opcodes']['STG.E.EF.64']==4
assert finish['opcodes']['LDG.E.EF.128']==8
assert finish['opcodes']['LDG.E.EF.64']==4
with tempfile.TemporaryDirectory(prefix='qsb-cache-policy-negative-') as td:
 p=Path(td);source=p/'candidate';shutil.copytree(base,source)
 h=source/'tests/gpu_epochs/compact_table_device.cuh';s=h.read_text();needle='if(c>=12)return;';assert s.count(needle)==1;h.write_text(s.replace(needle,''))
 r=subprocess.run([sys.executable,'-B',str(HERE.parents[1]/'wide_windows/check_wide.py'),
  '--compact','--widths','18,17,17,17,17,17,17,17,17,17,17,17,26,25','--prefetch-chunks','12',
  '--source',str(source),'--report',str(p/'result.json')],capture_output=True,text=True,timeout=90)
 assert r.returncode!=0 and 'check failed' in r.stderr,(r.returncode,r.stderr[-2000:])
 assert not (p/'result.json').exists()
assert source_identity(base)==identity
fields=['registers','shared_bytes','stack_bytes','spill_store_bytes','spill_load_bytes','non_nop']
changes={name:{'control':{f:control['kernels'][name][f] for f in fields},'variant':{f:v[f] for f in fields},
 'memory_opcodes':{op:count for op,count in v['opcodes'].items() if op.startswith(('LDG','STG','CCTL'))}}
 for name,v in native['kernels'].items() if v!=control['kernels'][name]}
reports={}
for name in ['curve-results.json','pipeline-results.json','builder-pipeline-results.json']:
 p=HERE/name;d=json.loads(p.read_text());assert d['source_fingerprint']==identity['source_fingerprint'] and d['status']=='PASS'
 reports[name]=hashlib.sha256(p.read_bytes()).hexdigest()
report={'status':'PASS','source_fingerprint':identity['source_fingerprint'],'cpu_reports_sha256':reports,
 'native_report_sha256':hashlib.sha256((HERE/'native-results.json').read_bytes()).hexdigest(),
 'cold_prefetch_negative_control':'DETECTED: restoring ordinary L2 prefetch on cold windows violates the independent address/policy oracle.',
 'changed_kernel_reports':changes,'gpu_executed':False,
 'limits':'CPU helper paths use ordinary loads/stores and do not simulate cache; native report establishes emitted64/128bit evict-first forms and static resources only. No throughput or cache residency measurement.',
 'decision':'Retain geometry+policy comparison candidate. Neither variant has sufficient evidence for a substantial expected lead over current/pending submissions. Do not pursue further tiny cache variants without new evidence.'}
(HERE/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS: source-bound CPU/native reports, full vector/scalar evict-first forms, zero ranked spills and cold-prefetch mutation caught.')
