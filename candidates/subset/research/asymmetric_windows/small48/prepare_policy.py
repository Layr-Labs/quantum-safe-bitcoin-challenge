#!/usr/bin/env python3
"""Copy the frozen geometry control and add an optional runtime cache policy."""
import json,shutil,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT))
from preflight import source_identity
base=HERE/'candidate';identity=source_identity(base)
assert identity['source_fingerprint']=='df16d1bd5cd279bb881f29049cdd797e2f0da6be1f065e530dae926c472cd117'
dest=HERE/'policy_candidate';assert not dest.exists(),'Preserve frozen source'
for name in [*identity['source_sha256'],'COPYING']:
 p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base/name,p)
h=dest/'tests/gpu_epochs';shutil.copyfile(HERE/'l2_policy.cuh',h/'l2_policy.cuh')
p=h/'tree.cu';s=p.read_text();marker='#include "table_resources.cuh"';assert s.count(marker)==1
s=s.replace(marker,marker+'\n#include "l2_policy.cuh"')
marker='    compact_build_table(&d_gtX,&d_gtY,dp.neg_r_inv);';assert s.count(marker)==1
s=s.replace(marker,marker+'\n    if(se_mode)qsb_enable_small_l2_policy(d_gtX,gpu_index);')
p.write_text(s)
report={'inherited_source':identity,'candidate_source':source_identity(dest),
 'changes':'Host-only checked persisting L2 policy on48MiBtableprefix after successful build; default stream ranked mode. Unsupported/small capacity leaves ordinary policy.',
 'source_inspiration':{'submission':'e2fd8093-2ba5-4d40-8f25-dabb0a4807c5','solver':'ercumentyildirim','evidence':'Author4090reports50MiBpersisting limit and+1.203%policy-only on64MiBpinning; not independent evidence for thisgeometry.','coauthor_if_substantially_used_before_promotion':'ercumentyildirim'},
 'gpu_executed':False,'limits':'No GPU execution or cache residency/speed claim; outside-window cache policy is unchanged.'}
(HERE/'policy-source.json').write_text(json.dumps(report,indent=2)+'\n');print(report['candidate_source']['source_fingerprint'])
