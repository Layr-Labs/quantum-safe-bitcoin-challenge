#!/usr/bin/env python3
"""Integrate the promoted54/56 window selection with the guarded48MiB pipeline.

This ports the frontend selection, not the entire PR77 grinder or its score.
"""
import json,shutil,sys,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
from preflight import source_identity
parent=HERE.parent/'policy_candidate';identity=source_identity(parent)
assert identity['source_fingerprint']=='76c07307fc835215e0974b0fe73118da9c2375ca0c9cf757661b73fb53678d5f'
frontier=ROOT/'research/resident_projective/pr77/control';frontier_id=source_identity(frontier)
assert frontier_id['source_fingerprint']=='e18e355f7d8c90452e6ad61a6cbf4be1a495b5bb7d0817a1e8d00a69d935c918'
dest=HERE/'candidate';assert not dest.exists(),'Preserve frozen source'
for name in [*identity['source_sha256'],'COPYING']:
 p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent/name,p)
h=dest/'tests/gpu_epochs';f=frontier/'tests/gpu_epochs'
old=(h/'window_schedule_shared.cuh').read_text();new=(f/'window_schedule_shared.cuh').read_text()
marker='static int qsb_prepare_window_schedule('
assert old[old.index(marker):]==new[new.index(marker):],'Review any changed hash/schedule body'
(h/'window_schedule_shared.cuh').write_text(new)
front=(f/'tree.cu').read_text();p=h/'tree.cu';s=p.read_text()
start='        uint8_t h_win3[QSB_SE_PER_EPOCH][QSB_SE_TWIN];'
end='        if (qsb_prepare_window_schedule(dp.dummy_sigs, h_win3, h_const_words)) return 1;'
assert s.count(start)==s.count(end)==front.count(start)==front.count(end)==1
selection=front[front.index(start):front.index(end)]
s=s[:s.index(start)]+selection+s[s.index(end):];p.write_text(s)
report={'inherited_source':identity,'promoted_frontend_reference':frontier_id,'candidate_source':source_identity(dest),
 'promoted_origin':'PR77, promotiond2772418e0f372767b4c59f7382d71f9142585fe',
 'changes':'Exact promoted256window selection and second/first-key lane grouping;54first/56second classes. Existing schedule/hash bodies identical. Guarded14pointpath, checked arithmetic, externalinverse and48MiBpolicy unchanged.',
 'selection_sha256':hashlib.sha256(selection.encode()).hexdigest(),
 'gpu_executed':False,'limits':'The488210159officialscore describes the whole originalPR77source, not this frontend port or our corrected controls. No performance additivity or GPU measurement claimed.'}
(HERE/'prepared-source.json').write_text(json.dumps(report,indent=2)+'\n');print(report['candidate_source']['source_fingerprint'])
