#!/usr/bin/env python3
"""Immutable single-qualifier root-outlining experiment; no build or selection."""
import hashlib,json,shutil,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
from preflight import source_identity
BASE=HERE.parent/'candidate';OUT=HERE/'candidate';READY=ROOT/'research/weak_field/ordinary_region/candidate'
BASE_FP='f47ad8241adac7aac14e63ad947735ada3bf3565ae60266fa7153b6744d5583a'
READY_FP='3ac276e408a4bb399ea805a56fc9e27831a7bbd6da8df5159f16e95f311c5912'
HEADER='tests/gpu_epochs/hm43_warp_inverse.cuh'
BEFORE='__device__ __forceinline__ bool hm43_warp_inverse(uint64_t result[5],int lane){'
AFTER='__device__ __noinline__ bool hm43_warp_inverse(uint64_t result[5],int lane){'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=source_identity(BASE);ready=source_identity(READY)
 assert old['source_fingerprint']==BASE_FP and ready['source_fingerprint']==READY_FP
 assert old['source_sha256'][HEADER]=='675d2b4cca914de576113b1013bac320610ce968c5bc873bf5fc2caeb205dc27'
 assert not OUT.exists(),'Preserve existing outlined snapshot; refusing overwrite'
 text=(BASE/HEADER).read_text();assert text.count(BEFORE)==1
 changed=text.replace(BEFORE,AFTER,1)
 assert changed.replace(AFTER,BEFORE,1)==text
 for rel in list(old['source_sha256'])+['COPYING']:
  dst=OUT/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/rel,dst)
 (OUT/HEADER).write_text(changed)
 identity=source_identity(OUT)
 changed_files=[n for n,v in old['source_sha256'].items() if identity['source_sha256'][n]!=v]
 assert changed_files==[HEADER] and len(identity['source_sha256'])==19
 assert source_identity(BASE)==old and source_identity(READY)==ready
 (OUT/'PROTOTYPE_NOT_QUALIFIED.md').write_text('Isolated bounded HM43 noinline root experiment. Only root declaration qualifier changes. Native call/register/stack effects and GPU behavior unmeasured; do not replace ready3ac from source-only evidence. See outlined/prepared-source.json.\n')
 reports={}
 for name in ['helper-results.json','tree-check/results.json']:
  p=HERE.parent/name
  if p.exists():
   d=json.loads(p.read_text());assert d['source_fingerprint']==BASE_FP
   reports[name]={'sha256':sha(p),'status':d['status'],'transfer_scope':'Arithmetic/control source unchanged; no fresh outlined host/native execution claimed.'}
 d={'status':'SOURCE_PREPARED_FOR_NATIVE_OUTLINING_SCREEN_NOT_QUALIFIED',**identity,'base_source_fingerprint':BASE_FP,'ready_source_fingerprint':READY_FP,'changed_files':changed_files,'exact_reversible_change':{'before':BEFORE,'after':AFTER},'all_other18_files_byte_identical':True,'helper_body_byte_identical':True,'caller_cap_fallback_and_barriers_byte_identical':True,'original_COPYING_preserved':sha(OUT/'COPYING')==sha(BASE/'COPYING'),'baseline_evidence':reports,'generator_sha256':sha(Path(__file__)),'qualification_needed':['Verify compiler emitted actual out-of-line device call in sm89 and official default builds.','Compare wholekernel80register launch bound, call stack/caller-save/root[5] ABI traffic and actual13-add loop locals.','Use host __noinline__ projection only if rerunning arithmetic; attribute-only source equality transfers algorithm evidence, not CUDA codegen.','Audit build and exact include closure, then package checks separately if selected.'],'hypothesis':'A separate root allocation/scheduling region may avoid repeated EC-loop local spills caused by fully inlined root code. noinline does not allocate registers per warp or guarantee reduction.','risk':'Direct device call can materialize40B root array per participating thread, incur parameter/caller-save/return stack traffic, or raise propagated register needs. Fullmask collectives still require every physical64..95lane.','gpu_executed':False,'native_compiled':False,'submission_performed':False,'base_and_ready_preserved':True,'coauthor_if_used':'AbdelStark','license':'Existing GPL-3.0-only header and COPYING preserved'}
 (HERE/'prepared-source.json').write_text(json.dumps(d,indent=2)+'\n')
 print(json.dumps({'status':d['status'],'source_fingerprint':identity['source_fingerprint'],'header_sha256':identity['source_sha256'][HEADER]}))
if __name__=='__main__':main()
