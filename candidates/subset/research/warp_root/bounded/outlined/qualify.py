#!/usr/bin/env python3
"""Verify existing exact-source CPU/native receipts; never stage, build or submit."""
import hashlib,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];SOURCE=HERE/'candidate';BOUNDED=HERE.parent;READY=ROOT/'research/weak_field/ordinary_region'
sys.path[:0]=[str(ROOT),str(ROOT/'research')]
from preflight import source_identity
from ptx_field_model import function
FP='a36071b7a5b06c22d16172f8f29ec0d86eb1cd90620e3716526b40349a890047';BFP='f47ad8241adac7aac14e63ad947735ada3bf3565ae60266fa7153b6744d5583a';RFP='3ac276e408a4bb399ea805a56fc9e27831a7bbd6da8df5159f16e95f311c5912'
HEADER='tests/gpu_epochs/hm43_warp_inverse.cuh';TREE='tests/gpu_epochs/tree_inverse.cuh'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
textsha=lambda s:hashlib.sha256(s.encode()).hexdigest()
def main():
 identity=source_identity(SOURCE);bounded=source_identity(BOUNDED/'candidate');ready=source_identity(READY/'candidate')
 assert identity['source_fingerprint']==FP and bounded['source_fingerprint']==BFP and ready['source_fingerprint']==RFP
 files=identity['source_sha256'];reports={}
 def record(path):
  d=json.loads(path.read_text());reports[str(path.relative_to(ROOT))]={'sha256':sha(path),'status':d.get('status')};return d
 header=(SOURCE/HEADER).read_text();bh=(BOUNDED/'candidate'/HEADER).read_text()
 before='__device__ __forceinline__ bool hm43_warp_inverse(';after='__device__ __noinline__ bool hm43_warp_inverse('
 assert header.count(after)==1 and header.replace(after,before,1)==bh
 assert all(files[n]==v for n,v in bounded['source_sha256'].items() if n!=HEADER)
 tree=(SOURCE/TREE).read_text();rt=(READY/'candidate'/TREE).read_text();sig='__device__ __forceinline__ void qsb_ec192_inverse_tree_scratch('
 body=function(tree,sig);rbody=function(rt,sig)
 currentroot=body[body.index('    // Only the first EC warp'):body.index('    qsb_ec192_barrier(sync);',body.index('    // Only the first EC warp'))]
 oldroot=rbody[rbody.index('    if(tid==0){'):rbody.index('    qsb_ec192_barrier(sync);',rbody.index('    if(tid==0){'))]
 assert body.replace(currentroot,oldroot,1)==rbody
 restored=tree.replace('#include "hm43_warp_inverse.cuh"\n','',1).replace(body,rbody,1);assert restored==rt
 unchanged=[n for n,v in ready['source_sha256'].items() if n!=TREE]
 assert len(unchanged)==17 and all(files[n]==ready['source_sha256'][n] for n in unchanged)
 assert sha(SOURCE/'COPYING')==sha(READY/'candidate/COPYING')
 helper=record(BOUNDED/'helper-results.json');tr=record(BOUNDED/'tree-check/results.json')
 assert helper['status']=='PASS_ACTUAL_BOUNDED_HM43_HOST_SUCCESS_FAILURE_AND_MUTATIONS' and tr['status']=='PASS_BOUNDED_HM43_TREE_AND_ACTUAL_LEGACY_FALLBACK_PROJECTION'
 for d in [helper,tr]:
  assert d['source_fingerprint']==BFP and d['source_sha256']==bounded['source_sha256']
  for f,v in d['support_sha256'].items():assert sha(BOUNDED.parent/f)==v
 assert helper['checker_sha256']==sha(BOUNDED/'check_helper.py') and tr['checker_sha256']==sha(BOUNDED/'check_tree.py')
 assert helper['header_sha256']==textsha(bh)
 assert helper['forced_failure_lane_returns_checked']==7296 and helper['forced_failure_output_words_unchanged_checked']==36480
 for name,d in helper['reports'].items():
  assert d['compiled'] and not d['timeout']
  if name in ['cap16_positive','cap0_no_publication','cap1_no_publication']:assert d['exit_code']==0 and d['result']['all32_lane_counts_equal']
  else:assert d['exit_code']==2 and 'MISMATCH' in d['diagnostic']
 for d in tr['results']:
  assert d['compiled'] and d['actual_tree_helper_sha256']==textsha(body)
  if not d['mutation']:
   assert d['returncode']==0 and d['legacy_projection_independent_roots']==520
   for c in d['cases']:
    assert c['multiply']==669 and c['subgroup_joins']==5 and c['root_callers']==32 and c['outputs']==192
    assert c['actual_scalar_fallback_calls']==(0 if d['cap']==16 else 1)
  else:assert d['returncode']!=0 and d['diagnostics']
 assert tr['tails']==[0,1,31,32,33,191,192]
 proof=record(BOUNDED.parent/'range-review.json')
 assert proof['proof_results']['signed_working_bits']==384 and proof['proof_results']['RS_after16_abs_strict_upper']=='2^260'
 assert '#define HM43_WARP_MAX_BATCHES 16' in header and 'if(completed_batches==HM43_WARP_MAX_BATCHES)return false;' in header
 assert 'if(!root_complete)' in currentroot and 'root[4]=0;' in currentroot and '_ModInv(root);' in currentroot
 # Both separate build closures together must cover every current source file.
 prod=record(HERE/'production-native-results.json');audit=record(HERE/'audit-native-results.json');union={}
 for d in [prod,audit]:
  assert d['status']=='PASS' and d['default_flags_build'] and not d['gpu_executed']
  assert all(v['target']=='sm_89' for v in d['kernels'].values())
  for f,v in d['source_sha256'].items():assert files[f]==v;union[f]=v
 assert union==files and len(union)==19
 policy=record(HERE/'policy-host-results.json');assert policy['status']=='PASS' and policy['source_fingerprint']==FP
 assert policy['checker_sha256']==sha(ROOT/'research/specialist_warps/resident_tables/small32/check_policy.py')
 assert policy['actual_after_build_call_verified'] and all(v['detected'] for v in policy['negative_controls'].values())
 for f,v in policy['component_sha256'].items():assert files['tests/gpu_epochs/'+f]==v
 # The current native review is required; never substitute the inline-f47 review.
 review=record(HERE/'native-review.json');assert review['source']==identity
 assert review['native_report_sha256']['outlined']==sha(HERE/'production-native-results.json') and review['native_report_sha256']['audit']==sha(HERE/'audit-native-results.json')
 assert review['script_sha256']==sha(HERE/'review_native.py')
 assert review['production_audit_union_files']==19 and review['both_entries_sm89_and_default_pass']
 loop=review['ordinary_loops']['outlined']
 assert loop['instruction_slots']==1427 and not loop['local_instructions']
 rq=record(READY/'qualification.json');assert rq['source_fingerprint']==RFP and rq['source_sha256']==ready['source_sha256']
 transferred={}
 # Bind every ancestor evidence digest; do not turn old tests into fresh outlined runs.
 for rel,v in rq['reports'].items():
  path=ROOT/rel;assert sha(path)==v['sha256'];record(path)
 for label,name in [('weak_products','product-results.json'),('weak_helper','helper-results.json'),('point_chain_recovery','chain-check/results.json')]:
  path=READY/name;d=record(path);assert d['source_fingerprint']==RFP
  transferred[label]={'report':str(path.relative_to(ROOT)),'sha256':sha(path),'fresh_outlined_run':False,'binding':'Compact helper, weak primitives, canonical math and complete tree.cu/recovery are byte-identical3ac. Inverse-tree root is separately checked onboundedf47, samebody/callerhere.'}
 for label,evidence in rq['transferred_component_evidence'].items():
  if label=='inverse-check-results.json':continue # old whole-header match no longer applies
  assert sha(ROOT/evidence['report'])==evidence['sha256'];record(ROOT/evidence['report'])
  transferred['ready_'+label]={'fresh_outlined_run':False,'inherited_receipt':evidence,'current_binding':('Outer tree.cu/arena protocol unchanged; nonroot inverse/barrier reconstructed exactly, newroot separately checked. Prior wholecontrol was mocked and is not a freshoutlinedrun.' if label=='protocol' else 'Ready3ac component source unchanged; inverse-root modification is outside this component evidence.')}
 transferred['inverse_root']={'fresh_outlined_run':False,'supersedes_old_inverse_whole_header_transfer':True,'report':str((BOUNDED/'tree-check/results.json').relative_to(ROOT)),'sha256':sha(BOUNDED/'tree-check/results.json'),'current_binding':'Actual tree helper and cap/fallbackcaller byte-identicalboundedf47; only HM43 declarationqualifier differs.'}
 # Existing CPU protocol keeps same barriers, arena layout and outer kernel;
 # numerical root replacement is composed with separately checked actual tree.
 result={'status':'CPU_AND_NATIVE_COMPOSITE_QUALIFIED_PACKAGE_AND_NOTE_PENDING',**identity,'qualifier_sha256':sha(Path(__file__)),'bounded_source_fingerprint':BFP,'ready_source_fingerprint':RFP,'source_equality':{'only_qualifier_differs_from_bounded':True,'all_other18_files_equal_bounded':True,'root_body_and_caller_identical_bounded':True,'ready3ac_nonroot_tree_reconstructed_exactly':True,'other17_ready_files_equal':unchanged,'COPYING_preserved':True},'reports':reports,'native_production_audit_union_files':19,'native_summary':{'resources':review['resources']['outlined'],'ordinary_loop_slots':loop['instruction_slots'],'ordinary_loop_local_instructions':len(loop['local_instructions']),'root_cost_limit':'First EC warp materializes40B per lane; root matrix includes a4B context reload per traversed outer batch, plus root/canonicalization/call costs. Static counts are not timings.'},'production_and_audit_default_and_sm89_pass':True,'fresh_current_startup_policy':{'cases':policy['host_cases'],'injected_failures':policy['injected_failures'],'extra_paths':policy['extra_path_cases'],'all5_mutations_detected':True},'bounded_actual_host_evidence':{'fresh_outlined_run':False,'transfer':'Exact unchanged helperbody/caller; the only root declaration qualifier change affects codegen, separatelynativechecked.','cap16_helper_cases':1452,'forced_failure_returns':7296,'unchanged_failure_words':36480,'natural_tree_outputs':tr['natural_values_checked'],'fault_injected_positive_outputs':tr['fault_injected_positive_values_checked'],'tree_canaries':tr['canaries_checked'],'tail_sizes':tr['tails'],'root_participants':32,'tree_multiply_count':669,'subgroup_joins':5,'scalar_fallback':'Actual legacy source/functions with scalar carry projections, independently520roots perrun; not OpenSSL replacement.','fault_injection_limit':'Private state poison tests defensive reload/reset/publication; naturalfalsecontract leavesprivateinputunchanged.'},'transferred_ready_components':transferred,'qualification_scope':'Exact source composition plus finite actual CPU helper/tree arithmetic and legacyfallback, fresh currentpolicy, native19fileunion andsourceboundreview. No fresh wholeoutlinedkernel/protocol execution.','remaining':['No CUDA/GPU execution, runtime race/scheduling/occupancy or throughput measurement.','Package/public note/selection are separate pending decisions.','Source preparation markers remain historical; this receipt supersedes only qualification status.'],'gpu_executed':False,'source_modified':False,'stage_modified':False,'submission_performed':False,'coauthor_if_used':'AbdelStark'}
 assert source_identity(SOURCE)==identity and source_identity(BOUNDED/'candidate')==bounded and source_identity(READY/'candidate')==ready
 (HERE/'qualification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'source_fingerprint':FP,'reports_bound':len(reports)}))
if __name__=='__main__':main()
