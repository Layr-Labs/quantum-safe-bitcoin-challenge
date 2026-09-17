#!/usr/bin/env python3
"""Source-bound review of saved outlined HM43 sm89 SASS. No builds or GPU work."""
import importlib.util,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('inline_review',HERE.parent/'review_native.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r=m.r

def main():
 paths={'outlined':HERE/'production-native-results.json','audit':HERE/'audit-native-results.json','inline':HERE.parent/'production-native-results.json','baseline':m.BASE/'native-results.json'}
 reports={k:json.loads(v.read_text()) for k,v in paths.items()}
 roots={'outlined':HERE/'candidate','inline':HERE.parent/'candidate','baseline':m.BASE/'candidate'}
 identities={k:m.f.source_identity(v) for k,v in roots.items()}
 expected={'outlined':'a36071b7a5b06c22d16172f8f29ec0d86eb1cd90620e3716526b40349a890047','inline':'f47ad8241adac7aac14e63ad947735ada3bf3565ae60266fa7153b6744d5583a','baseline':'3ac276e408a4bb399ea805a56fc9e27831a7bbd6da8df5159f16e95f311c5912'}
 for k in identities:assert identities[k]['source_fingerprint']==expected[k]
 for k,n in reports.items():
  sid=identities['outlined' if k=='audit' else k]
  assert n['status']=='PASS'
  if k!='baseline':assert n['default_flags_build']
  assert all(sid['source_sha256'][p]==v for p,v in n['source_sha256'].items())
 assert set(reports['outlined']['source_sha256'])|set(reports['audit']['source_sha256'])==set(identities['outlined']['source_sha256'])
 changed=[k for k,v in identities['outlined']['source_sha256'].items() if identities['inline']['source_sha256'][k]!=v]
 assert len(changed)==1
 text=(roots['outlined']/changed[0]).read_text();oldtext=(roots['inline']/changed[0]).read_text()
 assert text.replace('__noinline__ bool hm43_warp_inverse','__forceinline__ bool hm43_warp_inverse')==oldtext
 sass={k:Path(reports[k]['build_directory'])/'sass.txt' for k in roots}
 rows={k:r.parse(v) for k,v in sass.items()};new=rows['outlined'];pc=dict(new)
 loops={'outlined':r.region(new,0x4450,0x9d70),'baseline':r.region(rows['baseline'],0x4450,0x9d70),'inline':r.region(rows['inline'],0x4490,0x9ee0)}
 assert loops['outlined']['instruction_slots']==1427 and not loops['outlined']['local_instructions']
 loop_same=m.basem.normalized(new,0x4450,0x9d70,0)==m.basem.normalized(rows['baseline'],0x4450,0x9d70,0)
 assert loop_same and pc[0x9d70]=='BRA 0x4450' and pc[0x9d60]=='@P1 CALL.REL.NOINC 0x9d80'
 producer=r.region(new,0x29690,0x37370)
 producer_same=m.basem.normalized(new,0x29690,0x37370,0x41130)==m.basem.normalized(rows['baseline'],0x29500,0x371e0,0x3c700)
 assert producer_same and producer['instruction_slots']==3535 and not producer['local_instructions']
 poll_same=m.basem.normalized(new,0x4d0,0x640,0x41130,True)==m.basem.normalized(rows['baseline'],0x4d0,0x640,0x3c700,True)
 assert poll_same and pc[0x530]=='ATOMS.ADD R0, [R0.X4+0x8018], RZ'
 assert pc[0x1ab60]=='CALL.REL.NOINC 0x37690' and pc[0x1ac80]=='CALL.REL.NOINC 0x3bdc0'
 assert pc[0x37af0]=='ISETP.NE.AND P2, PT, R8, 0x10, PT' and pc[0x37b20]=='@!P2 BRA 0x39b10'
 assert pc[0x39260]=='IADD3 R8, R8, 0x1, RZ' and pc[0x39280]=='BRA 0x37af0'
 assert pc[0x1ac30]=='STL.64 [R1+0x20], RZ'
 assert all(pc[p].startswith('LDS.64') for p in range(0x1abf0,0x1ac30,16))
 assert all(pc[p].startswith('STS.64') for p in range(0x1ace0,0x1ad20,16))
 matrix=r.region(new,0x37af0,0x39280);outlines=r.region(new,0x3a0d0,0x3b390)
 assert matrix['static_local_load_instruction_bytes']==4 and matrix['static_local_store_instruction_bytes']==0
 assert matrix['local_instructions']==[{'pc':'0x38e10','instruction':'LDL R17, [R1+0x9c]'}]
 assert not outlines['local_instructions']
 keys=['registers','shared_bytes','stack_bytes','spill_store_bytes','spill_load_bytes','instruction_slots','non_nop']
 resources={k:{f:n['kernels'][r.KERNEL][f] for f in keys} for k,n in reports.items() if k!='audit'}
 result={'status':'SOURCE_BOUND_OUTLINED_HM43_NATIVE_REVIEW_COMPLETE','source':identities['outlined'],'comparison_sources':{k:v for k,v in identities.items() if k!='outlined'},'native_report_sha256':{k:r.sha(v) for k,v in paths.items()},'sass_sha256':{k:r.sha(v) for k,v in sass.items()},'script_sha256':r.sha(Path(__file__)),'production_audit_union_files':len(identities['outlined']['source_sha256']),'both_entries_sm89_and_default_pass':True,'only_source_change_from_inline':{'file':changed[0],'change':'hm43_warp_inverse __forceinline__ to __noinline__; exact reverse replacement equals inline file'},'kernel':r.KERNEL,'resources':resources,'ordinary_loops':loops,'ordinary_loop':{'same_as_ready3ac_after_return_target_normalization':loop_same,'iterations':13,'local_load_store_B_per_candidate':[0,0],'inline_local_load_store_B_per_candidate':[104,104],'table_load_sites':4,'shared_load64_sites':22,'shared_store64_sites':12,'late_prefetch_sites':2,'decision':'Outlining restores the exact ready3ac ordinary loop; it removes the inline integration spill regression.'},'producer':{'region':producer,'same_as_ready3ac_after_relocation':producer_same,'local_load_store_B':[0,0]},'polling':{'consumer_same_modulo_relocation_stackoffsets':poll_same,'consumer_listing':r.listing(new,0x4d0,0x640),'producer_listing':r.listing(new,0x35ee0,0x35f60),'one_leader_atomic_per_spin':True,'consumer_leader_local_reads_B_per_spin':8,'producer_local_B_per_spin':0},'root':{'caller':r.region(new,0x1a9d0,0x1ad20),'caller_listing':r.listing(new,0x1a9d0,0x1ad60),'hm43_call':'0x1ab60 -> 0x37690','legacy_fallback_call':'0x1ac80 -> 0x3bdc0','callee':r.region(new,0x37690,0x3bdb0),'matrix_main':matrix,'matrix_outlines':outlines,'exchange_helpers':r.region(new,0x41080,0x41120),'fullwarp_join':r.listing(new,0x41130,0x41160),'cap':r.listing(new,0x37af0,0x37b50),'batch_backedge':r.listing(new,0x39230,0x392a0),'return':r.listing(new,0x39af0,0x39b50),'abi':'First EC warp (physical64..95) materializes40B local root perlane; caller initializes allfive64bit words, leader overwritesfour inputwords. HM43 input/output and canonicalization access thatarray. Leader publishesfour resultwords. Failure leader reloads originalsharedroot, zerosword4, invokesexisting scalarinverse. OtherECwarps waitat unchanged publicationjoin.','context_reload':'0x38e10 LDL4B [R1+0x9c] is inside the outerbatch mainrange, outside its innerdivstep loop, but0x38e00 @P0 BRA0x38fc0 bypasses it. Only lanes reaching this conditional path execute the reload;4B is its operandwidth, not unconditional traffic for all32lanes or allbatches. The same contextslot is stored at0x37890 and alternative0x39e10 during setup. Outlinedmatrixregions havezero locals.','context_reload_branch_listing':r.listing(new,0x38dc0,0x38e70),'count_limit':'Caller/callee staticcounts include mutuallyexclusive success/fallback/canonicalization and BRA.DIV alternatives. They are not summed as dynamicbytes or cycles perroot. Firstwarproot workoccurs per192candidatecohort, unlike pointloop work peractivecandidate.'},'assessment':'Prefer outlined a360 over inline f47 for further qualification: it preserves ready3ac ordinary loop and producer while adding bounded cooperative root. Same80regs/32832shared;344Bstack versus352baseline/368inline. RootABI/finalization localtraffic, exchangehelpers, capfallback and fullwarp work remain material costs. Neither native screen proves a net speedup over ready3ac nor root criticalpath reduction on GPU.','gpu_executed':False,'candidate_modified':False,'limitations':['StaticSASSslots/operandbytes are not cycles,transactions or timings.','Aggregate compiler552/388spilltotals mustnot substitute for actual repeatedlooptraffic.','No unconditional16batchtermination assumption; capfallback remains.','Same reportedresources doesnot establish achievedoccupancy or runtime scheduling.','CPU arithmetic/protocol evidence is separate from this native review.']}
 assert m.f.source_identity(HERE/'candidate')==identities['outlined']
 (HERE/'native-review.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'source':expected['outlined'],'loop_same':loop_same,'producer_same':producer_same,'loop_local_B':[0,0],'matrix_main_local_read_B':4,'resources':resources['outlined']},indent=2))
if __name__=='__main__':main()
