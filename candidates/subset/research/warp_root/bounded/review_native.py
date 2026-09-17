#!/usr/bin/env python3
"""Read saved f47/3ac sm89 SASS; never starts a compiler, VM or GPU run."""
import hashlib,importlib.util,json,re,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
RESEARCH=HERE.parents[1]
BASE=RESEARCH/'weak_field/ordinary_region'
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=module('sass_parser',RESEARCH/'specialist_warps/two_six_ring/review_sass.py')
f=module('source_preflight',RESEARCH.parent/'preflight.py')
basem=module('base_review',BASE/'review_native.py')
def main():
 np=HERE/'production-native-results.json';bp=BASE/'native-results.json';ap=HERE/'audit-native-results.json'
 n=json.loads(np.read_text());b=json.loads(bp.read_text());audit=json.loads(ap.read_text())
 identity=f.source_identity(HERE/'candidate');bi=f.source_identity(BASE/'candidate')
 assert identity['source_fingerprint']=='f47ad8241adac7aac14e63ad947735ada3bf3565ae60266fa7153b6744d5583a'
 assert bi['source_fingerprint']=='3ac276e408a4bb399ea805a56fc9e27831a7bbd6da8df5159f16e95f311c5912'
 for native in [n,audit]:
  assert native['status']=='PASS' and native['default_flags_build']
  assert all(identity['source_sha256'][k]==v for k,v in native['source_sha256'].items())
 assert set(n['source_sha256'])|set(audit['source_sha256'])==set(identity['source_sha256'])
 assert all(bi['source_sha256'][k]==v for k,v in b['source_sha256'].items())
 sass=Path(n['build_directory'])/'sass.txt';bsass=Path(b['build_directory'])/'sass.txt'
 rows=r.parse(sass);old=r.parse(bsass);pc=dict(rows)
 loop=r.region(rows,0x4490,0x9ee0);oldloop=r.region(old,0x4450,0x9d70)
 assert pc[0x4420]=='IMAD.MOV.U32 R70, RZ, RZ, RZ'
 assert pc[0x9e70]=='ISETP.GE.U32.AND P1, PT, R70, 0xd, PT'
 assert pc[0x9ed0]=='@P1 CALL.REL.NOINC 0x9ef0' and pc[0x9ee0]=='BRA 0x4490'
 assert len(loop['control_edges'])==2 and loop['instruction_slots']==1446
 assert loop['static_local_load_instruction_bytes']==loop['static_local_store_instruction_bytes']==8
 assert all(not x['instruction'].startswith('@') for x in loop['local_instructions'])
 assert oldloop['instruction_slots']==1427 and not oldloop['local_instructions']
 for op,count in [('LDG.E.128',4),('LDS.64',22),('STS.64',12),('CCTL.E.PF2',2)]:assert loop['opcodes'][op]==oldloop['opcodes'][op]==count
 producer=r.region(rows,0x2bb10,0x39800);oldproducer=r.region(old,0x29500,0x371e0)
 assert producer['instruction_slots']==3536 and oldproducer['instruction_slots']==3535
 assert producer['static_local_load_instruction_bytes']==producer['static_local_store_instruction_bytes']==0
 pn=basem.normalized(rows,0x2bb10,0x39800,0x40b70);po=basem.normalized(old,0x29500,0x371e0,0x3c700)
 assert pn!=po
 ops=producer['opcodes'];oo=oldproducer['opcodes'];producer_delta={k:ops.get(k,0)-oo.get(k,0) for k in sorted(set(ops)|set(oo)) if ops.get(k,0)!=oo.get(k,0)}
 assert producer_delta=={'ISETP.NE.AND':2,'LOP3.LUT':-1}
 helper_same=basem.normalized(rows,0x40b70,0x40ba0,0)==basem.normalized(old,0x3c700,0x3c730,0)
 assert helper_same
 pollsame=basem.normalized(rows,0x530,0x6a0,0x40b70,True)==basem.normalized(old,0x4d0,0x640,0x3c700,True)
 assert pollsame
 assert pc[0x1af10]=='ISETP.NE.AND P2, PT, R11, 0x10, PT'
 assert pc[0x1af40]=='@!P2 BRA 0x1d0d0'
 assert pc[0x1c860]=='IADD3 R11, R11, 0x1, RZ'
 assert pc[0x1c870]=='@!P2 CALL.REL.NOINC 0x1c890' and pc[0x1c880]=='BRA 0x1af10'
 assert pc[0x1d190]=='STL.64 [R1+0x20], RZ' and pc[0x1d1e0]=='CALL.REL.NOINC 0x3b650'
 assert all(pc[p].startswith('LDS.64') for p in range(0x1d150,0x1d190,16))
 assert all(pc[p].startswith('STS.64') for p in range(0x1d240,0x1d280,16))
 rootmain=r.region(rows,0x1aa80,0x1d280);batchmain=r.region(rows,0x1af10,0x1c880)
 outlines=r.region(rows,0x39aa0,0x3b640);batchoutlines=r.region(rows,0x39e90,0x3aea0)
 assert all(z['static_local_load_instruction_bytes']==z['static_local_store_instruction_bytes']==0 for z in [batchmain,batchoutlines])
 assert batchmain['instruction_slots']==408 and batchoutlines['instruction_slots']==258
 keys=['registers','shared_bytes','stack_bytes','spill_store_bytes','spill_load_bytes','instruction_slots','non_nop']
 resources={k:n['kernels'][r.KERNEL][k] for k in keys};baseline={k:b['kernels'][r.KERNEL][k] for k in keys}
 d={'status':'SOURCE_BOUND_BOUNDED_HM43_NATIVE_REVIEW_COMPLETE','source':identity,'baseline_source':bi,'native_report_sha256':r.sha(np),'audit_report_sha256':r.sha(ap),'baseline_report_sha256':r.sha(bp),'sass_sha256':r.sha(sass),'baseline_sass_sha256':r.sha(bsass),'script_sha256':r.sha(Path(__file__)),'kernel':r.KERNEL,'production_audit_union_files':len(identity['source_sha256']),'sm89_and_default_both_entries_pass':True,'resources':resources,'baseline_resources':baseline,'resource_delta':{k:resources[k]-baseline[k] for k in keys},'ordinary_loop':loop,'baseline_ordinary_loop':oldloop,'loop_change':{'iterations':13,'slot_delta':19,'new_local_load_store_B_per_active_candidate':[104,104],'baseline_local_load_store_B_per_active_candidate':[0,0],'table_and_shared_sites_unchanged':True,'inferred_spill_roles':[{'offset':'0x78','load':'0x6b30','store':'0x9e40','role':'deferredY word6,bits192..223'},{'offset':'0x7c','load':'0x69b0','store':'0x9e50','role':'deferredY word7,bits224..255'}],'extra_preloop_store_bytes':8,'preloop_evidence':r.listing(rows,0x4340,0x4370),'limit':'HighY roles inferred from final product output carry chain and nextiteration subtraction. Exact addresses/localcounts observed. Source point helper unchanged; wholekernel allocation changed.'},'prefetch':{'pcs':['0x9e10','0x9e30'],'late':True,'limit':'No new early-prefetch or effectiveness claim.'},'producer':{'current':producer,'baseline':oldproducer,'instruction_identical_after_relocation':False,'source_tree_and_SHA_header_unchanged':identity['source_sha256']['tests/gpu_epochs/tree.cu']==bi['source_sha256']['tests/gpu_epochs/tree.cu'] and identity['source_sha256']['tests/gpu_epochs/window_schedule_shared.cuh']==bi['source_sha256']['tests/gpu_epochs/window_schedule_shared.cuh'],'opcode_delta':producer_delta,'finding':'Remainslocalfree;oneextraslot,registerassignmentandloadschedulechange. Cannotreuseoldinstruction-identicalclaim.'},'polling':{'consumer_same_modulo_relocation_and_stackoffsets':pollsame,'consumer':r.listing(rows,0x530,0x6a0),'producer':r.listing(rows,0x38360,0x383e0),'fullwarp_helper_same':helper_same,'one_leader_atomic_per_spin':True,'consumer_local_read_B_per_leader_spin':8,'producer_local_B_per_spin':0},'root':{'main_region':rootmain,'matrix_main':batchmain,'outlined_root_regions':outlines,'outlined_matrix_regions':batchoutlines,'shared_exchange_helpers':r.region(rows,0x40ac0,0x40b60),'shared_exchange_helper_listing':r.listing(rows,0x40ac0,0x40b60),'matrix_main_direct_SHFL32_sites':24,'matrix_main_direct_ballot_sites':13,'matrix_outlines_call_sites':39,'matrix_regions_local_read_store_bytes':[0,0],'cap_listing':r.listing(rows,0x1af10,0x1af50),'success_and_backedge':r.listing(rows,0x1c810,0x1c890),'fallback_reload_zero_call_publish':r.listing(rows,0x1d0d0,0x1d2d0),'scalar_fallback_callee':r.region(rows,0x3b650,0x40ab0),'baseline_scalar_root_main':r.region(old,0x1aa30,0x1ab90),'interpretation':'OnlyfirstECwarpcooperates. Native16capandVzeroexit present; failureleaderreloadsoriginalshared4words,zerosfifth,callsscalarModInv,thenpublishes. HM43matrices localfree,rootsetup/finalization/fallbackmaterializeslocalarrays. BRA.DIV outlines and sharedWARPSYNC helpers mustnotbeomittedfrominstructionwork. Main/outlines arealternative/convergentcompilerpaths plusinnerloops, not additiveperbatchdynamic counts.'},'assessment':'Boundedcooperative root trades scalarlanecriticalpath for fullwarpdistributedarithmetic/collectives. Rootmatrixloopislocalfree but integrationreintroduces104Breads+104Bwrites percandidatein13pointadds,adds19loopslots, and1095wholekernelnonNOPslots. Same80regs/32832sharedcapacitydoesnotestablishbenefit. No timingorfull dynamic root-vs-spillcost comparison available. A separatelyisolatednoinline variant maytestallocationcoupling; thisf47source ispreserved.','gpu_executed':False,'candidate_modified':False,'limits':['Staticinstructioncountsnotcycles,physicaltransactionsorGPUperformance.','Compiler556/344spilltotalsarenotrepeated-looptraffic;locatedlocalsreportedseparately.','Numerical/protocolCPUqualificationandactualGPUschedulingremainseparate.','DefaultbuildPASSdoesnotproveactualdriverJITresourcesorachievedoccupancy.']}
 assert f.source_identity(HERE/'candidate')==identity
 (HERE/'native-review.json').write_text(json.dumps(d,indent=2)+'\n')
 print(json.dumps({'source':identity['source_fingerprint'],'resources':resources,'loop_slots':1446,'loop_local_B':[104,104],'producer_slots':3536,'producer_unchanged':False,'matrix_main_and_outlines_local_free':True},indent=2))
if __name__=='__main__':main()
