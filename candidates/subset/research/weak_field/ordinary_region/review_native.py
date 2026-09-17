#!/usr/bin/env python3
"""Reproduce source-bound production sm89 comparison; no compiler/GPU invocation."""
import hashlib,importlib.util,json,re,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
RESEARCH=HERE.parents[1]
BASE=RESEARCH/'specialist_warps/resident_tables/packed_digits/scoped_load'
PARSER=RESEARCH/'specialist_warps/two_six_ring/review_sass.py'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=module('sass_parser',PARSER)
f=module('source_preflight',RESEARCH.parent/'preflight.py')
def normalized(rows,lo,hi,helper,stack=False):
 out=[]
 for pc,i in rows:
  if not lo<=pc<=hi:continue
  if re.search(r'\b(?:BRA|CALL|BSSY)\b',i) or re.match(r'MOV R4, 0x',i):
   i=re.sub(r'0x[0-9a-f]+',lambda m:'FULLWARP_HELPER' if int(m[0],16)==helper else hex(int(m[0],16)-lo),i)
  if stack:i=re.sub(r'\[R1\+0x[0-9a-f]+\]','[STACK_SLOT]',i)
  out.append(i)
 return out

def main():
 np=HERE/'native-results.json';bp=BASE/'native-results.json';n=json.loads(np.read_text());b=json.loads(bp.read_text())
 ident=f.source_identity(HERE/'candidate');bi=f.source_identity(BASE/'candidate')
 assert ident['source_fingerprint']=='3ac276e408a4bb399ea805a56fc9e27831a7bbd6da8df5159f16e95f311c5912'
 assert bi['source_fingerprint']=='cd3d11f24e6fd42459eed3ba3a431c7f952de506cb1d22dfddc9b291e4c98667'
 assert all(ident['source_sha256'][p]==h for p,h in n['source_sha256'].items())
 assert all(bi['source_sha256'][p]==h for p,h in b['source_sha256'].items())
 sass=Path(n['build_directory'])/'sass.txt';bsass=Path(b['build_directory'])/'sass.txt'
 rows=r.parse(sass);old=r.parse(bsass);pc=dict(rows)
 loop=r.region(rows,0x4450,0x9d70);ol=r.region(old,0x4430,0xa7c0)
 assert loop['instruction_slots']==1427 and loop['static_local_load_instruction_bytes']==loop['static_local_store_instruction_bytes']==0
 assert pc[0x43a0]=='IMAD.MOV.U32 R64, RZ, RZ, RZ'
 assert pc[0x4570]=='IADD3 R64, R64, 0x1, RZ'
 assert pc[0x9cf0]=='ISETP.GE.U32.AND P1, PT, R64, 0xd, PT'
 assert pc[0x9d60]=='@P1 CALL.REL.NOINC 0x9d80' and pc[0x9d70]=='BRA 0x4450'
 assert len(loop['control_edges'])==2
 assert loop['opcodes']['LDG.E.128']==4
 assert loop['opcodes']['LDS.64']==ol['opcodes']['LDS.64']==22
 assert loop['opcodes']['STS.64']==ol['opcodes']['STS.64']==12
 assert pc[0x4550]=='LOP3.LUT R20, R17, 0x8000, RZ, 0xc0, !PT'
 assert pc[0x9cd0]=='CCTL.E.PF2 [R2]' and pc[0x9d10]=='CCTL.E.PF2 [R2+0x20]'
 producer=r.region(rows,0x29500,0x371e0);op=r.region(old,0x29950,0x37630)
 same=normalized(rows,0x29500,0x371e0,0x3c700)==normalized(old,0x29950,0x37630,0x3cbb0)
 helper_same=normalized(rows,0x3c700,0x3c730,0)==normalized(old,0x3cbb0,0x3cbe0,0)
 assert same and helper_same and producer['instruction_slots']==3535
 assert producer['static_local_load_instruction_bytes']==producer['static_local_store_instruction_bytes']==0
 consumer_poll_same=normalized(rows,0x4d0,0x640,0x3c700,True)==normalized(old,0x4b0,0x620,0x3cbb0,True)
 assert consumer_poll_same
 device=(HERE/'candidate/tests/gpu_epochs/compact_table_device.cuh').read_text()
 body=device.split('__device__ void compact_ec192_fixed_xyzz_packed_shared(')[1]
 assert 'for(int i=0;i<13;++i)' in body
 assert body.count('qsb_ec192_PointAddXYZZ_shared_z_weak(')==1
 assert 'qsb_weak_normalize(X); qsb_weak_normalize(Y);' in body
 assert 'qsb_weak_normalize(ZZ); qsb_weak_normalize(ZZZ);' in body
 assert body.index('qsb_weak_normalize(X)')<body.index('qsb_asym_last_add(')
 labels=['entry_seed','ordinary13add_loop','postloop_canonical_boundary_and_guard','later_consumer_inverse_recovery_hash_hit_control','sha_producer','out_of_line_helpers']
 nr=[(0,0x4440),(0x4450,0x9d70),(0x9d80,0x18800),(0x18810,0x294f0),(0x29500,0x371e0),(0x371f0,0x3c730)]
 br=[(0,0x4420),(0x4430,0xa7c0),(0xa7d0,0x18cf0),(0x18d00,0x29940),(0x29950,0x37630),(0x37640,0x3cbe0)]
 parts={}
 for label,(lo,hi),(blo,bhi) in zip(labels,nr,br):
  a=r.region(rows,lo,hi);z=r.region(old,blo,bhi)
  parts[label]={'current':a,'baseline':z,'static_delta':{k:a[k]-z[k] for k in ['instruction_slots','non_nop','static_local_load_instruction_bytes','static_local_store_instruction_bytes']}}
 norms=[('ZZZ',0xa2d0,'R16','R17'),('Y',0xae20,'R69','R70'),('ZZ',0xb0a0,'R30','R31'),('X',0xbc60,'R18','R62')]
 for name,start,low,high in norms:assert re.search(r'IADD3 R\d+, P\d, '+low+r', 0x3d1, RZ',pc[start]),name
 keys=['registers','shared_bytes','stack_bytes','spill_store_bytes','spill_load_bytes','instruction_slots','non_nop']
 resources={k:n['kernels'][r.KERNEL][k] for k in keys};baseline={k:b['kernels'][r.KERNEL][k] for k in keys}
 d={'status':'SOURCE_BOUND_SM89_NATIVE_REVIEW_COMPLETE','source':ident,'baseline_source':bi,'native_report_sha256':r.sha(np),'baseline_native_report_sha256':r.sha(bp),'native_production_files_verified':len(n['source_sha256']),'sass_sha256':r.sha(sass),'baseline_sass_sha256':r.sha(bsass),'script_sha256':r.sha(Path(__file__)),'parser_sha256':r.sha(PARSER),'kernel':r.KERNEL,'resources':resources,'baseline_resources':baseline,'resource_delta':{k:resources[k]-baseline[k] for k in keys},'hot_loop':loop,'baseline_hot_loop':ol,
 'loop_comparison':{'iterations':13,'slot_delta':-167,'slot_change_percent':100*(1427/1594-1),'local_read_write_B_per_candidate':[0,0],'baseline_local_read_write_B_per_candidate':[104,104],'shared_LDS64_sites':22,'shared_STS64_sites':12,'global_LDG128_sites':4,'source_order':[14,15]+list(range(13))+[13],'native_control':r.listing(rows,0x9cd0,0x9d80),'decode_and_sign_evidence':r.listing(rows,0x4450,0x46b0),'limit':'Static loop includes conditional exit transfer and final skipped backedge. No out-of-line helper on repeated path. Counts are instruction sites/logical operands, not cycles, physical transactions or throughput.'},
 'prefetch':{'hint_pcs':['0x9cd0','0x9d10'],'position':'near loop tail; no restored early-prefetch claim','sites_between_last_hint_and_next_first_table_load':18,'baseline_sites':24,'limit':'Intervening instruction sites are not latency or overlap duration; final iteration uses a different guarded-final path.'},
 'producer':{'current':producer,'baseline':op,'identical_after_relocation':same,'fullwarp_helper_identical':helper_same,'normalization':'Producer base delta -0x450; six external full-warp CALL targets delta -0x4b0. External helper matched separately, not ignored.','sha_region_normalized_sha256':hashlib.sha256('\n'.join(normalized(rows,0x29500,0x371e0,0x3c700)).encode()).hexdigest()},
 'polling':{'consumer_identical_modulo_relocation_and_stack_offsets':consumer_poll_same,'consumer_listing':r.listing(rows,0x4d0,0x640),'producer_listing':r.listing(rows,0x35d50,0x35dd0),'fullwarp_helper':r.listing(rows,0x3c700,0x3c730),'one_active_atomic_lane_per_spin':True,'consumer_logical_local_read_B_per_leader_spin':8,'baseline_consumer_logical_local_read_B_per_leader_spin':8,'producer_local_read_write_B_per_spin':[0,0],'interpretation':'Lane0 polls, reconvergence then full-warp WARPSYNC and CTA fence precede payload reads. Physical requests/serialization unmeasured.'},
 'static_region_accounting':parts,'region_accounting_limit':'Static local operand sums include branches, rare paths, explicit inverse arrays and repetitions. Neither these nor ptxas aggregate spills are a per-candidate total. Regions are source/CFG boundaries, not isolated timed components.',
 'boundary_normalization':{'four_source_calls':['X','Y','ZZ','ZZZ'],'native_chains':[{'value_inferred_from_dataflow':name,'carry_chain_start':hex(start),'low_input':low,'high_input':high} for name,start,low,high in norms],'evidence':r.listing(rows,0xa2d0,0xa6f0)+r.listing(rows,0xae20,0xb430)+r.listing(rows,0xbc60,0xbfe0),'effect':'Four full-width +c carry chains with final carry-controlled XOR/select appear interleaved with unchanged guarded final arithmetic. ZZZ normalization feeds first(Y) product, Y feeds subtraction, ZZ feeds second(X) product and X feeds subtraction. The combined postloop boundary/guard region grows86slots,44B static local-load operands and60B static stores. These differences cannot be assigned exclusively to normalization because register allocation/scheduling of final arithmetic changes.','new_deferred_ZZZ_spills':{'store_pcs':['0xa470','0xa4f0','0xa530','0xa5a0','0xa5e0','0xa620','0xa6c0','0xa6f0'],'interpretation':'Normalized shared-loaded ZZZ eight words materialize into stack slots around the first final product, outside repeated loop.'}},
 'assessment':'Corrected weak-region native screen is stronger than invalid global elision: repeated point loop loses167slots and all104B+104B local traffic while producer/shared operands/poll protocol remain unchanged. Boundary/final and later control spill costs rise; no full dynamic total or measured gain established. Separate arithmetic/curve/protocol qualification still required.',
 'gpu_executed':False,'candidate_modified':False,'qualification_limit':'Production sm89 native review only; separate default/audit and actual-helper CPU reports decide qualification. This report does not select or upload source.'}
 assert f.source_identity(HERE/'candidate')==ident
 (HERE/'native-review.json').write_text(json.dumps(d,indent=2)+'\n')
 print(json.dumps({'fingerprint':ident['source_fingerprint'],'loop_slots':1427,'loop_local_B':[0,0],'producer_identical':same,'consumer_poll_same':consumer_poll_same,'regions':{k:v['static_delta'] for k,v in parts.items()}},indent=2))
if __name__=='__main__':main()
