from pathlib import Path
import json,hashlib,random
W=Path(__file__).resolve().parent;C=W.parent
# Reversal of a bounded reviewed change set proves all other production text unchanged.
changes=json.loads((W/'source-changes.json').read_text())
for name in ['pinning.cu','PackedRecovery.cuh']:
 s=(C/name).read_text()
 for file,old,new in reversed(changes):
  if name==file:assert s.count(new)==1;s=s.replace(new,old)
 assert s==(W/'baseline'/name).read_text()
s=(C/'pinning.cu').read_text();h=(C/'InterleavedCarry.cuh').read_text();p=(C/'PackedRecovery.cuh').read_text()
for token in ['QSB_S0_THREADS==256 && QSB_S0_BLOCKS==2','storage[8*QSB_S0_THREADS]','blockIdx.x*256u+(threadIdx.x&1u)*128u+(threadIdx.x>>1)','uint32_t lt_hi = (start_lt >> 8) + (uint32_t)blockIdx.x','uint32_t low = (threadIdx.x&1u)*128u+(threadIdx.x>>1)','int idx = STAGE==0 ? (int)qsb_prepare_candidate_index()','QSB_S2_THREADS==128']:
 assert token in s
assert s.count('codes[(size_t)c*QSB_S0_THREADS+threadIdx.x]')==2
assert 'qsb_prepare_scratch' not in s
assert 'qsb_interleaved_carry<QSB_RECOVERY_N,2>' in p and 'i=(size_t)qsb_prepare_candidate_index()' in p
assert '((size_t)blockIdx.x*G+g)*4+k' in h and '(size_t)blockIdx.x*G+g<root_count' in h
assert 'pr[k][(off+(i^half))*G+g]=out[k]' in h
checks=0;negative=0;sectors=0
for batch in [1,2,31,32,63,64,127,128,129,255,256,257,383,384,511,512,513,1023,8388608]:
 root_count=(batch+127)//128
 blocks=(batch+255)//256
 for block in sorted(set([0,blocks-1])):
  mapping=[block*256+(t&1)*128+(t>>1) for t in range(256)]
  assert sorted(mapping)==list(range(block*256,(block+1)*256))
  assert len([i for i in mapping if i<batch])==max(0,min(256,batch-block*256))
  for t,idx in enumerate(mapping):
   assert idx//128==block*2+(t&1) and idx%128==t>>1
   for start in [0,256,500000000,0xffffff00]:
    lt=(start+idx)&0xffffffff;hi=((start>>8)+block)&0xffffff;lo=(t&1)*128+(t>>1)
    assert ((hi<<8)|lo)==lt;checks+=1
    # Original128 CTA expression must not survive on the new geometry.
    if (((start>>8)+(block>>1))<<8 | (((block&1)<<7)|t))&0xffffffff !=lt:negative+=1
   for plane in range(15):assert (plane*256+t)*4+4<=16384
  for g in range(2):assert (block*2+g<root_count)==(block*256+g*128<batch)
# Full-batch stores retain the same number of 32-byte sectors per warp for each16-byte plane.
for warp in range(8):
 idx=[((t&1)*128+(t>>1)) for t in range(32*warp,32*warp+32)]
 new={(i*16+b)//32 for i in idx for b in [0,15]}
 old={(i*16+b)//32 for i in range(32*warp,32*warp+32) for b in [0,15]}
 assert len(new)==len(old)==16;sectors+=1
assert negative>0
# Frozen promoted runtime files not involved in the collective integration must match recorded hashes.
hashes=json.loads((W/'unchanged-hashes.json').read_text())
for file,expected in hashes.items():assert hashlib.sha256((C/file).read_bytes()).hexdigest()==expected
r={'status':'PASS','source_closure_changes':len(changes),'unchanged_runtime_files':len(hashes),'SHA_index_wrap_checks':checks,'old_index_negative_control_differences':negative,'global_store_sector_comparisons':sectors,'arena_bytes':16384,'digit_bytes':15360,'root_and_active_tail_bounds':'PASS','native_compile':False,'performance_measured':False}
(W/'integration-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
