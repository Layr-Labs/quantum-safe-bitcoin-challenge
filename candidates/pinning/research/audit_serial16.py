from pathlib import Path
import json,runpy,random,re,hashlib
w=Path(__file__).resolve().parent;p=w.parent;root=w.parents[1];src=(p/'pinning.cu').read_text()
C=runpy.run_path(str(w/'crypto.py'));N=C['N'];P=C['P'];B=1<<256;L=1<<240;K=65535*L;mul=C['point_mul'];add=C['point_add'];rng=random.Random(165716)
for token in ['#define GT_CHUNKS 16','#define GT_TOTAL_ENTRIES (1u << 19)','#define GT_HI 256','return 16*c;','return (unsigned)c << 15;','const unsigned pos=16u*c+1u;','const unsigned bits=16u;','base+=1u<<15;','int ch=(int)(t>>15);','gt_batch_ladder(grp,step,GT_HI-1,','#define QSB_L2_SKIP 0']:
 assert token in src,token
assert src.count('BN_set_word(shift, 1u<<16)')==2
assert src.count('size_t pipeline_tree_bytes=((size_t)BATCH+31u)/32u*sizeof(uint32_t);')==2
assert src.count('qsb_s16_publish_identity(identity_flags,idx,n,special);')==1
assert 'if(special){qsb_serial16_special_point(special,X,Y,U,V);return;}' in src
assert '#if QSB_TREE_OFFLOAD || QSB_TREE_OFFLOAD2' in src
assert 'if(identity && !qsb_s16_identity_flag((const uint32_t*)tree,(unsigned)idx))return;' in src
assert 'identity ? qsb_serial16_identity_finish(u2rx,u2ry,q1x,q2x) : qsb_packed_finish(' in src
# Both allocations and launch flow use same original slot pointer; root intermediates never touch it.
assert 'cudaMalloc(&d_pipeline_tree[s],pipeline_tree_bytes)' in src and 'cudaMalloc(&d_pipeline_tree,pipeline_tree_bytes)' in src
assert src.count('saved,roots,tree,tp);')==2
for n in [64,128,256]:assert 16*n*4+4*n*8<=12*n*8
addresses=0
for ch in range(16):
 for i in range(32768):
  t=(ch<<15)+i;assert (t>>15)==ch and t-(ch<<15)==i
  m=2*i+1;hi,lo=divmod(m,256);assert lo&1 and 0<lo<256 and hi<256
  assert t*64+64<=32*(1<<20);addresses+=1

def digits(k):
 d=2*(k%N)-N;out=[]
 for i in range(15):v=(d&0x1ffff)-65536;out.append(v);d=(d-v)>>16
 out.append(d);return out
curve=0;normalsteps=0;exceptions=0
for a in [1,2,N-1,rng.randrange(1,N)]:
 A=mul(a);base=mul(pow(2,-1,N),A);bases=[]
 for ch in range(16):bases.append(base);base=mul(1<<16,base)
 specialpoint=mul(K,A)
 ks=[0,N,K,N-K,1,N-1,B-1]+[rng.randrange(B) for _ in range(17)]
 for k in ks:
  ds=digits(k);total=None
  for ch,d in enumerate(ds):
   # Exact table-model signed-y load; negative multiply only applies the y sign.
   point=mul(abs(d),bases[ch]);assert point is not None
   if d<0:point=(point[0],P-point[1])
   if total is not None and total[0]==point[0]:
    assert ch==15 and k%N in [0,K,N-K];exceptions+=1
   else:normalsteps+=1
   total=add(total,point)
  assert total==mul(k,A)
  if k%N in [K,N-K]:assert total==(specialpoint if k%N==K else (specialpoint[0],P-specialpoint[1]))
  if k%N==0:assert total is None
  curve+=1
result={'status':'PASS','all_table_addresses_and_ladder_indices':addresses,'full_curve_cases':curve,'normal_add_checks':normalsteps,'expected_final_exceptions':exceptions,'shared_widths':[64,128,256],'default_bitmap_bytes_per_slot':1048576,'default_slots':2,'scope':'Source-bound geometry, allocation/launch references and exact group model. No native compile, GPU perf, or proof of all inherited approximate raw field operations.'}
(w/'audit-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
