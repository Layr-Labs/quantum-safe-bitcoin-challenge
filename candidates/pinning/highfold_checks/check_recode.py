import random,json,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent
M=(1<<64)-1;Z=1<<30; widths=[25]*3+[26]*6
pos=[];p=0
for w in widths:pos.append(p);p+=w
counts=[(1<<25)+1]+[1<<(w-1) for w in widths]
def literal(z,bad=False):
 q=[(z>>(64*i))&M for i in range(4)];carry=0;codes=[];zero=0
 for j,(w,p) in enumerate(zip(widths,pos),1):
  wi=p>>6;sh=p&63;v=q[wi]>>sh
  if sh+w>64 and wi<3:v|=(q[wi+1]<<(64-sh))&M
  v=(v&((1<<w)-1))+carry;neg=int(v>(1<<(w-1)))
  if bad and j==9:neg=0
  m=(1<<w)-v if neg else v;carry=neg
  c=((m-1)&0xffffffff)|(neg<<31)
  if m==0:c=Z;zero|=1<<j
  codes.append(c)
 return [(q[3]>>39)+carry]+codes,zero
def decode(cs):
 return (cs[0]<<231)+sum((0 if c==Z else (-1 if c>>31 else 1)*((c&0x3fffffff)+1))<<p for c,p in zip(cs[1:],pos))
r=random.Random(27444);vals=[0,1,(1<<256)-1]
for p,w in zip(pos,widths):
 for v in [0,1,(1<<(w-1))-1,1<<(w-1),(1<<(w-1))+1,(1<<w)-1]:
  vals.extend([v<<p,((1<<256)-1)^(v<<p)])
vals += [r.getrandbits(256) for _ in range(50000)]
neg=0;extra=0;zero=0
for z in vals:
 cs,zm=literal(z);assert decode(cs)==z;assert cs[0]<counts[0]
 for j,c in enumerate(cs[1:],1):
  if c!=Z:assert (c&0x3fffffff)<counts[j]
 extra+=cs[0]==(1<<25);zero+=bool(zm);neg+=any(c!=Z and (c&0x3fffffff)>=counts[j] for j,c in enumerate(literal(z,True)[0][1:],1))
# SIMD lane operations are 64-bit masked shifts/add/sub, signed compare: all
# compared values <= 2^26. Packing permute indices 0,2,4,6 must select low32.
for i in range(0,len(vals)-3,4):
 lanes=[literal(z)[0] for z in vals[i:i+4]]
 for j in range(10):
  words=sum(([a[j]&0xffffffff,0] for a in lanes),[])
  assert [words[k] for k in [0,2,4,6]]==[a[j] for a in lanes]
# New tail: count 2^25+1, final segment at 2^25 has exactly one entry.
seg=4096;cnt=counts[0];last=((cnt+seg-1)//seg)-1
assert cnt-last*seg==1
s=(D.parent/cpu_cogrind.h').read_text();t=(D.parent/cg_table.h').read_text()
for a in ['j < k || L.pos[0] != 0','j == k && L.pos[0] == 0','q[3] >> 39','_mm256_srli_epi64(q[3],39)','j == L.nwin - 1 && L.pos[0] == 0']:assert a in s
assert 'row_count > 1 && batch_add' in t
assert neg>0 and extra>0
result=dict(cases=len(vals),negative_last_signed_disabled_out_of_bounds=neg,extra_top_record=extra,zero_cases=zero,bytes=sum(counts)*64,tail_records=1,sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (D.parent').glob('*.h')},scope='Python word and packing model plus source anchors; no native compilation or timing')
(D/'recode-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
