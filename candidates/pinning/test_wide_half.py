#!/usr/bin/env python3
"""Source-bound integer/PTX checks only; no native build or GPU execution."""
from pathlib import Path
import runpy,re,json,random,hashlib
r=Path(__file__).resolve().parent;m=runpy.run_path(str(r/'wide_half_research/ptx_model.py'));m['check_semantics']()
s=(r/'WideHalf256.cuh').read_text();raw=m['extract_ptx'](m['function'](s,'void qsb_widehalf256_raw(')).strip();old=(r/'wide_half_research/baseline_raw.ptx').read_text()
def flatten(p):
 p=re.sub(r'/\*.*?\*/','',p,flags=re.S);packs=[];p=re.sub(r'\{[^{};]+,[^{};]+\}',lambda x:(packs.append(x[0]) or f'PACK{len(packs)-1}'),p);p=p.replace('{','').replace('}','');return '{\n'+re.sub(r'PACK(\d+)',lambda x:packs[int(x[1])],p).strip()+'\n}'
old=flatten(old);cut='.reg .u64 r0,r1,r2,r3,h0';assert raw[raw.index(cut):]==old[old.index(cut):]
prefix=raw[:raw.index(cut)]+''.join(f'mov.b64 %{i},{{x{2*i},x{2*i+1}}};\n' for i in range(8))+'}'
p=m['Program'](prefix);rp=m['Program'](raw);op=m['Program'](old);bp=m['Program']((r/'wide_half_research/baseline_product.ptx').read_text())
assert sum(c=='mul.wide.u32' for c,_ in p.ops)==64
assert not any(c.startswith(('mad.','madc.')) for c,_ in p.ops)
assert sum(c.startswith('add') for c,_ in p.ops)==127
assert all(c.endswith('u32') for c,_ in p.ops if c.startswith('add'))
ms1=re.sub(r'addc.u32 (x\d+),0,0;',r'mov.u32 \1,0;',prefix);assert ms1!=prefix
ms2=re.sub(r'addc.cc.u32 (x\d+),(x\d+),(sp_h\d+);',r'add.cc.u32 \1,\2,\3;',prefix);assert ms2!=prefix
mut=[m['Program'](ms1),m['Program'](ms2)];fails=[0,0]
B=1<<256;mask=(1<<64)-1;K=(1<<32)+977
v=[0,1,2,(1<<32)-1,mask,(1<<96)-1,(1<<128)-1,B-K-1,B-K,B-1,B-2]
rows=[(a,b) for a in v for b in v];rng=random.Random(23075022);rows.extend((rng.getrandbits(256),rng.getrandbits(256)) for _ in range(1024))
for a,b in rows:
 ins=[(v>>(64*i))&mask for v in [a,b] for i in range(4)]
 expected=[(a*b>>(64*i))&mask for i in range(8)]
 assert p.run(ins,outputs=8)==expected==bp.run(ins,input_base=8,outputs=8)
 assert rp.run(ins)==op.run(ins)
 for k,q in enumerate(mut):fails[k]+=q.run(ins,outputs=8)!=expected
assert min(fails)>0
math=(r/'GPUMath.h').read_text();assert '#define QSB_WIDE_HALF256 1' in math and 'qsb_widehalf256_raw(r,a,b);' in math
cu=(r/'pinning.cu').read_text()
for part in ['#define QSB_BATCH 16777216','#define QSB_SLOTS 4 ','#define QSB_S2_BLOCKS 8 ']:assert part in cu
print(json.dumps({'passed':True,'source_bound_full512_and_raw_pairs':len(rows),'plain_wide_products':64,'complete_word_adds':127,'missing_low_endcarry_failures':fails[0],'missing_high_passcarry_failures':fails[1],'header_sha256':hashlib.sha256(s.encode()).hexdigest(),'native_compilation':False,'GPU_execution':False},indent=2))
