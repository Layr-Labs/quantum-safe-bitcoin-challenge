#!/usr/bin/env python3
"""Pure Python execution of the actual emitted PTX, never a native build."""
from pathlib import Path
import runpy,re,random,json,hashlib
r=Path(__file__).resolve().parent;pr=r/'mul48_research'
m=runpy.run_path(str(pr/'ptx_model.py'));m['check_semantics']()
def flatten(ptx):
 ptx=re.sub(r'/\*.*?\*/','',ptx,flags=re.S);packs=[]
 ptx=re.sub(r'\{[^{};]+,[^{};]+\}',lambda x:(packs.append(x[0]) or f'PACK{len(packs)-1}'),ptx)
 ptx=ptx.replace('{','').replace('}','')
 return '{\n'+re.sub(r'PACK(\d+)',lambda x:packs[int(x[1])],ptx).strip()+'\n}'
header=(r/'PredicatedMul48.cuh').read_text()
ptx=m['extract_ptx'](m['function'](header,'void qsb_predicated_raw_mul48('))
new=m['Program'](flatten(ptx));old=m['Program'](flatten((pr/'baseline_raw.ptx').read_text()))
prod=m['Program']((pr/'product.ptx').read_text())
ref=m['Program']((pr/'baseline_product.ptx').read_text())
# The guard's entire carry chain uses one predicate. A disabled chain preserves CC.
for pa in range(2):
 for pb in range(2):
  q=m['Program'](('''{.reg .pred pa,pb; .reg .u32 a,b;
  setp.ne.u32 pa,PA,0;setp.ne.u32 pb,PB,0;mov.u32 a,0;
  sub.cc.u32 b,0,1;@pa add.cc.u32 b,0,0;
  @pb subc.u32 a,0,0;mov.b64 %0,{a,b};}''').replace('PA',str(pa)).replace('PB',str(pb)))
  assert q.run([],outputs=1)[0]==((0xffffffff if pb and not pa else 0)+((0 if pa else 0xffffffff)<<32))
B=1<<256;R=1<<128;W=(1<<64)-1;K=(1<<32)+977
edges=[0,1,2,(1<<32)-1,W,R-1,R,R+1,B-K-1,B-K,B-K+1,B-2,B-1]
rows=[(a,b) for a in edges for b in edges]
rng=random.Random(23032842);rows.extend((rng.getrandbits(256),rng.getrandbits(256)) for _ in range(1024))
mutant=m['Program']((pr/'product.ptx').read_text().replace('@pb subc.u32 d4,d4,ma32;', '@pb subc.u32 d4,d4,0;'))
negative=0
for a,b in rows:
 ins=[(v>>(64*i))&W for v in [a,b] for i in range(4)]
 def integer(p):return sum(v<<(64*i) for i,v in enumerate(p.run(ins,input_base=8,outputs=8)))
 assert integer(prod)==integer(ref)==a*b
 assert new.run(ins)==old.run(ins)
 negative+=integer(mutant)!=a*b
assert negative>0
math=(r/'GPUMath.h').read_text();body=m['function'](math,'void _ModMultCore(')
assert '#if QSB_PRED_MUL48 && QSB_C31\n    qsb_predicated_raw_mul48(r,a,b);\n#else' in body
assert '#define QSB_PRED_MUL48 1' in math
assert 'QSB_C31 && QSB_SHORT_CARRY' in header
# Same fold, not merely equal modulo p; inherited approximations are preserved.
assert flatten(ptx).split('.reg .u64 r0,r1,r2,r3,h0')[1]==flatten((pr/'baseline_raw.ptx').read_text()).split('.reg .u64 r0,r1,r2,r3,h0')[1]
result={'passed':True,'source_tied_full_product_and_raw_pairs':len(rows),'predicate_CC_cases':4,
 'missing_fused_sign_negative_control_failures':negative,'header_sha256':hashlib.sha256(header.encode()).hexdigest(),
 'native_compilation':False,'GPU_execution':False}
print(json.dumps(result,indent=2))
