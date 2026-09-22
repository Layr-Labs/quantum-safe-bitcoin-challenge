#!/usr/bin/env python3
"""Execute the actual emitted integer PTX in Python; never compiles CUDA."""
from pathlib import Path
import runpy,re,random,json,hashlib
r=Path(__file__).resolve().parent;pr=r/'rowmac_research'
m=runpy.run_path(str(pr/'ptx_model.py'));m['check_semantics']()
def flatten(ptx):
 ptx=re.sub(r'/\*.*?\*/','',ptx,flags=re.S);packs=[]
 ptx=re.sub(r'\{[^{};]+,[^{};]+\}',lambda x:(packs.append(x[0]) or f'PACK{len(packs)-1}'),ptx)
 ptx=ptx.replace('{','').replace('}','')
 return '{\n'+re.sub(r'PACK(\d+)',lambda x:packs[int(x[1])],ptx).strip()+'\n}'
header=(r/'RowMac256.cuh').read_text()
ptx=m['extract_ptx'](m['function'](header,'void qsb_rowmac256_raw('))
new=m['Program'](flatten(ptx));old=m['Program'](flatten((pr/'baseline_raw.ptx').read_text()))
product=(pr/'product.ptx').read_text();prod=m['Program'](product);ref=m['Program']((pr/'baseline_product.ptx').read_text())
no_row_carry=m['Program'](product.replace('mov.b64 odd_t,{cy,0};','mov.b64 odd_t,{0,0};'))
no_merge_carry=m['Program'](product.replace('addc.cc.u32','add.cc.u32').replace('addc.u32','add.u32'))
q=m['Program']('''{.reg .u64 t,seed;
mov.b64 seed,{4294967295,0};mad.wide.u32 t,4294967295,4294967295,seed;
mov.b64 %0,t;}''')
assert q.run([],outputs=1)==[(1<<64)-(1<<32)]
B=1<<256;R=1<<128;W=(1<<64)-1;K=(1<<32)+977
edges=[0,1,2,(1<<32)-1,W,R-1,R,R+1,B-K-1,B-K,B-K+1,B-2,B-1]
rows=[(a,b) for a in edges for b in edges]
rng=random.Random(23054142);rows.extend((rng.getrandbits(256),rng.getrandbits(256)) for _ in range(1024))
negative_row=negative_merge=0
for a,b in rows:
 ins=[(v>>(64*i))&W for v in [a,b] for i in range(4)]
 def integer(p):return sum(v<<(64*i) for i,v in enumerate(p.run(ins,input_base=8,outputs=8)))
 assert integer(prod)==integer(ref)==a*b
 assert new.run(ins)==old.run(ins)
 negative_row+=integer(no_row_carry)!=a*b;negative_merge+=integer(no_merge_carry)!=a*b
assert negative_row>0 and negative_merge>0
math=(r/'GPUMath.h').read_text();body=m['function'](math,'void _ModMultCore(')
assert '#if QSB_ROW_MAC256 && QSB_C31\n    qsb_rowmac256_raw(r,a,b);\n#else' in body
assert '#define QSB_ROW_MAC256 1' in math
assert flatten(ptx).split('.reg .u64 r0,r1,r2,r3,h0')[1]==flatten((pr/'baseline_raw.ptx').read_text()).split('.reg .u64 r0,r1,r2,r3,h0')[1]
assert not any(guard for guard in new.guards)
result={'passed':True,'source_tied_full_product_and_raw_pairs':len(rows),'tight_row_MAC_boundary':True,
 'removed_row_carry_failures':negative_row,'removed_merge_carry_failures':negative_merge,
 'header_sha256':hashlib.sha256(header.encode()).hexdigest(),'native_compilation':False,'GPU_execution':False}
print(json.dumps(result,indent=2))
