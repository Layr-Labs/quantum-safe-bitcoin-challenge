from pathlib import Path
import ast,re,random,json
from ptx_field_model import Program,function
W=Path(__file__).resolve().parent;s=(W.parent/'GLV608Chain.cuh').read_text()
f=function(s,'__device__ __forceinline__ void g14_sub(');a=f.index('asm(')+4;b=f.index('\n :',a)
t=''.join(ast.literal_eval(x) for x in re.findall(r'"(?:[^"\\]|\\.)*"',f[a:b]));pr=Program(t)
p=2**256-2**32-977;mask=2**64-1;r=random.Random(202609210725)
v=[0,1,p-1,p,2**256-1];pairs=[(a,b) for a in v for b in v]+[(r.randrange(2**256),r.randrange(2**256)) for _ in range(10000)]
for a,b in pairs:
 a%=p;b%=p;out=pr.run([(v>>(64*i))&mask for v in (a,b) for i in range(4)])
 assert sum(v<<(64*i) for i,v in enumerate(out))==(a-b)%p
result={'status':'PASS','actual_new_subtract_ptx_cases':len(pairs),'scope':'integer PTX semantic model, no CUDA compilation'}
(W/'new-sub-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
