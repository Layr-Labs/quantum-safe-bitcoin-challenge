#!/usr/bin/env python3
"""Compare actual approximate PTX multiplication associations, with no GPU."""
from pathlib import Path
import ast,collections,hashlib,itertools,json,random,re,sys
ROOT=Path(__file__).resolve().parent
source_path=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT.parent/'GPUMath.h'
source=source_path.read_text()
body=source[source.index('void _ModMultCore('):]
asm=re.search(r'\basm\s*\((.*?)\n\s*:',body,re.S).group(1)
# Default C31 expands QSB_SECOND_FOLD_TAIL to the empty string.
assert 'QSB_SECOND_FOLD_TAIL' in asm
ptx=''.join(ast.literal_eval(s) for s in re.findall(r'"(?:\\.|[^"\\])*"',asm))
scope={'__file__':str(ROOT/'audit_seeded_sub.py')}
audit_prefix=(ROOT/'audit_seeded_sub.py').read_text().split('pops=parse(product_ptx)')[0]
saved_argv=sys.argv
sys.argv=[str(ROOT/'audit_seeded_sub.py')]
try:exec(audit_prefix,scope)
finally:sys.argv=saved_argv
ops=scope['parse'](ptx);execute=scope['execute'];inputs=scope['inputs'];value=scope['value']
B,K,P=[scope[k] for k in ('B','K','P')]
def mul(a,b):return value(execute(ops,inputs(a,b,0)))
edge=[0,1,2,976,977,K-1,K,K+1,(1<<32)-1,1<<32,(1<<64)-1,1<<64,
 (1<<96)-1,1<<96,(1<<128)-1,1<<128,B//2-1,B//2,P-1,P,P+1,B-2,B-1]
rng=random.Random(0x4153534f43)
counts=collections.Counter();examples=[]
for index,(x,u,s) in enumerate(itertools.chain(itertools.product(edge,repeat=3),
 ((rng.getrandbits(256),rng.getrandbits(256),rng.getrandbits(256)) for _ in range(6000)))):
 kind='directed' if index<len(edge)**3 else 'random'
 a=mul(x,u);old=mul(a,s);b=mul(u,s);new=mul(x,b)
 errors=(a%P!=x*u%P,old%P!=a*s%P,b%P!=u*s%P,new%P!=x*b%P)
 counts[kind+'_cases']+=1
 if old!=new:counts[kind+'_raw_differences']+=1
 if old%P!=new%P:
  counts[kind+'_field_differences']+=1;assert any(errors)
  if len(examples)<2:examples.append({'x':hex(x),'u':hex(u),'s':hex(s),'primitive_errors':errors})
 if not any(errors):assert old%P==new%P==x*u*s%P
counts.setdefault('random_raw_differences',0);counts.setdefault('random_field_differences',0)
result={'source':str(source_path),'sha256':hashlib.sha256(source.encode()).hexdigest(),
 'checks':dict(counts),'unexplained_errors':0,'gpu_execution':False,'examples':examples,
 'scope':'Actual PTX product/reducer under default empty C31 tail; associations are not universally equal for this inherited approximate arithmetic.'}
(ROOT/'reassociation_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
