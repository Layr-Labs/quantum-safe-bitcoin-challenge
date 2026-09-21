from pathlib import Path
import re,runpy,json,random,hashlib
w=Path(__file__).resolve().parent
q=runpy.run_path(str(w/'split_window.py'))
source=(w.parent/'Parity25.cuh').read_text()
body=source.split('uint32_t neg, uint32_t &out) {',1)[1].rsplit('}',1)[0]
def expr(s):
 s=re.sub(r'(?<=[0-9a-f])u\b','',s)
 return s.replace('uint64_t(', 'u64(').replace('uint32_t(', 'u32(').replace(' || ',' or ').replace(' && ',' and ')
py=['def model(a,b,offset,neg):']
for line in body.splitlines():
 s=line.strip()
 if not s or s.startswith('//'):continue
 if s.startswith('if ('):
  assert s.endswith(') return false;'),s
  py.append('    if '+expr(s[4:-15])+': return (False,None)');continue
 if s=='return true;':py.append('    return (True,out)');continue
 # Join the two-line final expression below after line processing.
 if s.startswith('out ='):
  pending=s;continue
 if 'pending' in locals():s=pending+' '+s;del pending
 decl=re.fullmatch(r'(?:const )?(uint32_t|uint64_t) (\w+) = (.*);',s)
 if decl:
  typ,var,rhs=decl.groups();py.append('    '+var+' = '+('u32' if typ=='uint32_t' else 'u64')+'('+expr(rhs)+')');continue
 aug=re.fullmatch(r'(\w+) (\+=|>>=) (.*);',s)
 if aug:
  var,op,rhs=aug.groups();py.append('    '+var+' = u64('+var+(' + (' if op=='+=' else ' >> (')+expr(rhs)+'))');continue
 assign=re.fullmatch(r'(carry|out) = (.*);',s)
 assert assign,s
 var,rhs=assign.groups();py.append('    '+var+' = '+('u32' if var=='out' else 'u64')+'('+expr(rhs)+')')
code='\n'.join(py)+'\n';(w/'header_semantics.py').write_text(code)
u32=lambda x:x&((1<<32)-1);u64=lambda x:x&((1<<64)-1)
ns={'u32':u32,'u64':u64,'__umulhi':lambda a,b:(a*b)>>32};exec(code,ns)
model=ns['model'];B=q['B'];P=q['P'];raw=q['q']['raw'];rng=random.Random(131925)
def limbs(v):return [u64(v>>(64*i)) for i in range(4)]
cases=list(q['q']['cases']);edges=q['q']['edges']
for a in edges:
 for b in edges:
  for top in [0,1,2,0xfffffffd,0xfffffffe,0xffffffff]:
   for low in [0,1,(1<<224)-1]:
    c=((top<<224)|low)%P
    for n in [0,1]:cases.append((a,b,c,n))
cases += [(rng.randrange(B),rng.randrange(B),rng.randrange(P),rng.randrange(2)) for _ in range(100000)]
fast=0;fallback=0
for a,b,c,n in cases:
 hit,bit=model(limbs(a),limbs(b),limbs(c),n)
 expected=(raw(a,b)+c)%P;expected=((-expected)%P if n else expected)&1
 if hit:assert bit==expected,(a,b,c,n,bit,expected);fast+=1
 else:fallback+=1
assert source.count('__umulhi(')==11
assert len(re.findall(r'uint64_t\(a[0-7]\) \* b[0-7]',source))==14
result={'status':'PASS','header_sha256':hashlib.sha256(source.encode()).hexdigest(),'source_translated_cases':len(cases),'fast':fast,'fallback':fallback,'product_counts':{'wide':14,'high':11,'bit_and':7,'constant_977':4},'scope':'Every executable header statement translated to Python with u32/u64 wrap, compared to promoted raw contract. Not CUDA compilation, ABI validation, or GPU performance evidence. Caller must use original full fallback.'}
(w/'header-audit-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
