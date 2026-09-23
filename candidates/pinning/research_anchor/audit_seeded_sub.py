#!/usr/bin/env python3
"""Independent interpreter of the proposed positive-bias subtract MAC.

Extracts actual inline PTX, proves the integer product in directed/random
fixtures, and accounts for inherited reduction and final-short-borrow errors.
No CUDA execution, gate hit recall, or performance claim is made.
Interpreter structure follows the prior local test_seeded_mac.py audit.
"""
import ast,collections,hashlib,itertools,json,random,re,sys
from pathlib import Path
B=1<<256; K=(1<<32)+977; P=B-K
M32=(1<<32)-1;M64=(1<<64)-1;M96=(1<<96)-1
ROOT=Path(__file__).resolve().parent
source_path=(Path(sys.argv[1]) if len(sys.argv)>1 else
 ROOT.parent/'research_arithmetic/subtract_seed_foldbias.cuh')
source=source_path.read_text()
sub=source[source.index('void qsb_mulsub_seed('):]
MODE=('foldbias' if 'sub.u32 z8,z8,1;' in sub else
      'signed' if 'subc.u64 bias_carry, 0, 0;' in sub else 'bias')
asm=re.search(r'\basm\s*\((.*?)\n\s*:',sub,re.S).group(1)
ptx=''.join(ast.literal_eval(s) for s in re.findall(r'"(?:\\.|[^"\\])*"',asm))
marker='.reg .u64 r0,r1,r2,r3,h0,h1,h2,h3'
product_ptx,tail_ptx=ptx.split(marker,1)

def parse(code):
 code=re.sub(r'/\*.*?\*/','',code,flags=re.S)
 code=re.sub(r'\.reg[^;]*;','',code)
 code=re.sub(r'(?m)^\s*[{}]\s*$','',code)
 code=re.sub(r';\s*}',';',code)
 out=[]
 for line in code.split(';'):
  line=line.strip().lstrip('{').strip()
  if not line or line=='}':continue
  op,args=line.split(None,1)
  assert op in {'mov.b64','mov.u64','mov.u32','mul.wide.u32',
   'add.cc.u64','addc.cc.u64','addc.u64','add.cc.u32','addc.cc.u32','addc.u32',
   'sub.cc.u64','subc.cc.u64','subc.u64','sub.cc.u32','subc.cc.u32','subc.u32','sub.u32'},op
  out.append((op,re.findall(r'\{[^}]*\}|[^,]+',args.replace(' ',''))))
 return out

def execute(ops,initial,alias=None):
 regs=dict(initial);carry=0
 def name(x):return '%'+str(alias+int(x[1:])) if alias is not None and x in {'%0','%1','%2','%3'} else x
 def get(x):
  x=name(x)
  if x in regs:return regs[x]
  assert re.fullmatch(r'(?:0x[0-9a-fA-F]+|[0-9]+)',x),x
  return int(x,0)
 def put(x,v):regs[name(x)]=v
 for op,args in ops:
  if op=='mov.b64':
   if args[0].startswith('{'):
    lo,hi=args[0][1:-1].split(',');v=get(args[1]);put(lo,v&M32);put(hi,v>>32)
   elif args[1].startswith('{'):
    lo,hi=args[1][1:-1].split(',');put(args[0],get(lo)|(get(hi)<<32))
   else:put(args[0],get(args[1]))
  elif op.startswith('mov.'):put(args[0],get(args[1]))
  elif op=='mul.wide.u32':
   a,b=map(get,args[1:]);assert 0<=a<=M32 and 0<=b<=M32;put(args[0],a*b)
  else:
   bits=int(op.rsplit('u',1)[1]);a,b=map(get,args[1:]);cy=carry if op.startswith(('addc.','subc.')) else 0
   v=a-b-cy if op.startswith('sub') else a+b+cy
   if '.cc.' in op:carry=int(v<0) if op.startswith('sub') else int(v>=1<<bits)
   put(args[0],v&((1<<bits)-1))
 return regs

def inputs(a,b,c):return {'%'+str(base+i):(v>>(64*i))&M64 for base,v in ((4,a),(8,b),(12,c)) for i in range(4)}
def value(regs,base=0):return sum(regs['%'+str(base+i)]<<(64*i) for i in range(4))
def product(regs):return sum(regs['x'+str(i)]<<(32*i) for i in range(16))
def model(prod):
 x=[(prod>>(32*i))&M32 for i in range(16)]
 even=sum(x[8+2*i]<<(64*i) for i in range(4));odd=sum(x[9+2*i]<<(64*i) for i in range(4))
 f=(prod&(B-1))+977*even;g=(prod>>256)+977*odd
 g_lost=g>>256;first=f+((g&(B-1))<<32);z_lost=first>>288;first&=(1<<288)-1
 lo=first&(B-1)
 first_borrow=0
 if MODE=='foldbias':
  first_borrow=int((first>>256)==0)
  first=lo+((((first>>256)-1)&M32)<<256)
 second=(lo&M96)+K*(first>>256);tail_lost=second>>96
 raw=(lo&~M96)|(second&M96)
 lost=(g_lost+z_lost)*(1<<288)+tail_lost*(1<<96)
 if MODE=='foldbias':
  assert (raw-(prod-B)+lost-first_borrow*(1<<288))%P==0
  return raw,(g_lost,z_lost,tail_lost,first_borrow)
 assert (prod-raw-lost)%P==0
 borrow=int((raw&M64)<K)
 out=(raw&~M64)|(((raw&M64)-K)&M64)
 assert (out-(prod-K)+lost-borrow*(1<<64))%P==0
 return out,(g_lost,z_lost,tail_lost,borrow)

pops=parse(product_ptx);tops=parse(marker+tail_ptx);allops=pops+tops
if MODE=='signed':
 for a,b,c in ((0,0,1),(1,1,2),(3,5,7)):
  regs=execute(pops,inputs(a,b,c));observed=product(regs);expected=(a*b-c)%(B*B)
  if observed!=expected:
   result={'source':str(source_path),'sha256':hashlib.sha256(source.encode()).hexdigest(),
    'rejected':True,'reason':'Signed high seed borrow does not propagate beyond e4.',
    'a':a,'b':b,'c':c,'integer_product':hex(observed),'expected_mod_512':hex(expected),
    'gpu_execution':False}
   (ROOT/'seeded_sub_signed_result.json').write_text(json.dumps(result,indent=2)+'\n')
   print(json.dumps(result,indent=2));sys.exit(0)
 raise AssertionError('expected signed-seed counterexample missing')
edges=[0,1,2,976,977,K-1,K,K+1,(1<<32)-1,1<<32,(1<<64)-1,1<<64,
 (1<<96)-1,1<<96,(1<<128)-1,1<<128,B//2-1,B//2,P-1,P,P+1,B-2,B-1]
rng=random.Random(0x53554253454544)
cases=itertools.chain(itertools.product(edges,repeat=3),
 ((rng.getrandbits(256),rng.getrandbits(256),rng.getrandbits(256)) for _ in range(12000)))
counts=collections.Counter();examples=[]
for index,(a,b,c) in enumerate(cases):
 category='directed' if index<len(edges)**3 else 'random'
 initial=inputs(a,b,c);regs=execute(pops,initial);prod=product(regs)
 assert prod==a*b+B-c,('integer-product',index)
 assert 1<=prod<=B*B-B+1
 out=value(execute(tops,regs));wanted,loss=model(prod)
 assert out==wanted,('unexplained-field',index)
 counts['exact_product_cases']+=1;counts[category+'_cases']+=1
 if out%P!=(a*b-c)%P:
  counts[category+'_field_differences']+=1;assert any(loss)
  if len(examples)<4:examples.append({'a':hex(a),'b':hex(b),'c':hex(c),'losses':loss})
 if index%101==0:
  for alias in (4,8,12):
   assert value(execute(allops,initial,alias),alias)==out
   counts['aliases_checked']+=1

corrupt=parse(product_ptx.replace('subc.u64 bias_carry, 1, 0;','subc.u64 bias_carry, 0, 0;'))
assert product(execute(corrupt,inputs(3,5,7)))!=3*5+B-7
counts['negative_controls']=1
counts.setdefault('random_field_differences',0)
result={'source':str(source_path),'sha256':hashlib.sha256(source.encode()).hexdigest(),
 'checks':dict(counts),'unexplained_errors':0,'gpu_execution':False,'examples':examples,
 'conclusion':('512-bit biased product exact on corpus; all field differences accounted for by inherited tails or '+
               ('first-fold z8 decrement underflow.' if MODE=='foldbias' else 'final low64 borrow.'))}
(ROOT/('seeded_sub_'+MODE+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
