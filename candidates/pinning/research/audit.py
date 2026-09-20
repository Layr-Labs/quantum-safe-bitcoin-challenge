from pathlib import Path
import re,sys,json,random,hashlib
from ptx_field_model import Program,function,extract_ptx,check_semantics
W=Path(__file__).resolve().parent;C=W/'candidate/candidates/pinning'
if not C.exists():C=W.parent
import curve_ref as g
P=g.P;N=g.N;B=1<<256;mask=(1<<64)-1
cu=(C/'pinning.cu').read_text();h=(C/'HybridPair.cuh').read_text();f=(C/'HybridField.cuh').read_text();math=(C/'GPUMath.h').read_text()
check_semantics()
def prog(body):return Program(extract_ptx(body).replace('.reg .pred take;',''))
M=prog(function(cu,'void qsb_field_mul('));S=prog(function(f,'void hy_square('));SUB=prog(function(f,'void hy_sub('))
limbs=lambda x:[(x>>(64*i))&mask for i in range(4)]
join=lambda x:sum(v<<(64*i) for i,v in enumerate(x))
def mul(a,b,ptx=False):return join(M.run(limbs(a)+limbs(b)))%P if ptx else a*b%P
def sq(a,ptx=False):return join(S.run(limbs(a)))%P if ptx else a*a%P
def sub(a,b,ptx=False):return join(SUB.run(limbs(a)+limbs(b))) if ptx else (a-b)%P
rng=random.Random(1004)
edges=[0,1,2,(1<<32)+976,(1<<32)+977,P-2,P-1,P,P+1,B-2,B-1]
for a in edges+[rng.randrange(B) for _ in range(512)]:
 assert sq(a,True)==a*a%P
 for b in edges[:3]+edges[-3:]:assert mul(a,b,True)==a*b%P
for a in edges[:7]+[rng.randrange(P) for _ in range(512)]:
 for b in edges[:7]:assert sub(a,b,True)==(a-b)%P
# Carry-complete canonical add model mirrors the actual two four-limb loops.
def add_limb(a,b):
 v=[];carry=0
 for aa,bb in zip(limbs(a),limbs(b)):
  t=aa+bb+carry;v.append(t&mask);carry=t>>64
 k=carry*((1<<32)+977);carry=0;out=[]
 for j,x in enumerate(v):
  t=x+(k if j==0 else 0)+carry;out.append(t&mask);carry=t>>64
 assert carry==0
 return join(out)%P
for _ in range(4096):
 a=rng.randrange(P);b=rng.randrange(P);assert add_limb(a,b)==(a+b)%P
for a in edges[:7]:
 for b in edges[:7]:assert add_limb(a,b)==(a+b)%P
assert 'for(int j=0;j<4;j++)' in function(h,'void hy_add(')
# Parse the production call sequences, not copies of the pair/point formulas.
def pp(s):
 active=[True];out=[]
 for line in s.splitlines():
  if line.startswith('#if '):active.append(active[-1] and line.strip().split()[-1] in ['QSB_LAZY','QSB_FUSE_SQRADDSUB2'])
  elif line.startswith('#else'):active[-1]=active[-2] and not active[-1]
  elif line.startswith('#endif'):active.pop()
  elif active[-1]:out.append(line)
 return '\n'.join(out)
def body(source,sig):
 b=function(source,sig);b=b[b.index('{')+1:-1];return re.sub(r'//[^\n]*|/\*.*?\*/','',b,flags=re.S)
def choose(s,condition,yes):
 pattern=r'if\s*\('+re.escape(condition)+r'\)\s*\{([^{}]*)\}\s*else\s*\{([^{}]*)\}'
 s,n=re.subn(pattern,lambda m:m[1] if yes else m[2],s);assert n==1,(condition,n);return s
pair=body(h,'void hy_affine_pair(')
seed=body(math,'void _PointAddXYZZ_mm(')
# Skip forward declaration before seeking definition.
match=list(re.finditer(r'void _PointAddXYZZT\([^;{]*\)\s*\{',math))[-1]
madd=body(math[match.start():],'void _PointAddXYZZT(')
madd=choose(pp(madd),'DEFER_Y',True)
double=body(h,'void hy_double(')
allowed={'hy_load','hy_mul','hy_sub','hy_add','hy_square','_ModInv','Load256','_ModSub256','_ModAdd256','_ModAddLazy','_ModMult','_ModSqr','_ModSqrAddSub2'}
def calls(s):
 s=re.sub(r'\(uint64_t\s*\*\)','',s)
 s=re.sub(r'uint64_t\s+[^;]+;','',s)
 s=s.replace('inv[4]=0;','')
 out=[]
 for statement in s.split(';'):
  st=statement.strip()
  if not st:continue
  m=re.fullmatch(r'(\w+)\(([^()]*)\)',st);assert m,st
  name,args=m.groups();assert name in allowed,name;out.append((name,[a.strip() for a in args.split(',')]))
 return out
seedcalls=calls(seed);maddcalls=calls(madd);doublecalls=calls(double)
assert len(seedcalls)==12 and len(maddcalls)==15,(len(seedcalls),len(maddcalls))
def run(cs,state,ptx=False,table=None):
 for name,args in cs:
  out=args[0]
  if name=='hy_load':
   idx=eval(args[1],{},state);x,y=table[idx];state[args[3]]=x;state[args[4]]=y;continue
  vals=[state[x] for x in args[1:]]
  if name in ['hy_mul','_ModMult']:
   if len(vals)==1:vals=[state[out],vals[0]]
   v=mul(*vals,ptx=ptx) if name=='hy_mul' else vals[0]*vals[1]%P
  elif name in ['hy_sub','_ModSub256']:v=sub(*vals,ptx=ptx) if name=='hy_sub' else (vals[0]-vals[1])%P
  elif name in ['hy_add','_ModAdd256','_ModAddLazy']:v=add_limb(*vals)
  elif name in ['hy_square','_ModSqr']:v=sq(vals[0],ptx) if name=='hy_square' else vals[0]*vals[0]%P
  elif name=='_ModSqrAddSub2':v=(vals[0]*vals[0]+vals[1]-2*vals[2])%P
  elif name=='Load256':v=vals[0]
  elif name=='_ModInv':v=pow(state[out],-1,P)
  else:raise AssertionError(name)
  state[out]=v
 return state
# Direct digit extraction bound to actual source constants and sign handling.
codebody=function(h,'uint32_t hy_code(')
for text in ['17u*c+2u','bits=c==0?18u:17u','c==14?-negative','mask<0']:
 assert text in codebody,text
def digits(k):
 k%=N;D=2*k-N;raw=D%B;negative=int(D<0);terms=[]
 for c in range(15):
  bits=18 if c==0 else 17;pos=1 if c==0 else 17*c+2
  ff=(raw>>pos)&((1<<bits)-1)
  tm=-negative if c==14 else (ff>>(bits-1))-1
  idx=(ff^tm)&((1<<(bits-1))-1);sgn=-1 if tm<0 else 1
  terms.append((sgn*(2*idx+1))<<(0 if c==0 else 17*c+1))
 assert sum(terms)==D
 return terms
def exception(k):
 v=limbs(k)
 n=limbs(N)
 if v[3]==n[3] and (v[2]>n[2] or(v[2]==n[2] and(v[1]>n[1] or(v[1]==n[1] and v[0]>=n[0])))):
  k-=N;v=limbs(k)
 if (v[1]|v[2]|v[3])==0:
  if v[0]==0:return 1
  if v[0]==0x4D0364141:return 2
 if v==[0xbfd25e8800000000,0xbaaedce6af48a03b,0xfffffffffffffffe,0xffffffffffffffff]:return 2
 return 0
for x in ['0x4D0364141ULL','0xbfd25e8800000000ULL','0xbaaedce6af48a03bULL']:
 assert x in function(h,'unsigned hy_exception(')
ks=[0,1,2,3,N-1,N,N+1,B-1,0x4d0364141,N-0x4d0364141,N+0x4d0364141]+[rng.randrange(B) for _ in range(85)]
for k in ks+[rng.randrange(B) for _ in range(10000)]:
 digits(k);assert exception(k)==(1 if k%N==0 else 2 if k%N in [0x4d0364141,N-0x4d0364141] else 0)

def point(k):return g.to_affine(g.jac_mul(k))
def evaluate(k,ptx=False):
 table=[point(t*pow(2,-1,N)%N) for t in digits(k)]
 ds=[(table[2*j+1][0]-table[2*j][0])%P for j in range(7)]
 T=1
 for d in ds:assert d;T=T*d%P
 r=pow(T,-1,P);prefix=[ds[0]]
 for d in ds[1:6]:prefix.append(mul(prefix[-1],d,ptx))
 def pair_at(j,rr):
  state={'j':j,'negative':0,'r':rr,'prefix':prefix[j-1] if j else 0}
  state=run(calls(choose(pair,'j',bool(j))),state,ptx,table)
  # Compare every paired point independently to Jacobian addition.
  expected=g.to_affine(g.jac_add((*table[2*j],1),(*table[2*j+1],1)))
  assert (state['x'],state['y'])==expected
  return state['x'],state['y'],state['r']
 x,y,r=pair_at(6,r);carry=table[14]
 st=run(seedcalls,{'X1':carry[0],'Y1':carry[1],'X2':x,'Y2':y})
 X,Y,U,V=st['X3'],st['Y3'],st['ZZ3'],st['ZZZ3'];anchor=carry[1];ex=exception(k)
 for j in range(5,-1,-1):
  x,y,r=pair_at(j,r)
  if j==0 and ex:
   if ex==2:
    st=run(doublecalls,{'x':x,'y':y},ptx);X,Y=st['x'],st['y'];U=V=1
   else:X=Y=U=V=0
  else:
   st=run(maddcalls,{'X1':X,'Y1':Y,'ZZ1':U,'ZZZ1':V,'X2':x,'Y2':y,'Yoff':anchor})
   X,Y,U,V=st['X1'],st['Y1'],st['ZZ1'],st['ZZZ1'];anchor=y
 if not ex:Y=(Y-anchor*V)%P
 got=None if U==0 or V==0 else (X*pow(U,-1,P)%P,Y*pow(V,-1,P)%P)
 assert got==point(k),(hex(k),got,point(k))
for k in ks:evaluate(k)
for k in ks[:11]+ks[-5:]:evaluate(k,True)
# Source-bound layout/launch/freeze checks.
for name,digest in {'GPUMath.h': '68717e37488c611bf8b9742eeb139864550fe21ec40daee414c379d9bbde1054', 'GPUHash.h': '8cf9b303b6f5a09051e433a8bc21e2f3a66631e3bd22b2b1a10e71fe0b21b2bc', 'PackedRecovery.cuh': '932941bb0d3d9f9a08397a74bb243e9e99e0f8902767e286fc769db4562d2413', 'LeafRecovery.cuh': 'a654bab8ecbac83b99a8c2c5b6cfea6b5f8d8163cf384bd50a8c3da67451d9e8', 'cofactor_checkpoint.h': '3377fd660fb8039a34f661fc66bac9e58000f12339db6c43a99a076f90de4961'}.items():
 assert hashlib.sha256((C/name).read_bytes()).hexdigest()==digest
assert 'storage[24*QSB_TREE_N]' in cu
assert 'qsb_field_normalize(rx);qsb_field_normalize(ry);' in cu
assert cu.count('qsb_root_group_prepare<<<')==2 and cu.count('qsb_root_group_finish<<<')==2
assert cu.count('hy_pair_prepare<<<')==1
assert 'if(j<5)hy_put(j,prefix)' in h and 'hy_put(5,M)' in h
assert 'hy_get((unsigned)j-1,prefix)' in h and 'for(int j=5;j>=0;j--)' in h
assert 'qsb_packed_prepare(D,U,Y,V,usable,active,n,saved,roots)' in h
assert 'if(active && exception==1){qsb_st_v2(saved+i,1,0);}' in h
assert 'hy_identity=true;qy[0]=0;' in cu
assert 'if(hy_identity)y_parities=' in cu
for b in [1,2,P-2,P-1]+[rng.randrange(1,P) for _ in range(128)]:
 assert ((b&1)|(((b&1)^1)<<1))==((b&1)|(((-b)%P&1)<<1))
assert (C/'HybridCofactor.cuh').read_text()==(C/'cofactor_checkpoint.h').read_text().replace('qsb_cofactor_prepare','hy_cofactor_prepare').replace('qsb_field_mul_sc','qsb_field_mul')
assert '_ModInv' not in function(h,'void hy_checkpoint(')
result={'status':'PASS','source_field_programs':'new square,subtraction and unchanged full multiply actual PTX semantics audited','square_PTX_cases':523,'multiply_PTX_cases':3138,'sub_PTX_cases':3633,'canonical_add_cases':4145,'raw_scalar_digit_exception_cases':10096,'source_call_full_scalar_cases':len(ks),'source_call_PTX_pair_scalar_cases':16,'shared_arena_bytes':24*128*8,'kernels_added':4,'source_SHA256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in C.iterdir() if x.is_file() and x.suffix in ['.cu','.cuh','.h']},'limits':'No native compile/GPU benchmark. Inherited mixed-add/seed arithmetic modeled under field contract, not certified exact; GPU scheduling/registers not modeled.'}
(W/'audit-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
