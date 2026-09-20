from pathlib import Path
import json,hashlib,re,random,math
import curve_ref as g
import source_calls as sc
from ptx_field_model import function,extract_ptx,Program,check_semantics
W=Path(__file__).resolve().parent;C=W.parent
cu=(C/'pinning.cu').read_text();mh=(C/'GPUMath.h').read_text();eh=(C/'WideExceptions.cuh').read_text();fh=(C/'WideField.cuh').read_text()
P,N=g.P,g.N;B=1<<256;mask=(1<<64)-1;delta=B-N;critical=(1<<236)-delta
widths=[19]*4+[20]*9;shifts=[sum(widths[:i]) for i in range(13)];entries=[1<<(w-1) for w in widths];offsets=[sum(entries[:i]) for i in range(13)]
assert sum(widths)==256 and sum(entries)*64==352*2**20
assert '#define GT_CHUNKS 13' in cu and '#define GT_HI 4096' in cu
for x in ['gt_shift(c)+1u','sh+bits>64u','bits=gt_width(c)','code&0x7ffffu','base+=gt_entries(c)']:assert x in cu,x
assert cu.count('1u<<gt_width(ch-1)')==2
assert '(int)((2u*gt_entries(ch))/GT_LO)-1' in cu
assert cu.count('qsb_root_group_prepare<<<')==1 and cu.count('qsb_root_group_finish<<<')==1
assert 'storage[12*QSB_TREE_N]' in cu
for c in range(13):
 for d in [0,1,entries[c]-1]:
  flat=offsets[c]+d;ch=flat>>18 if flat<(4<<18) else 4+((flat-(4<<18))>>19)
  assert ch==c and flat-offsets[ch]==d
  m=2*d+1;hi,lo=divmod(m,256)
  assert lo%2==1 and hi<4096 and hi<2*entries[c]//256
  assert (hi*256+lo)*(1<<shifts[c])==m*(1<<shifts[c])
# Entire partition: contiguous, no alias, GPUlaunch full256 entries perCTA.
for c in range(12):assert offsets[c]+entries[c]==offsets[c+1]
assert sum(entries)%256==0

def direct(k):
 k%=N;D=2*k-N;raw=D%B;neg=int(D<0);out=[]
 for c,(off,w) in enumerate(zip(shifts,widths)):
  pos=off+1;j,sh=divmod(pos,64);limb=(raw>>(64*j))&mask
  value=limb>>sh
  if j<3 and sh+w>64:value|=((raw>>(64*(j+1)))&mask)<<(64-sh)
  f=value&((1<<w)-1);tm=-neg if c==12 else (f>>(w-1))-1
  index=(f^tm)&((1<<(w-1))-1)
  assert index<entries[c]
  out.append((-1 if tm<0 else 1)*(2*index+1)*(1<<off))
 assert sum(out)==D
 return out

def recurrence(k):
 d=2*(k%N)-N;m=abs(d);sgn=1 if d>0 else -1;out=[]
 for off,w in zip(shifts[:-1],widths[:-1]):
  out.append(sgn*((m% (1<<(w+1)))-(1<<w))*(1<<off));m=2*(m>>(w+1))+1
 out.append(sgn*m*(1<<shifts[-1]));return out
# Guard constants are parsed directly from the actual function.
exbody=function(eh,'unsigned wide_exception(')
guards=re.findall(r'k\[0\]==0x([0-9A-F]+)ULL && k\[1\]==0x([0-9A-F]+)ULL && k\[2\]==0x([0-9A-F]+)ULL && k\[3\]==0x([0-9A-F]+)ULL',exbody)
assert len(guards)==2
guarded={sum(int(x,16)<<(64*j) for j,x in enumerate(a)) for a in guards}
assert guarded=={critical,N-critical}
def exception(k):return 1 if k%N==0 else 2 if k%N in guarded else 0
rng=random.Random(131920)
ks=[0,1,2,3,N//2,N//2+1,N-1,N,N+1,B-1,critical,N-critical]
ks += [x+d for x in [critical,N-critical,N] for d in [-2,-1,1,2] if 0<=x+d<B]
ks += [x for b in range(256) for x in [(1<<b)-1,1<<b,(1<<b)+1] if x<B]
ks += [rng.randrange(B) for _ in range(20000)]
exceptions=[]
for k in ks:
 terms=direct(k);assert terms==recurrence(k)
 current=terms[0]
 for j in range(1,13):
  equal=(current-terms[j])%N==0;opposite=(current+terms[j])%N==0
  if equal or opposite:
   assert j==12 and exception(k)
   assert (equal and exception(k)==2) or(opposite and exception(k)==1)
   exceptions.append((hex(k),j,'double' if equal else 'infinity'))
  current+=terms[j]
 assert current%N==2*k%N
# Source PTX primitives, newly used forbuilder normalization/rare doubles.
check_semantics()
prog=lambda src,sig:Program(extract_ptx(function(src,sig)).replace('.reg .pred take;',''))
M=prog(cu,'void qsb_field_mul(');S=prog(fh,'void wide_square(');SUB=prog(fh,'void wide_sub(')
limbs=lambda x:[(x>>(64*i))&mask for i in range(4)]
join=lambda x:sum(a<<(64*i) for i,a in enumerate(x))
def mul(a,b,ptx=False):return join(M.run(limbs(a)+limbs(b)))%P if ptx else a*b%P
def sq(a,ptx=False):return join(S.run(limbs(a)))%P if ptx else a*a%P
def sub(a,b,ptx=False):return join(SUB.run(limbs(a)+limbs(b))) if ptx else (a-b)%P
def add(a,b):
 carry=0;v=[]
 for aa,bb in zip(limbs(a),limbs(b)):
  t=aa+bb+carry;v.append(t&mask);carry=t>>64
 k=carry*((1<<32)+977);carry=0;out=[]
 for j,x in enumerate(v):t=x+(k if j==0 else 0)+carry;out.append(t&mask);carry=t>>64
 assert carry==0
 return join(out)%P
sc.P=P;sc.mul=mul;sc.sq=sq;sc.sub=sub;sc.add_limb=add
seed=sc.calls(sc.body(mh,'void _PointAddXYZZ_mm('))
match=list(re.finditer(r'void _PointAddXYZZT\([^;{]*\)\s*\{',mh))[-1]
madd=sc.calls(sc.choose(sc.pp(sc.body(mh[match.start():],'void _PointAddXYZZT(')),'DEFER_Y',True))
dbl=sc.calls(sc.body(eh,'void wide_double('));assert len(seed)==12 and len(madd)==15
edges=[0,1,2,P-2,P-1,P,P+1,B-1]
for a in edges+[rng.randrange(B) for _ in range(128)]:
 assert sq(a,True)==a*a%P
 for b in edges:assert mul(a,b,True)==a*b%P
for a in edges[:5]+[rng.randrange(P) for _ in range(128)]:
 for b in edges[:5]:assert sub(a,b,True)==(a-b)%P and add(a,b)==(a+b)%P
point=lambda k:g.to_affine(g.jac_mul(k%N))
pointcases=ks[:24]+[rng.randrange(B) for _ in range(72)]
for k in pointcases:
 pts=[point(t*pow(2,-1,N)%N) for t in direct(k)];ex=exception(k)
 if ex==1:got=None
 elif ex==2:
  st=sc.run(dbl,{'x':pts[-1][0],'y':pts[-1][1]},True);got=(st['x'],st['y'])
 else:
  st=sc.run(seed,{'X1':pts[0][0],'Y1':pts[0][1],'X2':pts[1][0],'Y2':pts[1][1]})
  X,Y,U,V=st['X3'],st['Y3'],st['ZZ3'],st['ZZZ3'];anchor=pts[0][1]
  for x,y in pts[2:]:
   st=sc.run(madd,{'X1':X,'Y1':Y,'ZZ1':U,'ZZZ1':V,'X2':x,'Y2':y,'Yoff':anchor})
   X,Y,U,V=st['X1'],st['Y1'],st['ZZ1'],st['ZZZ1'];anchor=y
  Y=(Y-anchor*V)%P;got=(X*pow(U,-1,P)%P,Y*pow(V,-1,P)%P)
 assert got==point(k),hex(k)
# Same infinity sentinel as priorauditedversion, +R/-R parities exact.
assert 'wide_identity=true;qy[0]=0;' in cu
assert 'wide_special==1)qsb_st_v2(saved+(size_t)idx,1,0)' in cu
for y in [1,2,P-2,P-1]+[rng.randrange(1,P) for _ in range(64)]:assert ((y&1)^1)==((-y)%P&1)
# Builder collective must precede any active-dependent return and normalizeas1/Z.
start=cu.index('kernel_build_gtable(');builder=function(cu[start:],'kernel_build_gtable(')
assert builder.index('qsb_block_inverse(pz);')<builder.index('if(!active)return;')
assert builder.count('return;')==1 and '_ModInv' not in builder
assert 'wide_mul(px,px,pz);wide_mul(py,py,pz);' in builder
assert 'bool active=t<GT_TOTAL_ENTRIES;' in builder
for name,digest in {'GPUMath.h':'68717e37488c611bf8b9742eeb139864550fe21ec40daee414c379d9bbde1054','GPUHash.h':'8cf9b303b6f5a09051e433a8bc21e2f3a66631e3bd22b2b1a10e71fe0b21b2bc','PackedRecovery.cuh':'932941bb0d3d9f9a08397a74bb243e9e99e0f8902767e286fc769db4562d2413','LeafRecovery.cuh':'a654bab8ecbac83b99a8c2c5b6cfea6b5f8d8163cf384bd50a8c3da67451d9e8','cofactor_checkpoint.h':'3377fd660fb8039a34f661fc66bac9e58000f12339db6c43a99a076f90de4961'}.items():assert hashlib.sha256((C/name).read_bytes()).hexdigest()==digest
result={'status':'PASS','scalar_cases':len(ks),'point_cases':len(pointcases),'complete_square_PTX_cases':136,'complete_product_PTX_cases':1088,'complete_sub_and_add_cases':665,'critical_scalar':hex(critical),'exception_hits':exceptions,'table_bytes':sum(entries)*64,'normal_chain_M_S':[81,24],'baseline_M_S':[95,28],'production_hashes':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in C.iterdir() if x.suffix in ['.cu','.cuh','.h']},'limits':'No nativecompile/GPUbenchmark; inheritedpointhelpers modeled as fieldcontracts, not certifiedfullcarry. No predicted score.'}
(W/'audit-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
