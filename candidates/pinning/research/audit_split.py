from pathlib import Path
import sys,random,json,runpy,re
W=Path(__file__).resolve().parent
import reference as h
M=2**32-1;D=2**256-h.N;rng=random.Random(202609210618)
def words(n,L):return [(n>>(32*i))&M for i in range(L)]
def val(a):return sum(x<<(32*i) for i,x in enumerate(a))
def signed(a):return val(a)-(1<<(32*len(a))) if a[-1]>>31 else val(a)
def mul(a,b,O):
 out=[0]*O
 for i in range(len(a)):
  carry=0
  for j in range(min(len(b),O-i)):
   t=a[i]*b[j]+out[i+j]+carry
   assert 0<=t<2**64
   out[i+j]=t&M;carry=t>>32
  if i+len(b)<O:out[i+len(b)]=carry
 assert val(out)==val(a)*val(b)%(1<<(32*O))
 return out

def add(a,b):
 carry=0
 for i in range(len(a)):t=a[i]+b[i]+carry;a[i]=t&M;carry=t>>32

def sub(a,b):
 borrow=0
 for i in range(len(a)):t=b[i]+borrow;ai=a[i];a[i]=(ai-(t&M))&M;borrow=int(ai<t)

def inc(a,c):
 for i in range(len(a)):t=a[i]+c;a[i]=t&M;c=t>>32

def less(a,b):
 for i in range(len(a)-1,-1,-1):
  if a[i]!=b[i]:return a[i]<b[i]
 return False

def sless(a,b):
 sa,sb=a[-1]>>31,b[-1]>>31
 return sa>sb if sa!=sb else less(a,b)

hh=[0,0,0];corr=0

def coefficient(k,b):
 global corr
 t=mul(words(k,8),words(b,4),12);q=t[8:];r=mul(q,words(D,5),9)
 t[8]=0
 add(r,t[:9]);add(r,words(h.N//2,9));hi=r[8];r[8]=0
 assert 0<=hi<=2;hh[hi]+=1
 before=val(r);carry=0;dn=words(D,5)
 for i in range(9):
  d=dn[i] if i<5 else 0;v=hi*d+r[i]+carry;r[i]=v&M;carry=v>>32
 assert carry==0 and val(r)==before+hi*D and val(r)<2*h.N
 correction=int(not less(r,words(h.N,9)));corr+=correction
 inc(q,hi+correction)
 assert val(q)==(k*b+h.N//2)//h.N
 return q

branches={s:0 for s in ['0','-v2','-v1','+v2','+v1']}

def split(k):
 kk=words(k,8)
 if not less(kk,words(h.N,8)):sub(kk,words(h.N,8))
 k=val(kk);c1=coefficient(k,h.B2);c2=coefficient(k,h.B1)
 x=kk[:5];sub(x,mul(c1,words(h.A1,5),5));sub(x,mul(c2,words(h.A2,5),5))
 y=mul(c1,words(h.B1,5),5);sub(y,mul(c2,words(h.B2,5),5));which='0'
 if sless(words(h.R,5),x):
  if not sless(y,words(h.T,5)):sub(x,words(h.A2,5));sub(y,words(h.B2,5));which='-v2'
  else:sub(x,words(h.A1,5));add(y,words(h.B1,5));which='-v1'
 elif sless(x,words(-h.R,5)):
  if not sless(words(-h.T,5),y):add(x,words(h.A2,5));add(y,words(h.B2,5));which='+v2'
  else:add(x,words(h.A1,5));sub(y,words(h.B1,5));which='+v1'
 branches[which]+=1
 result=signed(x),signed(y),which;assert result==h.split_corr(k)
 return result

# Prove numeric bounds used by the single comparison, not only sample them.
for b in [h.B1,h.B2]:
 maxq=((h.N-1)*b)>>256
 rbound=(2**256-1)+maxq*D+h.N//2
 assert rbound<3*2**256
 assert (2**256-1)+2*D<2*h.N
 assert b<2**128
# Confirm generated CUDA constants, including signed-160 R and T, from the source.
s=(W.parent/'Split608.cuh').read_text()
for name,n,L in [('dn',D,5),('half',h.N//2,9),('a1',h.A1,5),('a2',h.A2,5),('b1',h.B1,5),('b2',h.B2,5),('r',h.R,5),('nr',-h.R,5),('t',h.T,5),('nt',-h.T,5)]:
 m=re.search(r'\b'+name+r'\['+str(L)+r'\]=\{([^}]+)\}',s);assert m,name
 assert [int(z.removesuffix('u'),16) for z in m[1].split(',')]==words(n,L)
# Near every rounding threshold selected adversarially via modular inverses.
ks=[0,1,2,h.N-1,h.N,h.N+1,2**256-1]
for i in range(256):ks.extend([2**i-1,2**i,min(2**256-1,2**i+1)])
for b in [h.B1,h.B2]:
 inv=pow(b,-1,h.N)
 for offset in range(-64,65):ks.append(((h.N//2+offset)*inv)%h.N)
ks+=[rng.randrange(2**256) for _ in range(20000)]
for k in ks:split(k)
assert all(v>0 for v in branches.values())
# End-to-end fixed-u32 split feeds independently audited fixed-u32 recoder.
ns=runpy.run_path(str(W/'audit_recode.py'))
for k in ks[:1033]+ks[-1000:]:x,y,_=split(k);ns['audit_pair'](x,y)
r={'status':'PASS','split_cases':len(ks),'coefficient_checks':2*(len(ks)+2033),'rounding_boundary_inputs':258,'split_correction_branches':branches,'quotient_high_histogram':hh,'final_quotient_corrections':corr,'end_to_end_fixed_word_cases':2033,'max_quotient_high_proved':2,'single_comparison_bound_proved':True,'cuda_constant_checks':10,'source_u32_multiply_count':'coefficient 32+20+5=57 each; split low160 products 14 each x4=56; total170 before compiler zero-constant folding','scope':'Exact u32 semantic mirror and bigint oracle. Integrated source; no local C++/CUDA compile or GPU timing.'}
(W/'split-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
