"""Pure Python, source-bound IFMA limb interpreter and integration invariants. No native compilation."""
from pathlib import Path
import re,random,json,hashlib,sys,subprocess
r=Path(__file__).resolve().parent;p=r.parent
field=(p/'cpu_ifma_field.h').read_text();P=2**256-2**32-977;M=2**52-1;U=2**64-1
class F:
 def __init__(self,v=0):self.l=[(v>>(52*k))&M for k in range(5)]
def val(a):return sum(x<<(52*k) for k,x in enumerate(a.l))
def body(name,source=field):
 pos=source.index('void '+name+'(');start=source.index('{',pos);n=1;i=start+1
 while n:n+=(source[i]=='{')-(source[i]=='}');i+=1
 return source[start+1:i-1]
def split(s):
 out=[];n=0;j=0
 for i,c in enumerate(s):
  n+=(c=='(')-(c==')')
  if c==',' and not n:out.append(s[j:i].strip());j=i+1
 return out+[s[j:].strip()]
def num(x):return re.sub(r'(?i)(0x[0-9a-f]+|\b\d+)(?:ull|ll|u)\b',r'\1',x)
ops={
 '_mm512_setzero_si512':lambda:0,'_mm512_set1_epi64':lambda x:x&U,
 '_mm512_add_epi64':lambda a,b:(a+b)&U,'_mm512_sub_epi64':lambda a,b:(a-b)&U,
 '_mm512_and_si512':lambda a,b:a&b,'_mm512_or_si512':lambda a,b:a|b,
 '_mm512_srli_epi64':lambda a,b:a>>b,'_mm512_slli_epi64':lambda a,b:(a<<b)&U,
 '_mm512_madd52lo_epu64':lambda c,a,b:(c+((a&M)*(b&M)&M))&U,
 '_mm512_madd52hi_epu64':lambda c,a,b:(c+(((a&M)*(b&M))>>52))&U,
}
funcs=['fe8_carry','fe8_red','fe8_mul','fe8_sqr','fe8_sub','fe8_carry_m','fe8_sub_m','fe8_mul_sub','fe8_sqr_sub3']
signatures={
'fe8_carry':['r'],'fe8_red':['r']+['c'+str(i) for i in range(10)],'fe8_mul':['r','a','b'],'fe8_sqr':['r','a'],
'fe8_sub':['r','a','b'],'fe8_carry_m':['r'],'fe8_sub_m':['r','a','b'],'fe8_mul_sub':['r','a','b','s'],'fe8_sqr_sub3':['r','a','d','x']}
programs={}
for name in funcs:
 s=body(name);s=re.sub(r'/\*.*?\*/','',s,flags=re.S);s=re.sub(r'^\s*#.*$','',s,flags=re.M)
 programs[name]=[t.strip() for t in s.split(';') if t.strip()]
def run(name,*args):
 e=dict(zip(signatures[name],args));e.update(ops);e['F8_M52']=M
 for f in funcs:e[f]=lambda *a,f=f:run(f,*a)
 for t in programs[name]:
  dec=re.match(r'(?:const )?__m512i\s+(.*)',t,re.S)
  if dec: parts=split(dec.group(1))
  else:parts=[t]
  for line in parts:
   m=re.fullmatch(r'(LO|HI)\((.*)\)',line,re.S)
   if m:
    v,a,b=split(m[2]);fun=ops['_mm512_madd52'+('lo' if m[1]=='LO' else 'hi')+'_epu64'];e[v]=fun(e[v],eval(num(a),{'__builtins__':{}},e),eval(num(b),{'__builtins__':{}},e));continue
   if '=' in line:
    lhs,expr=line.split('=',1);lhs=lhs.strip();v=eval(num(expr),{'__builtins__':{}},e)
    m=re.fullmatch(r'(\w+)\.l\[(\d+)\]',lhs)
    if m:e[m[1]].l[int(m[2])]=v
    else:e[lhs]=v
   elif '(' in line:eval(num(line),{'__builtins__':{}},e)
   else:e[line]=0
 return args[0]
random.seed(260926)
checks=0
edge=[0,1,2,P-1,P,P+1,2**256-1,2**256,2**257-1]
for i in range(2200):
 a=F(random.choice(edge) if i<200 else random.randrange(2**257));b=F(random.choice(edge) if i<200 else random.randrange(2**257));s=F(random.choice(edge) if i<200 else random.randrange(2**257))
 for fn,args,expected in [('fe8_mul',(a,b),val(a)*val(b)),('fe8_sqr',(a,),val(a)**2),('fe8_sub',(a,b),val(a)-val(b)),('fe8_sub_m',(a,b),val(a)-val(b)),('fe8_mul_sub',(a,b,s),val(a)*val(b)-val(s))]:
  o=run(fn,F(),*args);assert val(o)%P==expected%P,(fn,i);assert all(0<=v<2**52 for v in o.l),(fn,o.l);checks+=1
 d=run('fe8_sub',F(),b,a);o=run('fe8_sqr_sub3',F(),s,d,a);assert val(o)%P==(val(s)**2-val(d)-2*val(a))%P;assert all(0<=v<2**52 for v in o.l);checks+=1

print({"source_bound_ifma_limb_checks":checks,"native_execution":False})
