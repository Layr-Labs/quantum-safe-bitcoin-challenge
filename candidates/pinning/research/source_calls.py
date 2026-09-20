import re
from ptx_field_model import function
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
allowed={'wide_load','wide_mul','wide_sub','wide_add','wide_square','_ModInv','Load256','_ModSub256','_ModAdd256','_ModAddLazy','_ModMult','_ModSqr','_ModSqrAddSub2'}
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
def run(cs,state,ptx=False,table=None):
 for name,args in cs:
  out=args[0]
  if name=='wide_load':
   idx=eval(args[1],{},state);x,y=table[idx];state[args[3]]=x;state[args[4]]=y;continue
  vals=[state[x] for x in args[1:]]
  if name in ['wide_mul','_ModMult']:
   if len(vals)==1:vals=[state[out],vals[0]]
   v=mul(*vals,ptx=ptx) if name=='wide_mul' else vals[0]*vals[1]%P
  elif name in ['wide_sub','_ModSub256']:
   if len(vals)==1:vals=[state[out],vals[0]]
   v=sub(*vals,ptx=ptx) if name=='wide_sub' else (vals[0]-vals[1])%P
  elif name in ['wide_add','_ModAdd256','_ModAddLazy']:v=add_limb(*vals)
  elif name in ['wide_square','_ModSqr']:v=sq(vals[0],ptx) if name=='wide_square' else vals[0]*vals[0]%P
  elif name=='_ModSqrAddSub2':v=(vals[0]*vals[0]+vals[1]-2*vals[2])%P
  elif name=='Load256':v=vals[0]
  elif name=='_ModInv':v=pow(state[out],-1,P)
  else:raise AssertionError(name)
  state[out]=v
 return state
