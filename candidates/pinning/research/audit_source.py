from pathlib import Path
import re,json,hashlib,random
W=Path(__file__).resolve().parent;P=W.parent;B=W/'baseline'
def function(s,name):
 match=re.search(r'__device__[^\n]* void '+re.escape(name)+r'[^;{]*\)\s*\{',s);assert match,name
 a=s.index('{',match.start());depth=1;i=a+1
 while depth:
  depth+=(s[i]=='{')-(s[i]=='}');i+=1
 return s[a+1:i-1]
def pp(s,flags):
 active=True;stack=[];out=[]
 for line in s.splitlines():
  t=line.strip()
  if t.startswith('#if '):
   key=t[4:];value=flags[key[1:]]==0 if key.startswith('!') else bool(flags[key]);stack.append((active,value));active=active and value
  elif t.startswith('#elif '):
   parent,previous=stack[-1];value=(not previous) and bool(flags[t[6:]]);stack[-1]=(parent,previous or value);active=parent and value
  elif t=='#else':parent,previous=stack[-1];active=parent and not previous;stack[-1]=(parent,True)
  elif t=='#endif':parent,_=stack.pop();active=parent
  elif active:out.append(line)
 assert not stack
 return '\n'.join(out)
def branch(s,value):
 a=s.index('if (DEFER_Y)');lo=s.index('{',a);mid=s.index('} else {',lo);end=s.index('}',mid+8)
 return s[:a]+(s[lo+1:mid] if value else s[mid+8:end])+s[end+1:]
def trace(s,names):
 s=re.sub(r'/\*.*?\*/|//[^\n]*','',s,flags=re.S);state={n:('input',n) for n in names};events=[]
 for fn,inside in re.findall(r'(_Mod\w+|Load256)\(([^;]+)\)\s*;',s):
  args=[re.sub(r'\([^)]*\*\)','',x).strip() for x in inside.split(',')];out=args[0]
  if fn=='Load256':state[out]=state[args[1]];continue
  ins=args[1:] if len(args)>2 else [out,args[1]]
  if fn=='_ModSqr':ins=[args[1]]
  vals=tuple(state[x] for x in ins);event=(fn,vals);events.append(event);state[out]=hashlib.sha256(repr(event).encode()).hexdigest()
 return state,events
old=(B/'GPUMath.h').read_text();new=(P/'GPUMath.h').read_text();cases=0
for name,outs,inputs in [('_PointAddXYZZT(',['X1','Y1','ZZ1','ZZZ1'],['X1','Y1','ZZ1','ZZZ1','X2','Y2','Yoff']),('_PointAddXYZZ_mm(',['X3','Y3','ZZ3','ZZZ3'],['X1','Y1','X2','Y2'])]:
 a=function(old,name);b=function(new,name)
 for yoff in (0,1):
  for lazy in (0,1):
   for fuse in (0,1):
    for defer in (False,True):
     flags={'QSB_YOFF':yoff,'QSB_LAZY':lazy,'QSB_FUSE_SQRADDSUB2':fuse};aa=pp(a,flags);bb=pp(b,flags)
     if 'if (DEFER_Y)' in aa:aa=branch(aa,defer);bb=branch(bb,defer)
     sa,ta=trace(aa,inputs);sb,tb=trace(bb,inputs)
     assert ta==tb,(name,flags,defer,[(i,x,y) for i,(x,y) in enumerate(zip(ta,tb)) if x!=y],len(ta),len(tb));assert all(sa[n]==sb[n] for n in outs)
     cases+=1
 # Restore only these complete function bodies: everyprimitive/macro/otherfunction unchanged.
 new=new.replace(b,a,1)
assert new==old

for f in B.iterdir():
 if f.name not in ['pinning.cu','GPUMath.h','PackedRecovery.cuh','SUBMISSION.md','SOURCE-MANIFEST.json','RESEARCH.md']:
  assert (P/f.name).read_bytes()==f.read_bytes(),f.name
oldpack=(B/'PackedRecovery.cuh').read_text();newpack=(P/'PackedRecovery.cuh').read_text()
a=newpack.index('// Certified partial product;');b=newpack.index('// Combine the public cofactor traversal',a)
newpack=newpack[:a]+newpack[b:]
newpack=newpack.replace('const uint32_t parity0=qsb_packed_product_parity(l,s,b,1u);','qsb_packed_raw_mul(u,l,s);').replace('const uint32_t parity1=qsb_packed_product_parity(m,s,b,0u);','qsb_packed_raw_mul(v,m,s);').replace('return parity0|(parity1<<1);','return qsb_sum_parity(u,b,1u)|(qsb_sum_parity(v,b,0u)<<1);')
assert newpack==oldpack
result={'status':'PASS','point_symbolic_configs':cases,'raw_primitive_order_and_inputs_unchanged':True,'source_reversal':True,'field_operation_trace_preserved':True}
(W/'source-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

reversed_source=(P/'pinning.cu').read_text()
for old,new,count in reversed(json.loads((W/'integration-edits.json').read_text())):
 assert reversed_source.count(new)==count,(new,count)
 reversed_source=reversed_source.replace(new,old)
assert reversed_source==(B/'pinning.cu').read_text()
print('Serial16 source reversal PASS')
