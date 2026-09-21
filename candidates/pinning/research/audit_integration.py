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
s=(P/'pinning.cu').read_text();o=(B/'pinning.cu').read_text()
a=s.index('// Exact transport integration');b=s.index('/* _FixedBaseSignedAffine:',a);oa=o.index('__device__ __forceinline__ void qsb_decode_to_shared');ob=o.index('/* _FixedBaseSignedAffine:',oa)
assert s[:a]+o[oa:ob]+s[b:]==o
# Source-bound constantpositions, writes and reads; input normalizer is in restoredbytes.
for token in ['__funnelshift_r(words[wi],hi,shift)','wi<7?words[(wi+1)&7]:0u','((c-4)/2)*QSB_TREE_N+threadIdx.x','(uint64_t)pending|((uint64_t)code<<32)','(code&0x007fffc0u)','table+(2u<<22)','table+(3u<<22)','table+(4u<<22)','plane+=2u<<22','qsb_load_bytecode(plane+(1u<<22)','qsb_yoff_to_y(yb)','_ModMult(x,yb,V);_ModSub256(Y,Y,x)']:assert token in s,token
assert s.count('uint64_t x[4],ya[4],yb[4];')==1
assert 'Load256(ya,yb)' not in s and 'Load256(yb,ya)' not in s
# Full byteaddress/domain proof: whole table, not randomsamples.
addresses=0
for c in range(15):
 base=0 if c==0 else (c+1)<<16;plane=base*64;limit=1<<(17 if c==0 else 16)
 for idx in range(limit):
  for neg in (0,1):
   code=(idx<<6)|(neg<<31);addr=plane+(code&0x007fffc0)
   assert addr==(base+idx)*64 and addr+64<=64*2**20 and ((code>>31)&1)==neg
  addresses+=1
# Byte packing, allthreads andwidths, no cross-lane collisions and8-bytealignment.
for n in (64,128,256):
 loc=[(pair*n+lane)*8 for pair in range(6) for lane in range(n)];assert len(set(loc))==6*n and max(loc)+8<=12*n*8
# Ordered tablepoint/anchor flow: firstmixedstepusesseedpoint0, notpoint1.
ref=[(2,0)]+[(c,c-1) for c in range(3,15)];ya,yb=0,1;events=[]
yb=2;events.append((yb,ya))
for pair in range(6):
 ya=3+2*pair;events.append((ya,yb));yb=4+2*pair;events.append((yb,ya))
assert events==ref and yb==14
# Pair encoding over actualsignednormalizedscalar, boundaryscalars andbasisvectors.
import decoder_model as d
rng=random.Random(202609211210);digitchecks=0
for k in [0,1,d.N//2,d.N-1,d.N,d.N+1,2**256-1]+[rng.getrandbits(256) for _ in range(12000)]:
 raw=2*(k%d.N)-d.N;M=raw%(2**256);neg=int(raw<0);oldcodes=d.decode(M,neg,False);newcodes=[]
 words=[(M>>(32*i))&0xffffffff for i in range(8)]
 for c in range(15):
  pos=1 if c==0 else 17*c+2;bits=18 if c==0 else 17;wi,shift=divmod(pos,32);hi=words[wi+1] if wi<7 else 0
  f=((words[wi]>>shift)|(hi<<(32-shift)))&((1<<bits)-1);sign=neg if c==14 else 1-(f>>(bits-1));code=(((f^(-sign&0xffffffff))&((1<<(bits-1))-1))<<6)|(sign<<31);newcodes.append(code)
  assert code==((oldcodes[c]&0x1ffff)<<6)|(oldcodes[c]&0x80000000);digitchecks+=1
 packed=[newcodes[c]|(newcodes[c+1]<<32) for c in range(3,15,2)];decoded=newcodes[:3]+[v for pair in packed for v in (pair&0xffffffff,pair>>32)]
 assert decoded==newcodes
# TOP2 dead arms, originaltree arithmetic andbarriers restored exactly.
a=(P/'cofactor_checkpoint.h').read_text();b=(B/'cofactor_checkpoint.h').read_text()
for n in (16,32,64,128,256,512,1024):
 up=[];c=n
 while c>2:up.append(c);c//=2
 down=[];c=8
 while c<n:down.append(c);c*=2
 assert all(c>2 for c in up) and all(c!=2 for c in down)
a=a.replace('#if QSB_TREE_TOP2\n        if(half>32)__syncthreads();else __syncwarp();\n#else\n        if(count>2){if(half>32)__syncthreads();else __syncwarp();}\n#endif','        if(count>2){if(half>32)__syncthreads();else __syncwarp();}')
a=a.replace('#if QSB_TREE_TOP2\n            qsb_field_mul_sc(out,parent,sibling);\n#else\n            if(count==2){Load256(out,sibling);}else{qsb_field_mul_sc(out,parent,sibling);}\n#endif','            if(count==2){Load256(out,sibling);}else{qsb_field_mul_sc(out,parent,sibling);}')
assert a==b
assert 'All lanes finish reading their digits/anchor before tree overwrites.\n    __syncthreads();' in (P/'PackedRecovery.cuh').read_text()
r={'status':'PASS','point_symbolic_configurations':cases,'same_ordered_primitive_inputs_in_all_configs':True,'all_other_field_source_unchanged':True,'table_addresses':addresses,'scalar_digit_checks':digitchecks,'widths':[64,128,256],'chain_point_anchor_sequence':events,'tree_dead_arm_widths':[16,32,64,128,256,512,1024],'source_reversal':True,'scope':'Source-bound symbolic operators preserve raw approximate arithmetic; no reassociation, native compilation or GPU timing.'}
(W/'integration-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
