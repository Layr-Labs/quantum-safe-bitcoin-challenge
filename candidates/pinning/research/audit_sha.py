"""Execute generated C assignment semantics in Python; no native compilation."""
from pathlib import Path
import re,random,hashlib,json,struct
C=Path(__file__).resolve().parent.parent
text=(C/'ExactSha.cuh').read_text();cu=(C/'pinning.cu').read_text()
MASK=(1<<32)-1
ror=lambda x,n:((x>>n)|(x<<(32-n)))&MASK
S0=lambda x:ror(x,2)^ror(x,13)^ror(x,22)
S1=lambda x:ror(x,6)^ror(x,11)^ror(x,25)
s0=lambda x:ror(x,7)^ror(x,18)^(x>>3)
s1=lambda x:ror(x,17)^ror(x,19)^(x>>10)
Ch=lambda x,y,z:z^(x&(y^z))
Maj=lambda x,y,z:(x&y)|(z&(x|y))
IV=[0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]
K=[int(x,16) for x in re.search(r'uint32_t K\[\] = \{(.*?)\};',(C/'GPUHash.h').read_text(),re.S)[1].replace('\n','').replace('\t','').split(',') if x.strip()]
def body(text,name):
 p=text.index(name+'(');b=text.index('{',p);level=1;e=b+1
 while level:
  level+=(text[e]=='{')-(text[e]=='}');e+=1
 return text[b+1:e-1]
def translate(name,args,source=text):
 b=body(source,name);b=re.sub(r'/\*.*?\*/','',b,flags=re.S)
 b=b.replace('{','').replace('}','')
 lines=[]
 for stmt in b.split(';'):
  stmt=stmt.strip()
  if not stmt:continue
  decl=re.fullmatch(r'uint32_t (\w+)(?:\[(\d+)\])?',stmt)
  if decl:
   if decl[2]:lines.append(f'{decl[1]} = [None]*{decl[2]}')
   continue
  stmt=re.sub(r'^(?:const )?uint32_t ','',stmt)
  m=re.fullmatch(r'(\w+(?:\[\d+\])?)\s*(\+=|=)\s*(.*)',stmt,re.S)
  assert m,(name,stmt)
  dst,op,expr=m.groups();expr=re.sub(r'\b(0[xX][\da-fA-F]+|\d+)[uU]\b',r'\1',expr)
  if op=='+=':expr=f'{dst} + ({expr})'
  lines.append(f'{dst} = ({expr}) & MASK')
 namespace=dict(globals(),qsb_host_S0=S0,qsb_host_S1=S1)
 exec('def run('+args+'):\n'+''.join(' '+l+'\n' for l in lines)+' return locals()\n',namespace)
 return namespace['run']
pre=translate('qsb_sha_tail_precompute','p,w2')
tail=translate('_SHA256TransformFastTail11','state,w0,w1,w2,pre')
digest=translate('_SHA256TransformDigest32','out,m')
pub=translate('_SHA256TransformPubkey33','out,m')
def compress(state,words):
 w=list(words)
 for i in range(16,64):w.append((s1(w[i-2])+w[i-7]+s0(w[i-15])+w[i-16])&MASK)
 a,b,c,d,e,f,g,h=state
 for k,x in zip(K,w):
  t=(h+S1(e)+Ch(e,f,g)+k+x)&MASK
  a,b,c,d,e,f,g,h=(t+S0(a)+Maj(a,b,c))&MASK,a,b,c,(d+t)&MASK,e,f,g
 return [(x+y)&MASK for x,y in zip(state,[a,b,c,d,e,f,g,h])]
def hashblock(b):return list(struct.unpack('>8I',hashlib.sha256(b).digest()))
rng=random.Random(202609201735)
N=16384
edges=[0,1,MASK,0x80000000,0x7fffffff,0xaaaaaaaa,0x55555555]
for i in range(N):
 m=[rng.getrandbits(32) for _ in range(9)] if i>=len(edges) else [edges[i]]*9
 out=[None]*8;digest(out,m[:8]);assert out==compress(IV,m[:8]+[1<<31]+[0]*6+[256])
 assert out==hashblock(struct.pack('>8I',*m[:8]))
 # Arbitrary nine words audit the exact original transform contract, while
 # properly padded 33-byte cases separately check hashlib.
 pub(out,m);assert out==compress(IV,m+[0]*6+[264])
 b=bytes([2+(i&1)])+rng.randbytes(32)
 pm=list(struct.unpack('>8I',b[:32]))+[(b[32]<<24)|0x800000]
 pub(out,pm);assert out==hashblock(b)
 state=[rng.getrandbits(32) for _ in range(8)] if i>=len(edges) else [edges[i]]*8
 p=state+[0]*5;pre(p,m[2]);s=state[:];tail(s,*m[:3],p)
 assert s==compress(state,m[:3]+[0]*12+[79960])
 assert p[:8]==state
# End-to-end 9995-byte hashes, varied whole-message midstates and padding.
for i in range(96):
 data=rng.randbytes(9995);state=IV[:]
 for off in range(0,9984,64):state=compress(state,struct.unpack('>16I',data[off:off+64]))
 w=list(struct.unpack('>3I',data[9984:]+b'\x80'));p=state+[0]*5;pre(p,w[2]);tail(state,*w,p)
 assert state==hashblock(data)
# Byte selector 0x1234 maps output bytes [B0,A3,A2,A1]. Check the production
# expression and arbitrary locktimes, every byte value in every position.
assert 'uint32_t w1 = __byte_perm(lt, pin_tail_words[1], 0x1234);' in cu
assert 'pp.suffix[71],' in cu
sel=int(re.search(r'uint32_t w1 = __byte_perm\(lt, pin_tail_words\[1\], (0x\w+)\)',cu)[1],16)
def perm(a,b,sel):
 bs=a.to_bytes(4,'little')+b.to_bytes(4,'little')
 return sum(bs[(sel>>(4*i))&7]<<(8*i) for i in range(4))
def oldword(lt,b):return ((lt&0xff00)<<16)|(lt&0xff0000)|((lt>>16)&0xff00)|b
ltests=[v<<(8*j) for j in range(4) for v in range(256)]+[rng.getrandbits(32) for _ in range(65536)]
for lt in ltests:
 b=rng.randrange(256);assert perm(lt,b,sel)==oldword(lt,b)
for lt in [499999999,500000000,500000127,500000128,1744599999,1744600000,MASK]:
 for b in range(256):assert perm(lt,b,sel)==oldword(lt,b)
# Host helper rotations are explicitly unsigned with fixed nonzero shift counts.
assert 'return (x >> n) | (x << (32 - n));' in body(text,'qsb_host_ror')
assert 'return qsb_host_ror(x,2) ^ qsb_host_ror(x,13) ^ qsb_host_ror(x,22);' in body(text,'qsb_host_S0')
assert 'return qsb_host_ror(x,6) ^ qsb_host_ror(x,11) ^ qsb_host_ror(x,25);' in body(text,'qsb_host_S1')
assert '#define QSB_SHA_STATE_WORDS 13' in text
for token in ['cudaMalloc(&d_mid, QSB_SHA_STATE_WORDS*sizeof(uint32_t))','cudaMalloc(&d_mid_slot[s], QSB_SHA_STATE_WORDS*sizeof(uint32_t))','QSB_SLOTS*QSB_SHA_STATE_WORDS*sizeof(uint32_t)','memcpy(h_mid + (size_t)s*QSB_SHA_STATE_WORDS, cur_mid, sizeof(cur_mid));','cudaMemcpyAsync(d_mid_slot[s], h_mid + (size_t)s*QSB_SHA_STATE_WORDS, sizeof(cur_mid), cudaMemcpyHostToDevice, st);','qsb_sha_tail_precompute(cur_mid, tail_word2);','qsb_sha_tail_precompute(mid_pre, tail_word2);','cudaMemcpy(d_mid,mid_pre,sizeof(mid_pre),cudaMemcpyHostToDevice)']:
 assert token in cu,token
assert cu.index('if (drain_slot(s)) return 1;')<cu.index('memcpy(h_mid + (size_t)s*QSB_SHA_STATE_WORDS')<cu.index('cudaMemcpyAsync(d_mid_slot[s]')<cu.index('launch_pinning_pipeline<true>(')
# Source structural sanity; dense interleave unchanged and mutable K gone only
# from the selected specialized functions (generic fallback remains untouched).
assert 'K[' not in text and 'S2Round(' not in text and 'WMIX(' not in text
assert text.count('t2 = S0(')==191 # tail round zero now entirely precomputed
assert text.count('uint32_t w[16];')==3
# Negative controls: wrong literal constant, wrong precompute, wrong selector.
bad=translate('_SHA256TransformDigest32','out,m',text.replace('0x71374491u','0x71374490u'))
out=[0]*8;bad(out,[0]*8);assert out!=hashblock(bytes(32))
p=IV+[0]*5;pre(p,0);p[8]^=1;st=IV[:];tail(st,0,0,0,p);assert st!=compress(IV,[0]*15+[79960])
assert perm(0x12345678,0xab,0x1243)!=oldword(0x12345678,0xab)
result={'status':'PASS','three_transform_cases':N,'full_9995_byte_hashes':96,'byte_permutation_cases':len(ltests)+7*256,'negative_controls':3,'native_compile':False,'gpu_timing':None,'limitations':'Python executes source assignment semantics, not nvcc/SASS. Source checks cover allocation sizes and stream ordering, not CUDA runtime races.','sha256':{n:hashlib.sha256((C/n).read_bytes()).hexdigest() for n in ['pinning.cu','ExactSha.cuh']}}
(C/'research/sha-audit-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
