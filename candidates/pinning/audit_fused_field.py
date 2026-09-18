from pathlib import Path
import random,struct,subprocess,json,sys,itertools
P=(1<<256)-(1<<32)-977;B=1<<256;r=random.Random(0x202609180607)
edge=[0,1,2,976,977,(1<<32)-1,1<<32,(1<<64)-1,1<<64,1<<128,1<<255,P-65537,P-977,P-2,P-1,P,P+1,B-977,B-2,B-1]
cases=list(itertools.product(edge,repeat=3))
cases += [(r.getrandbits(256),r.getrandbits(256),r.getrandbits(256)) for _ in range(30000)]
cases += [(a,a,b) for a,b in itertools.product(edge,repeat=2)]
# Force the rare carry in the final pseudo-Mersenne fold.
K=(1<<32)+977
for a in edge+[r.randrange(B) for _ in range(512)]:
 first=(a*a)%B+K*((a*a)//B)+3*P
 for high in (first//B,first//B+1):
  e=high*B+B-high*K-first
  for offset in (0,1,977,65537):
   if 0<=e+offset<B:cases.append((a,e+offset,0))
inp=Path('logs/field-input.bin');out=Path('logs/field-output.bin')
inp.write_bytes(struct.pack('<I',len(cases))+b''.join(x.to_bytes(32,'little') for t in cases for x in t))
subprocess.run([sys.argv[1],str(inp),str(out)],check=True,timeout=120)
data=out.read_bytes();assert len(data)==len(cases)*128
for i,(a,b,c) in enumerate(cases):
 got=[int.from_bytes(data[i*128+32*j:i*128+32*(j+1)],'little') for j in range(4)]
 assert len(set(got))==1,(i,'alias')
 assert all(x%P==w for x,w in zip(got,[(a*a+b-2*c)%P]*4)),(i,a,b,c,got)
print(json.dumps({'passed':True,'native_triples':len(cases),'comparisons':len(cases)*4,'full_width_inputs':True,'raw_alias_equality':True,'result':'congruent modulo p; not claiming all outputs canonical'}))
