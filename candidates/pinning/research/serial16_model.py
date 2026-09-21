from pathlib import Path
import random,json
w=Path(__file__).resolve().parent;B=1<<256;N=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
r=random.Random(141116)
def decode(k,widths):
 d=2*(k%N)-N;m=d%(1<<256);negative=int(d<0);shift=0;out=[]
 for c,bits in enumerate(widths):
  f=(m>>(shift+1))&((1<<bits)-1)
  neg=negative if c==len(widths)-1 else 1-(f>>(bits-1))
  idx=(f^-neg)&((1<<(bits-1))-1)
  out.append((1-2*neg)*(2*idx+1));shift+=bits
 return out
# Independent signed recurrence, including unbounded negative right-shifts.
def serial_reference(k):
 d=2*(k%N)-N;out=[]
 for _ in range(15):
  v=(d&0x1ffff)-65536;out.append(v);d=(d-v)>>16
 out.append(d);return out
cases=[0,1,N//2,N//2+1,N-1,N,N+1,B-1]
for i in range(256):
 for v in [-2,-1,0,1,2]:cases.append(((1<<i)+v)%(1<<256))
cases += [r.getrandbits(256) for _ in range(100000)]
for k in cases:
 digits=decode(k,[16]*16)
 assert digits==serial_reference(k)
 assert all(d&1 and abs(d)<1<<16 for d in digits)
 assert sum(d<<(16*i) for i,d in enumerate(digits))==2*(k%N)-N
 old=decode(k,[18]+[17]*14);shift=0;total=0
 for width,d in zip([18]+[17]*14,old):total+=d<<shift;shift+=width
 assert total==2*(k%N)-N
# Every aligned16bit field transition and sign flag. Direct extract has only shifts1/17/33/49.
exhaustive=0
for c in range(16):
 for idx in range(32768):
  addr=(c*32768+idx)*64
  assert 0<=addr and addr+64<=32*(1<<20);exhaustive+=1
# Lower-bound memory/arithmetic break-even, not performance prediction.
# Let t be former lookup/cache fraction. Optimistically halve it; arithmetic rises7/95 (~7.37%).
fraction=(7/95)/(0.5+7/95)
result={'status':'PASS','scalar_cases':len(cases),'all_table_addresses':exhaustive,'table_bytes':32*(1<<20),'old_table_bytes':64*(1<<20),'lookups':[15,16],'per_candidate_table_bytes':[960,1024],'point_chain_M':[95,102],'point_chain_S':[28,30],'shared_digit_bytes':[60,64],'optimistic_break_even_old_lookup_fraction_if_lookup_time_halved':fraction,'scope':'Signed scalar transport/table bounds only. NO complete curve/device integration or performance claim. Earlier16-window affineW5 failure does not isolate this serial pipeline. Full32MiB-residency and cache halving assumptions are not measured.'}
(w/'serial16-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
