from pathlib import Path
from fractions import Fraction
import runpy,random,json
w=Path(__file__).resolve().parent
q=runpy.run_path(str(w/'parity_window.py'))
B=q['B'];P=q['P'];K=q['K'];beta=1<<32;mask=beta-1;S=1<<224
rng=random.Random(1319860);stats={};cases=0;max_tail=0
# Discarded fraction bound for top product cutoff at diagonal12.
top_bound=Fraction(4*(beta-1),beta)
for d in range(11):top_bound+=Fraction(min(d+1,15-d)*(beta-1)**2,beta**(12-d))
assert top_bound<9

def windows(a,b):
 x=[a>>(32*i)&mask for i in range(8)];y=[b>>(32*i)&mask for i in range(8)]
 # Only bit0 of diagonal8 is needed, so use AND/XOR rather than multiplies.
 cross=0
 for i in range(1,8):cross^=x[i]&y[8-i]
 lo=sum((x[i]*y[6-i])>>32 for i in range(7))+sum(x[i]*y[7-i] for i in range(8))+((cross&1)<<32)
 lo&=(1<<33)-1
 top=sum((x[i]*y[j])<<(32*(i+j-12)) for i in range(8) for j in range(8) if i+j>=12)+sum((x[i]*y[11-i])>>32 for i in range(4,8))
 return lo,top

def parity(a,b,c,neg):
 global cases,max_tail
 cases+=1;lo,top=windows(a,b)
 t=a*b;fullhead=q['head'](a,b)
 assert lo==fullhead% (1<<33)
 assert top<=Fraction(t,1<<384)<top+9
 def fallback(reason):
  stats[reason]=stats.get(reason,0)+1
  z=(q['raw'](a,b)+c)%P
  return ((-z)%P if neg else z)&1
 # Carry uncertainty at bit256; same fallback is necessary even if wrap changes bit257.
 if (lo&mask)+12>=beta:return fallback('low_carry')
 hi_bit=(lo>>32)&1
 assert hi_bit==(t>>256)&1
 # hi=T>>256. top*2^128 <= hi < (top+9)*2^128.
 zlo=(K*top)>>96;zhi=(K*(top+9)-1)>>96
 if zlo!=zhi:return fallback('high_fold_window')
 assert zlo==(K*(t>>256))>>224
 f=(lo&mask)+zlo
 if f>>32 != (f+13)>>32:return fallback('fold_carry')
 h=f>>32
 lower=(f&mask)*S;upper=((f&mask)+14)*S-1
 raw=q['raw'](a,b)
 assert 0<=lower<=raw<=upper<B
 qlo=(lower+c)//P;qhi=(upper+c)//P
 if qlo!=qhi or lower+c<=qlo*P:return fallback('parity_zero_boundary')
 result=((a&b)&1)^hi_bit^(h&1)^(c&1)^(qlo&1)^neg
 z=(raw+c)%P;expected=((-z)%P if neg else z)&1
 assert result==expected
 stats['fast']=stats.get('fast',0)+1
 return result
for a,b,c,n in q['cases']:parity(a,b,c,n)
# Force boundaries in bit256 and in top-fold certification with synthetic limbs.
for d in [6,7,8,11,12,13,14]:
 for i in range(8):
  j=d-i
  if not 0<=j<8:continue
  for x in [1,2,977,mask-1,mask]:
   for y in [1,2,977,mask-1,mask]:
    a=x<<(32*i);b=y<<(32*j)
    for c in [0,1,P-1,(-q['raw'](a,b))%P]:
     for neg in [0,1]:parity(a,b,c,neg)
# Uniform warp sample for the additional certificate, not a device-distribution model.
warp_fallback=0;random_fallback=0
for warp in range(2048):
 before=cases-stats.get('fast',0)
 for lane in range(32):parity(rng.getrandbits(256),rng.getrandbits(256),rng.randrange(1,P),lane&1)
 delta=cases-stats.get('fast',0)-before;warp_fallback+=delta!=0;random_fallback+=delta
result={'status':'PASS','cases':cases,'paths':stats,'random_warps':2048,'random_warps_with_fallback':warp_fallback,'random_lane_fallbacks':random_fallback,'low_window_wide_products':8,'low_window_high_products':7,'high_window_wide_products':6,'high_window_high_products':4,'operand_products_total':25,'low_window_bit_products':7,'top_tail_strict_bound':9,'top_tail_fraction':str(top_bound),'limits':'Python model only. Constant fold multiplies/guards/accumulation excluded. Uniform samples do not model actual field operands. Ambiguity always uses original full multiplier.'}
(w/'split-window-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
