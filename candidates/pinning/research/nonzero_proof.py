from pathlib import Path
import sys,json,random
W=Path(__file__).resolve().parent
import reference as h
assert (h.LAMBDA*h.LAMBDA+h.LAMBDA+1)%h.N==0
maxima={};checks=0
for m in [512,608]:
 maximum=0
 for a in range(m):
  for b in range(m):
   rank,j,neg,dx,dy=h.fast(a,b,m)
   assert bool(rank)==bool(dx or dy)
   maximum=max(maximum,abs(dx),abs(dy));checks+=1
 assert maximum<=2*m//3;maxima[m]=maximum
# Any relation a+lambda*b=0 modN forces Q=a^2-a*b+b^2=0 modN.
# If |a|,|b|<=M and3M^2<N, positive definiteness ofQ excludes a nonzero relation.
weight=1;prefix=0;bounds=[]
for i,m in enumerate([512]*4+[608]*9):
 D=2*m//3
 assert prefix<weight
 M=prefix+D*weight
 assert 3*M*M<h.N
 bounds.append({'digit':i,'weight':weight,'prefix_bound':prefix,'difference_coordinate_bound':M,'Q_upper':3*M*M})
 prefix=M;weight*=m
assert prefix<weight
assert 3*h.R*h.R<h.N
assert 3*(h.R+2*prefix)**2<h.N
# For regular steps, current point is a nonzero integer multiple ofweight in
# at least onecoordinate; prefixcoordinates are strictly smaller thanweight.
# Thus prefix +/- current cannot vanish overZ, nor modN bytheQbound.
# Final plus isinitialsplitpair r: boundR. Final minus is2prefix-r: boundR+2L.
# Sameintegernonzero argument applies becausefinaldigit iscertifiednonzero.
rng=random.Random(202609210859);fallback=0;warpfallback=0;trials=32000
for block in range(trials//32):
 anyzero=False
 for lane in range(32):
  k=rng.randrange(2**256);x,y,_=h.split_corr(k);u=v=0;w=1;ok=True
  for i in range(14):
   if i<13:
    m=512 if i<4 else 608
    rank,j,n,dx,dy=h.fast(x%m,y%m,m);x,y=(x-dx)//m,(y-dy)//m
   else:dx,dy=x,y;rank=bool(x or y)
   ok &= bool(rank)
   if i and ok:
    # All prefixes withnonzero digits: neither equal nor opposite to next affine point.
    assert (u+h.LAMBDA*v)%h.N != (w*(dx+h.LAMBDA*dy))%h.N
    assert (u+h.LAMBDA*v)%h.N != (-w*(dx+h.LAMBDA*dy))%h.N
   u+=w*dx;v+=w*dy
   if i<13:w*=m
  if not ok:fallback+=1;anyzero=True
  assert (u+h.LAMBDA*v)%h.N==k%h.N
 warpfallback+=anyzero
r={'status':'PASS','exhaustive_digit_pairs':checks,'coordinate_digit_maxima':maxima,'regular_difference_Q_bounds':bounds,'final_weight':weight,'final_prefix_bound':prefix,'final_sum_Q_upper':3*h.R*h.R,'final_difference_Q_upper':3*(h.R+2*prefix)**2,'order':h.N,'final_difference_bound_ratio':3*(h.R+2*prefix)**2/h.N,'sample_scalar_count':trials,'sample_zero_digit_scalars':fallback,'sample_warps_with_fallback':warpfallback,'universal_nonzero_digit_group_certificate':True,'scope':'Exact group theorem with finite exhaustive digit-bound proof. Inherited approximate device field primitives and GPU performance are outside this theorem.'}
(W/'nonzero-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='regular_difference_Q_bounds'},indent=2))
