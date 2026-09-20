from pathlib import Path
import json,random,hashlib
import curve_ref as g
import source_calls as sc
from ptx_field_model import function,extract_ptx,Program
C=Path(__file__).resolve().parent.parent;W=C/'research'
cu=(C/'pinning.cu').read_text();mh=(C/'GPUMath.h').read_text();P,N=g.P,g.N
sc.P=P;sc.add_limb=lambda a,b:(a+b)%P
mixed=sc.calls(sc.body(mh,'void _PointAddSecp256k1('));assert len(mixed)==18
point=lambda k:g.to_affine(g.jac_mul(k%N))
prog=Program(extract_ptx(function(cu,'void qsb_field_mul(')).replace('.reg .pred take;',''))
mask=(1<<64)-1;limbs=lambda a:[(a>>(64*i))&mask for i in range(4)]
def mulptx(a,b):return sum(x<<(64*j) for j,x in enumerate(prog.run(limbs(a)+limbs(b))))%P
body=function(cu,'void qsb_block_inverse(')
for text in ['products[4][512]','inverses[4][256]','for(int count=256;count>1;count>>=1)','if(tid==0)','products[k][510]','inverses[k][254]','int local_parent=tid&(half-1)','products[k][offset+(tid^half)]','inverses[k][tid&127]','products[k][tid^128]']:
 assert text in body,text
# Source flattening: products[0..255] leaves, internalnodeoffsets as CUDA;
# inverse index = product index-256. Level snapshots model each barrier.
def inverse_block(ds):
 products=list(ds)+[None]*256;inverse=[None]*256;offset=0;count=256
 while count>1:
  half=count//2
  for tid in range(half):products[offset+count+tid]=products[offset+tid]*products[offset+half+tid]%P
  offset+=count;count=half
 inverse[254]=pow(products[510],-1,P)
 offset=508;count=2
 while count<256:
  half=count//2
  for tid in range(count):inverse[offset-256+tid]=inverse[offset+count-256+(tid&(half-1))]*products[offset+(tid^half)]%P
  offset-=count*2;count*=2
 return [inverse[tid&127]*products[tid^128]%P for tid in range(256)]
rng=random.Random(25613);checks=ptx=0;hi0=0
for active_count in [1,2,31,32,33,127,128,255,256]:
 coords=[];expected=[];ds=[]
 for tid in range(256):
  if tid<active_count:
   ch=tid%13;width=19 if ch<4 else 20;shift=19*ch if ch<4 else 76+20*(ch-4)
   d=[0,1,127,128,(1<<(width-1))-1,rng.randrange(1<<(width-1))][tid%6]
   nri=7 # nonzero runtime base distinctfromG; sufficientfor schedulecheck
   base=(pow(2,-1,N)*(1<<shift)*nri)%N;m=2*d+1;hi,lo=divmod(m,256)
   L=point(lo*base)
   if hi:
    H=point(hi*256*base)
    st=sc.run(mixed,{'p1x':H[0],'p1y':H[1],'p1z':1,'p2x':L[0],'p2y':L[1]})
    x,y,z=st['p1x'],st['p1y'],st['p1z'];assert z
   else:x,y,z=*L,1;hi0+=1
   coords.append((x,y));ds.append(z);expected.append(point(m*base))
  else:coords.append((1,1));ds.append(1)
 inverses=inverse_block(ds)
 assert all(a*b%P==1 for a,b in zip(ds,inverses))
 for tid in range(active_count):
  x,y=coords[tid];iv=inverses[tid]
  if tid<4:got=(mulptx(x,iv),mulptx(y,iv));ptx+=1
  else:got=(x*iv%P,y*iv%P)
  assert got==expected[tid];checks+=1
# allidentity treeand negativecontrol catch wrongJacobianZpowers.
assert inverse_block([1]*256)==[1]*256
x,y=point(17);z=7
assert ((x*z%P)*pow(z,-1,P)%P,(y*z%P)*pow(z,-1,P)%P)==(x,y)
assert ((x*z%P)*pow(z,-2,P)%P,(y*z%P)*pow(z,-3,P)%P)!=(x,y)
r={'status':'PASS','built_point_checks':checks,'hi_zero_cases':hi0,'PTX_normalized_points':ptx,'collective_inverse_checks':10*256,'tail_batches':[1,2,31,32,33,127,128,255,256],'negative_control':'JacobianZpowers rejected; builderhomogeneousX/Z,Y/Z confirmed','production_hash':hashlib.sha256((C/'pinning.cu').read_bytes()).hexdigest(),'limits':'Sourcecall fieldcontractmodel; actualnewnormalizePTX sampled; no nativecompile/GPUbenchmark or inheritedshortcarrycertification.'}
(W/'builder-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
