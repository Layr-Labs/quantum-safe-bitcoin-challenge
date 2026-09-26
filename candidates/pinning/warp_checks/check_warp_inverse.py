"""Independent Python divsteps + source-derived LUT + exact 32-lane integer model.
Never compiles or executes C++/CUDA. Checks the selected promoted mechanism.
"""
from pathlib import Path
import random,re,json,hashlib
D=Path(__file__).resolve().parent;U=2**32-1;P=2**256-2**32-977
M=2**30-1;MM=0xD2253531;R=random.Random(0x92632)
z=(D/'promoted-subset-zinv32.cuh').read_text()
lut=[int(x,16) for x in re.search(r'ZI_BY_LUT\[832\]=\{(.*?)\};',z,re.S)[1].replace('ULL','').split(',') if x.strip()]
assert len(lut)==832 and (MM*P)&M==M

def sbyte(x):return (x&255)-256 if x&128 else x&255

def decode(delta,f,g):
    fi=(f*(2-f*f))&U;ratio=g*fi&63
    word=lut[(max(-6,min(6,delta))+6)*64+ratio]
    a,b,c,d=[sbyte(word>>(8*k)) for k in range(4)]
    flag=word>>32
    return (-delta if flag>>31 else delta)+sbyte(flag),(a,b,c,d)

def steps(delta,f,g,n):
    a,b,c,d=1,0,0,1
    for _ in range(n):
        odd=g&1
        if delta>0 and odd:
            delta=1-delta;f,g=g,(g-f)//2
            a,b,c,d=2*c,2*d,c-a,d-b
        else:
            delta+=1;g=(g+odd*f)//2
            a,b,c,d=2*a,2*b,c+odd*a,d+odd*b
    return delta,(a,b,c,d),f,g

def column(delta,f,g):
    u,v,q,r=1,0,0,1
    for _ in range(5):
        delta,(a,b,c,d)=decode(delta,f,g)
        nf=((a*f+b*g)&U)>>6;g=((c*f+d*g)&U)>>6;f=nf
        u,v,q,r=a*u+b*q,a*v+b*r,c*u+d*q,c*v+d*r
    assert abs(u)+abs(v)<=2**30 and abs(q)+abs(r)<=2**30
    return delta,(u,v,q,r)

lut_checks=0
for delta in list(range(-8,9))+[-100,-32,32,100]:
 for f in range(1,64,2):
  for g in range(64):
   got=decode(delta,f,g);want=steps(delta,f,g,6)
   assert got==want[:2];lut_checks+=1

def limb_row(x,y,a,b,modp,drop_final_carry=False):
    xs=[(x>>(32*j))&U for j in range(8)];ys=[(y>>(32*j))&U for j in range(8)]
    xt=x>>256;yt=y>>256
    m=((a*xs[0]+b*ys[0])*MM)&M if modp else 0
    acc=[a*xs[j]+b*ys[j]-m*([977,1][j] if j<2 else 0) for j in range(8)]
    assert all(-(2**63)<=v<2**63 for v in acc)
    biased=[v+2**63-(2**31 if j else 0) for j,v in enumerate(acc)]
    assert all(0<=v<2**64 for v in biased)
    lo=[v&U for v in biased];hi=[v>>32 for v in biased]
    sums=[(v+(hi[j-1] if j else 0))&U for j,v in enumerate(lo)]
    gen=sum(int(v<lo[j])<<j for j,v in enumerate(sums))
    prop=sum(int(v==U)<<j for j,v in enumerate(sums))
    carry=(prop+(gen<<1))^prop
    low=[(v+((carry>>j)&1))&U for j,v in enumerate(sums)]
    high=a*xt+b*yt+m+hi[7]-2**31+(0 if drop_final_carry else (carry>>8)&1)
    nexts=low[1:]+[high&U]
    out=[((v>>30)|(nexts[j]<<2))&U for j,v in enumerate(low)]
    represented=sum(v<<(32*j) for j,v in enumerate(out))+((high>>30)<<256)
    exact=a*x+b*y+m*P
    assert exact&M==0
    if not drop_final_carry:assert represented==exact>>30
    return represented, (carry>>8)&1

def canon(x):
    # Literal wide-top single pseudo-Mersenne fold followed by add/sub p.
    X=[(x>>(32*j))&U for j in range(9)];hi=X[8]-(2**32 if X[8]>>31 else 0)
    acc=X[0]+hi*977;X[0]=acc&U;acc>>=32
    acc+=X[1]+hi;X[1]=acc&U;acc>>=32
    for j in range(2,8):acc+=X[j];X[j]=acc&U;acc>>=32
    X[8]=acc&U
    signed=sum(v<<(32*j) for j,v in enumerate(X))-(2**288 if X[8]>>31 else 0)
    if signed<0:signed+=P
    if signed>=P:signed-=P
    assert 0<=signed<P and signed==x%P
    return signed

max_batch=0;rows=0;negative_carry=0;max_top=0;max_rs=0

def inverse(x,scale,cap=32):
    global max_batch,rows,negative_carry,max_top,max_rs
    orig=x;f,g=P,x;r,s=0,scale;delta=1
    for batch in range(cap):
        ref=steps(delta,f,g,30)
        delta,(a,b,c,d)=column(delta,f&U,g&U)
        assert (delta,(a,b,c,d))==ref[:2]
        old=(f,g,r,s)
        coefficients=[(a,b),(d,c),(a,b),(d,c)]
        result=[]
        for row,((aa,bb),xx,yy) in enumerate(zip(coefficients,old,[g,f,s,r])):
            nv,carry=limb_row(xx,yy,aa,bb,row>=2)
            bad,_=limb_row(xx,yy,aa,bb,row>=2,True)
            negative_carry+=bad!=nv
            max_top=max(max_top,abs(nv>>256));result.append(nv);rows+=1
        f,g,r,s=result
        assert (f,g)==ref[2:]
        assert (r*orig-f*scale)%P==0 and (s*orig-g*scale)%P==0
        # L1 coefficient norm <=2^30; correction m<2^30 gives +p at most per batch.
        assert max(abs(r),abs(s))<(batch+2)*P
        max_rs=max(max_rs,max(abs(r),abs(s))//P)
        if g==0:
            assert orig==0 or abs(f)==1
            max_batch=max(max_batch,batch+1)
            return canon(-r if f<0 else r),False
    # Header's fallback works on original input, never the partially updated rows.
    return pow(orig,P-2,P)*scale%P,True

cases=[0,1,2,P-2,P-1]+[1<<i for i in range(256)]+[P-(1<<i) for i in range(256)]+[R.randrange(P) for _ in range(600)]
fallback_checks=0
for x in cases:
    scale=R.choice([1,P-1,R.randrange(1,P)])
    got,fb=inverse(x,scale)
    assert not fb and got==(pow(x,-1,P)*scale%P if x else 0)
for x in cases[:40]:
 for cap in [0,1,2]:
    scale=R.randrange(1,P);got,fb=inverse(x,scale,cap)
    assert got==(pow(x,-1,P)*scale%P if x else 0)
    fallback_checks+=fb
canon_cases=0
for hi in range(-33,34):
 for low in [0,1,977,2**32,2**256-1,R.randrange(2**256)]:
    canon(hi*2**256+low);canon_cases+=1
assert negative_carry>0
header=(D.parent/'WarpInverse.cuh').read_text();adapter=(D.parent/'BalancedWarpRoots.cuh').read_text()
assert 'QSB_ISO_FUSED_ROOT_SCALE' not in header and 'QsbInverseWords' not in header
assert 'pin_iso_invu_words[digit>>1]' in header and '#define QWR_ROOT_MAX_BATCHES 32' in header
assert header.count('0xFFFFFFFEFFFFFC2DULL')==1
assert 'if(__all_sync(0xffffffffu,tid<32))' in adapter
assert '_ModInv(root)' not in adapter and 'qsb_field_mul(scaled,root,invu)' not in adapter
assert re.search(r'__syncthreads\(\);\s*\}\s*if\(__all_sync',adapter)
# Each parent write is visible after a CTA barrier before the first warp reads.
# Do not use a warp-only barrier to publish the lane0 final root across independently scheduled warps.
result=dict(status='PASS_PYTHON_SOURCE_MODEL_ONLY',lut_cases=lut_checks,inverses=len(cases),
    lane_row_checks=rows,forced_fallback_checks=fallback_checks,max_batches=max_batch,
    max_abs_signed_top_word=max_top,max_rs_multiple_p=max_rs,
    canon_cases=canon_cases,negative_dropped_final_carry=negative_carry,
    exact_ISO_initialization=True,only_one_scale=True,full_warp_entry=True,
    native_compilation=False,GPU_execution=False,production_inclusion=True,
    source_commit='a137e289b236c3622eba80f1ad5e9a0c8a91eb67',
    header_sha256=hashlib.sha256(header.encode()).hexdigest(),adapter_sha256=hashlib.sha256(adapter.encode()).hexdigest())
(D/'warp-inverse-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
