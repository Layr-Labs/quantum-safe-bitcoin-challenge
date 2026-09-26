"""Full-tile dead-output scratch reuse, exact modular model and alias checks."""
from pathlib import Path
import json,random,hashlib
ROOT=Path(__file__).resolve().parent
P=2**256-2**32-977
R=random.Random(0x52C8)

def run(raw,scale,b,fixed_base=False):
    n=len(raw); assert 0<n<=1024
    scratch_base=1024 if fixed_base else n
    mem=list(raw)+[None]*(2048-n);written=set();tmpwrites=set();tmpreads=set()
    def load(i):
        if i>=n:return 1,False
        assert i not in written
        v=mem[i]%P
        return v or 1,bool(v)
    def put(lane,j,v):
        row=lane+128*j
        assert row%128==lane
        mem[scratch_base+row]=v;tmpwrites.add(row)
    def get(lane,j):
        row=lane+128*j
        assert row in tmpwrites and row not in written
        assert row not in tmpreads
        tmpreads.add(row)
        return mem[scratch_base+row]
    product=[]
    for lane in range(128):
        qs=[]
        for q in range(2):
            vs=[load(lane+512*q+128*j)[0] for j in range(4)]
            p01=vs[0]*vs[1]%P;p23=vs[2]*vs[3]%P
            put(lane,2+2*q,p01);put(lane,3+2*q,p23)
            qs.append(p01*p23%P);put(lane,q,qs[-1])
        product.append(qs[0]*qs[1]%P)
    # Independent serial batch inverse model, exactly one scalar inverse.
    acc=1;prefix=[]
    for v in product:prefix.append(acc);acc=acc*v%P
    iv=pow(acc,-1,P)*scale%P;inverses=[0]*128
    for lane in reversed(range(128)):
        inverses[lane]=iv*prefix[lane]%P;iv=iv*product[lane]%P
    lanes=list(range(128));R.shuffle(lanes)
    for lane in lanes:
        q0=get(lane,0);q1=get(lane,1)
        invq=[inverses[lane]*q1%P,inverses[lane]*q0%P]
        order=[0,1];R.shuffle(order)
        for q in order:
            p01=get(lane,2+2*q);p23=get(lane,3+2*q)
            ip=[invq[q]*p23%P,invq[q]*p01%P]
            for pair in range(2):
                ids=[lane+q*512+pair*256+j*128 for j in range(2)]
                (a,na),(bb,nb)=[load(i) for i in ids]
                vals=[ip[pair]*bb%P,ip[pair]*a%P]
                for i,v,nz in zip(ids,vals,[na,nb]):
                    if i>=n:continue
                    assert i%128==lane and i not in written
                    if i in tmpwrites:assert i in tmpreads
                    v=v if nz else 0
                    mem[i]=v;mem[n+i]=v*b%P;written.add(i)
    assert len(written)==n and tmpreads==tmpwrites and len(tmpreads)==768
    return mem[:2*n]

cases=total=zeros=0
for n in [1,2,31,32,127,128,129,255,256,257,383,384,385,511,512,513,639,640,641,767,768,769,895,896,897,1023,1024]:
    for mode in range(4):
        raw=[R.randrange(2**256) for _ in range(n)]
        if mode==1:raw=[0]*n
        elif mode==2:raw=[P]*n
        elif mode==3:
            for j in range(0,n,3):raw[j]=[0,P,P-1,P+1,2**256-1][j%5]
        scale=R.randrange(1,P);weight=R.randrange(P)
        out=run(raw,scale,weight)
        expected=[pow(x%P,-1,P)*scale%P if x%P else 0 for x in raw]
        assert out==expected+[x*weight%P for x in expected]
        assert n+768<=2048 and 2*n<=2048
        zeros+=sum(x%P==0 for x in raw);cases+=1;total+=n
# Restore the old fixed scratch offset as an independent negative control.
raw=[R.randrange(1,P) for _ in range(769)];bad=False
try:
    out=run(raw,1,11,True)
    want=[pow(x,-1,P) for x in raw]
    bad=out!=want+[x*11%P for x in want]
except AssertionError:bad=True
assert bad
src=(ROOT.parent/'BalancedWarpRoots.cuh').read_text()
assert src.count('(count+row)*4u')==2
assert 'unsigned i=lane+quartet*512u;' in src
assert 'i+128u,n' in src and 'i+256u,n' in src and 'i+384u,n' in src
assert 'quartet*512u+pair*256u' in src
assert not any(x in src for x in ['i+512u,n','i+768u,n','pair*512u'])
assert 'if (count<=0 || count>1024) return' in src
assert src.count('qbw_root_store(roots,n,')==2
assert src.count('qsb_block_inverse_warp_n<128>(total)')==1
result=dict(status='PASS_PYTHON_MODEL_ONLY',batches=cases,roots=total,zero_roots=zeros,
    tail_fixed_base_negative_control=True,random_lane_and_quartet_order=True,
    scratch_base='logical root count',capacity_rows=2048,max_scratch_end_row_exclusive=1792,
    scratch_rows_per_CTA=768,extra_allocation_bytes=0,
    native_compilation=False,included_in_candidate=True,
    sha256=hashlib.sha256(src.encode()).hexdigest())
(ROOT/'balanced-root-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
