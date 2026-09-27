"""Interleaved independent-warp packed trees, tails and scratch ownership. Python only."""
from pathlib import Path
import random,json,hashlib
D=Path(__file__).resolve().parent;P=2**256-2**32-977
R=random.Random(0x4128)
from prefix_model import product as mul
coop_calls=0;shared_addresses=set()

def warp_inverse(values,scale,warp):
    global coop_calls
    pb=64*warp;ib=32*warp;prod=[None]*256;inv=[None]*128
    def putp(i,x):assert i//64==warp;prod[i]=x;shared_addresses.add(('p',i))
    def puti(i,x):assert i//32==warp;inv[i]=x;shared_addresses.add(('i',i))
    for lane,x in enumerate(values):putp(pb+lane,x)
    offset=0;count=32
    while count>1:
        half=count//2
        for lane in range(half):
            a=prod[pb+offset+lane];b=prod[pb+offset+half+lane]
            putp(pb+offset+count+lane,mul(a,b) if half<=4 else a*b%P)
            coop_calls+=half<=4
        offset+=count;count//=2
    puti(ib+30,pow(prod[pb+62],-1,P)*scale%P)
    offset=60;count=2
    while count<32:
        half=count//2
        for lane in range(count):
            a=inv[ib+offset+count-32+(lane&(half-1))];b=prod[pb+offset+(lane^half)]
            puti(ib+offset-32+lane,mul(a,b) if count<=4 else a*b%P)
            coop_calls+=count<=4
        offset-=count*2;count*=2
    got=[inv[ib+(lane&15)]*prod[pb+(lane^16)]%P for lane in range(32)]
    assert got==[scale*pow(x,-1,P)%P for x in values]
    return got

def run(raw,scale,weight,schedule):
    n=len(raw);mem=list(raw)+[None]*(2048-n)
    scratch_written=set();scratch_read=set();written=set();events=0
    def load(i):
        if i>=n:return 1,False
        assert i not in written
        v=mem[i]%P;return v or 1,bool(v)
    def put(lane,row,v):
        j=lane+row*128;assert j%128==lane
        assert n+j<2048;mem[n+j]=v;scratch_written.add(j)
    def get(lane,row):
        j=lane+row*128;assert j in scratch_written and j not in scratch_read
        assert j not in written;scratch_read.add(j);return mem[n+j]
    def worker(warp):
        lanes=list(range(warp*32,(warp+1)*32));totals=[]
        for lane in lanes:
            q=[]
            for quartet in range(2):
                v=[load(lane+quartet*512+j*128)[0] for j in range(4)]
                p0=v[0]*v[1]%P;p1=v[2]*v[3]%P
                put(lane,2+2*quartet,p0);put(lane,3+2*quartet,p1)
                q.append(p0*p1%P);put(lane,quartet,q[-1])
            totals.append(q[0]*q[1]%P);yield
        inverses=warp_inverse(totals,scale,warp);yield
        R.shuffle(lanes)
        for lane in lanes:
            q0=get(lane,0);q1=get(lane,1);iv=inverses[lane%32]
            iq=[iv*q1%P,iv*q0%P]
            for quartet in [1,0]:
                p0=get(lane,2+2*quartet);p1=get(lane,3+2*quartet)
                ip=[iq[quartet]*p1%P,iq[quartet]*p0%P]
                for pair in range(2):
                    ids=[lane+quartet*512+pair*256+j*128 for j in range(2)]
                    (a,na),(b,nb)=[load(i) for i in ids]
                    for i,v,nz in zip(ids,[ip[pair]*b%P,ip[pair]*a%P],[na,nb]):
                        if i>=n:continue
                        assert i%128==lane and i not in written
                        if i in scratch_written:assert i in scratch_read
                        v=v if nz else 0;mem[i]=v;mem[n+i]=v*weight%P;written.add(i)
                yield
    workers={w:worker(w) for w in range(4)}
    while workers:
        if schedule=='serial':w=min(workers)
        elif schedule=='reverse':w=max(workers)
        else:w=R.choice(list(workers))
        try:next(workers[w]);events+=1
        except StopIteration:del workers[w]
    assert len(written)==n and scratch_read==scratch_written and len(scratch_read)==768
    expected=[scale*pow(x%P,-1,P)%P if x%P else 0 for x in raw]
    assert mem[:2*n]==expected+[x*weight%P for x in expected]
    return events
