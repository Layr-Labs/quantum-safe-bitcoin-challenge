"""Source-tied productive-work policy and slot models; no native compilation."""
from pathlib import Path
import math,re,json
pin=Path(__file__).resolve().parent
p=(pin/'ProductiveFinish.cuh').read_text()
# Policy model tied to source constants and ordering; it does not execute C++.
order=[int(x) for x in re.search(r'order\[8\]=\{([^}]+)\}',p)[1].split(',')]
assert order==[0,1,1,0,1,0,0,1]
assert 'const unsigned a=2u*pair+((pair==1u || pair==2u)?1u:0u);' in p
assert 'const unsigned b=a^1u;' in p
assert 'cost[step-4u]=completed?seconds/(double)completed:0.0;' in p
floor=float(re.search(r'cost\[a\]<([\d.]+)\*cost\[b\]',p)[1])
gain=float(re.search(r'log_gain>=4.0\*log\(([\d.]+)\)',p)[1])
assert floor==0.995 and gain==1.01
assert 'wins>=3u' in p and 'step<12u' in p and 'step==1u || step==2u' in p
cases=0
def choose(costs,valid=True):
    if not valid:return 0
    wins=0; loggain=0
    for pair in range(4):
        a=2*pair+(pair in [1,2]);b=a^1
        if not(costs[a]>0 and costs[b]>0):return 0
        if costs[a]<floor*costs[b]:return 0
        wins+=costs[a]>costs[b]
        loggain+=math.log(costs[a])-math.log(costs[b])
    return int(wins>=3 and loggain>=4*math.log(gain))
def run(measured,warmup=None,forced=-1):
    global cases
    cases+=1
    schedule=[0,1,1,0]+order
    traces=[];costs=[];valid=True;selected=0;completed_total=0
    warmup=warmup or [(1.,1000,True)]*4
    work=warmup+measured
    for step,(seconds,completed,clock_ok) in enumerate(work):
        route=forced if forced>=0 else schedule[step]
        completed_total+=completed
        traces.append((route,completed))
        if not (clock_ok and completed and seconds>0 and seconds<1e100):valid=False
        if step>=4:costs.append(seconds/completed if completed else 0)
    selected=forced if forced>=0 else choose(costs,valid)
    # Finite completion: 32 more ordinary sequences cannot change the winner.
    for _ in range(32):traces.append((selected,7));completed_total+=7
    assert sum(n for _,n in traces)==completed_total
    assert all(route==selected for route,_ in traces[12:])
    return selected,traces
def measurements(gains,counts=None,drift=0):
    out=[];counts=counts or [1000]*8
    for i,route in enumerate(order):
        cost=1e-9*(1+drift*i)/(gains[i//2] if route else 1.)
        out.append((cost*counts[i],counts[i],True))
    return out
for counts in [[1000]*8,[1,5,17,10,2,700,19,71],[1244600000]*8]:
    for ratio,expect in [(0.7,0),(.99,0),(1.,0),(1.005,0),(1.009,0),(1.02,1),(1.3,1)]:
        assert run(measurements([ratio]*4,counts))[0]==expect
    assert run(measurements([1.10,1.10,1.10,.99],counts))[0]==0
    assert run(measurements([1.10,1.,1.,1.10],counts))[0]==0
    assert run(measurements([1.10,1.10,1.10,.998],counts))[0]==1
for drift in [-.006,-.003,.003,.006]:
    assert run(measurements([1.]*4,drift=drift))[0]==0
    assert run(measurements([1.06]*4,drift=drift))[0]==1
for bad in [(0.,1000,True),(-1.,1000,True),(float('nan'),1000,True),
            (float('inf'),1000,True),(1.,0,True),(1.,1000,False),(1e100,1000,True)]:
    for idx in range(12):
        warm=[(1.,1000,True)]*4;m=measurements([1.10]*4)
        if idx<4:warm[idx]=bad
        else:m[idx-4]=bad
        assert run(m,warmup=warm)[0]==0
for forced in [0,1]:
    chosen,trace=run(measurements([.1]*4),forced=forced)
    assert chosen==forced and all(route==forced for route,_ in trace)

# Sequence/slot model: all work remains distinct, including tails and switching.
route_switch_cases=0
for n in [1,63,64,65,1025,1244600000]:
  for batch in [8388608,((n+2)//3)]:
    pending=[None,None];published=[];seen=set()
    for seq,route in enumerate([0,1,1,0]+order+[1,0]):
        assert pending==[None,None]
        measured_count=0
        for j,off in enumerate(range(0,n,batch)):
            slot=j%2
            if pending[slot] is not None:published.append(pending[slot])
            count=min(batch,n-off)
            ident=(seq,off,count)
            assert ident not in seen;seen.add(ident)
            pending[slot]=ident
            measured_count+=count
        for slot in range(2):
            if pending[slot] is not None:published.append(pending[slot]);pending[slot]=None
        assert measured_count==n
    assert set(published)==seen and len(published)==len(seen)
    route_switch_cases+=1

print(json.dumps({'passed':True,'policy_model_cases':cases,'slot_sequence_models':route_switch_cases,'native_compilation':False,'gpu_execution':False},indent=2))
