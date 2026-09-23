#!/usr/bin/env python3
"""Finite interleaving model of mailbox ownership, not CUDA execution."""
from collections import deque

# Lane state: worker pc, service pc, request, response, root0, root1,
# inverse0, inverse1, captured0, captured1, stopped.
INITIAL=(0,0,0,0,0,0,0,0,0,0,0)

def actions(state,epochs,mutation=None):
    w,s,req,rsp,r0,r1,i0,i1,c0,c1,stopped=state
    answer=[]
    def emit(**kwargs):
        names=('w','s','req','rsp','r0','r1','i0','i1','c0','c1','stopped')
        z=dict(zip(names,state));z.update(kwargs);answer.append(tuple(z[n] for n in names))
    if w<4*epochs:
        epoch=w//4+1;phase=w%4
        # Distinct halves and generations expose stale or torn observations.
        root=(1000+epoch,2000+epoch)
        order=(0,1,2,3) if mutation!='early_request' else (2,0,1,3)
        op=order[phase]
        if op==0:emit(w=w+1,r0=root[0])
        elif op==1:emit(w=w+1,r1=root[1])
        elif op==2:emit(w=w+1,req=epoch)
        elif rsp==epoch:
            assert (i0,i1)==(root[0]^0xabc,root[1]^0xabc),'stale/torn inverse'
            emit(w=w+1)
    elif w==4*epochs:
        assert req==rsp
        emit(w=w+1,req=-1)
    if not stopped:
        epoch=s//6+1;phase=s%6
        if phase==0:
            if req==-1:emit(stopped=1)
            elif req==epoch:emit(s=s+1)
        elif phase==1:emit(s=s+1,c0=r0)
        elif phase==2:emit(s=s+1,c1=r1)
        elif phase==3:
            if mutation=='early_response':emit(s=s+1,rsp=epoch)
            else:emit(s=s+1,i0=c0^0xabc)
        elif phase==4:
            if mutation=='early_response':emit(s=s+1,i0=c0^0xabc)
            else:emit(s=s+1,i1=c1^0xabc)
        elif phase==5:
            if mutation=='early_response':emit(s=s+1,i1=c1^0xabc)
            else:emit(s=s+1,rsp=epoch)
    return answer

def explore(lanes,epochs,mutation=None):
    initial=tuple(INITIAL for _ in range(lanes)); seen={initial}; queue=deque([initial]);terminals=0
    while queue:
        state=queue.popleft();next_states=[]
        for n,lane in enumerate(state):
            for nxt in actions(lane,epochs,mutation):
                successor=state[:n]+(nxt,)+state[n+1:]
                next_states.append(successor)
                if successor not in seen:seen.add(successor);queue.append(successor)
        if not next_states:
            assert all(x[0]==4*epochs+1 and x[-1] for x in state),'nonterminal deadlock'
            terminals+=1
    return len(seen),terminals

if __name__=='__main__':
    total=0
    for lanes,epochs in ((1,0),(1,1),(1,2),(1,3),(2,1),(2,2)):
        states,terminals=explore(lanes,epochs);total+=states
        print(f'lanes={lanes} epochs={epochs}: {states} reachable states, {terminals} final states')
    for mutation in ('early_request','early_response'):
        try:explore(1,2,mutation)
        except AssertionError as exc:print(f'negative control {mutation}: detected ({exc})')
        else:raise AssertionError('mutation unexpectedly passed')
    print(f'PASS: {total} reachable states; no corrupt transport or nonterminal deadlock; two negative controls')
