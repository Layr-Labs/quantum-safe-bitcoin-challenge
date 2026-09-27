"""Interpret the literal inline-PTX blocks, fail on unknown instructions. No compiler."""
from pathlib import Path
import re,ast,random,json,hashlib
D=Path(__file__).resolve().parent;h=(D.parent/'PrefixCyclicField.cuh').read_text()
blocks=[]
for source in re.findall(r'asm\((.*?)\s*:\s*"',h,re.S):
    literals=re.findall(r'"(?:[^"\\]|\\.)*"',source)
    blocks.append(''.join(ast.literal_eval(x) for x in literals))
assert len(blocks)==2
def interpret(block,regs):
    cc=0;pred={};regs=regs.copy()
    def value(x):return regs[x] if x.startswith('%') else int(x)
    for statement in block.replace('{','').replace('}','').split(';'):
        statement=statement.strip()
        if not statement:continue
        if statement.startswith('.reg .pred'):continue
        if statement.startswith('@'):
            guard,statement=statement.split(None,1)
            if not pred[guard[1:]]:continue
        op,args=statement.split(None,1);a=[x.strip() for x in args.split(',')]
        if op=='setp.eq.u32':pred[a[0]]=value(a[1])==value(a[2]);continue
        if op=='add.cc.u64':
            z=value(a[1])+value(a[2]);regs[a[0]]=z&((1<<64)-1);cc=z>>64
        elif op=='addc.u32':regs[a[0]]=(value(a[1])+value(a[2])+cc)&((1<<32)-1)
        elif op=='mov.u64':regs[a[0]]=value(a[1])
        elif op=='bfi.b32':
            src,base,start,length=map(value,a[1:]);mask=((1<<length)-1)<<start
            regs[a[0]]=(base&~mask)|((src<<start)&mask)
        elif op=='sub.cc.u64':
            x,y=value(a[1]),value(a[2]);regs[a[0]]=(x-y)&((1<<64)-1);cc=int(x<y)
        elif op=='subc.u32':regs[a[0]]=(value(a[1])-value(a[2])-cc)&((1<<32)-1)
        else:raise AssertionError('unknown PTX '+op)
    return regs
R=random.Random(0x9642);W=1<<64;cases=0;negative=0;carrycases=0;borrowcases=0
patterns=[[0]*8,[(1<<64)-1]*8]+[[0]*i+[W-1]+[0]*(7-i) for i in range(8)]
patterns += [[R.randrange(W) for _ in range(8)] for _ in range(3000)]
for terms in patterns:
    for d in range(8):
        total=counts=prefix=0
        for i,term in enumerate(terms):
            r=interpret(blocks[0],{'%0':total,'%1':counts,'%2':prefix,'%3':term,'%4':d,'%5':i})
            total,counts,prefix=r['%0'],r['%1'],r['%2']
            assert total+(counts&255)*W==sum(terms[:i+1])
            assert (counts&255)<=7
        pref=sum(terms[:d+1]);assert prefix+((counts>>8)&255)*W==pref
        r=interpret(blocks[1],{'%2':total,'%3':prefix,'%4':counts&255,'%5':(counts>>8)&255})
        assert r['%0']+r['%1']*W==sum(terms[d+1:])
        carrycases+=(counts&255)>0;borrowcases+=total<prefix
        bad=interpret(blocks[1].replace('subc.u32','addc.u32'),{'%2':total,'%3':prefix,'%4':counts&255,'%5':(counts>>8)&255})
        negative+=bad['%0']+bad['%1']*W!=sum(terms[d+1:]);cases+=1
assert negative>0 and carrycases>0 and borrowcases>0
result=dict(status='PASS_LITERAL_PTX_PYTHON_INTERPRETER',cases=cases,
            total_carry_cases=carrycases,prefix_subtraction_borrow_cases=borrowcases,
            wrong_subc_negative_failures=negative,interpreted_blocks=len(blocks),
            native_compilation=False,GPU_execution=False,
            sha256=hashlib.sha256(h.encode()).hexdigest())
(D/'prefix-ptx-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
