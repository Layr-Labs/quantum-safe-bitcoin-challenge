"""Usage: e2e.py <pinning.ptx built with QSB_KMUL != 0>.
End-to-end check on the compiled PTX: a _ModMultCoreK inline-asm block (Karatsuba product +
production tail) vs a production _ModMultCore block, interpreted on random/edge inputs."""
import re,random,sys
M32,M64=(1<<32)-1,(1<<64)-1
ptx=open(sys.argv[1]).read()
blocks=re.findall(r'// begin inline asm\n(.*?)// end inline asm',ptx,re.S)
def is_mul(b): return 'mul.wide.u32 t, x8, 977' in b and 'mov.b64 {a0,a1}' in b
kb=[b for b in blocks if is_mul(b) and 'kda0' in b]
ob=[b for b in blocks if is_mul(b) and 'kda0' not in b and 'mul.wide.u32 e0, a0, b0' in b]
print('K blocks',len(kb),'orig blocks',len(ob))
def stmts(b):
    b=re.sub(r'//[^\n]*','',b); b=re.sub(r'/\*.*?\*/','',b,flags=re.S)
    out=[]
    for s in b.split(';'):
        s=s.strip()
        while s.startswith('{') and not s.startswith('{a') and re.match(r'\{\s',s+' '): s=s[1:].strip()
        while s.startswith('}'): s=s[1:].strip()
        if s.endswith('}') and not re.search(r'\{[^}]*\}$',s): s=s[:-1].strip()
        if s: out.append(s)
    return out
def run(b,a,bb):
    S=stmts(b)
    ins=[];outs=[]
    for s in S:
        m=re.match(r'mov\.b64 \{([ab])(\d),\w+\}, (%rd\d+)',s)
        if m: ins.append((m.group(1),int(m.group(2)),m.group(3)))
        m=re.match(r'mov\.b64 (%rd\d+), \{z(\d),z\d\}',s)
        if m: outs.append((int(m.group(2))//2,m.group(1)))
    R={};cc=[0]
    for which,idx,reg in ins:
        v=a if which=='a' else bb
        R[reg]=(v>>(64*(idx//2)))&M64
    def val(t,bits):
        t=t.strip()
        if re.fullmatch(r'-?\d+',t): return int(t)&((1<<bits)-1)
        if t.startswith('0x'): return int(t,16)&((1<<bits)-1)
        return R[t]
    for s in S:
        if s.startswith('.reg'): continue
        op,rest=s.split(None,1)
        if op=='ld.const.u32': R[rest.split(',')[0].strip()]=0; continue
        if op=='mov.b64':
            if rest.startswith('{'):
                d=rest[:rest.index('}')+1]; src=rest[rest.index('}')+2:].strip()
                lo,hi=[t.strip() for t in d[1:-1].split(',')]; v=val(src,64); R[lo]=v&M32; R[hi]=v>>32
            else:
                d,src=[t.strip() for t in rest.split(',',1)]
                lo,hi=[t.strip() for t in src.strip()[1:-1].split(',')]; R[d]=R[lo]|(R[hi]<<32)
            continue
        args=[t.strip() for t in rest.split(',')]; d=args[0]
        if op in('mov.u32','mov.b32'): R[d]=val(args[1],32); continue
        if op=='mul.wide.u32': R[d]=val(args[1],32)*val(args[2],32); continue
        if op=='xor.b32': R[d]=val(args[1],32)^val(args[2],32); continue
        m=re.fullmatch(r'(add|addc|sub|subc)(\.cc)?\.u(32|64)',op)
        if not m: raise SystemExit('unknown op '+s)
        k,setcc,bits=m.group(1),m.group(2),int(m.group(3)); mask=(1<<bits)-1
        x,y=val(args[1],bits),val(args[2],bits)
        r={'add':x+y,'addc':x+y+cc[0],'sub':x-y,'subc':x-y-cc[0]}[k]
        R[d]=r&mask
        if setcc: cc[0]=(1 if r>>bits else 0) if k in('add','addc') else (1 if r<0 else 0)
    res=0
    for i,reg in outs: res|=R[reg]<<(64*i)
    return res
def regs(b): return re.findall(r'mov\.b64 \{[ab]\d,[ab]\d\}, (%rd\d+)',b)
def tail(b):
    b=re.sub(r'%rd\d+','%R',b); return b[b.index('.reg .u64 r0,r1,r2,r3,h0'):]
kt=tail(kb[0])
good=[b for b in ob if len(set(regs(b)))==8 and tail(b)==kt]
print('production blocks with 8 distinct inputs and the same tail text:',len(good))
p=(1<<256)-(1<<32)-977
rnd=random.Random(7)
cases=[(rnd.getrandbits(256),rnd.getrandbits(256)) for _ in range(4000)]
cases+=[(rnd.randrange(p),rnd.randrange(p)) for _ in range(2000)]
E=[0,1,(1<<128)-1,(1<<256)-1,p-1,p,1<<255,(1<<128)|((1<<128)-1),((1<<128)-1)<<128]
cases+=[(x,y) for x in E for y in E]
bad=sum(run(kb[0],x,y)!=run(good[0],x,y) for x,y in cases)
print('cases',len(cases),'K block != production block:',bad)
sys.exit(1 if bad else 0)
