import re,sys,collections
def load(path,want):
    s=open(path).read()
    fns=re.split(r"\n\s*Function : ",s)
    f=[x for x in fns[1:] if x.split("\n")[0].strip().startswith(want)][0]
    return [(int(m.group(1),16),m.group(2).strip()) for m in re.finditer(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;",f)]
def opc(t):
    t=re.sub(r"^@!?U?P\w+\s+","",t); return t.split()[0]
ALU=set('IADD3 LOP3 SHF LEA ISETP SEL PRMT IABS FLO POPC BREV LOP SHL SHR ICMP BMSK SGXT IMNMX VIADD PLOP3 P2R R2P CSET CSETP IADD MOV MOV32I'.split())
def cls(o):
    b=o.split('.')[0]
    if o.startswith('IMAD.WIDE'): return 'W'
    if b in('IMAD','IMUL'): return 'H'
    if b in ALU: return 'A'
    if b in('LDL','STL'): return 'SPILL'
    if b in('LDG','STG','LDS','STS','LD','ST','LDC','ATOM','ATOMG','RED','ATOMS'): return 'M'
    if b.startswith('U'): return 'U'
    return 'C'
def counts(ins,lo=0,hi=1<<40):
    c=collections.Counter()
    for a,t in ins:
        if lo<=a<=hi: c[cls(opc(t))]+=1; c['N']+=1
    return c
def loops(ins):
    L=[]
    for a,t in ins:
        o=opc(t)
        if o.startswith('BRA'):
            m=re.search(r"0x([0-9a-f]+)",t.split(o,1)[1])
            if m and int(m.group(1),16)<=a: L.append((int(m.group(1),16),a))
    return L
WIDE,HEAVY,ALUC=3.1,2.07,2.1
def cyc(c): return c['W']*WIDE+c['H']*HEAVY, c['A']*ALUC, c['N']
def report(path,tag,trips=8):
    ins=load(path,'_Z23kernel_pinning_pipelineILb1ELi0')
    L=sorted(loops(ins),key=lambda x:x[1]-x[0])
    big=[l for l in L if l[1]-l[0]>1000*16//4]
    inner=big[0]; outer=big[1] if len(big)>1 else big[0]
    ci=counts(ins,*inner); co=counts(ins,*outer); ca=counts(ins)
    dyn=collections.Counter()
    for k in set(ca)|set(ci):
        dyn[k]=ca[k]+(trips-1)*ci[k]+ (co[k]-ci[k])  # outer extra runs twice
    h,a,n=cyc(dyn); hi,ai,ni=cyc(ci)
    print(f"{tag:6s} static={ca['N']:5d} loop={ci['N']:4d} [W{ci['W']} H{ci['H']} A{ci['A']} spillLS{ci['SPILL']}] "
          f"loop heavy/alu cyc={hi:.0f}/{ai:.0f} | dyn N={dyn['N']} W={dyn['W']} A={dyn['A']} spill={dyn['SPILL']} "
          f"heavy={h:.0f} alu={a:.0f} bound={max(h,a,n):.0f}")
    return max(h,a,n)
if __name__=='__main__':
    base=None
    for p in sys.argv[1:]:
        tag=p.split('/')[-1].replace('.sass','')
        b=report(p,tag)
        if base is None: base=b
        else: print(f"        model prepare time vs first: {100*(b/base-1):+.2f}%")
