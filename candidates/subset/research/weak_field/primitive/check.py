#!/usr/bin/env python3
"""Actual host bodies, extracted PTX model, independent integer oracles/aliases."""
import ast,ctypes,hashlib,json,random,re,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));from ptx_model import Program,core,check_semantics
U=1<<256;C=(1<<32)+977;P=U-C;MASK=(1<<64)-1
source=(HERE/'qsb_weak_addsub.cuh').read_text()
def limbs(x):return [(x>>(64*i))&MASK for i in range(4)]
def integer(v):return sum(int(x)<<(64*i) for i,x in enumerate(v))
def weak(a,b,op):
    if op=='add':
        t=a+b;r=t%U+(t//U)*C;return r%U+(r//U)*C
    d=a-b;r=d%U-int(d<0)*C;return r%U-int(r<0)*C

def extract(op):
    body=core.function(source,'QSB_WEAK_INLINE void qsb_weak_'+op+'(')
    asm=body.split('asm(',1)[1].split(': "=l"',1)[0]
    return ''.join(ast.literal_eval(s) for s in re.findall(r'"(?:[^"\\]|\\.)*"',asm))
texts={op:extract(op) for op in ['add','sub','normalize']}
programs={op:Program(t) for op,t in texts.items()}
probe_count=check_semantics()
cpp='''#include "qsb_weak_addsub.cuh"
extern "C" {
void add(uint64_t*r,const uint64_t*a,const uint64_t*b){qsb_weak_add(r,a,b);}
void sub(uint64_t*r,const uint64_t*a,const uint64_t*b){qsb_weak_sub(r,a,b);}
void norm(uint64_t*r,const uint64_t*a){qsb_weak_normalize(r,a);}
void norm_inplace(uint64_t*r){qsb_weak_normalize(r);}
int zero(const uint64_t*a){return qsb_weak_is_zero(a);}
}
'''
build=HERE/'test-artifacts';build.mkdir(exist_ok=True)
(build/'host.cpp').write_text(cpp)
cmd=['c++','-std=c++17','-O2','-shared','-fPIC','-I',str(HERE),str(build/'host.cpp'),'-o',str(build/'host.dylib')]
subprocess.run(cmd,check=True,capture_output=True,text=True)
lib=ctypes.CDLL(str(build/'host.dylib'));A=ctypes.c_uint64*4;ptr=ctypes.POINTER(ctypes.c_uint64)
for name in ['add','sub']:getattr(lib,name).argtypes=[ptr,ptr,ptr]
lib.norm.argtypes=[ptr,ptr];lib.norm_inplace.argtypes=[ptr];lib.zero.argtypes=[ptr];lib.zero.restype=ctypes.c_int
edges={0,1,2,C-1,C,C+1,P-1,P,P+1,U-2,U-1}
for bit in range(0,256,32):
    for delta in [-1,0,1]:
        if 0<=(1<<bit)+delta<U:edges.add((1<<bit)+delta)
rng=random.Random(0x57123);vectors=[(a,b) for a in sorted(edges) for b in sorted(edges)]+[(rng.getrandbits(256),rng.getrandbits(256)) for _ in range(12000)]
folds={op:[0,0] for op in ['add','sub']}
for a,b in vectors:
    for op in ['add','sub']:
        expected=weak(a,b,op);assert 0<=expected<U and expected%P==(a+b if op=='add' else a-b)%P
        got=A();getattr(lib,op)(got,A(*limbs(a)),A(*limbs(b)));assert integer(got)==expected,(op,a,b,'host')
        assert integer(programs[op].run(limbs(a)+limbs(b)))==expected,(op,a,b,'ptx')
        if op=='add':f1=(a+b)//U;f2=int((a+b)%U+f1*C>=U)
        else:f1=int(a<b);f2=int((a-b)%U-f1*C<0)
        folds[op][0]+=f1;folds[op][1]+=f2
    got=A();lib.norm(got,A(*limbs(a)));assert integer(got)==a%P
    assert integer(programs['normalize'].run(limbs(a)))==a%P
    inplace=A(*limbs(a));lib.norm_inplace(inplace);assert integer(inplace)==a%P
    assert bool(lib.zero(A(*limbs(a))))==(a%P==0)
# All relative aligned-limb overlaps among both inputs and output, including
# a/b overlap and exact three-way alias. Expected inputs are snapshotted first.
aliases=0;canary_words=0
for seed in range(32):
    initial=[rng.getrandbits(64) for _ in range(20)]
    for ai in range(2,7):
      for bi in range(2,7):
       for ri in range(0,11):
        for op in ['add','sub']:
            buf=(ctypes.c_uint64*20)(*initial);before=list(buf)
            a=integer(before[ai:ai+4]);b=integer(before[bi:bi+4])
            at=lambda i:ctypes.cast(ctypes.byref(buf,8*i),ptr)
            getattr(lib,op)(at(ri),at(ai),at(bi));assert integer(list(buf)[ri:ri+4])==weak(a,b,op)
            for j in range(20):
                if not ri<=j<ri+4:assert buf[j]==before[j];canary_words+=1
            aliases+=1
for a in sorted(edges):
 for delta in range(-3,4):
    buf=(ctypes.c_uint64*12)(*[0xabcdef0123456789]*12)
    for i,v in enumerate(limbs(a)):buf[4+i]=v
    before=list(buf);at=lambda i:ctypes.cast(ctypes.byref(buf,8*i),ptr)
    lib.norm(at(4+delta),at(4));assert integer(list(buf)[4+delta:8+delta])==a%P
    for j in range(12):
        if not 4+delta<=j<8+delta:assert buf[j]==before[j];canary_words+=1
    aliases+=1
# PTX-only regressions prove both rare second folds and first full-width folds.
mutations={
 'add_drop_second_fold':('add',texts['add'].replace('add.cc.u32 a0, a0, lo;\naddc.u32 a1, a1, hi;','mov.u32 a0, a0;\nmov.u32 a1, a1;'),U-1,U-1),
 'sub_drop_second_fold':('sub',texts['sub'].replace('sub.cc.u32 a0, a0, lo;\nsubc.u32 a1, a1, hi;','mov.u32 a0, a0;\nmov.u32 a1, a1;'),0,U-1),
 'add_drop_high_first_fold':('add',texts['add'].replace('addc.cc.u32 a2, a2, 0;','addc.u32 a2, a2, 0;'),U-1,(1<<96)-1),
 'sub_drop_high_first_fold':('sub',texts['sub'].replace('subc.cc.u32 a2, a2, 0;','subc.u32 a2, a2, 0;'),0,U-(1<<64)),
 'normalize_drop_carry':('normalize',texts['normalize'].replace('addc.u32 f, 0, 0;','mov.u32 f, 0;'),P,0),
}
negative={}
for name,(op,text,a,b) in mutations.items():
    assert text!=texts[op];expected=a%P if op=='normalize' else weak(a,b,op)
    got=integer(Program(text).run(limbs(a)+([] if op=='normalize' else limbs(b))))
    # If a first-fold carry witness did not distinguish it, search adversarial
    # deterministic input corpus; report the found witness, never pass silently.
    if got==expected:
      for a,b in vectors:
        expected=a%P if op=='normalize' else weak(a,b,op);got=integer(Program(text).run(limbs(a)+([] if op=='normalize' else limbs(b))))
        if got!=expected:break
    assert got!=expected,name
    negative[name]={'a':hex(a),'b':hex(b),'expected':hex(expected),'mutant':hex(got),'residue_wrong':got%P!=expected%P}
assert folds['add'][1] and folds['sub'][1]
host_mutations={
 'host_add_second_fold':('low+=carry*UINT64_C(0x1000003d1);','low+=0;','add',U-1,U-1),
 'host_sub_second_fold':('low-=borrow*UINT64_C(0x1000003d1);','low-=0;','sub',0,U-1),
 'host_normalize_condition':('if(carry)for(unsigned i=0;i<8;i++)x[i]=t[i];','if(!carry)for(unsigned i=0;i<8;i++)x[i]=t[i];','norm',P,0),
 'host_zero_omits_p':('(a0==UINT64_C(0xfffffffefffffc2f)', '(false && a0==UINT64_C(0xfffffffefffffc2f)','zero',P,0),
}
host_negative={}
for name,(old,new,op,a,b) in host_mutations.items():
    assert source.count(old)==1
    folder=build/name;folder.mkdir(exist_ok=True)
    mutated=source.replace(old,new);(folder/'qsb_weak_addsub.cuh').write_text(mutated)
    target=folder/'mutant.dylib'
    subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC','-I',str(folder),str(build/'host.cpp'),'-o',str(target)],check=True,capture_output=True,text=True)
    mutant=ctypes.CDLL(str(target));got=A()
    if op=='zero':
        mutant.zero.argtypes=[ptr];mutant.zero.restype=ctypes.c_int
        value=int(mutant.zero(A(*limbs(a))));expected=int(a%P==0)
    elif op=='norm':
        mutant.norm.argtypes=[ptr,ptr];mutant.norm(got,A(*limbs(a)));value=integer(got);expected=a%P
    else:
        f=getattr(mutant,op);f.argtypes=[ptr,ptr,ptr];f(got,A(*limbs(a)),A(*limbs(b)));value=integer(got);expected=weak(a,b,op)
    assert value!=expected,name
    host_negative[name]={'compilation_completed':True,'a':hex(a),'b':hex(b),'expected':hex(expected),'mutant':hex(value),'mutated_header_sha256':hashlib.sha256(mutated.encode()).hexdigest()}
report={'status':'PASS_HOST_AND_EXTRACTED_PTX_SEMANTICS','header_sha256':hashlib.sha256(source.encode()).hexdigest(),'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'base_ptx_model_sha256':hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),'extension_sha256':hashlib.sha256((HERE/'ptx_model.py').read_bytes()).hexdigest(),'ptx_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in texts.items()},'binary_pair_cases':len(vectors),'host_and_ptx_binary_operations_each':2*len(vectors),'host_and_ptx_normalize_cases_each':len(vectors),'host_zero_cases':len(vectors),'host_inplace_normalize_cases':len(vectors),'host_overlap_cases':aliases,'canary_words':canary_words,'first_second_fold_occurrences':folds,'borrow_semantic_probe_cases':probe_count,'negative_controls':negative,'compiled_host_negative_controls':host_negative,'gpu_executed':False,'native_cuda_compiled':False,'limits':'CPU actual host C++ plus source-extracted straight-line PTX semantic model; not native CUDA execution, integration, timing or register allocation. Arbitrary overlaps are aligned uint64 limb ranges; no unaligned-pointer contract.'}
(build/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
