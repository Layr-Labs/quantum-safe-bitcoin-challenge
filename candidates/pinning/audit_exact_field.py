#!/usr/bin/env python3
"""Check literal PTX field instruction streams and host branches on CPU.

The deliberately small PTX-to-C translator models only instructions present
in these three field routines. This is instruction emulation, not CUDA device
execution. Python integers are the independent arithmetic oracle.
"""
import argparse
import ast
import ctypes as ct
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import tempfile

def definition(source, signature):
    start=source.index(signature); opening=source.index('{',start)
    depth=1; i=opening+1
    while depth:
        depth+=(source[i]=='{')-(source[i]=='}'); i+=1
    return source[start:i]


ROOT=Path(__file__).resolve().parents[2]
P=(1<<256)-(1<<32)-977
MASK=(1<<256)-1


def asm_text(func):
    text=func[func.index('asm(')+4:]
    text=text[:text.index(': "=l"')]
    return ''.join(ast.literal_eval(x) for x in re.findall(r'"(?:[^"\\]|\\.)*"',text))


def translate(ptx,name,square=False):
    declarations=[]
    for width,names in re.findall(r'\.reg\s+\.(u32|u64|pred)\s+([^;]+);',ptx):
        typ={'u32':'uint32_t','u64':'uint64_t','pred':'bool'}[width]
        declarations.append(typ+' '+names+';')
    ptx=re.sub(r'\.reg\s+\.[a-z0-9]+\s+[^;]+;','',ptx).strip()
    assert ptx.startswith('{') and ptx.endswith('}')
    ptx=re.sub(r'%(\d+)',r'arg\1',ptx[1:-1])
    lines=['extern "C" void '+name+'(uint64_t *out,const uint64_t *a,const uint64_t *b) {',
           'unsigned carry_bit=0; __uint128_t wide;',*declarations,
           'uint64_t arg0,arg1,arg2,arg3;',
           'uint64_t arg4=a[0],arg5=a[1],arg6=a[2],arg7=a[3];']
    if not square:lines.append('uint64_t arg8=b[0],arg9=b[1],arg10=b[2],arg11=b[3];')
    for instruction in ptx.split(';'):
        instruction=instruction.strip()
        if not instruction:continue
        op,operands=instruction.split(None,1)
        if op=='mov.b64':
            m=re.fullmatch(r'\{(\w+),\s*(\w+)\},\s*(\w+)',operands)
            if m:
                lo,hi,src=m.groups();lines.append(f'{lo}=(uint32_t){src}; {hi}=(uint32_t)({src}>>32);');continue
            m=re.fullmatch(r'(\w+),\s*\{(\w+),\s*(\w+)\}',operands)
            assert m,instruction
            dst,lo,hi=m.groups();lines.append(f'{dst}=(uint64_t){lo}|((uint64_t){hi}<<32);');continue
        args=[s.strip() for s in operands.split(',')]
        dst,*src=args
        if op in ['cvt.u64.u32','mov.u32']:
            lines.append(f'{dst}={src[0]};')
        elif op.startswith('add'):
            width=int(op[-2:]);assert width in [32,64]
            extra='+carry_bit' if op.startswith('addc.') else ''
            lines.append(f'wide=(__uint128_t){src[0]}+{src[1]}{extra}; {dst}=(uint{width}_t)wide;')
            if '.cc.' in op:lines.append(f'carry_bit=(unsigned)(wide>>{width});')
        elif op in ['mul.wide.u32','mul.lo.u32']:
            typ='uint64_t' if op=='mul.wide.u32' else 'uint32_t'
            lines.append(f'{dst}=({typ})((uint64_t){src[0]}*{src[1]});')
        elif op=='mad.lo.u32':
            lines.append(f'{dst}=(uint32_t)((uint64_t){src[0]}*{src[1]}+{src[2]});')
        elif op=='shf.l.wrap.b32':
            lo,hi,n=src
            lines.append(f'{dst}=(uint32_t)((((uint64_t){hi}<<32)|{lo})<<(({n})&31)>>32);')
        else:raise AssertionError(instruction)
    lines+=['out[0]=arg0;out[1]=arg1;out[2]=arg2;out[3]=arg3;','}']
    return '\n'.join(lines)


def build_library(repo,tmp):
    track=repo/'candidates/pinning';math=(track/'GPUMath.h').read_text();source=(track/'pinning.cu').read_text()
    mul=definition(math,'__device__ __forceinline__ void _ModMultCore')
    sqr=definition(math,'__device__ __forceinline__ void _ModSqr')
    tree=definition(source,'__device__ __forceinline__ void qsb_field_mul')
    code='#include <stdint.h>\n#define __device__\n#define __forceinline__ inline\n'
    code+=mul+'\n'+sqr+'\n'
    code+='extern "C" void host_mul(uint64_t*r,const uint64_t*a,const uint64_t*b){_ModMultCore(r,a,b);}\n'
    code+='extern "C" void host_sqr(uint64_t*r,const uint64_t*a,const uint64_t*b){_ModSqr(r,a);}\n'
    code+=translate_field(mul,'ptx_mul')+'\n'
    code+=translate_field(sqr,'ptx_sqr',True)+'\n'
    code+=translate(asm_text(tree),'ptx_tree')+'\n'
    cpp=tmp/'field.cpp';cpp.write_text(code);lib=tmp/'field.so'
    subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-fsanitize=undefined',
                    '-fno-sanitize-recover=all',str(cpp),'-o',str(lib)],check=True)
    return ct.CDLL(str(lib)),code


def translate_field(func,name,square=False):
    """Emulate literal assembly followed by the actual shared C++ postlude.

    Do not insert an oracle reduction: the production conditional subtraction
    after #endif is part of the function being tested, on both host and GPU.
    """
    code=translate(asm_text(func),name,square)
    postlude=func.rsplit('#endif',1)[1].rsplit('}',1)[0]
    assert '#' not in postlude
    postlude=re.sub(r'\br\b','out',postlude)
    return code[:-1]+postlude+'\n}'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',default=str(ROOT));ap.add_argument('--output',type=Path)
    ap.add_argument('--expect-broken',action='store_true');args=ap.parse_args()
    repo=(ROOT/args.repo).resolve()
    with tempfile.TemporaryDirectory(prefix='qsb-ptx-field-') as path:
        lib,code=build_library(repo,Path(path))
        functions={name:getattr(lib,name) for name in ['host_mul','host_sqr','ptx_mul','ptx_sqr','ptx_tree']}
        wordptr=ct.POINTER(ct.c_uint64)
        for f in functions.values():f.argtypes=[wordptr,wordptr,wordptr]
        rng=random.Random(2026091761)
        edges=sorted(set([0,1,2,3,P-1,P,P+1,MASK]+[P-d for d in [7,31,255,977,65535,65536,65537,65538,(1<<20)-1,1<<32]]))
        cases=[(a,b) for a in edges for b in edges]
        cases += [(1<<k,(1<<k)-1) for k in range(256)]
        cases += [(P-rng.randrange(1<<24),P-rng.randrange(1<<24)) for _ in range(10000)]
        cases += [(rng.randrange(1<<256),rng.randrange(1<<256)) for _ in range(20000)]
        cases += [(rng.randrange(P),rng.randrange(P)) for _ in range(20000)]
        failures={name:0 for name in functions};first={};aliases=0
        arr=lambda x:(ct.c_uint64*4)(*[(x>>(64*i))&((1<<64)-1) for i in range(4)])
        integer=lambda x:sum(int(x[i])<<(64*i) for i in range(4))
        for i,(a,b) in enumerate(cases):
            aa,bb=arr(a),arr(b);results={}
            for name,f in functions.items():
                out=arr(0);f(out,aa,bb);got=integer(out)
                expected=a*(a if name.endswith('sqr') else b)%P
                results[name]=got
                # Tree-internal products retain a residue contract; its root
                # and leaf consumers normalize separately. Point products
                # must be canonical before add/sub and compressed-key output.
                if (got%P if name=='ptx_tree' else got) != expected:
                    failures[name]+=1
                    first.setdefault(name,{'a':hex(a),'b':hex(b),'got':hex(got),'expected':hex(expected)})
                if i<500:
                    alias=arr(a);f(alias,alias,bb);assert integer(alias)==got,(name,'alias',i);aliases+=1
            assert results['host_mul']==results['ptx_mul'],('host/ptx mul',i)
            assert results['host_sqr']==results['ptx_sqr'],('host/ptx sqr',i)
        report={'repo':str(repo),'cases':len(cases),'results':len(cases)*len(functions),
                'alias_checks':aliases,'failures':failures,'first_failures':first,
                'gpu_executed':False,'backend':'literal PTX + actual C++ normalization postlude; actual host branches',
                'point_contract':'canonical equality with Python integer oracle',
                'tree_contract':'equality modulo p; root/leaf normalization is separate',
                'source_sha256':{f:hashlib.sha256((repo/'candidates/pinning'/f).read_bytes()).hexdigest()
                                  for f in ['GPUMath.h','pinning.cu']}}
        if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        if args.expect_broken:
            assert failures['ptx_mul'] and failures['ptx_sqr'] and not failures['ptx_tree']
        else:assert not any(failures.values()),failures


if __name__=='__main__':main()
