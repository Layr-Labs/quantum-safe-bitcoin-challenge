#!/usr/bin/env python3
"""Source-extracted PTX audit for the exact split-word seeded multiplier.

The native CPU translation executes each selected PTX instruction, preserving
register widths and CC. The existing Python PTX interpreter independently
cross-checks it on directed inputs. Every result is compared with Python's
arbitrary-precision a*b+c. This is not GPU execution or a throughput test.
"""
from pathlib import Path
import argparse,ast,ctypes,hashlib,itertools,json,random,re,subprocess,sys,tempfile
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from ptx_field_model import Program,check_semantics,extract_ptx
from prepare import SOURCE,variant,baseline,BASELINE_COMMIT

B=1<<256
MASK=(1<<64)-1

def ptx(data,mode):
    proc=subprocess.run(['g++','-E','-P','-x','c++',
        '-D__CUDA_ARCH__=890',f'-DQSB_MAC_HALF_SEED={mode}','-'],
        input=data,text=True,check=True,capture_output=True)
    return extract_ptx(proc.stdout)

def product(data):
    marker='.reg .u64 r0,r1,r2,r3,h0,h1,h2,h3'
    assert data.count(marker)==1
    prefix=data[:data.index(marker)]
    for i in range(8):prefix+=f'mov.b64 %{i}, {{x{2*i},x{2*i+1}}};\n'
    return Program(prefix+'}')

def cpp_program(program,name):
    def reg(v):return re.sub(r'%([0-9]+)',r'operand\1',v)
    def read(v):
        if v.startswith('{'):
            lo,hi=(x.strip() for x in v[1:-1].split(','))
            return f'((uint64_t){read(lo)}|((uint64_t){read(hi)}<<32))'
        return reg(v)
    def write(v,expr):
        if v.startswith('{'):
            lo,hi=(x.strip() for x in v[1:-1].split(','))
            return f'{lo}=(uint32_t)({expr});{hi}=(uint32_t)(({expr})>>32);'
        return f'{reg(v)}=({expr});'
    lines=[f'extern "C" void {name}(const uint64_t *input,uint64_t *output){{',
           'unsigned __int128 value;uint32_t carry=0;']
    lines += [f'uint{bits}_t {v}=0;' for v,bits in program.widths.items()]
    lines += [f'uint64_t operand{i}='+('0;' if i<4 else f'input[{i-4}];') for i in range(16)]
    for opcode,args in program.ops:
        dst,*src=args;vals=[read(v) for v in src]
        if opcode.startswith(('mov.','cvt.')):expr=vals[0]
        elif opcode.startswith('mul.'):
            assert opcode=='mul.wide.u32'
            expr=f'(unsigned __int128){vals[0]}*{vals[1]}'
        elif opcode.startswith(('mad.','madc.')):
            expr=f'((unsigned __int128){vals[0]}*{vals[1]})'
            if '.hi.' in opcode:expr='('+expr+'>>32)'
            elif '.lo.' in opcode:expr='('+expr+'&0xffffffff)'
            expr+='+'+vals[2]
            if opcode.startswith('madc.'):expr+='+carry'
        elif opcode.startswith('add'):
            expr=f'(unsigned __int128){vals[0]}+{vals[1]}'
            if opcode.startswith('addc.'):expr+='+carry'
        else:raise AssertionError(opcode)
        lines.append('value='+expr+';')
        if '.cc.' in opcode:
            width=int(opcode.rsplit('u',1)[1]);lines.append(f'carry=(uint32_t)(value>>{width});')
        lines.append(write(dst,'value'))
    lines += [f'output[{i}]=operand{i};' for i in range(8)]
    return '\n'.join(lines+['}'])

def words(v):return [(v>>(64*i))&MASK for i in range(4)]
def integer(words):return sum(int(v)<<(64*i) for i,v in enumerate(words))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--narrow',action='store_true')
    parser.add_argument('--production',action='store_true',help='Audit actual selected header against generated candidate')
    args=parser.parse_args()
    check_semantics()
    original=baseline();generated=variant(original,args.narrow)
    if args.production:
        changed=(SOURCE/'negative_y_mac.cuh').read_text()
        assert changed==generated, 'Selected production differs from the audited candidate'
    else:changed=generated
    control_ptx=ptx(original,0);off_ptx=ptx(changed,0);candidate_ptx=ptx(changed,1)
    assert control_ptx==off_ptx
    tail='.reg .u64 r0,r1,r2,r3,h0,h1,h2,h3'
    assert control_ptx[control_ptx.index(tail):]==candidate_ptx[candidate_ptx.index(tail):]
    control=product(control_ptx);candidate=product(candidate_ptx)
    assert (2**32-1)**2+(2**32-1)<2**64
    code='#include <stdint.h>\n'+cpp_program(control,'control')+cpp_program(candidate,'candidate')
    edges=[0,1,2,B-1,B-2,B-(1<<32)-977]
    cases=list(itertools.product(edges,repeat=3))
    for bit in range(0,256,32):
        v=(1<<bit)
        for delta in [-1,0,1]:
            if 0<=v+delta<B:
                cases.extend([(B-1,B-1,v+delta),(v+delta,B-1,B-1),
                              (B-1,v+delta,B-1),(0,0,v+delta)])
    # Alternating/all-high limb patterns exercise both first-row carry chains.
    patterns=[sum(word<<(32*i) for i in range(8) if mask&(1<<i))
              for word in [1,0x7fffffff,0x80000000,0xffffffff]
              for mask in [1,2,3,0x55,0xaa,0xff]]
    cases.extend((a,B-1,c) for a in patterns for c in patterns)
    directed=len(cases)
    rng=random.Random(0x4d41435eed)
    cases.extend(tuple(rng.getrandbits(256) for _ in range(3)) for _ in range(100000))
    python_cases=min(directed,512)
    with tempfile.TemporaryDirectory(prefix='qsb-mac-oracle-') as temp:
        temp=Path(temp);(temp/'audit.cpp').write_text(code)
        subprocess.run(['g++','-O2','-shared','-fPIC','-fsanitize=undefined',
                        '-fno-sanitize-recover=all',str(temp/'audit.cpp'),'-o',str(temp/'audit.so')],check=True)
        lib=ctypes.CDLL(str(temp/'audit.so'))
        functions=[lib.control,lib.candidate]
        for function in functions:
            function.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_uint64)]
            function.restype=None
        for index,(a,b,c) in enumerate(cases):
            inputs=words(a)+words(b)+words(c)
            inp=(ctypes.c_uint64*12)(*inputs)
            expected=a*b+c
            for function in functions:
                out=(ctypes.c_uint64*8)();function(inp,out)
                assert integer(out)==expected,(index,a,b,c,function.__name__)
            if index<python_cases:
                assert integer(control.run(inputs,output_count=8))==expected
                assert integer(candidate.run(inputs,output_count=8))==expected
    if args.narrow:
        broken=product(candidate_ptx.replace('mad.lo.cc.u32 seeded_lo, a0, b1, c1;',
                                            'mad.lo.cc.u32 seeded_lo, a0, b1, 0;'))
    else:
        broken=product(candidate_ptx.replace('mad.wide.u32 o0, a0, b1, bias;',
                                            'mul.wide.u32 o0, a0, b1;'))
    assert integer(broken.run(words(0)+words(0)+words(1<<32),output_count=8))!=1<<32
    result={
        'passed':True,'production_verified':args.production,'baseline_commit':BASELINE_COMMIT,'narrow_mad':args.narrow,'random_cases':100000,'directed_cases':directed,
        'total_full_product_cases':len(cases),'native_comparisons':2*len(cases),
        'independent_python_ptx_cases_per_mode':python_cases,
        'disabled_ptx_identical_to_original':True,'reduction_tail_identical':True,
        'overflow_bound':'(2^32-1)^2+(2^32-1)=2^64-2^32 < 2^64',
        'mutation_rejected':'omitted high-half seed in first odd product',
        'original_sha256':hashlib.sha256(original.encode()).hexdigest(),
        'candidate_sha256':hashlib.sha256(changed.encode()).hexdigest(),
        'control_product_ops':len(control.ops),'candidate_product_ops':len(candidate.ops),
        'sanitizer':'undefined-behavior','gpu_executed':False,
        'scope':'Exact full a*b+c product equivalence; inherited reduction unchanged.'}
    (HERE/('production-oracle-results.json' if args.production else 'narrow-oracle-results.json' if args.narrow else 'oracle-results.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
