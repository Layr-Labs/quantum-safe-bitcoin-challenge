#!/usr/bin/env python3
"""Check PR24 host primitives and model actual inline PTX, with a carry repair.

Creates an isolated corrected copy only when --output is specified. Never
changes the production candidate. PTX interpretation does not validate CUDA
compilation, compiler constraints, GPU execution, resources, or throughput.
"""
import argparse
import ctypes as CT
import hashlib
import itertools
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
sys.dont_write_bytecode = True
from ptx_field_model import Program, check_semantics, extract_ptx, function

HERE = Path(__file__).resolve().parent
P = (1 << 256)-(1 << 32)-977
C = (1 << 32)+977
MASK64 = (1 << 64)-1
U64 = CT.c_uint64
MUL_SIG = '__device__ __forceinline__ void _ModMultCore('
SQR_SIG = '__device__ __forceinline__ void _ModSqr('


def once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def repair(body, square):
    # If the second fold overflows, its low remainder is < C*C. The third
    # fold is < C*C+C < 2^96, so it cannot carry out of z2. With no overflow,
    # adding zero leaves all limbs unchanged. Five extra PTX instructions.
    old = 'addc.u32 z7, z7, 0;'
    end = (r'\n\t' if square else r'\n')
    new = ('addc.cc.u32 z7, z7, 0;' + end +
        '.reg .u32 cf;' + end + 'addc.u32 cf, 0, 0;' + end +
        'mul.lo.u32 m0, cf, 977;' + end + 'add.cc.u32 z0, z0, m0;' + end +
        'addc.cc.u32 z1, z1, cf;' + end + 'addc.u32 z2, z2, 0;')
    body = once(body, old, new)
    var = 'tv' if square else 't'
    old = f'{var}=(uint64_t)z[7]+c; z[7]=(uint32_t){var}; }}'
    new = f'''{var}=(uint64_t)z[7]+c; z[7]=(uint32_t){var}; c={var}>>32;
      {var}=(uint64_t)z[0]+c*977; z[0]=(uint32_t){var}; uint64_t cf={var}>>32;
      {var}=(uint64_t)z[1]+c+cf; z[1]=(uint32_t){var}; cf={var}>>32;
      {var}=(uint64_t)z[2]+cf; z[2]=(uint32_t){var};
    }}'''
    body = once(body, old, new)
    body = once(body, '#endif\n}', '''#endif
    // Canonical output is required when a coordinate's parity is consumed.
    if ((r[1]&r[2]&r[3]) == UINT64_MAX && r[0] >= 0xFFFFFFFEFFFFFC2FULL) {
        r[0] -= 0xFFFFFFFEFFFFFC2FULL;
        r[1] = r[2] = r[3] = 0;
    }
}''')
    return body


def words(x):
    return [x >> (64*i) & MASK64 for i in range(4)]


def integer(values):
    return sum(int(v) << (64*i) for i, v in enumerate(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    check_semantics()
    root = HERE/'pr24_reference'
    provenance = json.loads((root/'PROVENANCE.json').read_text())
    for name, digest in provenance['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    header = (root/'GPUMath.h').read_text()
    original = [function(header, sig) for sig in (MUL_SIG, SQR_SIG)]
    fixed = [repair(body, square) for body, square in zip(original, (False, True))]
    assert C*C+C < (1 << 96)
    rng = random.Random(260916622)
    edges = [0,1,2,(1 << 128)-1,1 << 128,1 << 255,P-65537,P-2,P-1,P,P+1,(1 << 256)-1]
    cases = list(itertools.product(edges, repeat=2))
    cases += [(P-i,P-j) for i in (1,65535,65536,65537,100000,1 << 31)
              for j in (1,65535,65536,65537,100000,1 << 31)]
    cases += [(rng.getrandbits(256),rng.getrandbits(256)) for _ in range(20000)]
    ptx_cases = cases[:180] + cases[180:2180]
    report = {'provenance':provenance,'validation_level':'CPU host and straight-line PTX semantic model',
              'gpu_executed':False,'production_modified':False,
              'ptx_results_scope':'ASM output before the explicit C++ canonical subtraction',
              'variants':{}}
    with tempfile.TemporaryDirectory(prefix='qsb-pinning-field-') as tmp:
        for name, bodies in (('original',original),('corrected',fixed)):
            cpp, so = Path(tmp)/(name+'.cpp'), Path(tmp)/(name+'.so')
            cpp.write_text('#include <stdint.h>\n#define __device__\n#define __forceinline__ inline\n'+
                '\n'.join(bodies)+
                '\nextern "C" void mul(uint64_t*r,const uint64_t*a,const uint64_t*b){_ModMultCore(r,a,b);}\n'+
                'extern "C" void square(uint64_t*r,const uint64_t*a){_ModSqr(r,a);}\n')
            subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC',str(cpp),'-o',str(so)],check=True)
            lib = CT.CDLL(str(so))
            lib.mul.argtypes = [CT.POINTER(U64)]*3
            lib.square.argtypes = [CT.POINTER(U64)]*2
            counts = {op:{'calls':0,'wrong_residue':0,'noncanonical':0}
                      for op in ('host_mul','host_square','ptx_mul','ptx_square')}
            programs = [Program(extract_ptx(body)) for body in bodies]
            for index,(a,b) in enumerate(cases):
                for opname in ('mul','square'):
                    expect = a*(b if opname=='mul' else a) % P
                    for alias in range(3 if opname=='mul' else 2):
                        aa,bb,out=(U64*4)(*words(a)),(U64*4)(*words(b)),(U64*4)()
                        if alias == 1:out=aa
                        if alias == 2:out=bb
                        if opname=='mul':lib.mul(out,aa,bb)
                        else:lib.square(out,aa)
                        value=integer(out)
                        count=counts['host_'+opname];count['calls']+=1
                        count['wrong_residue']+=value%P!=expect
                        count['noncanonical']+=value>=P
                        if name=='corrected':assert value==expect,(opname,alias,a,b,value,expect)
                    if index<len(ptx_cases):
                        inp=words(a)+(words(b) if opname=='mul' else [])
                        raw=integer(programs[opname=='square'].run(inp))
                        # The model executes the ASM body, before the explicit
                        # corrected C++ canonical-subtraction boundary.
                        count=counts['ptx_'+opname];count['calls']+=1
                        count['wrong_residue']+=raw%P!=expect
                        count['noncanonical']+=raw>=P
                        if name=='corrected':assert raw%P==expect,(opname,a,b,raw,expect)
                        if name=='original':assert raw==value,(opname,'host/PTX disagreement',a,b,raw,value)
            if name=='original':
                assert counts['host_mul']['wrong_residue']>0 and counts['host_square']['wrong_residue']>0
                assert counts['ptx_mul']['wrong_residue']>0 and counts['ptx_square']['wrong_residue']>0
            counts['ptx_instruction_counts']=[len(p.ops) for p in programs]
            report['variants'][name]=counts
    # Reject the tempting but incorrect patch that captures CC without first
    # making the high second-fold addition write carry-out. This witness has
    # carry-in to z7 but no carry-out; the stale bit adds an erroneous C.
    a,b=P-1,P-(1 << 224)
    correct_ptx=extract_ptx(fixed[0])
    mutant=once(correct_ptx,'addc.cc.u32 z7, z7, 0;','addc.u32 z7, z7, 0;')
    expected=a*b%P
    assert integer(Program(correct_ptx).run(words(a)+words(b)))%P==expected
    wrong=integer(Program(mutant).run(words(a)+words(b)))%P
    assert wrong!=expected
    report['carry_flag_mutation']={'rejected':True,'a':hex(a),'b':hex(b),
                                   'expected':hex(expected),'mutant':hex(wrong)}
    corrected = header
    for old,new in zip(original,fixed):corrected=once(corrected,old,new)
    # Correct stale claims adjacent to the imported primitive implementations.
    a=corrected.index('// secp256k1 field multiply');b=corrected.index(MUL_SIG,a)
    corrected=corrected[:a]+'''// 8x32 even/odd field product, final overflow folded and canonicalized.
// Carry repair checked with host source and a PTX semantic model, not a GPU.
'''+corrected[b:]
    corrected=corrected.replace('// [0,2^256), final 2^256 carry dropped. Device: inline PTX; host: __uint128_t C-ref of',
        '// Canonical output after final carry repair. Device: inline PTX; host: __uint128_t C-ref of')
    report['corrected_header_sha256']=hashlib.sha256(corrected.encode()).hexdigest()
    report['status']='PASS'
    if args.output:
        args.output.mkdir(parents=True,exist_ok=False)
        for name in provenance['files']:shutil.copyfile(root/name,args.output/name)
        (args.output/'GPUMath.h').write_text(corrected)
        report['materialized_files']={name:hashlib.sha256((args.output/name).read_bytes()).hexdigest()
                                      for name in provenance['files']}
    result=json.dumps(report,indent=2)
    if args.report:args.report.write_text(result+'\n')
    print(result)


if __name__=='__main__':main()
