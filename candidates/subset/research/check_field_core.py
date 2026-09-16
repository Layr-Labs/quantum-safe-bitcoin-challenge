#!/usr/bin/env python3
"""Audit the actual host field core and an isolated final-carry correction.

The current production header is never modified. An optional output path can
materialize a corrected header for later CUDA compilation. CPU host-branch
checks do not validate the corresponding PTX edit on a GPU.
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from check_candidate import function
from preflight import source_identity, source_files

P = (1 << 256)-(1 << 32)-977
U64 = CT.c_uint64


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Unsupported field-core source; inspect before adapting')
    return text.replace(before, after)


def corrected_core(core):
    old = r'\taddc.u32 z7, z7, 0;\n\tmov.b64 %0'
    # z0..z7, cy and m0 are declared by the existing inline PTX block.
    new = (r'\taddc.cc.u32 z7, z7, 0;\n\taddc.u32 cy, 0, 0;\n'
           r'\tmul.lo.u32 m0, cy, 977;\n\tadd.cc.u32 z0, z0, m0;\n'
           r'\taddc.cc.u32 z1, z1, cy;\n'
           + ''.join(r'\taddc.cc.u32 z'+str(i)+', z'+str(i)+r', 0;\n' for i in range(2, 7))
           + r'\taddc.u32 z7, z7, 0;\n\tmov.b64 %0')
    core = once(core, old, new)
    core = once(core,
        't=(uint64_t)z[7]+c; z[7]=(uint32_t)t; }',
        '''t=(uint64_t)z[7]+c; z[7]=(uint32_t)t; c=t>>32;
      // Final 2^256 carry represents 2^32+977 modulo p.
      t=(uint64_t)z[0]+c*977; z[0]=(uint32_t)t; uint64_t cf=t>>32;
      t=(uint64_t)z[1]+c+cf; z[1]=(uint32_t)t; cf=t>>32;
      for(int k=2;k<8;k++){t=(uint64_t)z[k]+cf;z[k]=(uint32_t)t;cf=t>>32;}
    }''')
    core = once(core, '#undef QSB_MW\n#endif\n}', '''#undef QSB_MW
#endif
    // The third fold is below 2^256; at most one subtraction canonicalizes it.
    if ((r[1]&r[2]&r[3]) == UINT64_MAX && r[0] >= 0xFFFFFFFEFFFFFC2FULL) {
        r[0] -= 0xFFFFFFFEFFFFFC2FULL;
        r[1] = r[2] = r[3] = 0;
    }
}''')
    return core


def words(x):
    return (U64*4)(*[x >> (64*i) & ((1 << 64)-1) for i in range(4)])


def corrected_square(square):
    old = r'\taddc.u32 z7, z7, 0;\n'
    new = (r'\taddc.cc.u32 z7, z7, 0;\n\t.reg .u32 cf;\n'
           r'\taddc.u32 cf, 0, 0;\n\tmul.lo.u32 m0, cf, 977;\n'
           r'\tadd.cc.u32 z0, z0, m0;\n\taddc.cc.u32 z1, z1, cf;\n'
           + ''.join(r'\taddc.cc.u32 z'+str(i)+', z'+str(i)+r', 0;\n' for i in range(2, 7))
           + r'\taddc.u32 z7, z7, 0;\n')
    square = once(square, old, new)
    start = square.index('        /* The third 977-fold')
    end = square.index('        "mov.b64 %0', start)
    square = square[:start]+'        // Capture the second-fold carry and fold it modulo p.\n'+square[end:]
    square = once(square,
        "    /* Conditional canonical subtract removed with the third fold above; the\n     * result now carries exactly _ModMultCore's contract. */",
        '''    if ((r1&r2&r3) == UINT64_MAX && r0 >= 0xFFFFFFFEFFFFFC2FULL) {
        r0 -= 0xFFFFFFFEFFFFFC2FULL;
        r1 = r2 = r3 = 0;
    }''')
    return square


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-corrected-header', type=Path)
    parser.add_argument('--output', type=Path, help='materialize a fresh isolated track copy with BOTH corrected primitives')
    args = parser.parse_args()
    identity = source_identity()
    header = (ROOT/'GPUMath.h').read_text()
    original = function(header, '__device__ __forceinline__ void _ModMultCore(')
    corrected = corrected_core(original)
    square = (ROOT/'square32.cuh').read_text()
    fixed_square = corrected_square(square)
    prefix = '#include <stdint.h>\n#define __device__\n#define __forceinline__ inline\n'
    wrapper = ('\nextern "C" void mul(uint64_t*r,const uint64_t*a,const uint64_t*b){_ModMultCore(r,a,b);}\n'
               'extern "C" void square(uint64_t*r,const uint64_t*a){qsb_square32(r,a);}\n')
    rng = random.Random(26091649)
    edges = [0, 1, 2, (1 << 128)-1, 1 << 128, (1 << 255),
             P-65537, P-2, P-1, P, P+1, (1 << 256)-1]
    cases = list(itertools.product(edges, repeat=2))
    # Deliberately target the near-p reduction region; random sampling misses it.
    cases += [(P-i, P-j) for i in (1, 65535, 65536, 65537, 100000, 1 << 31)
              for j in (1, 65535, 65536, 65537, 100000, 1 << 31)]
    cases += [(rng.getrandbits(256), rng.getrandbits(256)) for _ in range(20000)]
    results = {}
    with tempfile.TemporaryDirectory(prefix='qsb-field-source-') as directory:
        for name, core, sq in (('production_host', original, square), ('corrected_host', corrected, fixed_square)):
            cpp, so = Path(directory)/(name+'.cpp'), Path(directory)/(name+'.so')
            cpp.write_text(prefix+core+'\n'+sq+wrapper)
            subprocess.run(['c++', '-std=c++17', '-O2', '-shared', '-fPIC', str(cpp), '-o', str(so)], check=True)
            lib = CT.CDLL(str(so))
            lib.mul.argtypes = [CT.POINTER(U64)]*3
            lib.square.argtypes = [CT.POINTER(U64)]*2
            report = {'calls': 0, 'wrong_residue': 0, 'noncanonical': 0, 'examples': []}
            for a, b in cases:
                expected = a*b % P
                for mode in ('distinct', 'alias_a', 'alias_b'):
                    aa, bb, out = words(a), words(b), (U64*4)()
                    if mode == 'alias_a': out = aa
                    if mode == 'alias_b': out = bb
                    lib.mul(out, aa, bb)
                    actual = sum(out[i] << (64*i) for i in range(4))
                    wrong, noncanonical = actual % P != expected, actual >= P
                    report['calls'] += 1
                    report['wrong_residue'] += wrong
                    report['noncanonical'] += noncanonical
                    if wrong and mode == 'distinct' and len(report['examples']) < 4:
                        report['examples'].append({'a': hex(a), 'b': hex(b), 'actual': actual, 'expected': expected})
                    if name == 'corrected_host':
                        assert actual == expected, (mode, hex(a), hex(b), actual, expected)
            results[name] = report
            sr = {'calls': 0, 'wrong_residue': 0, 'noncanonical': 0}
            for a in sorted({a for a, b in cases} | {b for a, b in cases}):
                for inplace in (False, True):
                    aa = words(a)
                    out = aa if inplace else (U64*4)()
                    lib.square(out, aa)
                    actual = sum(out[i] << (64*i) for i in range(4))
                    sr['calls'] += 1
                    sr['wrong_residue'] += actual % P != a*a % P
                    sr['noncanonical'] += actual >= P
                    if name == 'corrected_host': assert actual == a*a % P
            results[name+'_square'] = sr
    assert source_identity() == identity
    fixed_header = once(header, original, corrected)
    start = fixed_header.index('// secp256k1 field multiply r = a*b mod p')
    end = fixed_header.index('__device__ __forceinline__ void _ModMultCore(', start)
    fixed_header = (fixed_header[:start] + '// Isolated correction: original 8x32 product schedule, final carry folded\n'
                    '// modulo secp256k1 p and output canonicalized in both host/PTX branches.\n'
                    '// GPU compilation and performance remain unverified.\n' + fixed_header[end:])
    if args.write_corrected_header:
        if args.write_corrected_header.exists():
            parser.error('output already exists; use a fresh path')
        if args.write_corrected_header.resolve() == (ROOT/'GPUMath.h').resolve():
            parser.error('refusing to replace production')
        args.write_corrected_header.write_text(fixed_header)
    variant_identity = None
    if args.output:
        if args.output.exists(): parser.error('output directory already exists; use a fresh path')
        args.output.mkdir(parents=True)
        for path in source_files(ROOT)+[ROOT/'COPYING']:
            target = args.output/path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        (args.output/'GPUMath.h').write_text(fixed_header)
        (args.output/'square32.cuh').write_text(fixed_square)
        variant_identity = source_identity(args.output)
    print(json.dumps({'status': 'CORRECTION_PASS_BASELINE_FINDINGS_RECORDED', **results,
                      'source_identity': identity,
                      'corrected_header_sha256': hashlib.sha256(fixed_header.encode()).hexdigest(),
                      'corrected_square_sha256': hashlib.sha256(fixed_square.encode()).hexdigest(),
                      'isolated_variant': variant_identity,
                      'validation_level': 'extracted_cxx_host_branch',
                      'cuda_compiled': False, 'gpu_executed': False,
                      'selected_for_production': False,
                      'limitations': 'Canonical corrected host results include alias tests; corresponding PTX correction has not been compiled or run. No speed comparison.'}, indent=2))


if __name__ == '__main__':
    main()
