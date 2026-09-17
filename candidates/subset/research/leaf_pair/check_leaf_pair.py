#!/usr/bin/env python3
"""Execute the actual fixed leaf-pair header plus a focused producer/join audit.

CPU threads emulate the block/warp collectives and OpenSSL supplies field math.
The source-checked join analysis rejects missing n=64 synchronization independent
of the CPU scheduler. Neither part is GPU execution or a performance test.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from check_candidate import BACKEND, WRAPPERS, function
from preflight import source_identity
P = (1<<256)-(1<<32)-977
MASK = (1<<64)-1


def join_audit(header):
    # This is deliberately a narrow analysis of the inspected source structure,
    # not a parser claiming to verify arbitrary CUDA. Unknown structure fails.
    required = ['for(int width=n>>2;width>0;width>>=1)',
                'if(width>32)__syncthreads();else __syncwarp();',
                'for(int width=1;width<(n>>1);width<<=1)',
                'if((width<<1)>32)__syncthreads();else __syncwarp();',
                'pair_inverse[k]=tree[k][(n>>1)+(tid>>1)];',
                'tree[k][2*node]=right[k];tree[k][2*node+1]=left[k];']
    for marker in required:
        assert header.count(marker) == 1, ('Review changed source structure', marker)
    guard = 'if(n==64)__syncthreads();'
    assert header.count(guard) in (0, 1)
    if guard in header:
        assert header.index(guard) > header.index('if((width<<1)>32)')
        assert header.index(guard) < header.index('pair_inverse[k]=')
    result = {}
    for n in [32, 64, 128, 256]:
        last_width = n//4
        block_join = 2*last_width > 32 or (n == 64 and guard in header)
        cross_warp = [(tid, tid//4) for tid in range(n) if tid//32 != (tid//4)//32]
        assert block_join or not cross_warp, ('MISSING_FINAL_JOIN', n, cross_warp[0])
        # Initial pair-publish + root-inverse joins; source thresholds account
        # for every intervening tree join, including the correction.
        widths = []
        width = n//4
        while width:
            widths.append(width)
            width //= 2
        expected = 2 + sum(w>32 for w in widths) + sum(2*w>32 for w in widths)
        expected += n == 64 and guard in header
        result[n] = {'last_width': last_width, 'cross_warp_final_consumers': len(cross_warp),
                     'final_block_join': block_join, 'expected_CTA_joins': expected}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=HERE/'candidate')
    ap.add_argument('--report', type=Path, default=HERE/'cpu-results.json')
    args = ap.parse_args()
    base = args.source.resolve()
    identity = source_identity(base)
    header = (base/'tests/gpu_epochs/tree_inverse.cuh').read_text()
    joins = join_audit(header)
    mutant = header.replace('if(n==64)__syncthreads();', '')
    assert mutant != header
    try:
        join_audit(mutant)
    except AssertionError as error:
        mutation = error.args[0]
        assert mutation[:2] == ('MISSING_FINAL_JOIN', 64)
    else:
        raise AssertionError('Missing-64-join mutation escaped')
    public = (HERE/'pr138-tree_inverse.cuh').read_text()
    try:
        join_audit(public)
    except AssertionError as error:
        assert error.args[0][:2] == ('MISSING_FINAL_JOIN', 64)
    else:
        raise AssertionError('Original public header unexpectedly passed')

    warp = '\nstatic void __syncwarp(unsigned mask=0xffffffffu){require(mask==0xffffffffu);warp_barriers[threadIdx.x/32]->wait();}\n'
    wrapper = function(WRAPPERS, 'extern "C" void check_inverse(')
    code = BACKEND + warp + header + '\n' + wrapper
    rng = random.Random(20260917138)
    batches = []
    total = 0
    with tempfile.TemporaryDirectory(prefix='qsb-leaf-pair-') as td:
        cpp, so = Path(td)/'audit.cpp', Path(td)/'audit.so'
        cpp.write_text(code)
        subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC','-pthread',
                        '-Wno-pragma-once-outside-header','-I/opt/homebrew/opt/openssl@3/include',
                        '-L/opt/homebrew/opt/openssl@3/lib',str(cpp),'-lcrypto','-o',str(so)], check=True)
        lib = C.CDLL(str(so))
        lib.check_inverse.argtypes = [C.POINTER(C.c_uint64),C.POINTER(C.c_uint64),C.c_int,C.POINTER(C.c_int)]
        for n, barriers in zip([32,64,128,256],[2,3,3,5]):
            trials = [([1]*n, 'all_identity'), ([2]*n, 'race_witness_all_two'),
                      ([P-1 if i%2 else P-2 for i in range(n)], 'near_p'),
                      ([rng.randrange(1,P) if i<n//3 else 1 for i in range(n)], 'inactive_identity_tail')]
            trials += [([rng.randrange(1,P) for _ in range(n)], 'random') for _ in range(4)]
            for values, kind in trials:
                inputs = (C.c_uint64*(5*n))(*[w for v in values for w in
                                            [*((v>>(64*i))&MASK for i in range(4)),0]])
                outputs = (C.c_uint64*(5*n))()
                counts = (C.c_int*3)()
                lib.check_inverse(inputs,outputs,n,counts)
                assert list(counts) == [3*n-3,1,barriers], (n,kind,list(counts))
                for i,v in enumerate(values):
                    got = sum(int(outputs[5*i+j])<<(64*j) for j in range(4))
                    assert got == pow(v,-1,P) and outputs[5*i+4] == 0, (n,kind,i)
                batches.append({'n':n,'kind':kind,'multiplications':counts[0],
                                'inversions':counts[1],'CTA_joins':counts[2]})
                total += n
    assert source_identity(base) == identity
    report = {'status':'PASS', **identity, 'inverse_outputs':total, 'batches':batches,
              'join_analysis':joins, 'missing_n64_join_mutation_rejected':True,
              'original_PR138_missing_n64_join_rejected':True,
              'race_witness':{'n':64,'all_leaves':2,'stale_result':8,'expected':hex((P+1)//2)},
              'projection_sha256':hashlib.sha256(code.encode()).hexdigest(),
              'audit_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'gpu_executed':False, 'limitations':__doc__}
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',total,'inverse outputs; CTA joins 2,3,3,5; missing n64 join rejected')


if __name__ == '__main__':
    main()
