"""CPU oracle for the actual conditional small-field PTX, including both paths.

Use beside audit_small_ptx.py and audit_ptx_model.py. This checks arithmetic
semantics on diagnostic inputs; native correctness and throughput are separate.
"""
import argparse
import hashlib
import json
from pathlib import Path

from audit_small_ptx import boundary_pairs, evaluate
from audit_ptx_model import extract_ptx, function


def audit(header):
    text = header.read_text()
    pairs = boundary_pairs()
    p = (1 << 256) - (1 << 32) - 977
    results = {}
    for name in ('_ModAdd256', '_ModSub256', '_ModAddLazy'):
        ptx = extract_ptx(function(text, '__device__ __forceinline__ void ' + name + '('))
        if name == '_ModAdd256':
            assert 'bra ' not in ptx
            for a, b in pairs:
                assert evaluate(ptx, a, b) % p == (a+b) % p
            results[name] = {'passed': True, 'pairs': len(pairs)}
        else:
            branch = 'setp.eq.u64 choose,k,0;\n@choose bra qsb_small_fold_done;\n'
            assert ptx.count(branch) == 1 and ptx.count('qsb_small_fold_done:') == 1
            prefix, remainder = ptx.split(branch)
            tail, packing = remainder.split('qsb_small_fold_done:\n')
            assert 'bra ' not in prefix + tail + packing
            capture = prefix + 'mov.u64 %0,k;mov.u64 %1,0;mov.u64 %2,0;mov.u64 %3,0;}'
            counts = [0, 0]
            for a, b in pairs:
                flag = evaluate(capture, a, b)
                assert flag in (0, (1 << 64)-1 if name == '_ModSub256' else 1)
                counts[int(bool(flag))] += 1
                selected = prefix + (tail if flag else '') + packing
                actual = evaluate(selected, a, b)
                expected = (a-b if name == '_ModSub256' else a+b) % p
                assert actual % p == expected, (name, a, b, actual, expected)
            assert all(counts)
            results[name] = {'passed': True, 'pairs': len(pairs), 'branch_counts': counts}
        results[name]['ptx_sha256'] = hashlib.sha256(ptx.encode()).hexdigest()
    return {'passed': True, 'functions': results,
            'header_sha256': hashlib.sha256(header.read_bytes()).hexdigest(),
            'scope': 'CPU actual-PTX branch specialization and integer oracle; not GPU or score evidence.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('header', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    proof = audit(args.header)
    args.output.write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
