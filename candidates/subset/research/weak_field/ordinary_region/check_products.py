#!/usr/bin/env python3
"""Actual weak-product host bodies/PTX vs bigint; not a complete curve test."""
import contextlib
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / 'candidate'
CHECKER = ROOT / 'research/check_production_field.py'
sys.path[:0] = [str(ROOT), str(ROOT / 'research')]

def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=HERE/'product-results.json')
    args = parser.parse_args()
    # Preserve all independent cases, alias modes, fold range bounds and the
    # stale-carry negative control of the existing actual-source field checker.
    text = CHECKER.read_text()
    text = once(text, "(ROOT/'GPUMath.h').read_text()", "(ROOT/'qsb_weak_product.cuh').read_text()")
    text = once(text, "(ROOT/'square32.cuh').read_text()", "(ROOT/'qsb_weak_product.cuh').read_text()")
    text = text.replace('_ModMultCore(', 'qsb_weak_mul(').replace('qsb_square32(', 'qsb_weak_square(')
    text = once(text, 'assert integer(out)==expected,', 'assert integer(out)%P==expected,')
    text = once(text, "'validation_level':'actual host primitives and PTX semantic model'", "'validation_level':'actual weak-product host primitives and PTX semantic model, checking residues and uint256 range'")
    namespace = {'__name__':'weak_product_projection', '__file__':str(CHECKER)}
    exec(compile(text, str(CHECKER)+'[weak-product-adapter]', 'exec'), namespace)
    namespace['ROOT'] = args.source.resolve()
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        namespace['main']()
    result = json.loads(output.getvalue())
    result.update({
        'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_checker_sha256':hashlib.sha256(CHECKER.read_bytes()).hexdigest(),
        'contract':'Inputs and output in W=[0,2^256); exact modular residue, not canonical output.',
        'limits':'Only products. No new add/sub, actual complete-chain, GPU or performance qualification follows.',
    })
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status':result['status'],'counts':result['counts'],'source_fingerprint':result['source_fingerprint']}, indent=2))

if __name__ == '__main__':
    main()
