#!/usr/bin/env python3
"""Check the fused PR77 fourteen-window source using existing independent oracles.

Only checker extraction is adapted, in memory. Production source is never
rewritten to satisfy a legacy test. Actual fused recoding, vector loads, cold
digits, deferred seed/intermediate/final point helpers, and direct recovery run
against OpenSSL field operations and independent curve multiplication. This is
not native field/PTX execution, a complete fused-kernel audit, or a GPU test.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'research/wide_windows')]
from audit_support import function
from preflight import source_identity


def replace_once(source, old, new):
    assert source.count(old) == 1, ('Checker extraction changed; review adapter', old)
    return source.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=HERE / 'frontier_fused/candidate')
    parser.add_argument('--output', type=Path, default=Path('/tmp/qsb-fused-geometry'))
    args = parser.parse_args()
    base, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = source_identity(base)
    header = (base / 'tests/gpu_epochs/compact_table_device.cuh').read_text()
    math = (base / 'GPUMath.h').read_text()
    tree = (base / 'tests/gpu_epochs/tree.cu').read_text()
    chain = function(header, '__device__ void compact_fixed_xyzz(')
    assert '_PointAddXYZZ_mm_def(' in chain
    assert re.search(r'_PointAddXYZZ_def\([^;]*,\s*true\s*\)', chain)
    assert 'qsb_asym_last_add(' in chain
    assert '_PointAddXYZZ<' not in chain, 'Do not accidentally audit the old chain'
    assert 'compact_fixed_xyzz(' in tree, 'Fused source must call the selected chain'

    wide_path = ROOT / 'research/wide_windows/check_wide.py'
    exception_path = ROOT / 'research/asymmetric_windows/cold_seed/check_exceptions.py'
    wide_original = wide_path.read_text()
    wide = replace_once(
        wide_original,
        "src=(base/'tests/gpu_epochs/tree.cu').read_text()+'\\n'+(base/'tests/gpu_epochs/ranked_pipeline.cuh').read_text()",
        "src=(base/'tests/gpu_epochs/tree.cu').read_text()")
    wide = replace_once(wide, "function(math,'template<bool DEFER_Y>')",
                        "function(math,'__device__ void _PointAddXYZZ_def(')")
    wide = replace_once(wide, "function(math,'__device__ void _PointAddXYZZ_mm(')",
                        "function(math,'__device__ void _PointAddXYZZ_mm_def(')")
    wide = replace_once(wide, "geometry=(base/'tests/gpu_epochs/wide_geometry.cuh').read_text()",
                        "geometry='' # --compact loads the actual compact geometry below")

    # Reuse all independent geometry, recoding, sparse-load, prefetch, chain,
    # and direct-recovery checks. Source identity still reads the actual tree.
    def run_chain(command):
        assert Path(command[2]).resolve() == wide_path.resolve()
        previous = sys.argv
        try:
            sys.argv = command[2:]
            namespace = {'__name__': '__main__', '__file__': str(wide_path)}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(wide, str(wide_path) + '[fused-extraction]', 'exec'), namespace)
        finally:
            sys.argv = previous

    # The guarded final helper is extracted unchanged. Its negative control is
    # the fused source's actual incomplete final add, not the older template.
    exceptional_original = exception_path.read_text()
    exceptional = replace_once(exceptional_original, "function(math,'template<bool DEFER_Y>')",
                               "function(math,'__device__ void _PointAddXYZZ_def(')")
    exceptional = replace_once(exceptional,
        '_PointAddXYZZ<false>(state,state+4,state+8,state+12,point,point+4,anchor);',
        '_PointAddXYZZ_def(state,state+4,state+8,state+12,point,point+4,anchor,false);')
    exceptional = replace_once(exceptional, 'subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)',
                               'run_chain(cmd)')
    previous = sys.argv
    try:
        sys.argv = [str(exception_path), '--source', str(base), '--first-width', '17',
                    '--output', str(output)]
        namespace = {'__name__': '__main__', '__file__': str(exception_path), 'run_chain': run_chain}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(exceptional, str(exception_path) + '[fused-extraction]', 'exec'), namespace)
    finally:
        sys.argv = previous

    assert source_identity(base) == identity, 'Source changed during checks'
    curve = json.loads((output / 'guarded-curve-results.json').read_text())
    exceptional_report = json.loads((output / 'exception-domain-results.json').read_text())
    for report in [curve, exceptional_report]:
        assert report['status'] == 'PASS'
        assert report['source_fingerprint'] == identity['source_fingerprint']
    assert curve['geometry_bits'] == [17] * 12 + [26] * 2
    assert curve['checked_chain_order'] == [12, 13] + list(range(12))
    assert curve['table_bytes'] == (48 << 20) + (4 << 30)
    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
    bodies = {
        'compact_fixed_xyzz': chain,
        'compact_load_signed': function(header, '__device__ __forceinline__ void compact_load_signed('),
        'qsb_asym_last_add': function(header, '__device__ __forceinline__ void qsb_asym_last_add('),
        '_PointAddXYZZ_def': function(math, '__device__ void _PointAddXYZZ_def('),
        '_PointAddXYZZ_mm_def': function(math, '__device__ void _PointAddXYZZ_mm_def('),
        'qsb_xyzz_finish_prepare': function(tree, '__device__ __forceinline__ void qsb_xyzz_finish_prepare('),
        'qsb_xyzz_finish_precomputed': function(tree, '__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed('),
    }
    result = {
        'status': 'PASS', **identity,
        'validation_level': 'Actual fused-source CPU geometry/chain/recovery projection with OpenSSL fields',
        'recoding_cases': curve['recoding_cases'], 'curve_chains': curve['curve_chains'],
        'recovered_keys_compared': curve['recovered_keys_compared'],
        'actual_loader_cases': curve['actual_loader_cases'],
        'prefetch_next_load_checks': curve['prefetch_next_load_checks'],
        'geometry_bits': curve['geometry_bits'], 'checked_chain_order': curve['checked_chain_order'],
        'actual_helper_cases': exceptional_report['actual_helper_cases'],
        'prefix_bound_cases': len(exceptional_report['prefix_bounds']),
        'added_chain_scalar_cases': exceptional_report['added_chain_scalar_cases'],
        'extracted_actual_function_sha256': {name: sha(body) for name, body in bodies.items()},
        'checker_original_sha256': {'check_wide.py': sha(wide_original), 'check_exceptions.py': sha(exceptional_original)},
        'checker_adapted_sha256': {'check_wide.py': sha(wide), 'check_exceptions.py': sha(exceptional)},
        'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'projection_changes': [
            'Read direct recovery from actual fused tree.cu; do not read old ranked_pipeline.cuh.',
            'Extract actual fused _PointAddXYZZ_def and _PointAddXYZZ_mm_def instead of old template/seed.',
            'Use actual fused _PointAddXYZZ_def(...,false) as exceptional negative control.',
            'Retain legacy independent scalar, geometry, order, loader, prefetch and affine oracles unchanged.',
        ],
        'gpu_executed': False, 'production_modified': False, 'limits': __doc__,
    }
    (output / 'fused-geometry-results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in
                      ['source_sha256', 'extracted_actual_function_sha256']}, indent=2))


if __name__ == '__main__':
    main()
