#!/usr/bin/env python3
"""Audit public hit density against searched coordinates, not extrapolated counts.

Reads public diagnostics through authenticated gh. It does not modify the
harness or infer startup timing directly. The last hit is only a lower bound
on the searched prefix; timing estimates assume the printed maximum rate.
"""
from pathlib import Path
import io
import json
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
REPOSITORY = 'repos/Layr-Labs/quantum-safe-bitcoin-challenge'
LT_MIN, LT_MAX, SEQ_MIN = 500000000, 1744600000, 2147483648
TRIALS = [
    ('cold_seed', 'd8481039-4eb4-415e-ae36-d9ff284a402c', 10820417826,
     '09fd2e244ca6c5dbe9161319b97a90d2ad1f894e', 929.0),
    ('narrow_mac', '02a2c09a-e8f4-4c09-9dad-e11936ff27f0', 10825895234,
     '10e87143c6df31ee0bb93490e27ecc80c738a164', 909.4),
]


def main():
    results = []
    for label, submission, artifact, commit, rate_mps in TRIALS:
        raw = subprocess.check_output([
            'rtk', 'proxy', 'gh', 'api',
            f'{REPOSITORY}/actions/artifacts/{artifact}/zip'])
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            name = next(n for n in archive.namelist() if n.endswith('run-pinning.json'))
            run = json.loads(archive.read(name))
            runner_name = next(n for n in archive.namelist() if n.endswith('runner.txt'))
            assert archive.read(runner_name).decode().strip().splitlines()[-1] == commit
        hits = run['hits']
        assert run['zeros_n'] == 24
        assert all(LT_MIN <= h['locktime'] < LT_MAX and h['sequence'] >= SEQ_MIN
                   for h in hits)
        last = max(hits, key=lambda h: (h['sequence'], h['locktime']))
        sequences = {h['sequence'] for h in hits}
        assert sequences == set(range(SEQ_MIN, last['sequence'] + 1))
        prefix = ((last['sequence'] - SEQ_MIN) * (LT_MAX - LT_MIN)
                  + last['locktime'] - LT_MIN + 1)
        expected_hits = prefix / 2**23  # two recovered keys per locktime candidate
        count = run['candidates']
        results.append({
            'label': label, 'submission_id': submission, 'artifact_id': artifact,
            'submission_commit': commit, 'verified_hits': len(hits),
            'last_hit': {k: last[k] for k in ('sequence', 'locktime', 'recid')},
            'searched_prefix_lower_bound': prefix,
            'expected_hits_from_prefix': expected_hits,
            'hit_density_ratio': len(hits) / expected_hits,
            'poisson_relative_standard_error': 1 / len(hits)**0.5,
            'extrapolated_count': count,
            'prefix_over_extrapolated_count': prefix / count,
            'printed_max_rate_mps': rate_mps,
            'elapsed_seconds': run['elapsed_s'],
            'prefix_seconds_at_max_rate': prefix / (rate_mps * 1e6),
            'elapsed_minus_prefix_seconds_at_max_rate':
                run['elapsed_s'] - prefix / (rate_mps * 1e6),
        })
    report = {
        'interpretation': 'Both hit densities are compatible with expected density. '
            'The inferred/extrapolated count ratio is not measured arithmetic recall.',
        'timing_limit': 'Elapsed minus prefix/max-rate includes startup and any '
            'steady-state time below the maximum rate; it does not isolate JIT time. '
            'The last hit is a lower bound, not the exact final candidate.',
        'geometry': {'locktime_min': LT_MIN, 'locktime_max_exclusive': LT_MAX,
                     'sequence_min': SEQ_MIN, 'single_gpu_stride': 1},
        'source_evidence': ['harness/gpu_wrap.py:candidate_count',
                            'candidates/pinning/pinning.cu:main search loops'],
        'trials': results,
    }
    (HERE / 'ranked-prefix-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
