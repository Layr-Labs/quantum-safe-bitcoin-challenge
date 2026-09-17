#!/usr/bin/env python3
"""Reproduce the cold12 hint-only negative control with explicit CPU diagnostics.

Run with python3 -B run_negative.py. Both source copies, generated C++ and checker
adaptations are temporary. Existing sources/checkers/reports remain untouched,
except the designated prefetch-negative-control.json result. No CUDA compilation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
from preflight import source_identity

EXPECTED = '5001cdc69fa2c5979ef225894874e2c9ef9ce30da167e64b35aff613b583bba3'
MUTATED = '24000fbf100895f8bfea6edbf1679188ac2bcc046827da670efd55f5fb93b256'
HEADER = 'tests/gpu_epochs/compact_table_device.cuh'
OLD = '(unsigned)mixed_shift(cold)+1u,25u,cold==12,'
NEW = '(unsigned)mixed_shift(cold)+1u,25u,false,'
CHECKERS = [
    'preflight.py', 'audit_integrated.py',
    'research/wide_windows/audit_support.py',
    'research/wide_windows/check_wide.py',
    'research/asymmetric_windows/cold_seed/check_exceptions.py',
    'research/pr120_digits/check_chain_recovery.py',
    'research/shared_all_state/check.py',
    'research/geometry_frontier/shared_thirteen/check.py',
    str((HERE/'check.py').relative_to(ROOT)),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, old, new):
    assert text.count(old) == 1, ('Diagnostic marker changed', old)
    return text.replace(old, new, 1)


def worker(source, output, expected, base_fingerprint, checker):
    """Use unchanged checker adapters; instrument only temporary generated C++."""
    output.mkdir()
    evidence = {'compilations': [], 'generated_projections': []}
    evidence_path = output/'worker-evidence.json'
    original_write = Path.write_text
    original_run = subprocess.run

    def save():
        original_write(evidence_path, json.dumps(evidence, indent=2)+'\n')

    def write(self, text, *args, **kwargs):
        if self.name == 'audit.cpp' and 'static uintptr_t prefetched[WIDE_CHUNKS];' in text:
            before = hashlib.sha256(text.encode()).hexdigest()
            text = once(text, 'static uintptr_t prefetched[WIDE_CHUNKS];',
                        'static uint64_t qsb_negative_scalar[4];\nstatic uintptr_t prefetched[WIDE_CHUNKS];')
            marker = 'extern "C" int audit_chain(const uint64_t*k,const uint64_t*base_scalar){'
            text = once(text, marker, marker+'\nmemcpy(qsb_negative_scalar,k,32);')
            # Keep the existing range, half-pair and load-order checks unchanged.
            # Only the exact expected-address assertion gains a named diagnostic.
            text = once(text,
                'require(prefetched[c]==((uintptr_t)wide_offset(c)+idx)*64);',
                '''if(prefetched[c]!=((uintptr_t)wide_offset(c)+idx)*64){
 fprintf(stderr,"QSB_PREFETCH_ADDRESS_MISMATCH window=%d actual=0x%llx expected=0x%llx idx=%u halves=%u scalar=0x%016llx%016llx%016llx%016llx\\n",
 c,(unsigned long long)prefetched[c],(unsigned long long)(((uintptr_t)wide_offset(c)+idx)*64),idx,prefetched_halves[c],
 (unsigned long long)qsb_negative_scalar[3],(unsigned long long)qsb_negative_scalar[2],
 (unsigned long long)qsb_negative_scalar[1],(unsigned long long)qsb_negative_scalar[0]);
 fflush(stderr);exit(86);
}''')
            evidence['generated_projections'].append({
                'before_diagnostic_sha256': before,
                'after_diagnostic_sha256': hashlib.sha256(text.encode()).hexdigest(),
            })
            save()
        return original_write(self, text, *args, **kwargs)

    def run(command, *args, **kwargs):
        result = original_run(command, *args, **kwargs)
        if isinstance(command, list) and command[0] == 'c++':
            cpp = next(Path(item) for item in command if str(item).endswith('/audit.cpp'))
            lib = Path(command[command.index('-o')+1])
            # Record only successful completion; failed compiler invocations never
            # create this evidence and cannot satisfy the parent assertions.
            assert result.returncode == 0 and lib.is_file()
            record = {'exit_code': 0, 'generated_cpp_sha256': digest(cpp),
                      'shared_library_created': True,
                      'flags': [str(v) for v in command[1:] if str(v) not in [str(cpp), str(lib)]]}
            evidence['compilations'].append(record)
            save()
            print('QSB_CPU_COMPILE_OK cpp_sha256='+record['generated_cpp_sha256'], file=sys.stderr, flush=True)
        return result

    Path.write_text = write
    subprocess.run = run
    code = checker.read_text()
    assert code.count("assert identity['source_fingerprint']=='"+base_fingerprint+"'") == 1
    if expected != base_fingerprint:
        code = once(code, base_fingerprint, expected)
    evidence['temporary_entry_checker_sha256'] = hashlib.sha256(code.encode()).hexdigest()
    save()
    sys.argv = [str(checker), '--source', str(source), '--output', str(output)]
    exec(compile(code, str(checker)+'[negative-diagnostics]', 'exec'),
         {'__name__': '__main__', '__file__': str(checker)})


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5], Path(sys.argv[6]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=HERE/'candidate')
    parser.add_argument('--checker', type=Path, default=HERE/'check.py')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    source, checker = args.source.resolve(), args.checker.resolve()
    if source != HERE/'candidate' or checker != HERE/'check.py':
        assert args.report, '--report required with nondefault source/checker; preserve original report'
    report_path = (args.report or HERE/'prefetch-negative-control.json').resolve()
    assert report_path.is_relative_to(HERE), 'Reports must remain under cold_prefetch'
    identity = source_identity(source)
    fingerprint = identity['source_fingerprint']
    if source == HERE/'candidate':
        assert fingerprint == EXPECTED
    checkers = {name: digest(ROOT/name) for name in CHECKERS}
    checkers[str(checker.relative_to(ROOT))] = digest(checker)
    previous = json.loads(report_path.read_text()) if report_path.exists() else None
    runs = {}
    with tempfile.TemporaryDirectory(prefix='qsb-prefetch-negative-') as tmp:
        work = Path(tmp)
        mutation = work/'mutated-source'
        shutil.copytree(source, mutation)
        original = (mutation/HEADER).read_text()
        changed = once(original, OLD, NEW)
        (mutation/HEADER).write_text(changed)
        changed_identity = source_identity(mutation)
        if fingerprint == EXPECTED:
            assert changed_identity['source_fingerprint'] == MUTATED
        changed_files = [name for name, value in identity['source_sha256'].items()
                         if changed_identity['source_sha256'][name] != value]
        assert changed_files == [HEADER]
        assert changed.replace(NEW, OLD, 1) == original
        for label, run_source, run_fingerprint in [('positive', source, fingerprint),
                                                  ('negative', mutation, changed_identity['source_fingerprint'])]:
            output = work/label
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                                     '--worker', str(run_source), str(output), run_fingerprint,
                                     fingerprint, str(checker)],
                                    capture_output=True, text=True)
            evidence = json.loads((output/'worker-evidence.json').read_text())
            diagnostics = [line for line in result.stderr.splitlines() if line.startswith('QSB_')]
            assert len(evidence['generated_projections']) == 1
            projection = evidence['generated_projections'][0]
            assert any(c['generated_cpp_sha256'] == projection['after_diagnostic_sha256']
                       and c['exit_code'] == 0 and c['shared_library_created'] for c in evidence['compilations'])
            runs[label] = {'source_fingerprint': run_fingerprint, 'exit_code': result.returncode,
                           'diagnostics': diagnostics, **evidence}
            if label == 'positive':
                assert result.returncode == 0, result.stderr[-2000:]
                checked = json.loads((output/'check-results.json').read_text())
                assert checked['status'] == 'PASS' and checked['source_fingerprint'] == fingerprint
                runs[label]['check_results'] = checked
                runs[label]['chain_results'] = json.loads((output/'chain_recovery/chain-recovery-results.json').read_text())
            else:
                assert result.returncode == 86, result.stderr[-2000:]
                mismatches = [v for v in diagnostics if v.startswith('QSB_PREFETCH_ADDRESS_MISMATCH ')]
                assert len(mismatches) == 1
                matched = re.fullmatch(r'QSB_PREFETCH_ADDRESS_MISMATCH window=(\d+) actual=(0x[0-9a-f]+) expected=(0x[0-9a-f]+) idx=(\d+) halves=(\d+) scalar=(0x[0-9a-f]+)', mismatches[0])
                assert matched and matched[1] == '12' and matched[2] != matched[3] and matched[5] == '3'
                runs[label]['mismatch'] = dict(zip(['window','actual','expected','idx','halves','scalar'], matched.groups()))
    assert source_identity(source) == identity
    assert all(digest(ROOT/name) == value for name, value in checkers.items())
    report = {'status': 'EXPECTED_WRONG_COLD_PREFETCH_DETECTED_AFTER_SUCCESSFUL_CPU_COMPILE',
              'base_source': identity, 'mutated_source': changed_identity,
              'mutation': {'file': HEADER, 'old': OLD, 'new': NEW, 'changed_files': changed_files,
                           'only_hint_last_flag_changed': True, 'actual_loads_and_chain_unchanged': True},
              'runner_sha256': digest(Path(__file__)), 'checker_sha256': checkers,
              'reproduce_argv': ['python3', '-B', str(Path(__file__).resolve().relative_to(ROOT.parent.parent)),
                                 '--source', str(source.relative_to(ROOT.parent.parent)),
                                 '--checker', str(checker.relative_to(ROOT.parent.parent)),
                                 '--report', str(report_path.relative_to(ROOT.parent.parent))],
              'runs': runs, 'gpu_executed': False, 'cuda_compiled': False,
              'scope': 'Actual source-derived prefetch offsets compared to later actual load indices in the existing OpenSSL CPU chain projection. A wrong hint is detected; this does not imply wrong arithmetic outputs or measure cache behavior.',
              'prior_evidence': previous if previous and previous.get('status') == 'EXPECTED_WRONG_COLD_PREFETCH_DETECTED' else (previous or {}).get('prior_evidence')}
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'report': str(report_path),
                      'negative': runs['negative']['diagnostics']}, indent=2))


if __name__ == '__main__':
    main()
