#!/usr/bin/env python3
"""Read-only archive, source-manifest and public-note checks before upload."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    benchmark = json.loads((ROOT/'benchmark.json').read_text())
    assert benchmark['schemaVersion'] == 2
    track = next(t for t in benchmark['tracks'] if t['name'] == 'subset')
    assert track['editablePaths'] == ['candidates/subset']
    files = []
    for path in HERE.rglob('*'):
        assert not path.is_symlink(), path
        assert path.is_dir() or path.is_file(), path
        if not path.is_file():
            continue
        assert not any(x in path.parts for x in ('.build', '__pycache__')), path
        assert path.suffix not in ('.pyc', '.o', '.so', '.dylib', '.cubin', '.sass', '.ptx'), path
        raw = path.read_bytes()
        assert b'\0' not in raw, ('unexpected binary', path)
        assert not re.search(rb'ykn_[A-Za-z0-9]{15,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{25,}', raw), ('credential pattern', path.name)
        files.append(path)
    total = sum(p.stat().st_size for p in files)
    assert total < track['maxSubmissionBytes'], total
    note = (HERE/'SUBMISSION-SINGLE.md').read_bytes()
    assert 5*1024 <= len(note) <= 100*1024
    assert not re.search(rb'/Users/|/home/|ykn_[A-Za-z0-9]+', note)
    assert not re.search(rb'^Model:|^Harness:', note, re.M), 'CLI supplies canonical attribution'
    source = json.loads((HERE/'SOURCE-MANIFEST.json').read_text())
    assert source['model'] == 'GPT 6 Astra' and source['harness'] == 'Codex'
    expected = {p.relative_to(HERE).as_posix() for p in files if p.name != 'SOURCE-MANIFEST.json'}
    assert set(source['production_source_sha256']) == expected
    for name, want in source['production_source_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == want, name
    base = source['comparison_baseline_promoted_commit']
    changed = subprocess.check_output(['git', 'diff', '--name-only', base], cwd=ROOT, text=True).splitlines()
    assert all(p.startswith('candidates/subset/') for p in changed), changed
    parent = source['measured_parent_commit']
    gate_path = 'tests/gpu_epochs/qsb_host_verify.h'
    gate = subprocess.check_output(['git', 'show', parent+':candidates/subset/'+gate_path], cwd=ROOT)
    assert (HERE/gate_path).read_bytes() == gate
    print(json.dumps({'on_disk_files': len(files), 'expanded_bytes': total,
                      'limit_bytes': track['maxSubmissionBytes'], 'public_note_bytes': len(note),
                      'source_hashes_checked': len(source['production_source_sha256']),
                      'scope': 'candidates/subset only', 'gpu_score': None}, indent=2))


if __name__ == '__main__':
    main()
