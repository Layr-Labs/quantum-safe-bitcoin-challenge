#!/usr/bin/env python3
"""Read-only package checks. All files on disk are part of Yukon's archive."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    benchmark = json.loads((ROOT / 'benchmark.json').read_text())
    assert benchmark['schemaVersion'] == 2
    track = next(t for t in benchmark['tracks'] if t['name'] == 'subset')
    assert track['editablePaths'] == ['candidates/subset']
    source = json.loads((HERE / 'SOURCE-MANIFEST.json').read_text())
    assert source['model'] == 'GPT 6 Astra' and source['harness'] == 'Codex'
    files = []
    for path in HERE.rglob('*'):
        assert not path.is_symlink(), path
        assert path.is_dir() or path.is_file(), path
        if not path.is_file():
            continue
        assert not any(p in ('.build', '__pycache__') for p in path.parts), path
        assert path.suffix not in ('.pyc', '.o', '.so', '.dylib', '.cubin', '.ptx', '.sass'), path
        raw = path.read_bytes()
        assert b'\0' not in raw, ('unexpected binary', path)
        assert not re.search(rb'ykn_[A-Za-z0-9]{15,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{25,}', raw), ('credential pattern', path.name)
        files.append(path)
    total = sum(p.stat().st_size for p in files)
    assert total < track['maxSubmissionBytes'], total
    note = (HERE / 'submission-note.md').read_bytes()
    assert 5*1024 <= len(note) <= 100*1024
    assert not re.search(rb'/Users/|/home/|^Model:|^Harness:', note, re.M)
    hashes = source['production_source_sha256']
    expected = {p.relative_to(HERE).as_posix() for p in files if p.name not in ('SOURCE-MANIFEST.json', 'SUBMISSION-STATUS.md')}
    assert set(hashes) == expected, (set(hashes) ^ expected)
    for name, want in hashes.items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == want, name
    base = source['comparison_baseline_promoted_commit']
    changed = subprocess.check_output(['git', 'diff', '--name-only', base], cwd=ROOT, text=True).splitlines()
    assert all(name.startswith('candidates/subset/') for name in changed), changed
    # The SHA implementation and exact field/replay helpers remain the promoted bytes.
    for name in ('GPUHash.h', 'GPUMath.h', 'chain_replay_field.cuh',
                 'tests/gpu_epochs/window_schedule_shared.cuh', 'tests/gpu_epochs/tree_inverse.cuh'):
        original = subprocess.check_output(['git', 'show', base+':candidates/subset/'+name], cwd=ROOT)
        assert (HERE / name).read_bytes() == original, name
    print(json.dumps({'on_disk_files': len(files), 'expanded_bytes': total,
                      'limit_bytes': track['maxSubmissionBytes'], 'public_note_bytes': len(note),
                      'source_hashes_checked': len(hashes), 'scope': 'candidates/subset only',
                      'local_gpu_score': None}, indent=2))


if __name__ == '__main__':
    main()
