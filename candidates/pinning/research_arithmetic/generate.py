#!/usr/bin/env python3
"""Recreate an isolated arithmetic screen from its pinned upstream commit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PINNING = ROOT.parent
REPO = PINNING.parent.parent

parser = argparse.ArgumentParser()
parser.add_argument('arm', choices=['baseline'] + sorted(p.stem for p in (ROOT/'patches').glob('*.patch')))
parser.add_argument('--out', type=Path)
args = parser.parse_args()
out = (args.out or ROOT/'_build'/args.arm).resolve()
if not out.is_relative_to(PINNING) or out == PINNING:
    parser.error('Output must be an isolated subdirectory of candidates/pinning.')
out.mkdir(parents=True, exist_ok=True)
manifest = json.loads((ROOT/'source_manifest.json').read_text())
for name, digest in manifest['source_files'].items():
    data = subprocess.check_output(['rtk', 'proxy', 'git', 'show', manifest['source_commit']+':candidates/pinning/'+name], cwd=REPO)
    assert hashlib.sha256(data).hexdigest() == digest, name
    (out/name).write_bytes(data)
if args.arm != 'baseline':
    subprocess.run(['rtk', 'proxy', 'patch', '--batch', '-p1', '-i', str(ROOT/'patches'/f'{args.arm}.patch')], cwd=out, check=True)
print(out)
