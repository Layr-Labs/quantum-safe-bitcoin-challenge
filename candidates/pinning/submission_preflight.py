#!/usr/bin/env python3
"""Read-only archive, source-manifest and public-note checks before upload."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    manifest=json.loads((ROOT/'benchmark.json').read_text())
    assert manifest['schemaVersion']==2
    track=next(t for t in manifest['tracks'] if t['name']=='pinning')
    assert track['editablePaths']==['candidates/pinning']
    files=[]
    for path in HERE.rglob('*'):
        assert not path.is_symlink(),path
        assert path.is_dir() or path.is_file(),path
        if not path.is_file():continue
        assert not any(x in path.parts for x in ['.build','__pycache__']),path
        assert path.suffix not in ['.pyc','.o','.so','.dylib','.cubin','.sass','.ptx'],path
        raw=path.read_bytes()
        assert b'\0' not in raw,('unexpected binary',path)
        assert not re.search(rb'ykn_[A-Za-z0-9]{15,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{25,}',raw),('credential pattern',path.name)
        files.append(path)
    total=sum(p.stat().st_size for p in files)
    assert total<track['maxSubmissionBytes'],total
    note=(HERE/'SUBMISSION.md').read_bytes()
    assert 5*1024<=len(note)<=100*1024
    text=note.decode()
    assert not re.search(r'ykn_[A-Za-z0-9]+|/Users/|/home/|sk-[A-Za-z0-9]{20,}',text)
    assert not re.search(r'^Model:|^Harness:',text,re.M),'CLI supplies canonical attribution'
    source=json.loads((HERE/'SOURCE-MANIFEST.json').read_text())
    assert source['model']=='GPT 6 Astra' and source['harness']=='Codex'
    expected={p.relative_to(HERE).as_posix() for p in files if p.name not in ('SOURCE-MANIFEST.json','SUBMISSION-STATUS.md')}
    assert set(source['production_source_sha256'])==expected,('incomplete archive manifest',set(source['production_source_sha256'])^expected)
    for name,want in source['production_source_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==want,name
    base=source['comparison_baseline_promoted_commit']
    changed=subprocess.check_output(['git','diff','--name-only',base],cwd=ROOT,text=True).splitlines()
    assert all(p.startswith('candidates/pinning/') for p in changed),changed
    cofactor=subprocess.check_output(['git','show',base+':candidates/pinning/cofactor_checkpoint.h'],cwd=ROOT)
    assert (HERE/'cofactor_checkpoint.h').read_bytes()==cofactor
    print(json.dumps({'on_disk_files':len(files),'expanded_bytes':total,
        'limit_bytes':track['maxSubmissionBytes'],'public_note_bytes':len(note),
        'source_hashes_checked':len(source['production_source_sha256']),
        'scope':'candidates/pinning only','gpu_score':None},indent=2))


if __name__=='__main__':main()
