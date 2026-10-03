#!/usr/bin/env python3
"""Restore byte-exact GPL historical probe headers before opt-in experiments."""
from pathlib import Path
import tarfile,json,hashlib
lab=Path(__file__).resolve().parent
with tarfile.open(lab/'iter48-cold-lab.tar.xz') as t:
 r=json.load(t.extractfile('manifest.json'))
 for n,v in r.items():
  if n.startswith('optional-source/'):
   b=t.extractfile(n).read();assert hashlib.sha256(b).hexdigest()==v['sha256']
   p=lab.parent/n.removeprefix('optional-source/');p.write_bytes(b)
print('restored exact historical optionals; restore exceeds submission cap')
