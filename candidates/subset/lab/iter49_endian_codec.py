#!/usr/bin/env python3
"""Reversible SASS-word byte order unifies textual encodings with ELF text bytes."""
import io,tarfile

def transform(blob):
 out=io.BytesIO()
 with tarfile.open(fileobj=io.BytesIO(blob)) as t,tarfile.open(fileobj=out,mode='w') as dest:
  for m in t.getmembers():
   b=t.extractfile(m).read()
   if m.name.endswith('.sass'):
    inner=io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(b)) as it,tarfile.open(fileobj=inner,mode='w') as ot:
     for a in it.getmembers():
      z=it.extractfile(a).read()
      if a.name=='words':
       assert len(z)%8==0;v=bytearray(len(z))
       for i in range(8):v[i::8]=z[7-i::8]
       z=bytes(v)
      ot.addfile(a,io.BytesIO(z))
    b=inner.getvalue()
   m.size=len(b);dest.addfile(m,io.BytesIO(b))
 return out.getvalue()
