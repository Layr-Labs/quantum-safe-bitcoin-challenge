#!/usr/bin/env python3
"""Losslessly factor instruction addresses from SASS text in historical archives."""
import hashlib,io,json,re,struct,tarfile
sha=lambda b:hashlib.sha256(b).hexdigest()
pat=re.compile(rb'/\*([0-9a-f]{4,8})\*/')
def enc(b):
    parts=[];words=[];lengths=[];pos=0
    for m in pat.finditer(b):
        parts.append(b[pos:m.start(1)]);words.append(int(m[1],16));lengths.append(len(m[1]));pos=m.end(1)
    parts.append(b[pos:]);out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w') as t:
        blobs={'text':b''.join(parts),'addresses':struct.pack('<'+'I'*len(words),*words),'recipe':json.dumps({'sha256':sha(b),'lengths':[len(p) for p in parts],'widths':lengths},separators=(',',':')).encode()}
        for n,z in blobs.items():i=tarfile.TarInfo(n);i.size=len(z);t.addfile(i,io.BytesIO(z))
    return out.getvalue()
def dec(b):
    with tarfile.open(fileobj=io.BytesIO(b)) as t:
        r=json.load(t.extractfile('recipe'));text=t.extractfile('text').read();a=t.extractfile('addresses').read();parts=[];pos=0
        for i,n in enumerate(r['lengths']):
            parts.append(text[pos:pos+n]);pos+=n
            if i<len(r['widths']):parts.append(format(struct.unpack_from('<I',a,4*i)[0],f"0{r['widths'][i]}x").encode())
        b=b''.join(parts);assert sha(b)==r['sha256'];return b

def transform(blob,inverse=False):
    out=io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(blob)) as t,tarfile.open(fileobj=out,mode='w') as ot:
        for m in t.getmembers():
            b=t.extractfile(m).read()
            if m.name.endswith('.sass'):
                inner=io.BytesIO()
                with tarfile.open(fileobj=io.BytesIO(b)) as it,tarfile.open(fileobj=inner,mode='w') as dest:
                    for a in it.getmembers():
                        z=it.extractfile(a).read()
                        if a.name=='text':z=dec(z) if inverse else enc(z)
                        a.size=len(z);dest.addfile(a,io.BytesIO(z))
                b=inner.getvalue()
            m.size=len(b);ot.addfile(m,io.BytesIO(b))
    return out.getvalue()
