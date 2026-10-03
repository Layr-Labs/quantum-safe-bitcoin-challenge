#!/usr/bin/env python3
"""Normalize SASS hexadecimal encoding comments into binary, reversible byte-exact."""
import hashlib,io,json,re,tarfile
sha=lambda b:hashlib.sha256(b).hexdigest()
pat=re.compile(rb'/\* (0x[0-9a-f]{16}) \*/')
def normalize(raw):
    parts=[];words=[];pos=0
    for m in pat.finditer(raw):
        parts.append(raw[pos:m.start(1)]);words.append(bytes.fromhex(m.group(1)[2:].decode()));pos=m.end(1)
    parts.append(raw[pos:]);out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w') as t:
        def add(n,b):
            info=tarfile.TarInfo(n);info.size=len(b);t.addfile(info,io.BytesIO(b))
        add('text',b''.join(parts));add('words',b''.join(words))
        add('recipe.json',json.dumps({'sha256':sha(raw),'bytes':len(raw),'lengths':[len(p) for p in parts]},separators=(',',':')).encode())
    return out.getvalue(),len(words)
def reconstruct(blob):
    with tarfile.open(fileobj=io.BytesIO(blob)) as t:
        m=json.load(t.extractfile('recipe.json'));text=t.extractfile('text').read();words=t.extractfile('words').read();out=[];pos=0
        for k,n in enumerate(m['lengths']):
            out.append(text[pos:pos+n]);pos+=n
            if k+1<len(m['lengths']):out.append(b'0x'+words[8*k:8*k+8].hex().encode())
        raw=b''.join(out);assert len(raw)==m['bytes'] and sha(raw)==m['sha256'];return raw
