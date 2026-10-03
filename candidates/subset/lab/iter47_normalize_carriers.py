#!/usr/bin/env python3
"""Losslessly normalize embedded base64 images across rawtar evidence to share binary dictionary matches."""
from pathlib import Path
import base64,hashlib,io,json,lzma,re,tarfile
w=Path('/work/tmp/qsb-subset-iter47-karat')
sha=lambda b:hashlib.sha256(b).hexdigest()
pat=re.compile(rb'(static const char \*const qsb_carrier_b64\[\] = \{)(.*?)(\};)',re.S)
def normalize(raw):
    # Whole rawtar byte stream: no member sizes/padding are altered in reconstructed evidence.
    out=io.BytesIO(); records=[]; pos=0
    with tarfile.open(fileobj=out,mode='w') as t:
        def add(n,b):
            info=tarfile.TarInfo(n);info.size=len(b);t.addfile(info,io.BytesIO(b))
        for i,m in enumerate(pat.finditer(raw)):
            body=m.group(2);tokens=list(re.finditer(rb'"([A-Za-z0-9+/=]+)"',body))
            if not tokens:continue
            image=base64.b64decode(b''.join(x.group(1) for x in tokens)); enc=base64.b64encode(image)
            assert enc==b''.join(x.group(1) for x in tokens)
            prefix=raw[pos:m.start(2)];parts=[];p=0
            for x in tokens:parts.append(base64.b64encode(body[p:x.start(1)]).decode());p=x.end(1)
            parts.append(base64.b64encode(body[p:]).decode())
            row={'id':i,'lengths':[len(x.group(1)) for x in tokens],'parts':parts,'prefix_sha256':sha(prefix),'image_sha256':sha(image)}
            add(f'{i}/prefix',prefix);add(f'{i}/image',image);records.append(row);pos=m.end(2)
        add('tail',raw[pos:]);add('reconstruction.json',(json.dumps({'raw_bytes':len(raw),'raw_sha256':sha(raw),'records':records},separators=(',',':'))+'\n').encode())
    return out.getvalue(),len(records)
def reconstruct(blob):
    with tarfile.open(fileobj=io.BytesIO(blob)) as t:
        m=json.load(t.extractfile('reconstruction.json'));pieces=[]
        for row in m['records']:
            pieces.append(t.extractfile(f"{row['id']}/prefix").read());enc=base64.b64encode(t.extractfile(f"{row['id']}/image").read());pos=0;body=[]
            for k,n in enumerate(row['lengths']):body.extend([base64.b64decode(row['parts'][k]),enc[pos:pos+n]]);pos+=n
            body.append(base64.b64decode(row['parts'][-1]));assert pos==len(enc);pieces.append(b''.join(body))
        pieces.append(t.extractfile('tail').read());raw=b''.join(pieces);assert len(raw)==m['raw_bytes'] and sha(raw)==m['raw_sha256'];return raw
if __name__=='__main__':
    p=w/'iter47-joint-capped.tar.xz';r=json.load(open(str(p)+'.json'));records={};counts={};out=io.BytesIO()
    with tarfile.open(p) as old,tarfile.open(fileobj=out,mode='w') as t:
        for m in old.getmembers():
            b=old.extractfile(m).read();n=m.name
            if n.endswith('rawtar'):
                nb,count=normalize(b);assert reconstruct(nb)==b;counts[n]=count
                if count:n+='.normalized';b=nb
            info=tarfile.TarInfo(n);info.size=len(b);t.addfile(info,io.BytesIO(b));records[n]={'bytes':len(b),'sha256':sha(b)}
    raw=out.getvalue();(w/'iter47-normalized.rawtar').write_bytes(raw)
    b=lzma.compress(raw,format=lzma.FORMAT_XZ,filters=[{'id':lzma.FILTER_LZMA2,'preset':9|lzma.PRESET_EXTREME,'dict_size':512<<20}]);assert lzma.decompress(b)==raw
    p=w/'iter47-normalized.tar.xz';p.write_bytes(b);r.update(bytes=len(b),sha256=sha(b),raw_tar_bytes=len(raw),raw_tar_sha256=sha(raw),records=records,normalized_counts=counts,normalization='rawtar .normalized members reconstruct byte-identically with iter47_normalize_carriers.reconstruct')
    Path(str(p)+'.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
