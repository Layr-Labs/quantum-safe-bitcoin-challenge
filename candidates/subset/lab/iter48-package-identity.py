#!/usr/bin/env python3
"""Read retained pre/post preprocessing; validate source-only comment compaction."""
from pathlib import Path
import hashlib,io,json,re,tarfile
from iter47c_source_layout import comments
w=Path('/work/tmp/qsb-subset-iter48-phases');base=Path(__file__).resolve().parent.parent
sha=lambda b:hashlib.sha256(b).hexdigest()
# Same tokenizer as prior identity audit; comments/line directives carry no executable tokens.
pat=re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_][A-Za-z_0-9]*|(?:\d+(?:\.\d*)?(?:[eEpP][+-]?\d+)?[A-Za-z_0-9]*)|[^\s]')
def tokens(p):
    s='\n'.join(x for x in p.read_text().splitlines() if not re.match(r'^#\s*\d+\s',x))
    return re.findall(r'(?:u8|u|U|L)?"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_$][A-Za-z0-9_$]*|(?:[0-9][A-Za-z0-9_.]*|\.[0-9][A-Za-z0-9_.]*)|>>=|<<=|->\*|\.\.\.|##|::|->|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||\+=|-=|\*=|/=|%=|&=|\^=|\|=|[^\s]',s)
r={'source':{},'preprocessed':{},'off_identity':{},'records':{}}
for p in (w/'before-source').rglob('*'):
    if not p.is_file():continue
    rel=p.relative_to(w/'before-source').as_posix();a=p.read_bytes();b=(base/rel).read_bytes()
    assert comments(a.decode()).encode()==b,rel
    assert a.count(b'\n')==b.count(b'\n'),rel
    r['source'][rel]={'before_sha256':sha(a),'after_sha256':sha(b),'before_bytes':len(a),'after_bytes':len(b),'saved':len(a)-len(b)}
for route in ('native','audit'):
    a=tokens(w/(route+'-before.ii'));b=tokens(w/(route+'-after.ii'));assert a==b,route
    r['preprocessed'][route]={'tokens':len(a),'sha256':sha('\n'.join(a).encode()),'equal':True}
for route in ('89','86'):
    a=(w/f'off{route}.cubin').read_bytes();b=(w/f'compact-off{route}.cubin').read_bytes();assert a==b
    r['off_identity']['sm'+route]={'sha256':sha(a),'bytes':len(a),'equal':True}
with tarfile.open(base/'lab/iter48-package-identity.tar.xz','w:xz',preset=9) as t:
    def add(n,b):
        i=tarfile.TarInfo(n);i.size=len(b);t.addfile(i,io.BytesIO(b));r['records'][n]={'bytes':len(b),'sha256':sha(b)}
    for n in ('compact-off89.build.log','compact-off86.build.log','native-before.pp.log','native-after.pp.log','audit-before.pp.log','audit-after.pp.log'):add(n,(w/n).read_bytes())
    add('comparison.py',Path(__file__).read_bytes())
    b=(json.dumps(r,indent=2)+'\n').encode();i=tarfile.TarInfo('identity.json');i.size=len(b);t.addfile(i,io.BytesIO(b))
print(json.dumps(r,indent=2))
