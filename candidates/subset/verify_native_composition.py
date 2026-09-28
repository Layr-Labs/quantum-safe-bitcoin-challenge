#!/usr/bin/env python3
"""Read-only source/archive validation. No CUDA, compiler, or runtime benchmark."""
from pathlib import Path
import base64,hashlib,json,re
ROOT=Path(__file__).resolve().parent
SHA=lambda b:hashlib.sha256(b).hexdigest()
def sources():
    files=sorted(set(ROOT.glob('*.cu'))|set(ROOT.glob('*.cuh'))|set(ROOT.glob('*.h'))|
                 set((ROOT/'tests/gpu_epochs').glob('*.cu'))|set((ROOT/'tests/gpu_epochs').glob('*.cuh'))|
                 set((ROOT/'tests/gpu_epochs').glob('*.h'))|{ROOT/'build_carrier.sh',ROOT/'verify_native_composition.py'})
    return {str(p.relative_to(ROOT)):SHA(p.read_bytes()) for p in files}
def archive(name):
    text=(ROOT/name).read_text()
    image=base64.b64decode(''.join(re.findall(r'"([A-Za-z0-9+/=]+)"',text.split('qsb_carrier_b64[] = {')[1])),validate=True)
    knobs=re.search(rb'QSB_ZEROS_N=[^\0]+',image).group().decode()
    return dict(cubin_sha256=SHA(image),bytes=len(image),
        original_source_sha256=re.search(r'source sha256 ([a-f0-9]{64})',text).group(1),
        fingerprint=knobs,kernel_names=re.findall(r'"(_Z[^\"]+)"',text.split('qsb_carrier_b64[] = {')[0]))
def verify():
    manifest=json.loads((ROOT/'NATIVE_COMPOSITION.json').read_text())
    assert manifest['source_files']==sources(),'host composition source changed; review and refresh binding explicitly'
    archives={n:archive(n) for n in manifest['archives']}
    assert archives==manifest['archives'],'archived image/provenance changed'
    a,b=[archives[n] for n in ['qsb_carrier_sm89.h','qsb_carrier_alt_sm89.h']]
    assert a['cubin_sha256']=='96ea66aad1f80feca26f060e0ebdf2c2e682a945eea7665162e3b4733a6050bc'
    assert b['cubin_sha256']=='d2f8160bc9303cf4a785036fac79b0846b820817eea0a3c6c2c53dbf3261d4b9'
    assert a['kernel_names']==b['kernel_names'] and len(a['kernel_names'])==6
    maps=[dict(t.split('=',1) for t in x['fingerprint'].split(';') if t) for x in [a,b]]
    delta={k:[maps[0].get(k),maps[1].get(k)] for k in maps[0].keys()|maps[1].keys() if maps[0].get(k)!=maps[1].get(k)}
    assert delta=={'QSB_Q_MIX':['4','1']}
    assert all(x['QSB_PK_OUTLINE']=='1' for x in maps)
    entry=(ROOT/'subset.cu').read_text()
    assert '#define QSB_PK_OUTLINE 1\n' in entry and '#define QSB_Q_MIX 4\n' in entry
    fp=(ROOT/'QsbNativeFingerprints.h').read_text()
    for key,x in zip(['current','alternate'],[a,b]):
        assert re.search(r'qsb_native_expected_'+key+r'\[\] = "([^\"]+)"',fp).group(1)==x['fingerprint']
    print('PASS: fresh host composition, exact PK-MIX4/PK-MIX1, full literal fingerprints and six identical kernel ABIs')
if __name__=='__main__':verify()
