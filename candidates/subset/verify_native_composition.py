#!/usr/bin/env python3
"""Read-only source/archive validation. No CUDA, compiler, or runtime benchmark."""
from pathlib import Path
import base64,hashlib,json,re,struct
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
    assert SHA(image)==re.search(r'qsb_carrier_cubin_sha256\[\] = "([a-f0-9]{64})"',text).group(1)
    assert len(image)==int(re.search(r'qsb_carrier_cubin_bytes = (\d+)',text).group(1))
    offset=struct.unpack_from('<Q',image,40)[0]
    size,count=struct.unpack_from('<HH',image,58)
    sections=[struct.unpack_from('<IIQQQQIIQQ',image,offset+i*size) for i in range(count)]
    objects={};functions=set()
    for section in sections:
        if section[1]!=2:continue
        strings_section=sections[section[6]]
        strings=image[strings_section[4]:strings_section[4]+strings_section[5]]
        for i in range(section[4],section[4]+section[5],section[9]):
            at,info,_,index,value,nbytes=struct.unpack_from('<IBBHQQ',image,i)
            if not index or index>=count:continue
            symbol=strings[at:].split(b'\0')[0].decode()
            if info&15==2:functions.add(symbol)
            if info&15!=1:continue
            data_section=sections[index]
            data=image[data_section[4]+value:data_section[4]+value+nbytes] if data_section[1]!=8 else bytes(nbytes)
            objects[symbol]={'bytes':nbytes,'sha256':SHA(data)}
    kernels=re.findall(r'"(_Z[^\"]+)"',text.split('qsb_carrier_b64[] = {')[0])
    assert set(kernels)<=functions,'header kernel names absent from archived ELF'
    return dict(cubin_sha256=SHA(image),bytes=len(image),
        original_source_sha256=re.search(r'source sha256 ([a-f0-9]{64})',text).group(1),
        fingerprint=knobs,kernel_names=kernels,globals=objects)
def verify():
    manifest=json.loads((ROOT/'NATIVE_COMPOSITION.json').read_text())
    assert manifest['source_files']==sources(),'host composition source changed; review and refresh binding explicitly'
    archives={n:archive(n) for n in manifest['archives']}
    assert archives==manifest['archives'],'archived image/provenance changed'
    a,b=[archives[n] for n in ['qsb_carrier_sm89.h','qsb_carrier_alt_sm89.h']]
    assert a['cubin_sha256']=='f74548427859ec03273f05e9151c810716c6475596e0213a0db3915e468aa5dc'
    assert b['cubin_sha256']=='3280df88b19baff3f234945b9be5ebf3859c8d731fe47aff4855fb09d5eea688'
    assert a['kernel_names']==b['kernel_names'] and len(a['kernel_names'])==6
    maps=[dict(t.split('=',1) for t in x['fingerprint'].split(';') if t) for x in [a,b]]
    delta={k:[maps[0].get(k),maps[1].get(k)] for k in maps[0].keys()|maps[1].keys() if maps[0].get(k)!=maps[1].get(k)}
    assert delta=={'QSB_SHA_FMA_ADD':['0','1'],'QSB_PK_HEAD_FMA':[None,'1'],
                  'QSB_GLV_FP32_CARRY':[None,'1'],'QSB_PK_OUTLINE':[None,'1']}
    assert maps[0]['QSB_Q_MIX']==maps[1]['QSB_Q_MIX']=='2'
    objects=[{k:v for k,v in x['globals'].items() if k!='qsb_carrier_knobs'} for x in [a,b]]
    assert len(objects[0])==28 and objects[0]==objects[1],'module-global state differs'
    bound={n:int(z) for n,z in re.findall(r'\{"([^"]+)", (\d+)\}',(ROOT/'QsbNativeGlobals.h').read_text())}
    assert bound=={n:v['bytes'] for n,v in objects[0].items()},'incomplete runtime global inventory'
    fp=(ROOT/'QsbNativeFingerprints.h').read_text()
    for key,x in zip(['current','alternate'],[a,b]):
        assert re.search(r'qsb_native_expected_'+key+r'\[\] = "([^\"]+)"',fp).group(1)==x['fingerprint']
    source=json.loads((ROOT/'SOURCE-MANIFEST.json').read_text())
    assert source['files']=={str(p.relative_to(ROOT)):SHA(p.read_bytes()) for p in sorted(ROOT.rglob('*'))
        if p.is_file() and p.name!='SOURCE-MANIFEST.json'},'package manifest is stale'
    print('PASS: fresh host/package composition, exact archivedf745/3280, full literal fingerprints, six identical kernel ABIs and 28 equal data objects')
if __name__=='__main__':verify()
