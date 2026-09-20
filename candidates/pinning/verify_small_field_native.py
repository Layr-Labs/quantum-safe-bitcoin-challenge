"""Independently check native add/sub outputs, including all input aliases."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def verify(inputs, outputs):
    raw=inputs.read_bytes();data=outputs.read_bytes()
    count=struct.unpack_from('<I',raw)[0]
    assert len(raw)==4+64*count and len(data)==288*count
    p=(1<<256)-(1<<32)-977
    counts={name:0 for name in ('add','sub','lazy')}
    examples=[]
    for i in range(count):
        a=int.from_bytes(raw[4+64*i:36+64*i],'little')
        b=int.from_bytes(raw[36+64*i:68+64*i],'little')
        values=[int.from_bytes(data[288*i+32*j:288*i+32*(j+1)],'little') for j in range(9)]
        for f,name in enumerate(counts):
            three=values[3*f:3*f+3]
            assert three[0]==three[1]==three[2],(i,name,'alias mismatch')
            expected=(a-b if name=='sub' else a+b)%p
            if three[0]%p!=expected:
                counts[name]+=1
                if len(examples)<8:
                    examples.append({'index':i,'function':name,'a':hex(a),'b':hex(b),'got':hex(three[0]),'expected_mod_p':hex(expected)})
    return {'passed':not any(counts.values()),'pairs':count,'outputs':9*count,
            'alias_equality_verified':True,'mismatching_pairs_by_function':counts,'examples':examples,
            'input_sha256':hashlib.sha256(raw).hexdigest(),
            'output_sha256':hashlib.sha256(data).hexdigest(),
            'scope':'Native arbitrary raw field-primitive diagnostic; not normal production-output evidence or a public-peer exclusion.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',type=Path);p.add_argument('outputs',type=Path);p.add_argument('proof',type=Path)
    p.add_argument('--parent-counterexample',action='store_true')
    a=p.parse_args();assert not a.proof.exists()
    result=verify(a.inputs,a.outputs);a.proof.write_text(json.dumps(result,indent=2)+'\n')
    if a.parent_counterexample:
        assert result['mismatching_pairs_by_function']=={'add':36,'sub':47,'lazy':36}
    else:
        assert result['passed']
    print(json.dumps({k:v for k,v in result.items() if k!='examples'}))
