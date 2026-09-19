# SPDX-License-Identifier: GPL-3.0-only
"""Generate independent raw scalar fixtures and check every window identity.
Binary output:131072 records of32 little-endian scalar bytes.
"""
from pathlib import Path
import random,sys,json
n=int('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141',16)
edges=[0,1,2,n-1,n,n+1,(1<<256)-1,n//2,n//2+1]
for bit in range(256):edges += [(1<<bit)-1,1<<bit,(1<<bit)+1]
rng=random.Random(792184)
counts={str(w):0 for w in range(11,17)}
with Path(sys.argv[1]).open('wb') as f:
    for i in range(131072):
        raw=edges[i] if i<len(edges) else rng.getrandbits(256)
        signed=2*(raw%n)-n;bits=signed%(1<<256);negative=int(signed<0)
        for windows in range(11,17):
            q,r=divmod(256,windows);prefix=0;shift=0
            for c in range(windows):
                width=q+(c<r);field=(bits>>(shift+1))&((1<<width)-1)
                sign=-negative if c==windows-1 else (field>>(width-1))-1
                index=(field^sign)&((1<<(width-1))-1)
                term=(-1 if sign<0 else 1)*(2*index+1)<<shift
                if c<windows-1 and c:
                    assert 0<abs(prefix)<1<<shift<=abs(term)
                    assert abs(prefix)+abs(term)<n
                prefix+=term;shift+=width
            assert shift==256 and prefix==signed
            counts[str(windows)]+=1
        f.write(raw.to_bytes(32,'little'))
print(json.dumps({'raw_scalars':131072,'boundary_inputs':len(edges),
    'exact_reconstruction_and_nonsingular_prefix_cases':counts}))
