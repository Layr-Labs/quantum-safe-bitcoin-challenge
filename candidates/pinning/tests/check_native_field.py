#!/usr/bin/env python3
"""Feed integer-oracle edge cases through real production CUDA primitives."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess
import tempfile

P=2**256-2**32-977
M=2**256

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('binary',type=Path)
    args=ap.parse_args()
    rng=random.Random(202609171122)
    edge=[0,1,2,2**32,2**64-1,2**128-1,2**255,P-65537,P-2,P-1,P,P+1,M-1]
    pairs=[(a,b) for a in edge for b in edge]
    pairs += [(P-i,P-j) for i in range(65472,65602) for j in range(65472,65602)]
    pairs += [(rng.getrandbits(256),rng.getrandbits(256)) for _ in range(32768)]
    with tempfile.TemporaryDirectory(prefix='qsb-native-field-') as d:
        src=Path(d)/'input.bin';out=Path(d)/'output.bin'
        data=struct.pack('<I',len(pairs))+b''.join(a.to_bytes(32,'little')+b.to_bytes(32,'little') for a,b in pairs)
        src.write_bytes(data)
        subprocess.run([str(args.binary.resolve()),str(src),str(out)],check=True)
        got=out.read_bytes();assert len(got)==len(pairs)*160
        for i,(a,b) in enumerate(pairs):
            want=[a*b%P]*3+[(a+b)%P,(a-b)%P]
            actual=[int.from_bytes(got[i*160+j*32:i*160+(j+1)*32],'little') for j in range(5)]
            assert actual==want,(i,hex(a),hex(b),actual,want)
    print(json.dumps({'passed':True,'pairs':len(pairs),'comparisons':len(pairs)*5,
                      'actual_cuda':True,'binary_sha256':hashlib.sha256(args.binary.read_bytes()).hexdigest()}))

if __name__=='__main__':main()
