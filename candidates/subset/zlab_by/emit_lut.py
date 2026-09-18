#!/usr/bin/env python3
"""Emit the 1152-entry __constant__ initialiser for zinv32.cuh."""
import sys
sys.path.insert(0, '/root/qsb/exp/sub-by/zlab_by')
from by_table import tab

def pack(entry):
    (a,b,c,d), C, s, _ = entry
    for v in (a,b,c,d): assert -32 <= v <= 31, v
    assert -8 <= C <= 7, C
    return ((a & 63) | ((b & 63) << 6) | ((c & 63) << 12) | ((d & 63) << 18)
            | ((C & 15) << 24) | (s << 31))

L = [pack(t) for t in tab]
assert len(L) == 1152
out = []
for i in range(0, 1152, 8):
    out.append('    ' + ','.join('0x%08Xu' % v for v in L[i:i+8]) + ',')
body = '\n'.join(out)
body = body.rstrip(',')          # drop the final comma
print(body)
