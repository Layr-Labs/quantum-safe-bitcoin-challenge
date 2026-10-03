#!/usr/bin/env python3
"""Lossless sampled CSV codec, canonical integers and warp-local deltas."""
import struct

def varint(n):
    n=2*n if n>=0 else -2*n-1
    b=bytearray()
    while n>=128:
        b.append((n&127)|128); n>>=7
    b.append(n); return b

def encode(raw):
    lines=raw.splitlines(keepends=True)
    assert all(l.endswith(b'\n') for l in lines)
    rows=[[int(x) for x in l.split(b',')] for l in lines[1:]]
    out=bytearray(lines[0])+struct.pack('<I',len(rows))
    cols=[bytearray() for _ in range(11)]; prev={}
    for r in rows:
        orig=r[:]
        if r[3]%4==0:
            base=orig[4:]; key=(r[1],r[3])
            r[4]=orig[4]-prev.get(key,0); prev[key]=orig[4]
            for j in range(5,11):r[j]=orig[j]-orig[j-1]
        else:
            for j in range(4,11):r[j]=orig[j]-base[j-4]
        for j,x in enumerate(r):cols[j]+=varint(x)
    for c in cols:out+=struct.pack('<I',len(c))+c
    return bytes(out)

def decode(blob):
    cut=blob.index(b'\n')+1; out=[blob[:cut]]
    count=struct.unpack_from('<I',blob,cut)[0]; pos=cut+4; cols=[]
    for _ in range(11):
        n=struct.unpack_from('<I',blob,pos)[0]; pos+=4
        end=pos+n; values=[]; v=0; shift=0
        while pos<end:
            x=blob[pos]; pos+=1; v|=(x&127)<<shift
            if x<128:
                values.append(v//2 if v%2==0 else -(v//2)-1)
                v=shift=0
            else:shift+=7
        assert len(values)==count and shift==0
        cols.append(values)
    assert pos==len(blob); prev={}
    for i in range(count):
        r=[c[i] for c in cols]
        if r[3]%4==0:
            key=(r[1],r[3]); r[4]+=prev.get(key,0); prev[key]=r[4]
            for j in range(5,11):r[j]+=r[j-1]
            base=r[4:]
        else:
            for j in range(4,11):r[j]+=base[j-4]
        out.append((','.join(map(str,r))+'\n').encode())
    return b''.join(out)
