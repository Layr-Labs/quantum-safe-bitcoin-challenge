#!/usr/bin/env python3
"""Byte-exact pooling of complete repeated subset hit JSON stanzas.
Not a hit verifier: preserves all original whitespace, ordering and duplicates.
Applied after historical reversible TAR codecs; inverse precedes them.
"""
import re,struct,hashlib
MAGIC=b'QSB50HITPOOL1\0';TOKEN=b'\x00Q5H\x00'
PAT=re.compile(rb'\{\s*"bench"\s*:\s*"subset"\s*,\s*"skip"\s*:\s*\[\s*\d+(?:\s*,\s*\d+){8}\s*\]\s*,\s*"recid"\s*:\s*[01]\s*\}')
def encode(raw):
 assert TOKEN not in raw
 pool=[];ids={};hits=0
 def sub(m):
  nonlocal hits
  b=m[0];hits+=1
  if b not in ids:ids[b]=len(pool);pool.append(b)
  return TOKEN+struct.pack('<I',ids[b])
 body=PAT.sub(sub,raw);out=bytearray(MAGIC);out+=struct.pack('<QII',len(raw),len(pool),hits)+hashlib.sha256(raw).digest()
 for b in pool:out+=struct.pack('<I',len(b))+b
 return bytes(out)+body
def decode(blob):
 assert blob.startswith(MAGIC);p=len(MAGIC);size,n,hits=struct.unpack_from('<QII',blob,p);p+=16;sha=blob[p:p+32];p+=32;pool=[]
 for _ in range(n):
  k=struct.unpack_from('<I',blob,p)[0];p+=4;pool.append(blob[p:p+k]);p+=k
 raw=[];count=0
 while True:
  q=blob.find(TOKEN,p)
  if q<0:raw.append(blob[p:]);break
  raw.append(blob[p:q]);i=struct.unpack_from('<I',blob,q+len(TOKEN))[0];assert i<n;raw.append(pool[i]);count+=1;p=q+len(TOKEN)+4
 out=b''.join(raw);assert count==hits and len(out)==size and hashlib.sha256(out).digest()==sha
 return out
