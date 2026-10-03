#!/usr/bin/env python3
"""Lossless NUL-terminated ELF base64 token normalization; no image bytes removed."""
import re,base64,struct,json,hashlib
MAGIC=b'QSB50B64\0'
PAT=re.compile(rb'(?<![A-Za-z0-9+/=])([A-Za-z0-9+/=]{100,120})(?=\x00)')
def encode(raw):
 parts=[];words=[];lengths=[];pos=0
 for m in PAT.finditer(raw):
  try:b=base64.b64decode(m[1],validate=True)
  except Exception:continue
  if base64.b64encode(b)!=m[1]:continue
  parts.append(raw[pos:m.start(1)]);words.append(b);lengths.append(len(m[1]));pos=m.end(1)
 parts.append(raw[pos:]);r=json.dumps({'lengths':[len(p) for p in parts],'tokens':lengths,'sha256':hashlib.sha256(raw).hexdigest()},separators=(',',':')).encode();a=b''.join(parts)
 return MAGIC+struct.pack('<IQ',len(r),len(a))+r+a+b''.join(words)
def decode(blob):
 assert blob.startswith(MAGIC);p=len(MAGIC);n,a=struct.unpack_from('<IQ',blob,p);p+=12;r=json.loads(blob[p:p+n]);p+=n;text=blob[p:p+a];p+=a;q=0;parts=[]
 for i,n in enumerate(r['lengths']):
  parts.append(text[q:q+n]);q+=n
  if i<len(r['tokens']):
   z=r['tokens'][i];k=z//4*3;token=base64.b64encode(blob[p:p+k]);p+=k;assert len(token)==z;parts.append(token)
 assert p==len(blob) and q==len(text);raw=b''.join(parts);assert hashlib.sha256(raw).hexdigest()==r['sha256'];return raw
