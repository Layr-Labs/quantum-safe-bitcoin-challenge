#!/usr/bin/env python3
"""Lossless CUDA fatbin PTX decompression + complete LZ4 control-stream recipe.
Reconstruction restores the ORIGINAL compressor tokens/literals/match choices,
not a recompressed approximation. No verification of grinder behavior here.
"""
import struct
MAGIC=b'QSB50LC1'
def split(data):
 controls=bytearray();plain=bytearray();p=0
 while p<len(data):
  tok=data[p];p+=1;controls.append(tok);lit=tok>>4
  if lit==15:
   while True:
    x=data[p];p+=1;controls.append(x);lit+=x
    if x!=255:break
  assert p+lit<=len(data);plain+=data[p:p+lit];p+=lit
  if p==len(data):break
  assert p+2<=len(data);controls+=data[p:p+2];off=data[p]|(data[p+1]<<8);p+=2
  assert 0<off<=len(plain);match=(tok&15)+4
  if tok&15==15:
   while True:
    x=data[p];p+=1;controls.append(x);match+=x
    if x!=255:break
  for _ in range(match):plain.append(plain[-off])
 return bytes(controls),bytes(plain)
def join(ctrl,plain):
 out=bytearray();p=q=0
 while p<len(ctrl):
  tok=ctrl[p];p+=1;out.append(tok);n=tok>>4
  if n==15:
   while True:
    x=ctrl[p];p+=1;out.append(x);n+=x
    if x!=255:break
  assert q+n<=len(plain);out+=plain[q:q+n];q+=n
  if p==len(ctrl):break
  off=ctrl[p]|(ctrl[p+1]<<8);assert 0<off<=q;out+=ctrl[p:p+2];p+=2;match=(tok&15)+4
  if tok&15==15:
   while True:
    x=ctrl[p];p+=1;out.append(x);match+=x
    if x!=255:break
  assert q+match<=len(plain)
  for i in range(match):assert plain[q+i]==plain[q+i-off]
  q+=match
 assert q==len(plain)
 return bytes(out)
def transform(raw,inverse=False):
 out=bytearray();p=0
 if inverse:
  while True:
   x=raw.find(MAGIC,p)
   if x<0:out+=raw[p:];break
   a,b=struct.unpack_from('<II',raw,x+8);end=x+16+a+b;assert end<=len(raw)
   v=join(raw[x+16:x+16+a],raw[x+16+a:end]);out+=raw[p:x]+v;p=end
  return bytes(out)
 ranges=[];pos=0
 while True:
  x=raw.find(bytes.fromhex('50ed55ba'),pos)
  if x<0:break
  pos=x+4
  if x+16>len(raw):continue
  _,ver,hdr,size=struct.unpack_from('<IHHQ',raw,x)
  if ver!=1 or hdr!=16 or size<64 or x+hdr+size>len(raw):continue
  t=x+hdr;end=t+size;good=[]
  while t+64<=end:
   kind,v,hs,n=struct.unpack_from('<HHIQ',raw,t)
   if v!=0x101 or kind not in (1,2) or hs not in (64,80) or t+hs+n>end:break
   if kind==1 and hs==80:
    packed=struct.unpack_from('<I',raw,t+16)[0]
    if 0<packed<=n:
     try:
      b=raw[t+hs:t+hs+packed];c,l=split(b);assert join(c,l)==b
      good.append((t+hs,t+hs+packed,c,l))
     except (IndexError,AssertionError):pass
   t+=hs+n
  else:
   if t==end:ranges+=good;pos=end
 for a,b,c,l in ranges:
  assert a>=p;out+=raw[p:a]+MAGIC+struct.pack('<II',len(c),len(l))+c+l;p=b
 out+=raw[p:];return bytes(out)
