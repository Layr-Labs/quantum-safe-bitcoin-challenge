from pathlib import Path
import sys,random,struct
n=int('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141',16)
lam=int('5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72',16)
a1=int('3086d221a7d46bcde86c90e49284eb15',16)
a2=int('114ca50f7a8e2f3f657c1108d9d44cfd8',16)
b1=int('e4437ed6010e88286f547fa90abfe4c3',16)
g1=int('3086D221A7D46BCDE86C90E49284EB153DAA8A1471E8CA7FE893209A45DBB031',16)
g2=int('E4437ED6010E88286F547FA90ABFE4C4221208AC9DF506C61571B4AE8AC47F71',16)
edges=[0,1,2,n-1,n,n+1,(1<<256)-1,lam,lam-1,lam+1]
for i in range(256):edges.extend([(1<<i)-1,1<<i,(1<<i)+1])
rng=random.Random(792184)
with Path(sys.argv[1]).open('wb') as f:
    for i in range(131072):
        raw=edges[i] if i<len(edges) else rng.getrandbits(256);k=raw%n
        c1=(k*g1+(1<<383))>>384;c2=(k*g2+(1<<383))>>384
        x=k-c1*a1-c2*a2;y=c1*b1-c2*a1
        assert abs(x)<(1<<128) and abs(y)<(1<<128) and (x+lam*y-k)%n==0
        f.write(raw.to_bytes(32,'little')+abs(x).to_bytes(16,'little')+abs(y).to_bytes(16,'little')+struct.pack('<II',(x<0)|((y<0)<<1),0))
print('Generated 131072 independently checked GLV scalar cases, including',len(edges),'boundary fixtures')
