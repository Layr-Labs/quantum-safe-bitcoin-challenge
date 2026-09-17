import sys, random, hashlib, struct, inspect, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'harness'))
import crypto as C
rng=random.Random(20260916)
# Reuse reference SHA compression with an explicit starting state.
src=inspect.getsource(C.sha256_midstate).replace('def sha256_midstate(data: bytes):','def resume(data, initial):').replace('h = list(_H0)','h = list(initial)')
ns=dict(vars(C));exec(src,ns);resume=ns['resume']
sha_cases=0
for seed in range(4):
 prefix=bytes(rng.getrandbits(8) for _ in range(9920))
 suffix=bytes(rng.getrandbits(8) for _ in range(75))
 mid=C.sha256_midstate(prefix)
 for seq in [0,1,0x80000000,0xffffffff,rng.getrandbits(32)]:
  block=bytearray(suffix[:64]);block[31:35]=seq.to_bytes(4,'little')
  state=resume(bytes(block),mid)
  for lt in [0,1,255,256,0xffffffff,500000000,1744599999]+[rng.getrandbits(32) for _ in range(10)]:
   tail=bytearray(64);tail[:11]=suffix[64:];tail[11]=128;tail[56:]=(9995*8).to_bytes(8,'big')
   w=list(struct.unpack('>16I',tail));be=int.from_bytes(lt.to_bytes(4,'little'),'big')
   w[0]=(w[0]&0xffffff00)|(be>>24);w[1]=(w[1]&255)|((be<<8)&0xffffffff)
   first=resume(struct.pack('>16I',*w),state)
   b=struct.pack('>8I',*first)+b'\x80'+bytes(23)+(256).to_bytes(8,'big')
   final=resume(b,C._H0); got=struct.pack('>8I',*final)
   actual=bytearray(suffix);actual[31:35]=seq.to_bytes(4,'little');actual[67:71]=lt.to_bytes(4,'little')
   assert got==C.sha256d(prefix+actual)
   z=[(final[6-2*i]<<32)|final[7-2*i] for i in range(4)]
   assert sum(x<<(64*i) for i,x in enumerate(z))==int.from_bytes(got,'big')
   sha_cases+=1

# Bind the production sparse-transform defaults this chain relies on.
source=Path(__file__).with_name('pinning.cu').read_text()
assert '#define QSB_SPARSE_TAIL 1' in source
assert '#define QSB_SPARSE_D 1' in source
assert '_SHA256TransformFastTail11(state, w0, w1, w2);' in source
assert '_SHA256TransformDigest32(s2, state);' in source
print(json.dumps(dict(status='PASS',sha256d_cases=sha_cases,limitation='Python SHA and scalar-layout models; no compilation/OpenSSL/GPU execution'),indent=2))
