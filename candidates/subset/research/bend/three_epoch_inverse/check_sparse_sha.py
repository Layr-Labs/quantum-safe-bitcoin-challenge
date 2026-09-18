#!/usr/bin/env python3
"""Independent fixed-padding SHA schedule and successor source audit."""

import hashlib
import random
import ctypes
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "k3_sparse_successor"
HEADER = SRC / "tests/gpu_epochs/sparse_sha_fixed.cuh"
TREE = SRC / "tests/gpu_epochs/tree.cu"
MASK = (1 << 32) - 1
K = (
    0x428A2F98,0x71374491,0xB5C0FBCF,0xE9B5DBA5,0x3956C25B,0x59F111F1,0x923F82A4,0xAB1C5ED5,
    0xD807AA98,0x12835B01,0x243185BE,0x550C7DC3,0x72BE5D74,0x80DEB1FE,0x9BDC06A7,0xC19BF174,
    0xE49B69C1,0xEFBE4786,0x0FC19DC6,0x240CA1CC,0x2DE92C6F,0x4A7484AA,0x5CB0A9DC,0x76F988DA,
    0x983E5152,0xA831C66D,0xB00327C8,0xBF597FC7,0xC6E00BF3,0xD5A79147,0x06CA6351,0x14292967,
    0x27B70A85,0x2E1B2138,0x4D2C6DFC,0x53380D13,0x650A7354,0x766A0ABB,0x81C2C92E,0x92722C85,
    0xA2BFE8A1,0xA81A664B,0xC24B8B70,0xC76C51A3,0xD192E819,0xD6990624,0xF40E3585,0x106AA070,
    0x19A4C116,0x1E376C08,0x2748774C,0x34B0BCB5,0x391C0CB3,0x4ED8AA4A,0x5B9CCA4F,0x682E6FF3,
    0x748F82EE,0x78A5636F,0x84C87814,0x8CC70208,0x90BEFFFA,0xA4506CEB,0xBEF9A3F7,0xC67178F2,
)
IV = (0x6A09E667,0xBB67AE85,0x3C6EF372,0xA54FF53A,0x510E527F,0x9B05688C,0x1F83D9AB,0x5BE0CD19)

def rr(x,n): return ((x >> n) | (x << (32-n))) & MASK

def compress(words):
    w=list(words)+[0]*48
    for i in range(16,64):
        s0=rr(w[i-15],7)^rr(w[i-15],18)^(w[i-15]>>3)
        s1=rr(w[i-2],17)^rr(w[i-2],19)^(w[i-2]>>10)
        w[i]=(w[i-16]+s0+w[i-7]+s1)&MASK
    a,b,c,d,e,f,g,h=IV
    for i in range(64):
        s1=rr(e,6)^rr(e,11)^rr(e,25)
        ch=(e&f)^((~e)&g)
        t1=(h+s1+ch+K[i]+w[i])&MASK
        s0=rr(a,2)^rr(a,13)^rr(a,22)
        maj=(a&b)^(a&c)^(b&c)
        t2=(s0+maj)&MASK
        h,g,f,e,d,c,b,a=g,f,e,(d+t1)&MASK,c,b,a,(t1+t2)&MASK
    return tuple((x+y)&MASK for x,y in zip(IV,(a,b,c,d,e,f,g,h)))

def fixed(message, bad_length=False):
    assert len(message) in (32,33)
    block=message+b"\x80"+b"\0"*(55-len(message))+((len(message)*8)+(8 if bad_length else 0)).to_bytes(8,"big")
    return compress([int.from_bytes(block[i:i+4],"big") for i in range(0,64,4)])

def compiled_source():
    """Compile and execute the actual successor header with host macro stubs."""
    gpu=(SRC/"GPUHash.h").read_text()
    prefix=gpu.split("//Take the last 8 bytes",1)[0]
    code=("#include <stdint.h>\n#define __device__\n#define __constant__\n"
          "#define __forceinline__ inline\n"+prefix+"\n"+HEADER.read_text()+
          '\nextern "C" void d32(uint32_t*o,const uint32_t*m){_SHA256TransformDigest32(o,m);}'
          '\nextern "C" void p33(uint32_t*o,const uint32_t*m){_SHA256TransformPubkey33(o,m);}\n')
    td=tempfile.TemporaryDirectory(prefix="qsb-sparse-sha-")
    root=Path(td.name); cpp=root/"audit.cpp"; so=root/"audit.so"
    cpp.write_text(code)
    subprocess.run(["clang++","-std=c++17","-O2","-shared","-fPIC",str(cpp),"-o",str(so)],check=True)
    lib=ctypes.CDLL(str(so)); wordp=ctypes.POINTER(ctypes.c_uint32)
    lib.d32.argtypes=[wordp,wordp];lib.p33.argtypes=[wordp,wordp]
    return td,lib

def main():
    hs=HEADER.read_text(); ts=TREE.read_text()
    required=("_SHA256TransformDigest32","_SHA256TransformPubkey33","s1(256u)","s0(0x80000000u)","s1(0x108u)")
    for s in required: assert s in hs,s
    assert '#include "sparse_sha_fixed.cuh"' in ts
    assert ts.count("_SHA256TransformDigest32(s2,state)")==1
    assert ts.count("_SHA256TransformPubkey33(hs,pb)")==1
    assert "#define ZLAB_LAUNCH_BLOCKS 262144" in ts
    assert "cudaDeviceSetLimit(cudaLimitStackSize, 2048)" in ts
    rng=random.Random(0x5A256)
    cases=0; mutations=0
    td,lib=compiled_source()
    for n in (32,33):
        samples=[bytes(n),bytes([0xff])*n,bytes(range(n))]+[rng.randbytes(n) for _ in range(4096)]
        for msg in samples:
            got=b"".join(x.to_bytes(4,"big") for x in fixed(msg))
            assert got==hashlib.sha256(msg).digest()
            words=(ctypes.c_uint32*(9 if n==33 else 8))(*[int.from_bytes(msg[i:i+4],"big") for i in range(0,n-1 if n==33 else n,4)],*(([int.from_bytes(msg[32:]+b"\x80\0\0","big")]) if n==33 else []))
            out=(ctypes.c_uint32*8)()
            (lib.p33 if n==33 else lib.d32)(out,words)
            assert b"".join(int(x).to_bytes(4,"big") for x in out)==got
            if b"".join(x.to_bytes(4,"big") for x in fixed(msg,True))!=got: mutations+=1
            cases+=1
    td.cleanup()
    print(f"PASS: {cases} source-executed fixed-padding SHA cases; {mutations} length mutations rejected")

if __name__=="__main__": main()
