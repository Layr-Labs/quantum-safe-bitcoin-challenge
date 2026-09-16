#!/usr/bin/env python3
"""Execute extracted split SHA/transport source on the CPU against hashlib.

This checks byte order, tile stride, epoch addressing and hit identity. OpenSSL
and CPU synchronization replace GPU primitives. EC arithmetic remains inherited;
this is not a full EC pipeline test, CUDA compilation, or GPU timing.
"""
import argparse
import ctypes as CT
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import shlex
import struct
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import check_candidate as audit
from preflight import source_identity
from split_experiment import materialize


WRAPPER = r'''
extern "C" void split_tile(uint64_t first,uint64_t total,int count,int offset,
        const uint32_t *mid,const uint8_t *pre,int prelen,const uint8_t *rows,
        const uint8_t *windows,uint64_t *scalars,uint8_t *combos,uint32_t *indices) {
    require(count>0 && offset>=0 && offset+count<=32768 && first+count<=total);
    memcpy(WIN3,windows,sizeof(WIN3));
    std::vector<epoch_desc_t> epochs(offset+count);
    const int stride=count*256;
    const uint32_t poison=0xa519dec7;
    std::vector<uint32_t> guarded(8*stride+32,poison);
    uint32_t *hashes=guarded.data()+16;
    for(int b=0;b<count;b++) {
        threadIdx.x=b;blockIdx.x=0;blockDim.x=256;
        kernel_build_epochs(first,total,137,6,mid,pre,prelen,rows,epochs.data()+offset);
    }
    for(int b=0;b<count;b++) launch(256,[&](int lane){
        blockIdx.x=b;gridDim.x=count;
        qsb_split_hash(epochs.data()+offset,hashes);
    });
    for(int i=0;i<16;i++)require(guarded[i]==poison && guarded[16+8*stride+i]==poison);
    for(int i=0;i<stride;i++) {
        qsb_split_load_z(hashes,stride,i,scalars+4*i);
        qsb_split_combo(combos+9*i,epochs.data()+offset+i/256,i%256);
        indices[i]=qsb_split_index(offset,i);
    }
}
extern "C" void split_decode(const uint32_t *hashes,int stride,uint64_t *out) {
    for(int i=0;i<stride;i++)qsb_split_load_z(hashes,stride,i,out+4*i);
}
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,default=HERE.parent)
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    audit.HERE=args.base.resolve()
    before=source_identity(audit.HERE)
    flags=[]
    prefix=os.environ.get("OPENSSL_PREFIX")
    if not prefix and Path("/opt/homebrew/opt/openssl@3").exists():
        prefix="/opt/homebrew/opt/openssl@3"
    if prefix:flags=[f"-I{prefix}/include",f"-L{prefix}/lib"]
    checked=decoded=tiles=0
    with tempfile.TemporaryDirectory(prefix="qsb-split-audit-") as tmp:
        root=Path(tmp);generated=root/"candidate"
        manifest=materialize(audit.HERE,generated)
        stage=(generated/"tests/gpu_epochs/split_stage.cuh").read_text()
        helpers=stage[:stage.index("__global__ void __launch_bounds__(256, 2) qsb_split_ec(")]
        source=audit.build_source()+"\n#define __launch_bounds__(...)\n"
        source+="static thread_local Dim gridDim;\nuint8_t WIN3[256][3];\n"+helpers+WRAPPER
        cpp,so=root/"split.cpp",root/"split.so"
        cpp.write_text(source)
        subprocess.run(shlex.split(os.environ.get("CXX","c++"))+
            ["-std=c++17","-O2","-shared","-fPIC","-pthread","-Wno-deprecated-declarations",
             *flags,str(cpp),"-lcrypto","-o",str(so)],check=True)
        lib=CT.CDLL(str(so))
        lib.init_hash.argtypes=[audit.P8,audit.P8,audit.P32]
        lib.split_tile.argtypes=[audit.U64,audit.U64,CT.c_int,CT.c_int,audit.P32,
            audit.P8,CT.c_int,audit.P8,audit.P8,audit.P64,audit.P8,audit.P32]
        lib.split_decode.argtypes=[audit.P32,CT.c_int,audit.P64]
        windows=list(itertools.islice(itertools.combinations(range(137,150),3),256))
        packed=audit.buf(bytes(itertools.chain.from_iterable(windows)))
        total=math.comb(137,6)
        for seed in (20260916,679162400):
            prob,_=audit.GEN.gen_subset(random.Random(seed))
            prebytes=bytes.fromhex(prob["fixed_prefix"]);aligned=len(prebytes)//64*64
            mid=(audit.U32*8)(*audit.C.sha256_midstate(prebytes[:aligned]))
            pre=audit.buf(prebytes[aligned:])
            rows=audit.buf(b"".join(bytes.fromhex(x) for x in prob["dummy_sigs"]))
            suffix=bytes.fromhex(prob["tail_section"]+prob["tx_suffix"])
            constant=suffix+b"\x80"+b"\x00"*5+(prob["total_preimage_len"]*8).to_bytes(8,"big")
            lib.init_hash(rows,packed,(audit.U32*69)(*struct.unpack(">69I",constant)))
            for first,count,offset in ((0,2,0),(32767,3,128),(total-1,1,32767)):
                n=count*256
                values=(audit.U64*(4*n))();combos=(audit.U8*(9*n))();indices=(audit.U32*n)()
                lib.split_tile(first,total,count,offset,mid,pre,len(pre),rows,packed,values,combos,indices)
                seen=set()
                for i in range(n):
                    skip=list(combos[9*i:9*i+9])
                    # Independent combinadic reference, not the device unranking helper.
                    rank=first+i//256;early=[];lo=0
                    for slot in range(6):
                        for pos in range(lo,137):
                            span=math.comb(136-pos,5-slot) if 136-pos>=5-slot else 0
                            if rank<span:early.append(pos);lo=pos+1;break
                            rank-=span
                    assert skip==early+list(windows[i%256]),(seed,first,i,skip)
                    assert indices[i]==offset*256+i
                    assert tuple(skip) not in seen;seen.add(tuple(skip))
                    digest=hashlib.sha256(hashlib.sha256(audit.PB.sub_preimage(prob,skip)).digest()).digest()
                    got=sum(int(values[4*i+k])<<(64*k) for k in range(4))
                    assert got==int.from_bytes(digest,"big"),(seed,first,i)
                    checked+=1
                tiles+=1
                print(f"Split SHA/transport: seed={seed}, epochs={count}, offset={offset}: PASS",flush=True)
        rng=random.Random(9182)
        edges=[0,1,audit.C.N-1,audit.C.N,audit.C.N+1,(1<<256)-1]
        for stride in (1,31,256,768):
            want=[edges[i%len(edges)] if i<len(edges) else rng.getrandbits(256) for i in range(stride)]
            words=[(v>>(32*(7-k)))&0xffffffff for k in range(8) for v in want]
            out=(audit.U64*(stride*4))()
            lib.split_decode((audit.U32*len(words))(*words),stride,out)
            for i,v in enumerate(want):
                assert sum(int(out[4*i+k])<<(64*k) for k in range(4))==v
                decoded+=1
        assert source_identity(audit.HERE)==before,"Base changed during audit"
        report={"status":"PASS","validation_level":"cpu_reference",
            "sha256d_and_candidate_identities":checked,"guarded_tiles":tiles,
            "additional_scalar_transports":decoded,"base":before,
            "experiment":manifest["experiment"],"generator_sha256":manifest["generator_sha256"],
            "audit_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cuda_compiled":False,"gpu_executed":False,"performance":"unknown",
            "limitations":"No CUDA/PTX, full EC pipeline execution, GPU synchronization validation or throughput measurement."}
        if args.report:args.report.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
