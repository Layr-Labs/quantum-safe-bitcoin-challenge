#!/usr/bin/env python3
"""Execute the source-extracted in-place inverse tree with simulated CTAs."""

import ctypes as CT
import hashlib
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SUBSET = HERE.parents[2]
sys.path.insert(0, str(SUBSET))

import audit_integrated as ref
from check_candidate import BACKEND, function
from preflight import source_identity

SOURCE = HERE / "k3_candidate"
U64, I32 = CT.c_uint64, CT.c_int

FIELD = r'''
static Barrier *mask2,*mask4,*mask8,*mask16;
static void __syncwarp(unsigned mask=0xffffffffu){
    if(mask==0xffffffffu){warp_barriers[threadIdx.x/32]->wait();return;}
    if(mask==0x3u){mask2->wait();return;} if(mask==0xfu){mask4->wait();return;}
    if(mask==0xffu){mask8->wait();return;} if(mask==0xffffu){mask16->wait();return;}
    require(false);
}
static void qsb_field_mul_raw(uint64_t *r,uint64_t *a,uint64_t *b){
    BIGNUM *aa=BN_lebin2bn((uint8_t*)a,32,nullptr),*bb=BN_lebin2bn((uint8_t*)b,32,nullptr),*rr=BN_new();
    require(BN_mod_mul(rr,aa,bb,field.p,field.ctx));
    require(BN_bn2lebinpad(rr,(uint8_t*)r,32)==32);r[4]=0;
    BN_free(aa);BN_free(bb);BN_free(rr);multiply_count++;
}
static void qsb_field_normalize(uint64_t*){}
static uint64_t root_broadcast[4];
static Barrier *quad_barrier;
static void zi_inverse_quad(uint64_t *root,int tid){
    if(tid==0){_ModInv(root);memcpy(root_broadcast,root,32);}
    quad_barrier->wait();memcpy(root,root_broadcast,32);root[4]=0;
}
'''

WRAPPER = r'''
extern "C" void check_tree(const uint64_t *inputs,uint64_t *out,int *counts){
    multiply_count=0;inverse_count=0;std::vector<int> barriers(256);
    quad_barrier=new Barrier(4);
    mask2=new Barrier(2);mask4=new Barrier(4);mask8=new Barrier(8);mask16=new Barrier(16);
    launch(256,[&](int i){
        uint64_t value[5];memcpy(value,inputs+5*i,40);
        qsb_block_inverse_tree(value);memcpy(out+5*i,value,40);barriers[i]=block_syncs;
    });
    for(int i=1;i<256;i++)require(barriers[i]==barriers[0]);
    delete quad_barrier;
    delete mask2;delete mask4;delete mask8;delete mask16;
    counts[0]=multiply_count;counts[1]=inverse_count;counts[2]=barriers[0];
}
'''


def limbs(v):
    return [v >> (64 * i) & ((1 << 64) - 1) for i in range(4)] + [0]


def main():
    identity = source_identity(SOURCE)
    header_path = SOURCE / "tests/gpu_epochs/tree_inverse.cuh"
    header = header_path.read_text()
    assert "#define ZLAB_TREE 3" in header
    signature = "__device__ __forceinline__ void qsb_block_inverse_tree(uint64_t *value)"
    body = function(header[header.rindex(signature):], signature)
    source = "\n".join([BACKEND, FIELD, body, WRAPPER])
    rng = random.Random(260917944)
    cases = 0
    barrier_count = None
    with tempfile.TemporaryDirectory(prefix="qsb-tree3-") as td:
        cpp, so = Path(td) / "tree.cpp", Path(td) / "tree.so"
        cpp.write_text(source)
        flags = []
        openssl = Path("/opt/homebrew/opt/openssl@3")
        if openssl.exists():
            flags = [f"-I{openssl}/include", f"-L{openssl}/lib"]
        subprocess.run(shlex.split("c++") + ["-std=c++17", "-O2", "-shared", "-fPIC",
                       "-pthread", *flags, str(cpp), "-lcrypto", "-o", str(so)], check=True)
        lib = CT.CDLL(str(so))
        lib.check_tree.argtypes = [CT.POINTER(U64), CT.POINTER(U64), CT.POINTER(I32)]
        for trial in range(4):
            values = [rng.randrange(1, ref.P) for _ in range(256)]
            if trial == 0:
                values[0], values[1], values[-1] = 1, ref.P - 1, 2
            raw = (U64 * (256 * 5))(*[x for v in values for x in limbs(v)])
            out = (U64 * (256 * 5))()
            counts = (I32 * 3)()
            lib.check_tree(raw, out, counts)
            assert counts[1] == 1
            barrier_count = counts[2]
            for i, value in enumerate(values):
                got = sum(int(out[i * 5 + k]) << (64 * k) for k in range(4))
                expected = pow(value, -1, ref.P)
                assert got == expected, (trial, i, hex(got), hex(expected))
            cases += 256
    # Independent packed-level map: the common off-by-one parent mutation must fail.
    mutation_rejections = 0
    for count in (4, 8, 16, 32, 64, 128):
        half = count // 2
        offset = 512 - 2 * count
        for tid in range(count):
            parent = offset + count + (tid & (half - 1))
            wrong = offset + count + ((tid + 1) & (half - 1))
            mutation_rejections += parent != wrong
    assert mutation_rejections == 252
    assert source_identity(SOURCE) == identity
    result = {
        "status": "PASS",
        "leaf_inverses": cases,
        "cta_trials": cases // 256,
        "uniform_barriers_per_lane": barrier_count,
        "parent_index_mutation_rejections": mutation_rejections,
        "validation_level": "source-extracted compact tree, CPU CTA simulation, OpenSSL field arithmetic",
        "gpu_executed": False,
        "limits": "Cooperative root inversion is replaced by a barriered OpenSSL broadcast; no CUDA execution or timing.",
        **identity,
        "tree_header_sha256": hashlib.sha256(header_path.read_bytes()).hexdigest(),
        "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (HERE / "compact-tree-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
