#!/usr/bin/env python3
"""Execute the source-extracted K3 split and recovery schedule on CPU.

OpenSSL supplies field arithmetic.  The production qsb_k2s_pre/post helpers
are compiled unchanged; the wrapper mirrors the six multiplications and the
C -> A -> B finish order in kernel_digest.  This is a correctness audit, not
a CUDA execution or throughput claim.
"""

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
from check_deferred_source import FIELD
from preflight import source_identity

SOURCE = HERE / "k3_candidate"
U64 = CT.c_uint64

WRAPPER = r'''
extern "C" void k3_recover(const uint64_t *states,const uint64_t *rx,
                            const uint64_t *ry,const uint64_t *cc,
                            const uint8_t *valid,uint64_t *xs,uint32_t *par){
    memcpy(QSB_U2R_C,cc,32);
    uint64_t den[3][5],m1[3][4],m2[3][4];
    for(int i=0;i<3;i++){
        const uint64_t *s=states+i*16;
        qsb_xyzz_finish_prepare((uint64_t*)s,(uint64_t*)s+8,
                                (uint64_t*)s+12,(uint64_t*)rx,den[i]);
        qsb_k2s_pre((uint64_t*)s+4,(uint64_t*)s+8,(uint64_t*)s+12,
                    (uint64_t*)ry,m1[i],m2[i]);
        if(!valid[i]){den[i][0]=1;den[i][1]=den[i][2]=den[i][3]=den[i][4]=0;}
    }
    uint64_t prodAB[5],leaf[5],invAB[5];
    qsb_field_mul_raw(prodAB,den[0],den[1]);
    qsb_field_mul_raw(leaf,prodAB,den[2]);
    _ModInv(leaf);
    qsb_field_mul_raw(invAB,leaf,den[2]);
    const int order[3]={2,0,1};
    for(int oi=0;oi<3;oi++){
        int i=order[oi];if(!valid[i])continue;
        uint64_t inv[5];
        if(i==2)qsb_field_mul_raw(inv,leaf,prodAB);
        else if(i==0)qsb_field_mul_raw(inv,invAB,den[1]);
        else qsb_field_mul_raw(inv,invAB,den[0]);
        par[i]=qsb_k2s_post(m1[i],m2[i],inv,(uint64_t*)rx,(uint64_t*)ry,
                            xs+i*8,xs+i*8+4);
    }
}
'''


def limbs(v):
    return [v >> (64 * i) & ((1 << 64) - 1) for i in range(4)]


def array(values):
    return (U64 * (4 * len(values)))(*[x for v in values for x in limbs(v)])


def integer(values):
    return sum(int(v) << (64 * i) for i, v in enumerate(values))


def main():
    identity = source_identity(SOURCE)
    tree = (SOURCE / "tests/gpu_epochs/tree.cu").read_text()
    extracted = "\n".join([
        "static uint64_t QSB_U2R_C[4];",
        function(tree, "__device__ __forceinline__ void qsb_xyzz_finish_prepare("),
        function(tree, "__device__ __forceinline__ void qsb_k2s_pre("),
        function(tree, "__device__ __forceinline__ uint32_t qsb_k2s_post("),
    ])
    # The production raw multiplier is PTX-only.  Bind its call sites to the
    # same exact modular operation used by this source-extraction framework.
    raw = """static void _ModSub256(uint64_t*r,uint64_t*b){field_op(r,r,b,2);}
static void qsb_field_mul_raw(uint64_t*r,uint64_t*a,uint64_t*b){qsb_field_mul(r,a,b);}
"""
    source = "\n".join([BACKEND, FIELD, raw, extracted, WRAPPER])
    rng = random.Random(260917317)
    R = ref.scalar_mult(1_984_321)
    c = 3 * R[0] * R[0] * pow(2 * R[1], -1, ref.P) % ref.P
    cases = 0
    singular_cases = 0
    with tempfile.TemporaryDirectory(prefix="qsb-k3-runtime-") as td:
        cpp, so = Path(td) / "audit.cpp", Path(td) / "audit.so"
        cpp.write_text(source)
        flags = []
        openssl = Path("/opt/homebrew/opt/openssl@3")
        if openssl.exists():
            flags = [f"-I{openssl}/include", f"-L{openssl}/lib"]
        subprocess.run(shlex.split("c++") + ["-std=c++17", "-O2", "-shared", "-fPIC",
                       "-pthread", *flags, str(cpp), "-lcrypto", "-o", str(so)], check=True)
        lib = CT.CDLL(str(so))
        lib.k3_recover.argtypes = [CT.POINTER(U64), CT.POINTER(U64), CT.POINTER(U64),
                                   CT.POINTER(U64), CT.POINTER(CT.c_uint8),
                                   CT.POINTER(U64), CT.POINTER(CT.c_uint32)]
        for case in range(4096):
            points = [ref.scalar_mult(rng.randrange(1, ref.N)) for _ in range(3)]
            valid = [1, 1, 1]
            if case < 7:
                valid[case % 3] = 0
                points[case % 3] = R
                singular_cases += 1
            states = []
            for point in points:
                z = rng.randrange(1, ref.P)
                zz, zzz = z * z % ref.P, z * z * z % ref.P
                states.extend((point[0] * zz % ref.P, point[1] * zzz % ref.P, zz, zzz))
            out = (U64 * 24)()
            par = (CT.c_uint32 * 3)()
            lib.k3_recover(array(states), array([R[0]]), array([R[1]]), array([c]),
                           (CT.c_uint8 * 3)(*valid), out, par)
            for i in range(3):
                if not valid[i]:
                    continue
                expected = [ref.affine_add(points[i], R),
                            ref.affine_add(points[i], (R[0], (-R[1]) % ref.P))]
                got = [integer(out[i * 8:i * 8 + 4]), integer(out[i * 8 + 4:i * 8 + 8])]
                assert got == [q[0] for q in expected], (case, i)
                assert par[i] == ((expected[0][1] & 1) | ((expected[1][1] & 1) << 1))
            cases += 1
    assert source_identity(SOURCE) == identity, "candidate changed during audit"
    result = {
        "status": "PASS",
        "triple_schedules": cases,
        "candidate_recoveries": cases * 3 - singular_cases,
        "identity_substituted_singular_candidates": singular_cases,
        "finish_order": "C,A,B",
        "validation_level": "source-extracted production pre/post with exact K3 split schedule",
        "gpu_executed": False,
        "limitations": "OpenSSL field operations replace PTX arithmetic; no CUDA execution or timing.",
        **identity,
        "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (HERE / "integrated-k3-runtime-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
