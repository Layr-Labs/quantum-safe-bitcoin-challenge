#!/usr/bin/env python3
"""Reproduce the CPU-only cost probe that motivated deferred hit publication.

From the benchmark repository root:
    python3 candidates/pinning/gate_cost_probe.py --problem problems/pinning.bin

Requirements: g++, OpenSSL development headers/library, and a generated problem.
The production parameter loader and exact host gate are extracted from the
adjacent pinning.cu, not reimplemented. Compilation and execution use a temporary
directory. No candidate or problem file is modified; CUDA is not used.

Original experiment: g++ -O3 -Wno-deprecated-declarations, linked with -lcrypto;
three rounds of 3,000 qsb_host_exact_hit calls each. Every round uses sequence
0x80000000, locktime 500000000 + i*1009, and recid i&1 for i=0..2999. Timers exclude
parameter loading and OpenSSL group setup. The historical results were 368.960,
383.934, and 374.496 microseconds/call, with zero hits in each round. These are
local CPU observations, not GPU speedups or promised timings on other machines.

This measures one exact recovery/hash attempt, not qsb_gate_accept's possible
second recid attempt, file publication, or CUDA scheduling. The candidates are
ordinary deterministic inputs, not preselected successful hits. No predicate is
weakened: the compiled leading-zero threshold is the production default N=24.

SPDX-License-Identifier: GPL-3.0-only
"""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HEADERS = """#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <openssl/sha.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#define QSB_ZEROS_N 24
"""

# This is the unchanged driver used for the original 3 x 3,000-call experiment.
DRIVER = r'''
int main(int argc, char **argv){
 pinning2_params_t pp; if(load_pinning2(argv[1], &pp))return 1;
 EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);BN_CTX *ctx=BN_CTX_new();BIGNUM *order=BN_new(),*nri=BN_new(),*rx=BN_new(),*ry=BN_new();EC_POINT *R=EC_POINT_new(grp);
 EC_GROUP_get_order(grp,order,ctx);BN_lebin2bn(pp.neg_r_inv,32,nri);BN_lebin2bn(pp.u2r_x,32,rx);BN_lebin2bn(pp.u2r_y,32,ry);EC_POINT_set_affine_coordinates_GFp(grp,R,rx,ry,ctx);
 const int N=3000;volatile unsigned hits=0;
 for(int rep=0;rep<3;rep++) { struct timespec a,b;clock_gettime(CLOCK_MONOTONIC,&a);
 for(int i=0;i<N;i++) hits+=qsb_host_exact_hit(&pp,0x80000000u,500000000u+ i*1009u,i&1,grp,ctx,order,nri,R);
 clock_gettime(CLOCK_MONOTONIC,&b);double sec=b.tv_sec-a.tv_sec+1e-9*(b.tv_nsec-a.tv_nsec);
 printf("exact_hit calls=%d sec=%.6f microseconds_per_call=%.3f hits=%u\n",N,sec,sec*1e6/N,hits);
 }
}
'''


def extract_source():
    """Fail on layout changes instead of silently benchmarking a stale copy."""
    source = (HERE / "pinning.cu").read_text()
    loader_start = source.index("typedef struct {\n    uint32_t midstate[8];")
    loader_end = source.index("\ntypedef struct {\n    uint64_t alpha[4];", loader_start)
    gate_start = source.index("static int qsb_host_zeros(")
    gate_end = source.index("\n#endif\n\n\nint main(", gate_start)
    loader = source[loader_start:loader_end]
    gate = source[gate_start:gate_end]
    assert "static int load_pinning2(" in loader
    assert "static int qsb_host_exact_hit(" in gate
    assert "static int qsb_gate_accept(" in gate
    return HEADERS + loader + gate + DRIVER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path, default=ROOT / "problems" / "pinning.bin")
    parser.add_argument("--compile-only", action="store_true",
                        help="check extraction and compilation without rerunning CPU timings")
    args = parser.parse_args()
    compiler = shutil.which("g++")
    if not compiler:
        parser.error("g++ is required")
    if not args.problem.is_file():
        parser.error("problem file is missing; run the benchmark setup first")
    source = extract_source()
    print("CPU-only probe; no CUDA execution or GPU throughput measurement.", flush=True)
    print("extracted_cpp_sha256=" + hashlib.sha256(source.encode()).hexdigest(), flush=True)
    print("problem_sha256=" + hashlib.sha256(args.problem.read_bytes()).hexdigest(), flush=True)
    with tempfile.TemporaryDirectory(prefix="qsb-gate-cost-") as directory:
        build = Path(directory)
        cpp = build / "gate_cost.cpp"
        binary = build / "gate_cost"
        cpp.write_text(source)
        subprocess.run([compiler, "-O3", "-Wno-deprecated-declarations", str(cpp),
                        "-lcrypto", "-o", str(binary)], check=True)
        if args.compile_only:
            print("Compilation passed; timing was not rerun.")
        else:
            subprocess.run([str(binary), str(args.problem.resolve())], check=True)


if __name__ == "__main__":
    main()
