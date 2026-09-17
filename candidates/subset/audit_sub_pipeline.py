#!/usr/bin/env python3
"""Audit the ranked short-epoch external inversion pipeline in tree.cu.

Covers the qsb_sub_* pipeline (prepare / checkpoint collectives / root
grouping / super-root inversion / finish / launcher / host wiring) that no
GPU is available to execute. This script is a CPU source-contract and
algebra audit, never a CUDA compile or GPU benchmark.

Checks:
 1. Collective lineage: checkpoint/group bodies match the promoted pinning
    pipeline (commit 4d39b5f) modulo function names, except the documented
    normalization difference.
 2. Single deliberate difference: no qsb_field_normalize anywhere (subset's
    qsb_field_mul canonicalizes every product, so leaves/roots are already
    canonical for _ModInv and recovery parity).
 3. Prepare lifting: the SHA/z/chain/C sequence is the ranked path moved
    verbatim; save precedes identity substitution precedes checkpoint.
 4. Finish lifting: pubkey loop is first-hash plus word gate only (no
    second hash, no DER paths); collective precedes the usable return;
    hit records use the epoch/WIN3 mapping verbatim.
 5. Usability: W-zero identity substitution in prepare and inv-identity in
    finish, mirroring the monolithic kernel's singular-lane handling.
 6. Host contract: five-launch order, bounds checks, pipeline selection
    iff (single_hash && !easy && !calibrate), monolithic fallback intact,
    harvest code untouched.
 7. Allocation math: 1 GiB state, 256 MiB tree, roots, group buffers;
    group counts within the super-root kernel's fixed-256 capacity.
 8. Barrier discipline: no early return before the collectives in prepare;
    exactly one post-collective return in finish; group/super kernels use
    uniform barriers with identity-padded inactive lanes.
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
ROOT = HERE.parents[1]

SRC = (HERE / "tests/gpu_epochs/tree.cu").read_text()


def check(name, cond):
    assert cond, f"FAILED: {name}"
    return 1


def body(name, anchor="void "):
    """Extract a function/kernel body by brace matching from its definition.

    anchor selects the definition occurrence: device functions ("void
    NAME("), kernels ("__launch_bounds__(256, N) NAME("), the host launcher
    ("cudaError_t ..."). A bare-name search would match header comments or
    call sites first. The returned slice starts at the name, matching
    pinbody() alignment for lineage comparisons.
    """
    key = anchor + name + "("
    start = SRC.index(key) + len(key) - len(name) - 1
    i = SRC.index("{", start) + 1
    depth = 1
    while depth:
        depth += (SRC[i] == "{") - (SRC[i] == "}")
        i += 1
    return SRC[start:i]


def main():
    n = 0
    # ---- 1. collective lineage vs promoted pinning -------------------------
    pin = subprocess.run(
        ["git", "show", "4d39b5f:candidates/pinning/pinning.cu"],
        capture_output=True, check=True, cwd=str(ROOT), text=True).stdout

    def pinbody(name):
        start = pin.index(name)
        i = pin.index("{", start) + 1
        depth = 1
        while depth:
            depth += (pin[i] == "{") - (pin[i] == "}")
            i += 1
        return pin[start:i]

    def norm(text, old, new):
        return text.replace(old, new)

    pairs = [
        ("qsb_block_product_checkpoint", "qsb_sub_product_checkpoint", "void "),
        ("qsb_root_group_prepare", "qsb_sub_group_prepare", "__launch_bounds__(256,2) "),
        ("qsb_root_group_finish", "qsb_sub_group_finish", "__launch_bounds__(256,2) "),
    ]
    for old, new, anchor in pairs:
        a = norm(pinbody(old), old, new)
        b = body(new, anchor)
        # checkpoint helper names and stride macros inside group bodies map over
        a = norm(norm(norm(a, "qsb_block_product_checkpoint", "qsb_sub_product_checkpoint"),
                      "qsb_block_inverse_checkpoint", "qsb_sub_inverse_checkpoint"),
                 "QSB_CHECKPOINT_STRIDE", "QSB_PIPE_CKPT_STRIDE")
        a = norm(a, "QSB_CHECKPOINT_NODES", "QSB_PIPE_CKPT_NODES")
        assert a == b, f"collective divergence in {new}"
        n += 1
    # inverse checkpoint: identical except the omitted normalization tail
    a = norm(pinbody("qsb_block_inverse_checkpoint"),
             "qsb_block_inverse_checkpoint", "qsb_sub_inverse_checkpoint")
    a = norm(norm(a, "QSB_CHECKPOINT_STRIDE", "QSB_PIPE_CKPT_STRIDE"),
             "QSB_CHECKPOINT_NODES", "QSB_PIPE_CKPT_NODES")
    tail = "    qsb_field_mul(value,parent_inv,sibling);\n    qsb_field_normalize(value);\n}"
    assert a.endswith(tail)
    a = a[: -len(tail)] + "    qsb_field_mul(value,parent_inv,sibling);\n}"
    assert a == body("qsb_sub_inverse_checkpoint"), "inverse checkpoint divergence"
    n += 1

    # ---- 2. canonicalization story ------------------------------------------
    n += check("mul canonicalizes",
               "if ((r1 & r2 & r3) == UINT64_MAX && r0 >= 0xFFFFFFFEFFFFFC2FULL) {" in SRC)
    n += check("no pinning normalize helper", "qsb_field_normalize" not in SRC)
    n += check("_ModInv contract",
               "0 < this < P" in (HERE / "GPUMath.h").read_text())

    # ---- 3. prepare lifting ---------------------------------------------------
    prep = body("qsb_sub_prepare", "__launch_bounds__(256,2) ")
    for stmt in (
        "qsb_scheduled_window_hash(first,desc,threadIdx.x);",
        "b2[15]=0x00000100;",
        "_SHA256Transform(s2,b2);",
        "z[0] = ((uint64_t)s2[6] << 32) | (uint64_t)s2[7];",
        "_FixedBaseSignedXYZZStream(C,Y,ZZ,ZZZ,z,gT);",
        "Load256(xR,QSB_U2R);",
        "qsb_xyzz_finish_prepare(C,ZZ,xR,W);",
        "_ModSqr(C,C); _ModMult(C,ZZ);",
        "qsb_sub_save_state(state,count,idx,C,Y,W,ZZZ);",
        "if(!(W[0]|W[1]|W[2]|W[3]))W[0]=1;",
        "qsb_sub_product_checkpoint(W,roots,tree);",
    ):
        n += check(f"prepare has {stmt[:40]}", stmt in prep)
    n += check("save before substitute before checkpoint",
               prep.index("qsb_sub_save_state") < prep.index("W[0]=1") <
               prep.index("qsb_sub_product_checkpoint"))
    n += check("prepare has no return", "return" not in prep)

    # ---- 4. finish lifting -----------------------------------------------------
    fin = body("qsb_sub_finish", "__launch_bounds__(256,3) ")
    for stmt in (
        "qsb_sub_load_state(state,count,idx,C,Y,W,ZZZ);",
        "bool usable=(W[0]|W[1]|W[2]|W[3])!=0;",
        "Load256(inv,W);inv[4]=0;",
        "if(!usable)inv[0]=1;",
        "qsb_sub_inverse_checkpoint(inv,roots,tree);",
        "if(!usable)return;",
        "Load256(xR,QSB_U2R);Load256(yR,QSB_U2R+4);",
        "qsb_xyzz_finish_precomputed(C,Y,W,ZZZ,inv,xR,yR,q1x,q2x);",
        "int vv = gpu_bench_valid_words(hs);",
        "hit_combos[p*MAX_T+i]=se_desc->early[i];",
        "hit_combos[p*MAX_T+6+i]=WIN3[threadIdx.x][i];",
    ):
        n += check(f"finish has {stmt[:40]}", stmt in fin)
    n += check("collective before usable return",
               fin.index("qsb_sub_inverse_checkpoint") < fin.index("if(!usable)return;"))
    n += check("single usable return", fin.count("return;") == 1)
    fin_region = SRC[SRC.index("qsb_sub_finish"):SRC.index("static cudaError_t qsb_launch_sub_pipeline")]
    n += check("no second hash in finish", "uint8_t pp[64]" not in fin_region)
    n += check("no DER paths in finish", "gpu_is_der_relaxed" not in fin_region
               and "gpu_is_der_easy" not in fin_region)

    # ---- 5. usability order -----------------------------------------------------
    mono = body("kernel_digest", "__launch_bounds__(256, 2) ")
    n += check("monolithic singular guard",
               "bool usable = active && ((prod[0]|prod[1]|prod[2]|prod[3]) != 0);" in mono)

    # ---- 6. host contract ---------------------------------------------------------
    launch = body("qsb_launch_sub_pipeline", "cudaError_t ")
    order = ["qsb_sub_prepare<<<", "qsb_sub_group_prepare<<<", "qsb_sub_invert_super<<<",
             "qsb_sub_group_finish<<<", "qsb_sub_finish<<<"]
    pos = [launch.index(k) for k in order]
    n += check("five-launch order", pos == sorted(pos))
    for guard in ("nblk<1 || nblk>QSB_SE_LAUNCH_BLOCKS || count!=nblk*QSB_SE_PER_EPOCH",
                  "groups>QSB_PIPE_MAX_GROUPS"):
        n += check(f"launcher guard {guard[:30]}", guard in launch)
    sel = "if (use_pipeline && single_hash && !easy && !calibrate)"
    n += check("pipeline selector", SRC.count(sel) == 1)
    n += check("monolithic fallback intact", SRC.count("kernel_digest<<<nblk, QSB_SE_PER_EPOCH>>>") == 1)
    n += check("harvest tag format",
               "d_hit_idx[p]=((uint32_t)idx)|(recid<<30)|(hash_choice<<31);" in mono)
    for const in ("#define QSB_PIPE_CKPT_NODES 254", "#define QSB_PIPE_CKPT_STRIDE 256",
                  "#define QSB_PIPE_MAX_GROUPS 256"):
        n += check(const, const in SRC)

    # ---- 7. allocation math ----------------------------------------------------------
    n += check("state 1GiB",
               "(size_t)QSB_SE_LAUNCH_BLOCKS * QSB_SE_PER_EPOCH * 8 * sizeof(ulonglong2)" in SRC)
    assert 32768 * 256 * 8 * 16 == 1024**3
    n += check("tree 256MiB",
               "(size_t)QSB_SE_LAUNCH_BLOCKS * 4 * QSB_PIPE_CKPT_STRIDE * sizeof(uint64_t)" in SRC)
    assert 32768 * 4 * 256 * 8 == 256 * 1024**2
    n += check("roots 1MiB",
               "(size_t)QSB_SE_LAUNCH_BLOCKS * 4 * sizeof(uint64_t)" in SRC)
    n += check("max groups fit", (32768 + 255) // 256 <= 256)

    # ---- 8. super-root fixed-256 pattern -------------------------------------------------
    sup = body("qsb_sub_invert_super", "__launch_bounds__(256,1) ")
    n += check("super bound", "__launch_bounds__(256,1) qsb_sub_invert_super" in SRC)
    n += check("super identity default", "active?super_roots[(size_t)tid*4u]:1ULL" in sup)
    n += check("super classic tree", "qsb_block_inverse_tree(r);" in sup)
    n += check("super gated storeback", "if(active){" in sup)

    fp = hashlib.sha256((HERE / "tests/gpu_epochs/tree.cu").read_bytes()).hexdigest()
    print(json.dumps({"status": "PASS", "checks": n, "tree_cu_sha256": fp,
                      "gpu_executed": False}))
    print(f"PASS checks={n}")


if __name__ == "__main__":
    main()
