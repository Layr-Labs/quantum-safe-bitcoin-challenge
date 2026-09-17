#!/usr/bin/env python3
"""Audit the ranked short-epoch external inversion pipeline in tree.cu.

Covers the qsb_sub_* pipeline (prepare / checkpoint collectives / root
grouping / super-root inversion / finish / launcher / host wiring / L2
policy) that no GPU is available to execute. CPU source-contract and
algebra audit, never a CUDA compile or GPU benchmark.

Checks:
 1. Collective lineage: checkpoint/group bodies match the promoted pinning
    pipeline (commit 4d39b5f) modulo names, stride macros, and the
    canonical->raw multiply swap (subset's ZLAB_TREE packed tree uses raw
    residues; the one _ModInv here lives inside qsb_block_inverse_tree,
    which normalizes its own root).
 2. Prepare lifting: the SHA/z/chain/C sequence is the ranked path moved
    verbatim; save precedes identity substitution precedes checkpoint;
    no early return before the collective.
 3. Finish lifting: first-hash plus word gate only (no second hash, no DER
    paths); collective precedes the single usable return; packed hit
    records (tag at [4p], combos at [16p..]) match the ZLAB_HITPATH
    harvest exactly.
 4. Usability: W-zero identity substitution in prepare and inv-identity in
    finish, mirroring the monolithic kernel's singular-lane handling.
 5. Host contract: five-launch order, bounds guards, pipeline selection
    iff (single_hash && !easy && !calibrate), monolithic fallback intact
    with zh_* pointers, harvest code untouched.
 6. Allocation math for the 65536-block geometry: 2 GiB state, 512 MiB
    tree, roots, group buffers; group counts within the super-root
    kernel's fixed-256 capacity.
 7. Super-root fixed-256 pattern with identity padding and gated
    storeback through the packed tree.
 8. L2 policy: table-range persisting window with guarded attributes and
    a run-visible print, mirroring the promoted pinning shape.
"""
import hashlib
import json
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
        ("qsb_block_product_checkpoint", "qsb_sub_product_checkpoint",
         "__device__ __forceinline__ void "),
        ("qsb_root_group_prepare", "qsb_sub_group_prepare",
         "__launch_bounds__(256,2) "),
        ("qsb_root_group_finish", "qsb_sub_group_finish",
         "__launch_bounds__(256,2) "),
    ]
    for old, new, anchor in pairs:
        a = norm(pinbody(old), old, new)
        b = body(new, anchor)
        a = norm(norm(norm(a, "qsb_block_product_checkpoint", "qsb_sub_product_checkpoint"),
                      "qsb_block_inverse_checkpoint", "qsb_sub_inverse_checkpoint"),
                 "QSB_CHECKPOINT_STRIDE", "QSB_PIPE_CKPT_STRIDE")
        a = norm(norm(a, "QSB_CHECKPOINT_NODES", "QSB_PIPE_CKPT_NODES"),
                 "qsb_field_mul(", "qsb_field_mul_raw(")
        assert a == b, f"collective divergence in {new}"
        n += 1
    # inverse checkpoint: same mapping, minus pinning's leaf normalization
    a = norm(pinbody("qsb_block_inverse_checkpoint"),
             "qsb_block_inverse_checkpoint", "qsb_sub_inverse_checkpoint")
    a = norm(norm(a, "QSB_CHECKPOINT_STRIDE", "QSB_PIPE_CKPT_STRIDE"),
             "QSB_CHECKPOINT_NODES", "QSB_PIPE_CKPT_NODES")
    a = norm(a, "qsb_field_mul(", "qsb_field_mul_raw(")
    tail = "    qsb_field_mul_raw(value,parent_inv,sibling);\n    qsb_field_normalize(value);\n}"
    assert a.endswith(tail)
    a = a[: -len(tail)] + "    qsb_field_mul_raw(value,parent_inv,sibling);\n}"
    assert a == body("qsb_sub_inverse_checkpoint"), "inverse checkpoint divergence"
    n += 1
    n += check("raw mul available",
               "__device__ __forceinline__ void qsb_field_mul_raw(uint64_t *out,uint64_t *a,uint64_t *b){" in SRC)
    n += check("super uses packed tree", "qsb_block_inverse_tree(r);" in body(
        "qsb_sub_invert_super", "__launch_bounds__(256,1) "))

    # ---- 2. prepare lifting ---------------------------------------------------
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
    n += check("prepare has no return", "return;" not in prep)

    # ---- 3. finish lifting -----------------------------------------------------
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
        "hit_idx[p*4]=((uint32_t)idx)|((uint32_t)recid<<30)|((uint32_t)hash_choice<<31);",
        "hit_combos[p*ZLAB_HIT_REC+i]=se_desc->early[i];",
        "hit_combos[p*ZLAB_HIT_REC+6+i]=WIN3[threadIdx.x][i];",
    ):
        n += check(f"finish has {stmt[:40]}", stmt in fin)
    n += check("collective before usable return",
               fin.index("qsb_sub_inverse_checkpoint") < fin.index("if(!usable)return;"))
    n += check("single usable return", fin.count("return;") == 1)
    fin_region = SRC[SRC.index("qsb_sub_finish"):SRC.index("static cudaError_t qsb_launch_sub_pipeline")]
    n += check("no second hash in finish", "uint8_t pp[64]" not in fin_region)
    n += check("no DER paths in finish", "gpu_is_der_relaxed" not in fin_region
               and "gpu_is_der_easy" not in fin_region)

    # ---- 4. monolithic singular guard preserved as reference --------------------
    mono = body("kernel_digest", "__launch_bounds__(256, 2) ")
    n += check("monolithic singular guard",
               "bool usable = active && ((prod[0]|prod[1]|prod[2]|prod[3]) != 0);" in mono)

    # ---- 5. host contract ---------------------------------------------------------
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
    n += check("fallback passes zh pointers", "zh_cnt, zh_idx,\n                zh_combos, d_hit_sighash," in SRC)
    n += check("harvest tag format",
               'snprintf(wb + wl, sizeof(wb) - wl, "indices=%d' in SRC)
    for const in ("#define QSB_PIPE_CKPT_NODES 254", "#define QSB_PIPE_CKPT_STRIDE 256",
                  "#define QSB_PIPE_MAX_GROUPS 256"):
        n += check(const, const in SRC)

    # ---- 6. allocation math (65536-block geometry) -----------------------------------
    n += check("state 2GiB",
               "(size_t)QSB_SE_LAUNCH_BLOCKS * QSB_SE_PER_EPOCH * 8 * sizeof(ulonglong2)" in SRC)
    assert 65536 * 256 * 8 * 16 == 2 * 1024**3
    n += check("tree 512MiB",
               "(size_t)QSB_SE_LAUNCH_BLOCKS * 4 * QSB_PIPE_CKPT_STRIDE * sizeof(uint64_t)" in SRC)
    assert 65536 * 4 * 256 * 8 == 512 * 1024**2
    n += check("roots 2MiB",
               "(size_t)QSB_SE_LAUNCH_BLOCKS * 4 * sizeof(uint64_t)" in SRC)
    n += check("max groups fit", (65536 + 255) // 256 <= 256)

    # ---- 7. super-root fixed-256 pattern -------------------------------------------------
    sup = body("qsb_sub_invert_super", "__launch_bounds__(256,1) ")
    n += check("super identity default", "active?super_roots[(size_t)tid*4u]:1ULL" in sup)
    n += check("super gated storeback", "if(active){" in sup)

    # ---- 8. L2 policy ----------------------------------------------------------------------
    for stmt in ("cudaDevAttrMaxPersistingL2CacheSize",
                 "av.accessPolicyWindow.base_ptr  = (void *)d_gt;",
                 "av.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;",
                 "av.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;",
                 "L2 persistence: %.0f MiB pinned"):
        n += check(f"l2 policy {stmt[:30]}", stmt in SRC)

    fp = hashlib.sha256((HERE / "tests/gpu_epochs/tree.cu").read_bytes()).hexdigest()
    print(json.dumps({"status": "PASS", "checks": n, "tree_cu_sha256": fp,
                      "gpu_executed": False}))
    print(f"PASS checks={n}")


if __name__ == "__main__":
    main()
