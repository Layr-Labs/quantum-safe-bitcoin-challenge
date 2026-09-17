#!/usr/bin/env python3
"""Bind the e2 L2 policy block, 1a structure, and the ported 1f425ad mechanisms.

The whole-file digests of the 1a archive no longer apply: GPUMath.h and
pinning.cu are intentionally modified to carry the device mechanisms from
promoted commit 1f425ad. The e2 policy block itself must stay byte-identical,
so it keeps its own sub-block digest.
"""

import hashlib
from pathlib import Path


ONE_A_HEAD = "5c85ae053bc0effa27db4df76fcbf09ee4aaa1b2"
E2_HEAD = "d87de9fb5cfb4840f29a29de511d455d25562f79"
SCARLETBRIGHT_HEAD = "1f425ad"
E2_POLICY_SHA256 = "44464c51382d00c04ce788133a6abab6ef88877c0a84a32412ab045e1cabbecf"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    source = (root / "pinning.cu").read_text()

    begin = source.index("    /* Pin the fixed-base table in L2.")
    end_token = "    }\n    uint32_t *d_hit_cnt"
    end = source.index(end_token, begin) + len("    }\n")
    policy = source[begin:end]
    assert digest(policy.encode()) == E2_POLICY_SHA256

    for required in (
        "cudaDevAttrMaxPersistingL2CacheSize",
        "cudaDevAttrMaxAccessPolicyWindowSize",
        "cudaLimitPersistingL2CacheSize",
        "cudaStreamAttributeAccessPolicyWindow",
        "cudaAccessPropertyPersisting",
        "cudaAccessPropertyStreaming",
        "av.accessPolicyWindow.base_ptr  = (void *)d_gt;",
        "av.accessPolicyWindow.hitRatio  = 1.0f;",
    ):
        assert source.count(required) == 1, f"missing or duplicated policy token: {required}"

    assert source.index("size_t gt_sz = (size_t)GT_TOTAL_ENTRIES*64;") < begin
    assert source.index("cudaMalloc(&d_gt,gt_sz);") < begin
    assert source.index("cudaDeviceSetLimit(cudaLimitStackSize, 32768);") < begin
    assert begin < source.index("launch_pinning_pipeline<true>")

    # The 1a structural mechanisms remain intact around the host-only policy.
    assert "#define QSB_TREE_N 128" in source
    assert "qsb_block_product_checkpoint<QSB_TREE_N>" in source
    assert "qsb_block_inverse_checkpoint<QSB_TREE_N>" in source
    gpu_math = (root / "GPUMath.h").read_text()
    assert "#define QSB_LAZY 1" in gpu_math
    assert "_ModAddLazy" in gpu_math
    assert "_ModX3Fused" in gpu_math

    # Kept 1f425ad device mechanisms that survive the spill-safe cut: read-only
    # table loads, sparse FastTail11 schedule, and the arithmetic-shift sign
    # mask. Compile-time deferred-Y peeling was dropped — on this frontier it
    # dual-inlines _PointAddXYZZ and forces prepare-kernel spills.
    assert "template<bool DEFER_Y>" not in gpu_math
    assert "if (defer_y)" in gpu_math
    assert source.count("__ldg(tx)") == 1
    assert "_SHA256TransformFastTail11" in source
    assert "int32_t mask = ec >> 31;" in source

    print("PASS: e2 policy block byte-identical, 1a structure intact, spill-safe 1f425ad mechanisms present")

if __name__ == "__main__":
    main()
