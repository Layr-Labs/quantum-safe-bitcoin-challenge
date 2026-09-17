#!/usr/bin/env python3
"""Bind the merged frontier stack: 1a lean arithmetic + exact e2 default-stream
L2 policy block + the e7a648c trio (FastTail11, templated XYZZ, host drain).

The full-source pins bind THIS tree (crown 240f329 plus the trio rebase); the
policy block itself must stay byte-identical to the promoted e2 form.
"""

import hashlib
from pathlib import Path


ONE_A_GPUMATH_SHA256 = "835b061d1a0b158778c0b576ebce23a9f1663f616a4a292ed7afe103ff6afecd"
E2_POLICY_SHA256 = "44464c51382d00c04ce788133a6abab6ef88877c0a84a32412ab045e1cabbecf"
MERGED_GPUMATH_SHA256 = "3dc13e6037b01fb6282362502e0ae0d180a5d6d6f112fb2886487dfe3e616960"
MERGED_PINNING_SHA256 = "d3ead1168eb11766f0b53a8914f275171d6247399e653394e2f3acbf0d2c59f6"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    gpu_math = (root / "GPUMath.h").read_bytes()
    source = (root / "pinning.cu").read_text()
    assert digest(gpu_math) == MERGED_GPUMATH_SHA256
    assert digest(source.encode()) == MERGED_PINNING_SHA256

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
    assert "#define QSB_LAZY 1" in (root / "GPUMath.h").read_text()
    assert "_ModAddLazy" in (root / "GPUMath.h").read_text()
    assert "_ModX3Fused" in (root / "GPUMath.h").read_text()

    print("PASS: merged sources bound; exact e2 policy block, 128-thread trees and lean arithmetic preserved")


if __name__ == "__main__":
    main()
