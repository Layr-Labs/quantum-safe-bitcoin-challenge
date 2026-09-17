#!/usr/bin/env python3
"""Bind exact 1a source plus the exact e2 default-stream L2 policy block."""

import hashlib
from pathlib import Path


ONE_A_HEAD = "5c85ae053bc0effa27db4df76fcbf09ee4aaa1b2"
E2_HEAD = "d87de9fb5cfb4840f29a29de511d455d25562f79"
ONE_A_GPUMATH_SHA256 = "835b061d1a0b158778c0b576ebce23a9f1663f616a4a292ed7afe103ff6afecd"
ONE_A_PINNING_SHA256 = "b1f818ce3c473db215a58248c4685fd2890566c904ab12fe73c06f0001f0d5eb"
E2_POLICY_SHA256 = "44464c51382d00c04ce788133a6abab6ef88877c0a84a32412ab045e1cabbecf"
FINAL_PINNING_SHA256 = "c23c83b02e82f170a4b538a6d185efa606efbe6f9063e579d6a3fee8972cc667"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    gpu_math = (root / "GPUMath.h").read_bytes()
    source = (root / "pinning.cu").read_text()
    assert digest(gpu_math) == ONE_A_GPUMATH_SHA256
    assert digest(source.encode()) == FINAL_PINNING_SHA256

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

    print("PASS: exact 1a GPUMath, exact e2 policy block, 128-thread trees and lean arithmetic preserved")


if __name__ == "__main__":
    main()
