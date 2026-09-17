#!/usr/bin/env python3
"""Bind e2 L2 policy + 128-tree/lazy frontier plus additive FT11/XYZZ/drain opts."""

import hashlib
from pathlib import Path


E2_POLICY_SHA256 = "44464c51382d00c04ce788133a6abab6ef88877c0a84a32412ab045e1cabbecf"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    math_txt = (root / "GPUMath.h").read_text()
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

    # Frontier structural mechanisms from cba939b / 1a + e2.
    assert "#define QSB_TREE_N 128" in source
    assert "qsb_block_product_checkpoint<QSB_TREE_N>" in source
    assert "qsb_block_inverse_checkpoint<QSB_TREE_N>" in source
    assert "#define QSB_LAZY 1" in math_txt
    assert "_ModAddLazy" in math_txt
    assert "_ModX3Fused" in math_txt

    # Additive mechanisms from promoted e7a648c stack.
    assert "_SHA256TransformFastTail11" in source
    assert "__ldg(" in source
    assert "template<bool DEFER_Y>" in math_txt
    assert "cudaMemset(d_hit_cnt, 0, 4)" in source
    assert 'printf("\\n  *** HIT! seq=0x%08X ***\\n", seq);' not in source

    print(
        "PASS: e2 L2 policy + 128-tree/lazy frontier preserved; "
        "FastTail11 + templated XYZZ/__ldg + host-drain present"
    )


if __name__ == "__main__":
    main()
