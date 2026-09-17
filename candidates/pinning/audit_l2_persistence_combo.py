#!/usr/bin/env python3
"""Bind this bundle's provenance: exact e2 policy block + 1a structural
mechanisms + this bundle's GPUMath delta (DEFER_Y template + __restrict__).

The promoted crown (cba939b, commit 240f329) shipped GPUMath.h byte-identical
to 1a. This bundle rebases onto that crown and changes GPUMath.h by exactly
the alvaroborras codegen port: `_PointAddXYZZ` becomes a template<bool
DEFER_Y> with __restrict__ pointers and `_PointAddXYZZ_mm` gains __restrict__
__forceinline__. The lazy arithmetic (_ModAddLazy, _ModX3Fused, lean
_ModSub256) is 1a's, unchanged. The hashes below bind the final files; the
e2 policy block inside pinning.cu remains byte-identical to the promoted e2
text and its position contract is re-checked.
"""

import hashlib
from pathlib import Path


CROWN_HEAD = "240f329"
ONE_A_GPUMATH_SHA256 = "835b061d1a0b158778c0b576ebce23a9f1663f616a4a292ed7afe103ff6afecd"
E2_POLICY_SHA256 = "44464c51382d00c04ce788133a6abab6ef88877c0a84a32412ab045e1cabbecf"
BUNDLE_GPUMATH_SHA256 = "35970bbe7d11c447b468d1c64a8f4f2aec696b87725405ef877dc20a5a46f04b"
BUNDLE_PINNING_SHA256 = "ee314daf5eecdb1634b1cb33baff040abb1029a96dfb922db0fc48b6320df577"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    gpu_math = (root / "GPUMath.h").read_bytes()
    source_text = (root / "pinning.cu").read_text()
    source = source_text.encode()

    assert digest(gpu_math) == BUNDLE_GPUMATH_SHA256, "GPUMath.h drifted from this bundle"
    assert digest(source) == BUNDLE_PINNING_SHA256, "pinning.cu drifted from this bundle"

    # The delta vs 1a is exactly the codegen port: the template signature,
    # the __restrict__ pointers, the compile-time branch, and nothing else
    # in this file's hot arithmetic beyond 1a's lazy forms.
    math = gpu_math.decode()
    assert "template<bool DEFER_Y>" in math
    assert "const uint64_t *__restrict__ Yoff)" in math
    assert "if (DEFER_Y) {" in math
    assert "template<bool DEFER_Y>" not in math.replace(
        "template<bool DEFER_Y>\n__device__ __forceinline__ void _PointAddXYZZ(", "", 1)
    assert "__device__ __forceinline__ void _PointAddXYZZ_mm(" in math

    begin = source_text.index("    /* Pin the fixed-base table in L2.")
    end_token = "    }\n    uint32_t *d_hit_cnt"
    end = source_text.index(end_token, begin) + len("    }\n")
    policy = source_text[begin:end]
    assert digest(policy.encode()) == E2_POLICY_SHA256, "e2 policy block modified"

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
        assert source_text.count(required) == 1, f"missing or duplicated policy token: {required}"

    assert source_text.index("size_t gt_sz = (size_t)GT_TOTAL_ENTRIES*64;") < begin
    assert source_text.index("cudaMalloc(&d_gt,gt_sz);") < begin
    # This bundle's stack limit (alvaroborras item 3) precedes the policy.
    assert source_text.index("cudaDeviceSetLimit(cudaLimitStackSize, 4096);") < begin
    assert source_text.count("cudaDeviceSetLimit(cudaLimitStackSize,") == 1
    assert begin < source_text.index("launch_pinning_pipeline<true>")

    # The 1a structural mechanisms remain intact around the host-only policy.
    assert "#define QSB_TREE_N 128" in source_text
    assert "qsb_block_product_checkpoint<QSB_TREE_N>" in source_text
    assert "qsb_block_inverse_checkpoint<QSB_TREE_N>" in source_text
    assert "#define QSB_LAZY 1" in math
    assert "_ModAddLazy" in math
    assert "_ModX3Fused" in math

    print("PASS: bundle provenance bound; exact e2 policy block preserved "
          "byte-for-byte; 128-thread trees + lean arithmetic intact; GPUMath "
          "delta = DEFER_Y template + __restrict__ (1a base "
          f"{ONE_A_GPUMATH_SHA256[:12]}... -> {BUNDLE_GPUMATH_SHA256[:12]}...; "
          f"base crown {CROWN_HEAD})")


if __name__ == "__main__":
    main()
