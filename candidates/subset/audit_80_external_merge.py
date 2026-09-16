#!/usr/bin/env python3
"""Source-bound audit for the promoted-80 + promoted-65 external merge.

This is deliberately a provenance/geometry audit, not a CUDA or throughput
claim.  It proves that the hot mixed-table functions remain byte-identical to
promoted commit 106a682, that the external hierarchy functions retain their
promoted-65 bodies, and that the only field changes are the bounded carry and
canonicalization tails exercised by check_production_field.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def function(text: str, signature: str) -> str:
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 0
    for end in range(brace, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start:end + 1]
    raise AssertionError(f"unclosed function: {signature}")


def main() -> None:
    tree = (ROOT / "tests/gpu_epochs/tree.cu").read_text()
    pipe = (ROOT / "tests/gpu_epochs/ranked_pipeline.cuh").read_text()
    math = (ROOT / "GPUMath.h").read_text()
    square = (ROOT / "square32.cuh").read_text()

    # Whole files untouched from promoted 106a682.
    exact80_files = {
        "GPUHash.h": "c8415e1ddc839e078421a1ee347a9db5036f84714b03edb700aa2e635e05fedd",
        "subset.cu": "e2bead6006809470d21f6beb8de58a5efe7af483ffcf74f31596a9da108446ef",
        "tests/gpu_epochs/prefix_cache.cuh": "2b289341d3ed1136c4ea157d2d68c2924da2b5a6538cf4ea49e8c720712977b5",
        "tests/gpu_epochs/tree_audit.cu": "c090eaab7bef88b0d9fe0eaa1f68ddd2fc3a30324d3e7e928416a9f13e2a25e1",
        "tests/gpu_epochs/tree_inverse.cuh": "a7206ef98b0e8cb42cb725f03092475d8581eaf57cb863317a6d6636b5ff426f",
        "tests/gpu_epochs/window_schedule_shared.cuh": "57dede3365f0d87bfa5307bd73440f9ae25773d913c96b46a3316363fd734cb8",
    }
    for relative, expected in exact80_files.items():
        assert sha((ROOT / relative).read_bytes()) == expected, relative

    # Byte-exact promoted-80 hot-path functions: table signed-load, streamed
    # mixed-15 recode/add chain, and GPU table builder.
    exact80_functions = {
        "__device__ __forceinline__ void gt_load_signed_flat(": "c5f9bc5121c604a34f2a6375a215f63988ae2d1c1ee367f2311262380ab0643b",
        "__device__ void _FixedBaseSignedXYZZStream(": "ec555dfaabc8e81733ba1a432966f0492e93e988cea43460e49f78eae47b8f95",
        "__global__ void kernel_build_gtable(": "12f65c654e34254a16e7dea4f845bac95c6ecb9cd1529dd34f80c58538c0227d",
    }
    for signature, expected in exact80_functions.items():
        assert sha(function(tree, signature)) == expected, signature

    exact80_math = {
        "__device__ void _PointAddXYZZ_def(": "484301cd7e1cd98f390315a58ee9cce15c9f9531f921e08c04157725c081b2af",
        "__device__ void _PointAddXYZZ_mm_def(": "e55a81dabdd1862d2b067cc8aaaba513659c15fc7e48a24a1210270d7b7d5225",
    }
    for signature, expected in exact80_math.items():
        assert sha(function(math, signature)) == expected, signature

    # Byte-exact promoted-65 external hierarchy/state/recovery consumers.
    exact65_functions = {
        "__device__ __forceinline__ void qsb_block_product_checkpoint(": "bbc9d7164db196dfbf8fd1a19ff2e40a247437dc7825e5826764d92c929d6682",
        "__device__ __forceinline__ void qsb_block_inverse_checkpoint(": "847895f6107379067bf6efe85e4213957aa3ccbfdbf72a5000afa52a218a5c01",
        "__global__ void __launch_bounds__(256,2) qsb_root_group_prepare(": "3c7d8e698e62c4fb5aa6d7d059a577ce828d15e52013e5cb431c1eec88e0abb8",
        "__global__ void __launch_bounds__(256,1) qsb_invert_super_roots(": "d43e20445d90a0d5bcb78195e6cb0bb41ee3e0c0aceda6b3026c1becfcdb3e53",
        "__global__ void __launch_bounds__(256,2) qsb_root_group_finish(": "eec1df07f76bb3482e5d7c58f25571be8873212109712c011463635ecc87a380",
        "__device__ __forceinline__ void qsb_save_state(": "e5366a1b7ad6b2986f59b202706588d8cb5eca663d9f4f0c15058af88b3ea484",
        "__device__ __forceinline__ void qsb_load_state(": "376dba0f9c27784bdec63c85c892b9698b4644c3eda0003df24fe168795fa08a",
        "__device__ __forceinline__ int qsb_ranked_pubkey_hit(": "d2fe31b7a0d02e65f3961aed3b0517da19c17e0712bc92ce3f49321b78b5234f",
        "__global__ void __launch_bounds__(256,3) qsb_ranked_finish(": "be7f61fa3de125e0cfe3b3192a849abfeb66c84d1c37acd3cfb6d22be9f82165",
    }
    for signature, expected in exact65_functions.items():
        assert sha(function(pipe, signature)) == expected, signature

    # The two adapted functions differ from promoted 65 only in replacing its
    # split 32 MiB X/Y table with promoted 80's interleaved 64 MiB table and
    # naming the exact-80 streamed producer.
    assert sha(function(pipe, "__global__ void __launch_bounds__(256,2) qsb_ranked_prepare(")) == \
        "9c302a90f33851e4c4034059258078a107107ccc92654ccf524c55c8eae76f49"
    assert sha(function(pipe, "static cudaError_t qsb_launch_ranked_pipeline(")) == \
        "838006a9b952bca7db66d9a43b17167980d7746a0ef4d06657ed1e9b0d7dc9a5"

    for source, markers in ((math, (
        "addc.cc.u32 z7, z7, 0", "mul.lo.u32 m0, cf, 977",
        "r[0] -= 0xFFFFFFFEFFFFFC2FULL")), (square, (
        "addc.cc.u32 z7, z7, 0", "mul.lo.u32 m0, cf, 977",
        "out[0] -= 0xFFFFFFFEFFFFFC2FULL"))):
        for marker in markers:
            assert marker in source, marker

    integration = (
        '#include "ranked_pipeline.cuh"',
        "d_pipe_state", "d_pipe_roots", "d_pipe_tree",
        "d_pipe_super", "d_pipe_root_tree",
        "qsb_launch_ranked_pipeline(nblk, batch_pos, d_epochs",
    )
    for marker in integration:
        assert marker in tree, marker

    blocks, lanes = 32768, 256
    groups = (blocks + lanes - 1) // lanes
    resources = {
        "mixed_table_bytes": 1 << 26,
        "state_bytes": blocks * lanes * 8 * 16,
        "block_roots_bytes": blocks * 4 * 8,
        "block_checkpoint_bytes": blocks * 4 * 256 * 8,
        "super_roots_bytes": groups * 4 * 8,
        "super_checkpoint_bytes": groups * 4 * 256 * 8,
    }
    resources["pipeline_scratch_bytes"] = sum(v for k, v in resources.items() if k != "mixed_table_bytes")
    assert resources["pipeline_scratch_bytes"] == 1_344_278_528
    print(json.dumps({
        "status": "PASS",
        "promoted80_hot_functions": len(exact80_functions) + len(exact80_math),
        "promoted65_external_functions": len(exact65_functions),
        "resources": resources,
        "cuda_compiled": False,
        "gpu_executed": False,
        "limitations": "Source/provenance, geometry and field-model audit only; ptxas registers/spills and GPU throughput remain unknown.",
    }, indent=2))


if __name__ == "__main__":
    main()
