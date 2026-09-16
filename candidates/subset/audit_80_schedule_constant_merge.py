#!/usr/bin/env python3
"""Provenance and exclusion audit for exact80 + exact60 schedule + constant R."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def function(text: str, signature: str) -> str:
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 0
    for end in range(brace, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start:end + 1]
    raise AssertionError(signature)


def main() -> None:
    tree = (ROOT / "tests/gpu_epochs/tree.cu").read_text()
    schedule = ROOT / "tests/gpu_epochs/window_schedule_shared.cuh"

    exact80 = {
        "GPUHash.h": "c8415e1ddc839e078421a1ee347a9db5036f84714b03edb700aa2e635e05fedd",
        "GPUMath.h": "2f99a04753920f29942cf15bb8153819306b259f35d2d518c3000a0d61fd9684",
        "square32.cuh": "a973d1dd58bf760ce5cbff79e69cbe8cd725d8ad1a642741e936fb2d948be771",
        "subset.cu": "e2bead6006809470d21f6beb8de58a5efe7af483ffcf74f31596a9da108446ef",
        "tests/gpu_epochs/prefix_cache.cuh": "2b289341d3ed1136c4ea157d2d68c2924da2b5a6538cf4ea49e8c720712977b5",
        "tests/gpu_epochs/tree_audit.cu": "c090eaab7bef88b0d9fe0eaa1f68ddd2fc3a30324d3e7e928416a9f13e2a25e1",
        "tests/gpu_epochs/tree_inverse.cuh": "a7206ef98b0e8cb42cb725f03092475d8581eaf57cb863317a6d6636b5ff426f",
    }
    for relative, expected in exact80.items():
        assert sha((ROOT / relative).read_bytes()) == expected, relative

    # This is byte-exact exact60 schedule-pack source, including selector,
    # compact uint8 class maps and the 64-slot transposed schedule arrays.
    assert sha(schedule.read_bytes()) == "be58e5bee80ca50679602e557754d32b26dad33931a0e9808a493319e6c61d50"

    exact80_hot = {
        "__device__ __forceinline__ void gt_load_signed_flat(": "c5f9bc5121c604a34f2a6375a215f63988ae2d1c1ee367f2311262380ab0643b",
        "__device__ void _FixedBaseSignedXYZZStream(": "ec555dfaabc8e81733ba1a432966f0492e93e988cea43460e49f78eae47b8f95",
        "__global__ void kernel_build_gtable(": "12f65c654e34254a16e7dea4f845bac95c6ecb9cd1529dd34f80c58538c0227d",
    }
    for signature, expected in exact80_hot.items():
        assert sha(function(tree, signature)) == expected, signature

    for marker in (
        "if (qsb_select_window_schedule(dp.dummy_sigs, h_win3)) return 1;",
        "__device__ __constant__ uint64_t QSB_U2R[8];",
        "uint64_t u2rx[4]={QSB_U2R[0],QSB_U2R[1],QSB_U2R[2],QSB_U2R[3]};",
        "uint64_t u2ry[4]={QSB_U2R[4],QSB_U2R[5],QSB_U2R[6],QSB_U2R[7]};",
        "cudaMemcpyToSymbol(QSB_U2R,h_u2r,sizeof(h_u2r))",
    ):
        assert marker in tree, marker

    # Ensure this remains an isolated schedule/constant experiment. The
    # promoted-80 base naturally contains its own single-chain table loader;
    # no separate 78 rider or other in-flight mechanism is added here.
    for forbidden in (
        'ranked_pipeline.cuh', "qsb_launch_ranked_pipeline", "QSB_PAIRED_RANKED",
        "qsb_sha256_pair", "cudaStreamCreate", "void qsb_warp_inverse(",
    ):
        assert forbidden not in tree, forbidden

    print(json.dumps({
        "status": "PASS",
        "base": "106a682",
        "exact80_whole_files": len(exact80),
        "exact80_hot_functions": len(exact80_hot),
        "exact60_schedule_sha256": sha(schedule.read_bytes()),
        "constant_symbol_bytes": 64,
        "excluded_riders": ["external", "paired", "warp", "dual-stream"],
        "cuda_compiled": False,
        "gpu_executed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
