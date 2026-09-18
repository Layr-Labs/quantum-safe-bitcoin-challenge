#!/usr/bin/env python3
"""Source-bound structural audit for an exact-K2-derived K3 candidate."""

import argparse
import hashlib
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "exact_k2"
DEFAULT_SOURCE = HERE / "k3_candidate"

EXPECTED_BASE = {
    "GPUMath.h": "958be6d3b63ba65dc7728fd300d583cadd57e14bf7861ac74060eb0cc41136b2",
    "tests/gpu_epochs/tree.cu": "fb48802fc0e4394a109830abaa3520e7696d2935a41fb786269e778c8139e8d5",
    "tests/gpu_epochs/window_schedule_shared.cuh": "fcd8875884cf0e21b0f24a81e76b5c555455b9aafe46aac7cd94199cd46280ab",
}

EXPECTED_K3 = {
    "tests/gpu_epochs/tree.cu": "ecd77206bab899eace1186e4d18c2c92da99ee8895053e4cd0c0996065fc4411",
    "tests/gpu_epochs/tree_inverse.cuh": "4ad557c3f7d4365e31ae4bf4bd7b17be3cbe139f20bd553bb4e58a44d1763525",
    "tests/gpu_epochs/zinv32.cuh": "e8385842bbddcdd7023dac5fcbbf4d01780cb128494896dc1375db52f359da11",
}

EXPECTED_SUCCESSOR = {
    **EXPECTED_K3,
    "tests/gpu_epochs/tree.cu": "c4b997fcbafd8e8c0c111724ec6d9305d3328cddce10d474fd113b4d88f28ff0",
    "tests/gpu_epochs/sparse_sha_fixed.cuh": "0a9f63eba13ae928ca1c2b4996294d81585fbe9c8a5e3a43c8ae144518d15300",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="candidate root (default: k3_candidate)",
    )
    args = parser.parse_args()
    k3 = args.source.resolve()
    expected_k3 = EXPECTED_SUCCESSOR if k3.name == "k3_sparse_successor" else EXPECTED_K3

    for rel, expected in EXPECTED_BASE.items():
        assert sha(BASE / rel) == expected, rel
        if rel not in ("tests/gpu_epochs/tree.cu", "tests/gpu_epochs/tree_inverse.cuh"):
            assert sha(k3 / rel) == expected, rel
    for rel, expected in expected_k3.items():
        assert sha(k3 / rel) == expected, rel

    source = (k3 / "tests/gpu_epochs/tree.cu").read_text()
    required = [
        "#define QSB_K2S_MUL (ZLAB_K3S ? 3 : (ZLAB_K2S ? 2 : 1))",
        "__shared__ uint64_t parkA[8][256]",
        "__shared__ uint64_t denA[4][256]",
        "__shared__ uint64_t denB[4][256]",
        "uint64_t *parkB=d_k3park+(size_t)blockIdx.x*8u*256u",
        "qsb_field_mul_raw(prodAB,prodA,prodB)",
        "qsb_field_mul_raw(leaf,prodAB,prodC)",
        "qsb_field_mul_raw(invAB,leaf,prodC)",
        "qsb_field_mul_raw(inv,invAB,b)",
        "qsb_field_mul_raw(inv,invAB,a)",
        "qsb_field_mul_raw(inv,leaf,prodAB)",
        "(size_t)QSB_SE_LAUNCH_BLOCKS * QSB_STM_SLOTS * 8 * 256 * sizeof(uint64_t)",
        "(size_t)stm_slot * QSB_SE_LAUNCH_BLOCKS * 8 * 256",
    ]
    for text in required:
        assert text in source, text
    tree_header = (k3 / "tests/gpu_epochs/tree_inverse.cuh").read_text()
    for text in ("#define ZLAB_TREE 3", "__shared__ uint64_t nodes[4][512]",
                 "__syncwarp(0x3u)", "if(count>32)__syncthreads()"):
        assert text in tree_header, text

    # Every in-place downward level reads parents from the next packed level,
    # reads siblings from the current level, then overwrites only that current
    # level.  The two ranges must be disjoint before the synchronization join.
    for count in (4, 8, 16, 32, 64, 128):
        half = count // 2
        offset = 512 - 2 * count
        parents = {offset + count + (tid & (half - 1)) for tid in range(count)}
        siblings = {offset + (tid ^ half) for tid in range(count)}
        children = {offset + tid for tid in range(count)}
        assert parents.isdisjoint(children)
        assert siblings == children
    for epoch in ("e0", "e1", "e2"):
        assert f"qsb_k2s_front({epoch}," in source
    assert source.count("qsb_k2s_front(e2,") == 1
    assert source.count("qsb_k2s_post(") == 6  # definition, K3 A/B/C, retained K2 A/B
    assert source.count("stm_frst, stm_k3park)") == 1
    assert source.count("NULL, NULL, NULL)") == 2

    # Exactly 16 KiB of global parked numerators per block, with disjoint slots.
    blocks, lanes, planes, slots = 32768, 256, 8, 2
    assert lanes * planes * 8 == 16 * 1024
    addresses = {
        ((slot * blocks + block) * planes + plane) * lanes + lane
        for slot in range(slots)
        for block in (0, 1, blocks - 1)
        for plane in range(planes)
        for lane in range(lanes)
    }
    assert len(addresses) == slots * 3 * planes * lanes

    epochs = math.comb(137, 6)
    assert epochs % 3 == 0  # no K3 tail is silently dropped
    assert re.search(r"epochs_left\s*/=\s*QSB_K2S_MUL", source) is not None
    assert "epochs_left >>= 1" not in source
    print(f"PASS: {k3.name}: exact K2 identity, K3 source structure, scratch slots, and {epochs} complete epochs")


if __name__ == "__main__":
    main()
