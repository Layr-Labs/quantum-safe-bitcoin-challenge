#!/usr/bin/env python3
"""Mechanical equivalence audit for 9c lane order plus 60-style storage."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from audit_9c_contract import selected_windows


ROOT = Path(__file__).resolve().parent
TREE_PATH = "candidates/subset/tests/gpu_epochs/tree.cu"
HEADER = ROOT / "tests" / "gpu_epochs" / "window_schedule_shared.cuh"
EXPECTED_ORDER_SHA256 = "c9a201e0c23d361eab28ca95767c6d2bb309819060684a23d60e9b66e42c8b74"
EXPECTED_HEADER_SHA256 = "e6231736dea6b06704fa0bd8d359b4c5050395f569ac748dd0fd7c158b17325e"
EXPECTED_PREPARE_SHA256 = "5c739dbe02814e6d82da45d243c1ff0601bfd03d1bb3c9276934d2458c0f484e"
EXPECTED_CONSUMER_SHA256 = "2cf03fa235a6005075458cebacf06459e78798ba32ff264ce765139275724650"


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def function(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for end in range(brace, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start:end + 1]
    raise AssertionError(signature)


def schedule(words: list[int]) -> list[int]:
    result = words[:]
    mask = 0xFFFFFFFF

    def ror(value: int, amount: int) -> int:
        return ((value >> amount) | (value << (32 - amount))) & mask

    for index in range(16, 64):
        x, y = result[index - 15], result[index - 2]
        small0 = ror(x, 7) ^ ror(x, 18) ^ (x >> 3)
        small1 = ror(y, 17) ^ ror(y, 19) ^ (y >> 10)
        result.append(
            (result[index - 16] + small0 + result[index - 7] + small1) & mask
        )
    return result


def complete_preimage(problem: dict, relative: tuple[int, ...]) -> bytes:
    rows = [bytes.fromhex(value) for value in problem["dummy_sigs"]]
    skipped = set(range(6)) | {137 + value for value in relative}
    return b"".join(
        [bytes.fromhex(problem["fixed_prefix"])]
        + [row for index, row in enumerate(rows) if index not in skipped]
        + [bytes.fromhex(problem["tail_section"]), bytes.fromhex(problem["tx_suffix"])]
    )


def main() -> None:
    tree = (ROOT / "tests" / "gpu_epochs" / "tree.cu").read_text()
    base_tree = subprocess.check_output(
        ["git", "show", f"HEAD:{TREE_PATH}"], cwd=ROOT.parent.parent, text=True
    )
    marker = "    /* Short-epoch tables:"
    end_marker = "    uint64_t *d_nri"
    tree_selection = tree[tree.index(marker):tree.index(end_marker, tree.index(marker))]
    base_selection = base_tree[
        base_tree.index(marker):base_tree.index(end_marker, base_tree.index(marker))
    ]
    assert tree_selection == base_selection, "9c selection or lane ordering changed"

    header = HEADER.read_text()
    assert sha(HEADER.read_bytes()) == EXPECTED_HEADER_SHA256
    assert sha(function(header, "static int qsb_prepare_window_schedule(")) == EXPECTED_PREPARE_SHA256
    assert sha(function(header, "__device__ __forceinline__ void qsb_scheduled_window_hash(")) == EXPECTED_CONSUMER_SHA256
    for marker_text in (
        "#define QSB_WINDOW_CLASS_CAP 64",
        "__device__ uint32_t QSB_WINDOW_SECOND[64][QSB_WINDOW_CLASS_CAP]",
        "__device__ uint8_t QSB_WINDOW_CLASS[256]",
        "__device__ uint8_t QSB_FIRST_CLASS[256]",
        "__device__ uint32_t QSB_FIRST_UNIQUE[14][QSB_WINDOW_CLASS_CAP]",
        "__shared__ uint32_t first_states[8][QSB_WINDOW_CLASS_CAP]",
    ):
        assert marker_text in header
    assert "QSB_WINDOW_FIRST" not in header
    assert "QSB_PAIRED_RANKED" not in tree
    assert "qsb_sha256_dual_recid" not in tree

    windows = selected_windows()
    order_bytes = bytes(index for window in windows for index in window)
    assert sha(order_bytes) == EXPECTED_ORDER_SHA256
    problem = json.loads((ROOT.parent.parent / "problems" / "subset.json").read_text())
    rows = [bytes.fromhex(value) for value in problem["dummy_sigs"]]

    # Model the exact messages consumed by qsb_prepare_window_schedule. The
    # final twenty bytes are lane-invariant; zero sentinels are sufficient for
    # proving class mapping and full 64-word schedule identity.
    first_unique: list[bytes] = []
    second_unique: list[bytes] = []
    first_classes: list[int] = []
    second_classes: list[int] = []
    direct_words: list[tuple[list[int], list[int]]] = []
    digests: list[bytes] = []
    for window in windows:
        kept = b"".join(rows[137 + index] for index in range(13) if index not in window)
        payload = b"\x00" * 8 + kept + b"\x00" * 20
        assert len(payload) == 128
        words = [int.from_bytes(payload[index:index + 4], "big") for index in range(0, 128, 4)]
        first, second = payload[8:64], payload[64:128]
        if first not in first_unique:
            first_unique.append(first)
        if second not in second_unique:
            second_unique.append(second)
        first_classes.append(first_unique.index(first))
        second_classes.append(second_unique.index(second))
        direct_words.append((words[2:16], schedule(words[16:32])))
        preimage = complete_preimage(problem, window)
        digests.append(hashlib.sha256(hashlib.sha256(preimage).digest()).digest())

    assert len(first_unique) == 54
    assert len(second_unique) == 56
    assert max(first_classes) < 64 and max(second_classes) < 64
    assert len(set(digests)) == 256
    for lane, (direct_first, direct_second) in enumerate(direct_words):
        compact_first = [
            int.from_bytes(first_unique[first_classes[lane]][index:index + 4], "big")
            for index in range(0, 56, 4)
        ]
        compact_second_bytes = second_unique[second_classes[lane]]
        compact_second_words = [
            int.from_bytes(compact_second_bytes[index:index + 4], "big")
            for index in range(0, 64, 4)
        ]
        assert compact_first == direct_first
        assert schedule(compact_second_words) == direct_second

    old_global = 14 * 256 * 4 + 64 * 256 * 4 + 256 * 4 + 256 * 4 + 14 * 256 * 4
    new_global = 64 * 64 * 4 + 256 + 256 + 14 * 64 * 4
    old_shared = 8 * 256 * 4
    new_shared = 8 * 64 * 4
    assert (old_global, new_global) == (96256, 20480)
    assert (old_shared, new_shared) == (8192, 2048)

    print(json.dumps({
        "status": "PASS",
        "selection_source_byte_exact": True,
        "lane_order_sha256": EXPECTED_ORDER_SHA256,
        "lanes_and_unique_sha256d": len(digests),
        "first_classes": len(first_unique),
        "second_classes": len(second_unique),
        "all_lane_first_and_second_schedules_equal": True,
        "global_schedule_bytes": {"before": old_global, "after": new_global},
        "shared_first_state_bytes": {"before": old_shared, "after": new_shared},
        "exact60_prepare_sha256": EXPECTED_PREPARE_SHA256,
        "exact60_consumer_sha256": EXPECTED_CONSUMER_SHA256,
        "paired_sha_absent": True,
    }, indent=2))


if __name__ == "__main__":
    main()
