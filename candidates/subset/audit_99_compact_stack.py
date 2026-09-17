#!/usr/bin/env python3
"""Mechanical audit for exact99 + residual compact schedule storage + 2 KiB stack limit."""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
TREE_PATH = "candidates/subset/tests/gpu_epochs/tree.cu"
HEADER_PATH = "candidates/subset/tests/gpu_epochs/window_schedule_shared.cuh"
BASE_TREE_SHA256 = "1611399449e5dc4d19e2f9cdd9fc2e0144c27e032946146b79aa4571cd1e4780"
BASE_HEADER_SHA256 = "edce85dcd53d8827cd76962e19da4b578e1f946d31f519304a94694f011214d7"
COMPACT_HEADER_SHA256 = "e6231736dea6b06704fa0bd8d359b4c5050395f569ac748dd0fd7c158b17325e"
EXACT60_PREPARE_SHA256 = "5c739dbe02814e6d82da45d243c1ff0601bfd03d1bb3c9276934d2458c0f484e"
EXACT60_CONSUMER_SHA256 = "2cf03fa235a6005075458cebacf06459e78798ba32ff264ce765139275724650"
EXPECTED_ORDER_SHA256 = "c9a201e0c23d361eab28ca95767c6d2bb309819060684a23d60e9b66e42c8b74"


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def function(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for end in range(brace, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start : end + 1]
    raise AssertionError(signature)


def second_key(window: tuple[int, int, int]) -> int:
    key = count = 0
    for index in range(12, -1, -1):
        if index not in window and count < 5:
            key = (key << 4) | index
            count += 1
    return key


def first_key(window: tuple[int, int, int]) -> int:
    key = count = 0
    for index in range(13):
        if index not in window and count < 6:
            key = (key << 4) | index
            count += 1
    return key


def selected_windows() -> list[tuple[int, int, int]]:
    values = []
    for window in itertools.combinations(range(13), 3):
        a, b, c = window
        if a >= 1 and c <= 7 and not (a == 1 and b == 2):
            continue
        values.append(window)
    values.sort(key=lambda window: (second_key(window), first_key(window)))
    return values


def schedule(words: list[int]) -> list[int]:
    result = words[:]
    mask = 0xFFFFFFFF

    def ror(value: int, amount: int) -> int:
        return ((value >> amount) | (value << (32 - amount))) & mask

    for index in range(16, 64):
        x, y = result[index - 15], result[index - 2]
        s0 = ror(x, 7) ^ ror(x, 18) ^ (x >> 3)
        s1 = ror(y, 17) ^ ror(y, 19) ^ (y >> 10)
        result.append((result[index - 16] + s0 + result[index - 7] + s1) & mask)
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
    tree_path = ROOT / "tests/gpu_epochs/tree.cu"
    header_path = ROOT / "tests/gpu_epochs/window_schedule_shared.cuh"
    tree = tree_path.read_text(encoding="utf-8")
    header = header_path.read_text(encoding="utf-8")
    base_tree = subprocess.check_output(["git", "show", f"HEAD:{TREE_PATH}"], cwd=REPO)
    base_header = subprocess.check_output(["git", "show", f"HEAD:{HEADER_PATH}"], cwd=REPO)
    assert sha(base_tree) == BASE_TREE_SHA256
    assert sha(base_header) == BASE_HEADER_SHA256
    assert sha(header_path.read_bytes()) == COMPACT_HEADER_SHA256

    assert tree.count("cudaDeviceSetLimit(cudaLimitStackSize, 2048);") == 1
    restored_tree = tree.replace(
        "cudaDeviceSetLimit(cudaLimitStackSize, 2048);",
        "cudaDeviceSetLimit(cudaLimitStackSize, 32768);",
    )
    assert sha(restored_tree) == BASE_TREE_SHA256

    marker = "    /* Short-epoch tables:"
    end_marker = "    uint64_t *d_nri"
    base_tree_text = base_tree.decode()
    assert tree[tree.index(marker) : tree.index(end_marker, tree.index(marker))] == base_tree_text[
        base_tree_text.index(marker) : base_tree_text.index(end_marker, base_tree_text.index(marker))
    ]

    assert sha(function(header, "static int qsb_prepare_window_schedule(")) == EXACT60_PREPARE_SHA256
    assert sha(function(header, "__device__ __forceinline__ void qsb_scheduled_window_hash(")) == EXACT60_CONSUMER_SHA256
    for token in (
        "#define QSB_WINDOW_CLASS_CAP 64",
        "__device__ uint32_t QSB_WINDOW_SECOND[64][QSB_WINDOW_CLASS_CAP]",
        "__device__ uint8_t QSB_WINDOW_CLASS[256]",
        "__device__ uint8_t QSB_FIRST_CLASS[256]",
        "__device__ uint32_t QSB_FIRST_UNIQUE[14][QSB_WINDOW_CLASS_CAP]",
        "__shared__ uint32_t first_states[8][QSB_WINDOW_CLASS_CAP]",
    ):
        assert token in header
    assert "QSB_WINDOW_FIRST" not in header

    windows = selected_windows()
    assert len(windows) == 256 and len(set(windows)) == 256
    assert sha(bytes(index for window in windows for index in window)) == EXPECTED_ORDER_SHA256
    problem = json.loads((REPO / "problems/subset.json").read_text(encoding="utf-8"))
    rows = [bytes.fromhex(value) for value in problem["dummy_sigs"]]
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
        words = [int.from_bytes(payload[index : index + 4], "big") for index in range(0, 128, 4)]
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

    assert len(first_unique) == 54 and len(second_unique) == 56
    assert max(first_classes) < 64 and max(second_classes) < 64
    assert len(set(digests)) == 256
    for lane, (direct_first, direct_second) in enumerate(direct_words):
        compact_first = [
            int.from_bytes(first_unique[first_classes[lane]][index : index + 4], "big")
            for index in range(0, 56, 4)
        ]
        second_bytes = second_unique[second_classes[lane]]
        compact_second = [
            int.from_bytes(second_bytes[index : index + 4], "big")
            for index in range(0, 64, 4)
        ]
        assert compact_first == direct_first
        assert schedule(compact_second) == direct_second

    old_global = 14 * 256 * 4 + 64 * 256 * 4 + 256 * 4 + 256 * 4 + 14 * 256 * 4
    new_global = 64 * 64 * 4 + 256 + 256 + 14 * 64 * 4
    old_shared, new_shared = 8 * 256 * 4, 8 * 64 * 4
    assert (old_global - new_global, old_shared - new_shared) == (75776, 6144)

    changed = subprocess.check_output(["git", "diff", "--name-only"], cwd=REPO, text=True).splitlines()
    assert changed == [TREE_PATH, HEADER_PATH]
    print(json.dumps({
        "status": "PASS",
        "base_commit": "65cd138c80bf8f9070117a991c049a9da71d1de8",
        "selector_and_lane_order_byte_exact": True,
        "selected_windows": len(windows),
        "first_classes": len(first_unique),
        "second_classes": len(second_unique),
        "unique_sha256d": len(set(digests)),
        "all_lane_first_and_second_schedules_equal": True,
        "global_bytes_saved": old_global - new_global,
        "shared_bytes_saved": old_shared - new_shared,
        "stack_limit": 2048,
        "changed_files": changed,
    }, indent=2))


if __name__ == "__main__":
    main()
