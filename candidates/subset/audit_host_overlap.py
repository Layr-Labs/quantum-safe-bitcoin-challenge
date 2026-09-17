#!/usr/bin/env python3
"""CPU/source audit for the exact PR59 depth-two short-epoch host overlap."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TREE = ROOT / "tests" / "gpu_epochs" / "tree.cu"
TREE_PATH = "candidates/subset/tests/gpu_epochs/tree.cu"
EXPECTED_HOST_BLOCK_SHA256 = "fe2c8b043cb54220919d23a6a87dd5c6f1329bcd2f54d456d1246f06ccf414ee"
EXPECTED_SERIAL_BLOCK_SHA256 = "329ca2a0bd84f32f7de31eb211e3a391a06024ae6fbfdcac914326f2d2992e11"
EXPECTED_COMPACT_PARENT_TREE_SHA256 = "a58c506a49aa24740c3e48846afd9f64f665275137299f1ac2b95cd35de268ec"


def short_epoch_range(source: str) -> tuple[int, int]:
    marker = source.index("/* Short-epoch path:")
    start = source.index("    if (se_mode) {", marker)
    end = source.index("    /* GPU-enum fast path:", start)
    return start, end


def main() -> None:
    source = TREE.read_text()
    start, end = short_epoch_range(source)
    block = source[start:end]
    assert hashlib.sha256(block.encode()).hexdigest() == EXPECTED_HOST_BLOCK_SHA256

    required = {
        "two nonblocking streams": (
            "cudaStreamCreateWithFlags(&streamP, cudaStreamNonBlocking)",
            "cudaStreamCreateWithFlags(&streamC, cudaStreamNonBlocking)",
        ),
        "two event pairs": (
            "cudaEvent_t evP[2], evC[2]",
            "cudaEventCreateWithFlags(&evP[slot], cudaEventDisableTiming)",
            "cudaEventCreateWithFlags(&evC[slot], cudaEventDisableTiming)",
        ),
        "baseline slot zero reuse": (
            "epoch_desc_t *d_epochs2[2] = {d_epochs, NULL}",
            "uint32_t *d_hit_cnt2[2] = {d_hit_cnt, NULL}",
            "uint32_t *d_hit_idx2[2] = {d_hit_idx, NULL}",
            "uint8_t *d_hit_combos2[2] = {d_hit_combos, NULL}",
        ),
        "producer consumer order": (
            "cudaStreamWaitEvent(streamP, evC[slot], 0)",
            "kernel_build_epochs<<<(nblk + 255) / 256, 256, 0, streamP>>>",
            "cudaEventRecord(evP[slot], streamP)",
            "cudaStreamWaitEvent(streamC, evP[slot], 0)",
            "kernel_digest<<<nblk, QSB_SE_PER_EPOCH, 0, streamC>>>",
        ),
        "asynchronous hit handoff": (
            "cudaMemsetAsync(d_hit_cnt2[slot]",
            "cudaMemcpyAsync(h_hit_pin + slot",
            "cudaEventRecord(evC[slot], streamC)",
            "cudaEventSynchronize(evC[previous])",
            "drain_slot(previous)",
        ),
        "final drain and cleanup": (
            "drain_slot(final_slot)",
            "cudaFreeHost(h_hit_pin)",
            "cudaFree(d_epochs2[1])",
            "cudaStreamDestroy(streamP)",
            "cudaStreamDestroy(streamC)",
        ),
    }
    for label, needles in required.items():
        missing = [needle for needle in needles if needle not in block]
        assert not missing, f"{label}: missing {missing}"
    assert "cudaDeviceSynchronize" not in block
    assert block.count("kernel_build_epochs<<<") == 1
    assert block.count("kernel_digest<<<") == 1
    assert block.count("completed_searched +=") == 2

    # Exhaustively model every pipeline depth from one launch through 4096.
    for launches in range(1, 4097):
        drained: list[int] = []
        launched = 0
        for _ in range(launches):
            slot = launched & 1
            launched += 1
            if launched >= 2:
                drained.append(slot ^ 1)
        drained.append((launched - 1) & 1)
        assert drained == [launch & 1 for launch in range(launches)]

    # Mechanically restore the exact serial short-epoch block from promoted
    # d277241. Compact schedule storage lives in the included header, so this
    # tree projection must become the exact promoted production tree.
    base = subprocess.check_output(
        ["git", "show", f"HEAD:{TREE_PATH}"], cwd=ROOT.parent.parent, text=True
    )
    base_start, base_end = short_epoch_range(base)
    serial_block = base[base_start:base_end]
    assert hashlib.sha256(serial_block.encode()).hexdigest() == EXPECTED_SERIAL_BLOCK_SHA256
    reconstructed = source[:start] + serial_block + source[end:]
    reconstructed_sha256 = hashlib.sha256(reconstructed.encode()).hexdigest()
    assert reconstructed_sha256 == EXPECTED_COMPACT_PARENT_TREE_SHA256

    extra_epoch_bytes = 32768 * 64
    extra_result_bytes = 4 + 1024 * 4 + 1024 * 10
    assert extra_epoch_bytes == 2 * 1024 * 1024
    assert extra_result_bytes < 16 * 1024

    print({
        "status": "PASS",
        "host_block_sha256": EXPECTED_HOST_BLOCK_SHA256,
        "slot_drain_launches": "1..4096",
        "reconstructed_compact_parent_tree_sha256": reconstructed_sha256,
        "incremental_device_bytes": extra_epoch_bytes + extra_result_bytes,
        "cuda_execution": "NOT RUN",
        "gpu_throughput": "NOT RUN",
    })


if __name__ == "__main__":
    main()
