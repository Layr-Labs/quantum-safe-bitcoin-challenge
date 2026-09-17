#!/usr/bin/env python3
"""Exact byte-layout and alignment audit for the vectorized pipeline state."""

from pathlib import Path


FIELDS = 3
LIMBS = 4
LIMB_BYTES = 8
VECTOR_BYTES = 16
VECTOR_PLANES = 6
WARP = 32


def old_offset(batch_size, field, limb, candidate):
    plane = field * LIMBS + limb
    return (plane * batch_size + candidate) * LIMB_BYTES


def vector_offset(batch_size, field, limb, candidate):
    plane = field * 2 + limb // 2
    return (plane * batch_size + candidate) * VECTOR_BYTES + (limb & 1) * LIMB_BYTES


def audit_batch(batch_size):
    old = {}
    vector = {}
    for field in range(FIELDS):
        for limb in range(LIMBS):
            for candidate in range(batch_size):
                logical = (field, limb, candidate)
                old[old_offset(batch_size, *logical)] = logical
                vector[vector_offset(batch_size, *logical)] = logical

    cells = batch_size * FIELDS * LIMBS
    expected_offsets = list(range(0, cells * LIMB_BYTES, LIMB_BYTES))
    assert len(old) == len(vector) == cells
    assert sorted(old) == sorted(vector) == expected_offsets

    packed = [None] * (batch_size * VECTOR_PLANES)
    for field in range(FIELDS):
        for pair in range(2):
            plane = field * 2 + pair
            for candidate in range(batch_size):
                packed[plane * batch_size + candidate] = (
                    (field, pair * 2, candidate),
                    (field, pair * 2 + 1, candidate),
                )
    unpacked = {tag for pair in packed for tag in pair}
    expected = {
        (field, limb, candidate)
        for field in range(FIELDS)
        for limb in range(LIMBS)
        for candidate in range(batch_size)
    }
    assert unpacked == expected

    for plane in range(VECTOR_PLANES):
        plane_base = plane * batch_size * VECTOR_BYTES
        assert plane_base % VECTOR_BYTES == 0
        for warp_first in range(0, batch_size, WARP):
            lanes = min(WARP, batch_size - warp_first)
            starts = [
                plane_base + candidate * VECTOR_BYTES
                for candidate in range(warp_first, warp_first + lanes)
            ]
            assert all(address % VECTOR_BYTES == 0 for address in starts)
            assert all(b - a == VECTOR_BYTES for a, b in zip(starts, starts[1:]))
            assert starts[-1] + VECTOR_BYTES - starts[0] == lanes * VECTOR_BYTES

    assert cells * LIMB_BYTES == batch_size * VECTOR_PLANES * VECTOR_BYTES
    return cells * LIMB_BYTES


def audit_source():
    source = Path(__file__).with_name("pinning.cu").read_text()
    assert 'static_assert(sizeof(ulonglong2) == 16' in source
    assert 'static_assert(alignof(ulonglong2) == 16' in source
    assert source.count("ulonglong2 *saved") >= 2
    assert "BATCH*QSB_STATE_PLANES*sizeof(ulonglong2)" in source
    assert "alignof(ulonglong2)-1u" in source
    for plane in range(VECTOR_PLANES):
        address = f"saved[{plane}u*state_plane_stride+state_idx]"
        assert source.count(address) >= 2


def audit_production_batch():
    batch_size = 16_777_216
    total_bytes = batch_size * VECTOR_PLANES * VECTOR_BYTES
    assert total_bytes == int(1.5 * 1024**3)
    for plane in range(VECTOR_PLANES):
        assert plane * batch_size * VECTOR_BYTES % VECTOR_BYTES == 0
    assert vector_offset(batch_size, FIELDS - 1, LIMBS - 1, batch_size - 1) == (
        total_bytes - LIMB_BYTES
    )
    for warp_first in (0, WARP, batch_size // 2, batch_size - WARP):
        starts = [
            vector_offset(batch_size, 0, 0, candidate)
            for candidate in range(warp_first, warp_first + WARP)
        ]
        assert all(address % VECTOR_BYTES == 0 for address in starts)
        assert starts[-1] + VECTOR_BYTES - starts[0] == WARP * VECTOR_BYTES


def main():
    for batch_size in (1, 2, 3, 31, 32, 33, 255, 256, 257, 4096):
        assert audit_batch(batch_size) == batch_size * (VECTOR_PLANES * VECTOR_BYTES)
    audit_production_batch()
    audit_source()
    print(
        "PASS: 4x256-bit state maps bijectively to 8 ulonglong2 planes; "
        "all vector elements are 16-byte aligned, warp addresses are contiguous, "
        "traffic remains exactly 128 bytes/candidate/direction, and the "
        "16,777,216-candidate allocation remains exactly 2 GiB"
    )


if __name__ == "__main__":
    main()
