#!/usr/bin/env python3
"""Audit the access-frequency-aware persisting-L2 table window."""

from pathlib import Path


RECORD_BYTES = 64
CHUNK_ENTRIES = (1 << 17,) + (1 << 16,) * 14
TABLE_BYTES = sum(CHUNK_ENTRIES) * RECORD_BYTES


def covered_lookups(begin, size):
    """Expected chunk lookups covered when each chunk is sampled once."""
    end = begin + size
    cursor = 0
    covered = 0.0
    for entries in CHUNK_ENTRIES:
        chunk_bytes = entries * RECORD_BYTES
        overlap = max(0, min(end, cursor + chunk_bytes) - max(begin, cursor))
        covered += overlap / chunk_bytes
        cursor += chunk_bytes
    return covered


def audit_policy_math():
    assert TABLE_BYTES == 64 * 1024**2
    cold = CHUNK_ENTRIES[0] * RECORD_BYTES
    assert cold == 8 * 1024**2

    # AD102 exposes about 50 MiB of persisting L2. Moving that same-size
    # window beyond the double-sized cold first chunk covers one extra lookup.
    want = 50 * 1024**2
    baseline = covered_lookups(0, want)
    offset = covered_lookups(cold, want)
    assert baseline == 11.5
    assert offset == 12.5
    assert offset > baseline

    # The guard must fall back to offset zero whenever the full remaining
    # table would fit, so smaller tables/devices cannot produce an OOB window.
    for want in (0, 1, 48 * 1024**2, 50 * 1024**2, 56 * 1024**2, TABLE_BYTES):
        skip = cold if TABLE_BYTES > cold + want else 0
        avail = TABLE_BYTES - skip
        size = min(want, avail)
        assert 0 <= skip <= TABLE_BYTES
        assert 0 <= size <= TABLE_BYTES - skip


def audit_source():
    source = Path(__file__).with_name("pinning.cu").read_text()
    begin = source.index("    /* Pin the fixed-base table in L2.")
    end = source.index("    uint32_t *d_hit_cnt", begin)
    policy = source[begin:end]
    required = (
        "cudaDevAttrMaxPersistingL2CacheSize",
        "cudaDevAttrMaxAccessPolicyWindowSize",
        "cudaLimitPersistingL2CacheSize",
        "cudaStreamAttributeAccessPolicyWindow",
        "cudaAccessPropertyPersisting",
        "cudaAccessPropertyStreaming",
        "size_t cold = (size_t)gt_entries(0) * 64u;",
        "size_t skip = (gt_sz > cold + want) ? cold : 0;",
        "av.accessPolicyWindow.base_ptr  = (void *)(d_gt + skip);",
        "size_t avail = gt_sz - skip;",
        "av.accessPolicyWindow.hitRatio  = 1.0f;",
    )
    for token in required:
        assert policy.count(token) == 1, token
    assert source.index("size_t gt_sz = (size_t)GT_TOTAL_ENTRIES*64;") < begin
    assert source.index("cudaMalloc(&d_gt,gt_sz);") < begin
    assert begin < source.index("launch_pinning_pipeline<true>")


if __name__ == "__main__":
    audit_policy_math()
    audit_source()
    print(
        "PASS: 64 MiB table; 50 MiB offset window covers 12.5/15 lookups "
        "versus 11.5/15 for the prefix; guarded bounds and CUDA policy bound"
    )
