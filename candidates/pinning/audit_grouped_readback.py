#!/usr/bin/env python3
"""Grouped sequence readback: record encoding, range safety, and decode
equivalence for the fast-tail single-readback host loop.

The fast path queues every 16M-candidate pipeline of one sequence before a
single blocking hit-counter readback, so its hit records must be
self-describing: the kernel stores the absolute locktime in bits 0..30 and
the recovery id in bit 31 (this specialization has no second hash, so bit 31
is free). The fallback keeps one readback per batch and stores batch-relative
indices with recid in bit 30 and hash_choice in bit 31.
"""

import math
import random
from pathlib import Path


LT_MIN = 500_000_000            # timestamp-interpreted locktimes only
LT_MAX = 1_744_600_000
BATCH = 16_777_216              # QSB_BATCH: one pipeline
HIT_BUF = 1024
QSB_ZEROS_N = 24


def batches():
    lt_range = LT_MAX - LT_MIN
    off = 0
    while off < lt_range:
        sz = BATCH if off + BATCH <= lt_range else lt_range - off
        yield LT_MIN + off, sz
        off += sz


def audit_source():
    source = Path(__file__).with_name("pinning.cu").read_text()

    # Kernel-side encodings, exactly once each.
    enc = ("d_hit_idx[pos] = FAST_TAIL\n"
           "                    ? (((uint32_t)idx + start_lt) | ((uint32_t)ri << 31))\n"
           "                    : (((uint32_t)idx) | (ri << 30));")
    assert enc in source, "grouped fast/fallback record encoding missing"
    assert source.count("if(pos<1024){") == 1, "hit-buffer overflow guard"
    assert "if(pos<1024)d_hit_idx[pos]=((uint32_t)idx)|(ri<<30)|(1u<<31);" in source, \
        "easy-mode fallback record form"

    # Host-side decodes.
    loop = source[source.index("/* Search all safe locktimes"):]
    fast = loop[:loop.index("} else {")]
    assert "uint32_t lt = raw & 0x7FFFFFFFu;" in fast
    assert "int ri = (int)(raw >> 31);" in fast
    assert fast.count("cudaMemcpy(&h_hit, d_hit_cnt, 4, cudaMemcpyDeviceToHost)") == 1
    # The counter reset precedes the queued pipelines; the copy is the wait.
    assert fast.index("cudaMemsetAsync(d_hit_cnt, 0, 4)") < \
        fast.index("launch_pinning_pipeline<true>")
    assert fast.index("launch_pinning_pipeline<true>") < \
        fast.index("cudaMemcpy(&h_hit, d_hit_cnt, 4, cudaMemcpyDeviceToHost)")
    assert "cudaDeviceSynchronize" not in fast

    fallback = loop[loop.index("} else {"):]
    fallback = fallback[:fallback.index("/* Check if another GPU found it")]
    assert "uint32_t lt = batch_lt + (raw & 0x3FFFFFFF);" in fallback
    assert "int ri = (raw >> 30) & 1;" in fallback
    assert "int hc = (raw >> 31) & 1;" in fallback
    assert fallback.count("cudaMemsetAsync(d_hit_cnt, 0, 4)") == 1
    assert fallback.count("launch_pinning_pipeline<false>") == 1

    # The multi-GPU stop check is per sequence now, not throttled per 50M.
    assert "total_searched % (50*1024*1024)" not in source
    assert source.count("GPU %d found hit, stopping.") == 1

    # The 31-bit absolute-locktime contract is only sound while the whole
    # canonical range fits under 2^31; pin the range constants it depends on.
    assert "uint32_t LT_MIN = 500000000;" in source
    assert "uint32_t LT_MAX = 1744600000;" in source


def audit_ranges():
    worst = 0
    n = 0
    for batch_lt, sz in batches():
        worst = max(worst, batch_lt + (sz - 1))
        n += 1
    assert n == 75, f"expected 75 pipelines per sequence, got {n}"
    assert worst == LT_MAX - 1
    assert worst < (1 << 31), "absolute locktime would collide with recid bit 31"
    # Batch-relative field: idx < BATCH always fits its 30-bit slot.
    assert BATCH - 1 < (1 << 30)


def audit_decode_roundtrip():
    rng = random.Random(0x6E0CBEA7)
    checked = 0
    for batch_lt, sz in batches():
        for idx in {0, 1, sz - 1, rng.randrange(sz)}:
            for ri in (0, 1):
                # Fast encoding: self-describing, no batch context needed.
                raw = ((idx + batch_lt) | (ri << 31)) & 0xFFFFFFFF
                assert raw & 0x7FFFFFFF == batch_lt + idx
                assert (raw >> 31) == ri
                # Fallback encoding: needs batch_lt to decode.
                raw2 = (idx | (ri << 30) | (0 << 31)) & 0xFFFFFFFF
                assert batch_lt + (raw2 & 0x3FFFFFFF) == batch_lt + idx
                assert ((raw2 >> 30) & 1) == ri
                assert ((raw2 >> 31) & 1) == 0
                # Easy-mode fallback sets the hash-choice bit.
                raw3 = (idx | (ri << 30) | (1 << 31)) & 0xFFFFFFFF
                assert ((raw3 >> 31) & 1) == 1
                checked += 1
    return checked


def audit_buffer_capacity():
    """Poisson tail of the hit count for one grouped sequence at N=24."""
    per_seq = (LT_MAX - LT_MIN)  # candidates per sequence
    lam = per_seq * (2.0 ** -QSB_ZEROS_N)
    # P(X > HIT_BUF) via the Poisson upper tail, computed stably.
    term = math.exp(-lam)
    cdf = term
    for k in range(1, HIT_BUF + 1):
        term *= lam / k
        cdf += term
        if cdf > 1.0 - 1e-18:
            break
    p_overflow = max(0.0, 1.0 - cdf)
    assert lam + 6.0 * math.sqrt(lam) < HIT_BUF, \
        "expected hits + 6 sigma must fit the buffer"
    return lam, p_overflow


def main():
    audit_source()
    audit_ranges()
    checked = audit_decode_roundtrip()
    lam, p_overflow = audit_buffer_capacity()
    print(f"PASS: grouped sequence readback; 75 pipelines/sequence; absolute "
          f"locktime+recid records decode without batch context; {checked} "
          f"encode/decode roundtrips incl. partial-batch edges; buffer model "
          f"lambda={lam:.1f} hits/sequence, P(>1024)={p_overflow:.2e}")


if __name__ == "__main__":
    main()
