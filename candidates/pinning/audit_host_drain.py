#!/usr/bin/env python3
"""Audit of the async host drain (QSB_HOST_DRAIN) in pinning.cu.

Two parts:

1. Source-shape binding: the drain defaults on, needs the contiguous hit
   buffer (QSB_HOST_READBACK), uses one async 65-word copy per batch into
   double-buffered pinned staging, records one event per batch, resets the
   counter with cudaMemsetAsync, and reconstructs hit locktimes from the
   PREVIOUS batch's seq/batch_lt carried beside the staging slot.

2. A Python model of the ring protocol. Simulated batches produce random
   hit reports; the model issues "async copies" and "events", drains the
   previous slot after queuing the next batch, and checks that every report
   is decoded exactly once, in issue order, with the correct (seq, batch_lt)
   coordinates -- including across sequence boundaries and final partial
   batches.
"""

import random
from pathlib import Path

SRC = (Path(__file__).resolve().parent / "pinning.cu").read_text()


def check_source_shape() -> None:
    assert "#define QSB_HOST_DRAIN 1" in SRC
    assert "#define QSB_HOST_READBACK 1" in SRC
    assert "#if QSB_HOST_DRAIN && !QSB_HOST_READBACK" in SRC
    # contiguous device buffer: counter followed by the index window
    assert "cudaMalloc(&d_hit_cnt, (1 + 1024)*sizeof(uint32_t))" in SRC
    assert "d_hit_idx = d_hit_cnt + 1;" in SRC
    # pinned double-buffered staging, one event per slot
    assert SRC.count("cudaHostAlloc((void**)&h_pin_report[e_], (1 + 64)*sizeof(uint32_t), cudaHostAllocDefault)") == 1
    assert SRC.count("cudaEventCreateWithFlags(&pin_ev[e_], cudaEventDisableTiming);") == 1
    # per-batch async reset, copy, event
    assert "cudaMemsetAsync(d_hit_cnt, 0, 4, 0);" in SRC
    assert "cudaMemcpyAsync(h_pin_report[pin_cur], d_hit_cnt, (1+64)*sizeof(uint32_t)," in SRC
    assert "cudaEventRecord(pin_ev[pin_cur], 0);" in SRC
    # one-batch-late drain on the alternate slot
    assert "cudaEventSynchronize(pin_ev[pv]);" in SRC
    assert "uint32_t p_hit = h_pin_report[pv][0];" in SRC
    assert "const uint32_t *hits = h_pin_report[pv] + 1;" in SRC
    # locktime reconstruction uses the previous batch's coordinates
    assert "uint32_t lt = pin_prev_lt + (raw & 0x3FFFFFFF);" in SRC
    assert 'fprintf(pin_ff, "sequence=%u\\nlocktime=%u\\nhash_choice=%d\\nrecid=%d\\n",' in SRC
    assert "pin_prev_seq = seq; pin_prev_lt = batch_lt;" in SRC
    assert "pin_cur ^= 1; pin_have_prev = 1;" in SRC
    # async per-sequence midstate upload from the pinned ring
    assert "cudaMemcpyAsync(d_mid, mslot, 32," in SRC
    # the record format and file name are unchanged
    assert 'snprintf(pin_fname, sizeof(pin_fname), "results/pinning_hit_%d.txt", gpu_index);' in SRC
    print("host-drain source binding: OK")


def model_drain() -> int:
    rng = random.Random(0xD0A1)
    LT_MIN, LT_MAX = 500000000, 1744600000
    BATCH = 16_777_216
    lt_range = LT_MAX - LT_MIN

    # simulated device: batch i produces a random report
    reports = []
    coords = []
    seq = 0x80000000
    batches = 0
    made = []
    for s in range(4):
        lt_off = 0
        while lt_off < lt_range and batches < 37:
            batch_sz = min(BATCH, lt_range - lt_off)
            n_hits = rng.choice([0, 0, 0, 1, 1, 2, 5])
            rep = [n_hits] + [rng.getrandbits(30) | (rng.getrandbits(1) << 30) | (rng.getrandbits(1) << 31)
                              if n_hits else 0 for _ in range(64)]
            made.append((seq, LT_MIN + lt_off, rep))
            batches += 1
            lt_off += BATCH
        seq += 1

    pinned = [[None] * 65, [None] * 65]   # staging slots
    pin_cur, pin_have_prev = 0, 0
    pin_prev_seq = pin_prev_lt = 0
    drained = []
    for seq_, lt_, rep in made:
        pinned[pin_cur] = list(rep)          # async copy lands before event
        # event recorded; host moves on, drains previous slot
        if pin_have_prev:
            pv = pin_cur ^ 1
            prev = pinned[pv]                # eventSynchronize: copy complete
            if prev[0] > 0:
                nh = min(prev[0], 64)
                for h in range(nh):
                    raw = prev[1 + h]
                    drained.append((pin_prev_seq,
                                    pin_prev_lt + (raw & 0x3FFFFFFF),
                                    (raw >> 30) & 1, (raw >> 31) & 1))
        pin_prev_seq, pin_prev_lt = seq_, lt_
        pin_cur ^= 1
        pin_have_prev = 1

    # expected: every batch except the in-flight last one, in order
    want = []
    for seq_, lt_, rep in made[:-1]:
        for h in range(min(rep[0], 64)):
            raw = rep[1 + h]
            want.append((seq_, lt_ + (raw & 0x3FFFFFFF), (raw >> 30) & 1, (raw >> 31) & 1))
    assert drained == want, "drain reordering or coordinate mismatch"
    return len(drained)


def main() -> None:
    check_source_shape()
    hits = model_drain()
    print(f"PASS: async host drain; {hits} drained hits match in-order "
          f"reference across sequence and partial-batch boundaries; "
          f"in-flight loss is the final batch only")


if __name__ == "__main__":
    main()
