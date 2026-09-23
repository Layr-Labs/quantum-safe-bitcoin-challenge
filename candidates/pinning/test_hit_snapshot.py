#!/usr/bin/env python3
"""Compile the production snapshot header and audit slot publication ordering.

This is a CPU memory/ordering check, not a CUDA performance measurement. The
simulated DMA deliberately overwrites the old pinned payload at every reuse,
before the deferred publisher reads its snapshot.

SPDX-License-Identifier: GPL-3.0-only
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
CPP = r'''
#include "HitSnapshot.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static uint32_t word(uint32_t batch, uint32_t i) {
    return ((batch + i) % 2u << 30) | (i * 7919u + batch * 3u);
}

static void memory_audit() {
    qsb::HitSnapshot empty;
    assert(empty.count == 0);
    empty.capture(71, 83, 0, NULL);
    assert(empty.count == 0 && empty.sequence == 71 && empty.locktime == 83);
    const uint32_t counts[] = {0, 1, 63, 64, 65, 1024, UINT32_MAX};
    for (uint32_t count : counts) {
        uint32_t pinned[64];
        for (uint32_t i = 0; i < 64; ++i) pinned[i] = word(9, i);
        qsb::HitSnapshot snapshot;
        snapshot.capture(0x80000009u, 500000009u, count, pinned);
        // Model a completed readback for the next batch in the same slot.
        for (uint32_t i = 0; i < 64; ++i) pinned[i] = word(103, i);
        assert(snapshot.sequence == 0x80000009u);
        assert(snapshot.locktime == 500000009u);
        const uint32_t n = count > 64u ? 64u : count;
        assert(snapshot.count == n);
        for (uint32_t i = 0; i < n; ++i) assert(snapshot.indices[i] == word(9, i));
        // Reusing the snapshot for an empty batch must erase the old count.
        snapshot.capture(19, 23, 0, NULL);
        assert(snapshot.count == 0 && snapshot.sequence == 19 && snapshot.locktime == 23);
    }
}

struct Record {
    uint32_t sequence, locktime, recid;
    bool operator==(const Record &r) const {
        return sequence == r.sequence && locktime == r.locktime && recid == r.recid;
    }
};

struct Slot {
    bool busy;
    uint32_t sequence, locktime, count, indices[64];
    Slot() : busy(false), sequence(0), locktime(0), count(0) {}
};

static void publish(std::vector<Record> &records, uint32_t sequence,
                    uint32_t locktime, uint32_t count, const uint32_t *indices) {
    for (uint32_t i = 0; i < count && i < 64; ++i) {
        const uint32_t raw = indices[i];
        Record record = {sequence, locktime + (raw & 0x3fffffffu), (raw >> 30) & 1u};
        records.push_back(record);
    }
}

static std::vector<Record> run(bool deferred, int width, int sequences) {
    Slot slots[2];
    std::vector<Record> records;
    uint32_t batch = 0;
    const uint32_t counts[] = {0, 1, 2, 63, 64, 65, 1024, UINT32_MAX};
    for (int seq = 0; seq < sequences; ++seq) {
        for (int j = 0; j < width; ++j, ++batch) {
            Slot &slot = slots[batch % 2];
            qsb::HitSnapshot completed;
            // eventSynchronize is represented by arriving here only after the
            // old slot's payload is complete. The publisher remains serial.
            if (slot.busy) {
                if (deferred) {
                    completed.capture(slot.sequence, slot.locktime, slot.count, slot.indices);
                } else {
                    publish(records, slot.sequence, slot.locktime, slot.count, slot.indices);
                }
                slot.busy = false;
            }
            // Enqueue and even complete the next batch before publication.
            // Both metadata and pinned DMA destination are now overwritten.
            slot.sequence = 0x80000000u + static_cast<uint32_t>(seq);
            slot.locktime = 500000000u + static_cast<uint32_t>(j) * 8388608u;
            slot.count = counts[batch % 8];
            for (uint32_t i = 0; i < 64; ++i) slot.indices[i] = word(batch, i);
            slot.busy = true;
            if (deferred) {
                publish(records, completed.sequence, completed.locktime,
                        completed.count, completed.indices);
            }
        }
        // Preserve the existing slot-index drain order on sequence rollover.
        // Do not update a shared tail table until all slots are drained.
        for (Slot &slot : slots) {
            if (slot.busy) {
                qsb::HitSnapshot completed;
                completed.capture(slot.sequence, slot.locktime, slot.count, slot.indices);
                slot.busy = false;
                publish(records, completed.sequence, completed.locktime,
                        completed.count, completed.indices);
            }
            assert(!slot.busy);
        }
    }
    return records;
}

static void ordering_audit() {
    unsigned scenarios = 0;
    // Odd and even sequence lengths include partially filled startup slots,
    // both rollover drain orders, empty batches, and clipped oversized counts.
    for (int width = 1; width <= 12; ++width) {
        for (int sequences = 1; sequences <= 11; ++sequences) {
            assert(run(false, width, sequences) == run(true, width, sequences));
            ++scenarios;
        }
    }
    printf("publication_scenarios=%u\n", scenarios);
}

int main(int argc, char **argv) {
    assert(argc == 2);
    if (strcmp(argv[1], "memory") == 0) memory_audit();
    else if (strcmp(argv[1], "ordering") == 0) ordering_audit();
    else return 2;
    puts("PASS (CPU only; no GPU timing)");
}
'''


class HitSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++")
        if not compiler:
            raise RuntimeError("g++ is required to test the production C++ header")
        cls.temp = tempfile.TemporaryDirectory(prefix="qsb-hit-snapshot-")
        build = Path(cls.temp.name)
        source = build / "snapshot_test.cpp"
        source.write_text(CPP)
        cls.binary = build / "snapshot_test"
        subprocess.run(
            [compiler, "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror",
             "-I", str(HERE), str(source), "-o", str(cls.binary)], check=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_snapshot_survives_dma_and_metadata_reuse(self):
        subprocess.run([str(self.binary), "memory"], check=True)

    def test_two_slot_publication_matches_original_order(self):
        subprocess.run([str(self.binary), "ordering"], check=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
