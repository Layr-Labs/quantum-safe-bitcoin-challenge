// SPDX-License-Identifier: GPL-3.0-only
#ifndef QSB_HIT_SNAPSHOT_H
#define QSB_HIT_SNAPSHOT_H

#include <stdint.h>
#include <string.h>

namespace qsb {

// Own the completed batch's records before its slot is reused. The caller must
// wait for the slot's readback event before capture(), and may then enqueue the
// next batch before checking/publishing these records on the CPU.
struct HitSnapshot {
    uint32_t sequence;
    uint32_t locktime;
    uint32_t count;
    uint32_t indices[64];

    HitSnapshot() : sequence(0), locktime(0), count(0) {}

    void capture(uint32_t seq, uint32_t lt, uint32_t reported,
                 const uint32_t *src) {
        sequence = seq;
        locktime = lt;
        count = reported > 64u ? 64u : reported;
        if (count != 0) memcpy(indices, src, count * sizeof(indices[0]));
    }
};

}  // namespace qsb

#endif  // QSB_HIT_SNAPSHOT_H
