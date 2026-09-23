// SPDX-License-Identifier: GPL-3.0-only
// The existing OpenSSL table-check sample order, shared by gather and verifier.
#pragma once
#include <stdint.h>

struct QsbTableSample {
    int chunk;
    unsigned index;
};

template<class EntryCount>
static inline QsbTableSample qsb_table_sample_position(
    int sample, unsigned &seed, int chunks, EntryCount entries, bool big_table) {
    QsbTableSample result;
    if (sample < chunks * 4) {
        result.chunk = sample / 4;
        const unsigned corners[4] = {0u, 1u, 2u, entries(result.chunk) - 1u};
        result.index = corners[sample % 4];
    } else {
        seed = seed * 1664525u + 1013904223u;
        result.chunk = (int)(seed >> 28) % chunks;
        if (big_table) {
            seed = seed * 1664525u + 1013904223u;
            result.index = seed % entries(result.chunk);
        } else {
            result.index = (seed >> 4) & (entries(result.chunk) - 1u);
        }
    }
    return result;
}
