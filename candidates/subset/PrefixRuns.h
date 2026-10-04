// SPDX-License-Identifier: GPL-3.0-only
// DenChi: preserve the epoch-prefix bytes while copying contiguous kept pushes.
#ifndef DENCHI_QSB_PREFIX_RUNS_H
#define DENCHI_QSB_PREFIX_RUNS_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>

namespace qsb_prefix {
// Preconditions: omitted[0..count) is sorted, distinct and inside [0,cut).
// omitted[0..next) is below begin; output has space for every remaining push.
template <size_t PushBytes>
static inline size_t copy_kept_runs(uint8_t *output, const uint8_t *pushes,
                                   int cut, int begin, const uint8_t *omitted,
                                   int count, int next) {
    size_t written = 0;
    for (; next < count; ++next) {
        const int stop = omitted[next];
        const size_t bytes = (size_t)(stop - begin) * PushBytes;
        if (bytes) {
            memcpy(output + written, pushes + (size_t)begin * PushBytes, bytes);
            written += bytes;
        }
        begin = stop + 1;
    }
    const size_t bytes = (size_t)(cut - begin) * PushBytes;
    if (bytes) {
        memcpy(output + written, pushes + (size_t)begin * PushBytes, bytes);
        written += bytes;
    }
    return written;
}
} // namespace qsb_prefix
#endif
