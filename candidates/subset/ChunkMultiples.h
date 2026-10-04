// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>

namespace qsb_chunk {
// out[j] = (j+1)*base. Each expansion uses one batch of additions instead
// of a separate binary scalar multiplication for every table chunk.
// batch(dst, src, count, addend) must handle equal points by doubling.
template<class Point, class Double, class Batch>
bool multiples(std::vector<Point>& out, const Point& base, std::size_t count,
               Double twice, Batch batch) {
    out.resize(count);
    if (!count) return true;
    out[0] = base;
    if (count == 1) return true;
    out[1] = twice(base);
    for (std::size_t have = 2; have < count;) {
        const std::size_t n = std::min(have, count - have);
        // Copy the addend: dst never aliases it while the batch is written.
        const Point addend = out[have - 1];
        if (!batch(out.data() + have, out.data(), n, addend)) {
            out.clear();
            return false;
        }
        have += n;
    }
    return true;
}
} // namespace qsb_chunk
