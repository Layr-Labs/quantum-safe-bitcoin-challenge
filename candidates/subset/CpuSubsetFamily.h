/* Experimental host-only 5+4 enumeration. GPL-3.0, as CpuGrindSubset.h.
 * GPU candidates omit six pushes below 137; this family omits five, so
 * the two sets cannot intersect, independently of seed and epoch order.
 * The first 46 window bytes cover five kept pushes (the fifth partially).
 * Select 12 complete prefix classes: 70 + 5*35 + 6*15 = 335 candidates.
 * C(137,5)*335 = 125144925570 distinct candidates before exhaustion.
 */
#pragma once
#include <stdint.h>
#include <string.h>

namespace qsb_family54 {
enum { COUNT = 335, EARLY = 5, WINDOW = 4 };

static int make(uint8_t out[COUNT][WINDOW]) {
    uint8_t all[715][WINDOW];
    uint32_t keys[126] = {};
    int sizes[126] = {}, group[715], ng = 0, n = 0;
    for (int a = 0; a < 13; ++a) for (int b = a + 1; b < 13; ++b)
    for (int c = b + 1; c < 13; ++c) for (int d = c + 1; d < 13; ++d) {
        uint32_t key = 0; int kept = 0;
        for (int i = 0; i < 13 && kept < 5; ++i) {
            if (i == a || i == b || i == c || i == d) continue;
            key = (key << 4) | (uint32_t)i; ++kept;
        }
        int g = 0; while (g < ng && keys[g] != key) ++g;
        if (g == ng) { if (ng == 126) return 0; keys[ng++] = key; }
        if (n == 715) return 0;
        group[n] = g; ++sizes[g];
        all[n][0] = (uint8_t)(137 + a); all[n][1] = (uint8_t)(137 + b);
        all[n][2] = (uint8_t)(137 + c); all[n][3] = (uint8_t)(137 + d); ++n;
    }
    if (n != 715 || ng != 126) return 0;
    bool used[126] = {}; int count = 0;
    for (int k = 0; k < 12; ++k) {
        int best = -1;
        // Ties retain first occurrence in lexicographic omission order.
        for (int g = 0; g < ng; ++g)
            if (!used[g] && (best < 0 || sizes[g] > sizes[best])) best = g;
        if (best < 0) return 0;
        used[best] = true;
        for (int i = 0; i < n; ++i) if (group[i] == best) {
            if (count == COUNT) return 0;
            memcpy(out[count++], all[i], WINDOW);
        }
    }
    return count == COUNT ? count : 0;
}
} // namespace qsb_family54
