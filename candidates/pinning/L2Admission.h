// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <stddef.h>

// CUDA's hitRatio selects the fraction of the window given the persisting
// property; it is an admission hint, not a measured cache hit rate.
static inline float qsb_l2_admission_ratio(size_t window, size_t budget) {
    if (!window || budget >= window) return 1.0f;
    return static_cast<float>(budget) / static_cast<float>(window);
}
