#ifndef QSB_L2_POLICY_H
#define QSB_L2_POLICY_H

#include <stddef.h>

struct qsb_l2_policy {
    size_t set_aside_bytes;
    size_t window_bytes;
    float hit_ratio;
};

/* Host-only policy: distribute the persistence budget across the dense table
 * suffix. CUDA treats hit_ratio as an approximate segment-selection hint. */
static inline qsb_l2_policy qsb_make_l2_policy(
    size_t table_bytes, size_t skip, int max_persist, int max_window) {
    qsb_l2_policy policy = {};
    if (skip >= table_bytes || max_persist <= 0 || max_window <= 0)
        return policy;
    const size_t available = table_bytes - skip;
    policy.set_aside_bytes = available < (size_t)max_persist
        ? available : (size_t)max_persist;
    policy.window_bytes = available < (size_t)max_window
        ? available : (size_t)max_window;
    policy.hit_ratio = policy.set_aside_bytes >= policy.window_bytes
        ? 1.0f : (float)policy.set_aside_bytes / (float)policy.window_bytes;
    return policy;
}

#endif
