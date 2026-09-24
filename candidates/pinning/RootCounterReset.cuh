// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#ifndef QSB_ROOT_COUNTER_RESET
#define QSB_ROOT_COUNTER_RESET 1
#endif
#if QSB_ROOT_COUNTER_RESET != 0 && QSB_ROOT_COUNTER_RESET != 1
#error "QSB_ROOT_COUNTER_RESET must be 0 or 1"
#endif

// Called only from the super-root kernel. Completion of that kernel, and
// the existing root-to-finish stream dependency, precedes every hit writer.
// This is not safe for a reset in the hit-producing kernel itself.
__device__ __forceinline__ void qsb_reset_hit_count_in_root(
    uint32_t *hit_count, unsigned global_thread_id) {
#if QSB_ROOT_COUNTER_RESET
    if (global_thread_id == 0u) *hit_count = 0u;
#else
    (void)hit_count;
    (void)global_thread_id;
#endif
}
