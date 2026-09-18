// Cached G-table limb loads. The mixed-add walks fifteen windows and
// reads one 64-byte X||Y record per window from the same immutable
// table. Those loads are ordinary global vector reads on this crown, so
// they share L1 with the hit-buffer atomics and the inverse tree.
// `__ldg` is the unused read-only-cache load on this walk: same
// little-endian limbs, different cache.
//
// One mechanism. Arithmetic, inverse, SHA, and the host loop are
// untouched. Host-drain (rejected 35f408d8) is not restacked.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ void gt_ldg_xy_ll2(
    ulonglong2 *x0,
    ulonglong2 *x1,
    ulonglong2 *y0,
    ulonglong2 *y1,
    const uint8_t *rec)
{
    const ulonglong2 *tx = (const ulonglong2 *)rec;
    const ulonglong2 *ty = (const ulonglong2 *)(rec + 32);
    *x0 = __ldg(tx);
    *x1 = __ldg(tx + 1);
    *y0 = __ldg(ty);
    *y1 = __ldg(ty + 1);
}
