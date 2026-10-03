#pragma once
// Runtime half-slice planner. Primary b59 GPU geometry
// remains compiled unchanged. Runtime 65536 slicing selects secondary roots only.
#include <stdint.h>
#include <stddef.h>
namespace qsb_half_schedule_research {
static constexpr uint32_t candidates_per_slice = 65536;
static constexpr uint32_t ring_depth = 8;
static constexpr uint32_t tree_leaves = 128;
static constexpr uint32_t root_rows_allocated = 1024; // fixed scratch, even for tails
struct Slice {
    uint32_t offset, count, blocks, ring, lane;
    uint32_t locktime_delta, hit_base;
    size_t state_bytes, root_bytes;
};
static bool plan(uint32_t batch_size, uint32_t offset, uint64_t generation, Slice &out) {
    if (!batch_size || offset >= batch_size || offset % candidates_per_slice) return false;
    uint32_t left = batch_size - offset;
    uint32_t n = left < candidates_per_slice ? left : candidates_per_slice;
    out = {offset,n,(n+tree_leaves-1)/tree_leaves,
           (uint32_t)(generation%ring_depth),(uint32_t)(generation&1),
           offset/4,offset,(size_t)candidates_per_slice*64,
           (size_t)root_rows_allocated*4*sizeof(uint64_t)};
    return out.blocks >= 1 && out.blocks <= 512;
}
// Routing contract: prepare/finish remain the PRIMARY library; roots only SECONDARY.
// Before prepare(r): wait prior finish(r), and the current slot's input/reset event.
// roots(r) waits prepare(r); finish(r) waits roots(r); slot readback waits both
// finish lanes. Constants may change only after all work using either library drains.
// Initialization chooses ORIGINAL or HALF mode before arrays/events are allocated.
// Failure after enqueue is fatal, never a silent mid-job return to original geometry.
// Graph paths/startup root validation require their own integration; this planner
// does not authorize routing old graph nodes to the new root or skipping validation.
}
