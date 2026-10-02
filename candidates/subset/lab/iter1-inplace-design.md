# In-place downward tree: code audit before the benchmark

Default ZLAB_TREE2, 128 leaves, wave-top16. Products indices: leaves0..127;
64-nodes128..191;32-nodes192..223;16-nodes224..239. Smaller top intermediates240..253.
Aliases inverse indexj to product128+j. Upward construction unchanged. Root-wave
fully consumes its input/intermediate products before a warp-wide inversion and
shuffles; then stores inverse16 at224..239. Down32 snapshots parents224..239 and
siblings192..223; warp barrier before overwrite192..223. Down64 snapshots
parents192..223 and siblings128..191; block barrier before overwrite128..191.
The existing subsequent block barrier publishes inverses64. Leaves return
inverse64[tid&63]*leaf[tid^64]; original leaves untouched. All blocks launch the
selected128 threads, all participate in tree even for inactive candidate lanes;
invalid denominators remain identity. Early return block-uniform. No new errors,
allocation or cleanup paths. Shared LUT/pre3/root warp experiments rejected at
compile time for this probe. New knob goes into carrier identity to prevent a
stale native image from masking an enabled probe. Default knob0 remains separate
arenas. Compiler resource proof:128registers,20480shared,0stack/spills.

Single existing scored120s pair vs cached exact fill2 control is running.
This is not yet a speed result; do not substitute compile success for verifier.
The lower shared use cannot increase residency at the current128reg/fourCTA
cap; subsequent measured direction is launch_bounds(128,5), requiring roughly
96regs to fit640threads and5*20KiBshared. Extra register pressure/spills may kill
it: earlier launch_bounds3(139regs) lost15.61%, so do not adopt unmeasured.
