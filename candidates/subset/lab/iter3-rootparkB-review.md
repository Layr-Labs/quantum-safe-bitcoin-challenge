# Root-warp-only B liveness probe: pre-verification source review

The current digest carries twelve B front words across a batch-inverse tree.
Only warp 0 runs the root divsteps; other warps wait on later tree barriers.
This probe passes B into the existing inlined wave-top tree and uses the
otherwise-dead inverse arena as scratch on those 32 root lanes only. It does
not change the inverse arithmetic or add shared storage or a CTA-wide barrier.

Scratch location j is inverses[j/4][32*(j%4)+lane], j=0..11, lane=0..31:
three rows by 128 columns, 384 uint64 words = 3072 bytes. Each lane owns all
of its locations. QSB_SE_BLOCK>=128 is required. No products are overwritten.
The inverse arena has not been published anywhere before the root inverse.
The twelve words are restored after the root's final shuffles and before the
first inv16 write. A warp barrier after reloads prevents one lane's inv16
publication from racing another lane's scratch read. Later down-sweep and
leaf inverse writes use the same arena and ordering as the old implementation.
Non-root warps never access this scratch; their B values are unchanged.

Volatile stores and loads prevent forwarding the saved B values across the
root routine. The compiler may nevertheless keep redundant live copies or
place its maximum pressure elsewhere; zero spills and a lower register count
are not performance evidence. Three KiB store + three KiB load traffic per
CTA may outweigh a scheduling benefit. This is distinct from the rejected
in-place inverse tree and PRE3 callback: no inverse alias arithmetic changes,
no useful B transform during the root, and no all-warp parking.

Zero/invalid candidates carry whatever front words the unchanged code already
produced; B is not used when okB is false. Active/partial-lane semantics and
identity denominators remain the same. Empty blocks return before the tree.
No new allocation, host error, cleanup, hit encoding or argument ABI changes.
Direct callers such as tree_audit use the default nullptr and get the same
mathematical tree. Compact/shared LUT, moved root warp and PRE3 combinations
are compile-time rejected. Knob defaults to zero and appends native identity
bytes only when enabled. The submitted launch4 native image remains untouched.

First relevant exact and throughput check: unchanged benchmark.sh subset,
against the submitted launch4 control at identical local geometry, after
resource and emitted shared-store/reload inspection. This check actually
exercises non-null B; the old tree_audit would not prove B preservation.
