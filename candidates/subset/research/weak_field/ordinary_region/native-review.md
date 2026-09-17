# Corrected ordinary-region native screen

Source `3ac276e408a4bb399ea805a56fc9e27831a7bbd6da8df5159f16e95f311c5912` is bound to the production CUDA12.8 sm89 compiler report and SASS. All17 production include hashes match; the full production/audit closure fingerprint is recorded separately. Baseline is frozen scoped resident32 `cd3d11f2`. Reproduce with `python3 -B candidates/subset/research/weak_field/ordinary_region/review_native.py` using the retained native build files. No compilation or GPU execution is performed by that script.

The actual ordinary-add loop is `0x4450–0x9d70`: **1,427 instruction slots, zero LDL/STL**, versus1,594 and8B local reads plus8B writes per iteration. The counter starts at0, increments once, compares against13, exits through `@P1 CALL.REL.NOINC 0x9d80`, otherwise branches to0x4450. The CALL is the loop-exit control transfer, not an uncounted arithmetic helper. Thus the old13-iteration104B read/104B write operand traffic disappears from this loop. Static slots decrease167 (10.48%); this is not a speed measurement.

Source retains order `[14,15,0..12,13]`, canonical seed/table signs and guarded final addition. Native packed-cell decode still uses low15bits for index and bit15 for sign, with four128-bit point loads per iteration. Shared accesses remain22 LDS.64 and12 STS.64 sites per iteration. Prefetches remain near the tail at0x9cd0/0x9d10; the next ordinary point load is18 intervening instruction sites away, versus24 previously. No claim of improved prefetch effectiveness follows.

Four canonical boundary conversions are visible as full-width carry/select chains, interleaved with final arithmetic rather than emitted as four isolated blocks. Their starts are0xa2d0 (ZZZ),0xae20 (Y),0xb0a0 (ZZ),0xbc60 (X), with names inferred from shared offsets and following arithmetic uses. ZZZ feeds the final Y product, Y its subtraction, ZZ the X product, and X its subtraction. Eight normalized ZZZ words spill around0xa470–0xa6f0, outside the repeated loop. The combined boundary/final region grows86slots; that is not a pure normalization cost because allocation and scheduling also change.

The table below counts static local operand bytes across whole regions once. It **does not** sum to per-candidate traffic: regions contain branches, rare paths, explicit inverse arrays and repeated operations.

| Region | Current load/store bytes | Baseline load/store bytes | Slot delta |
|---|---:|---:|---:|
| Entry, packet acquisition and seed |36 /38|28 /38|+2|
| One ordinary-add iteration |0 /0|8 /8|−167|
| Boundary normalization and guarded final addition |164 /390|120 /330|+86|
| Later consumer inverse/recovery/hash/hit/control |216 /164|188 /160|+10|
| SHA producer |0 /0|0 /0|0|
| Out-of-line helpers |140 /416|144 /424|−6|

Producer0x29500–0x371e0 has3,535slots and is instruction-identical after relocation. Six full-warp-helper CALL targets move by a different displacement; the four-instruction helper itself is independently identical. Consumer polling also matches after address/stack-offset rebasing: lane0 performs the atomic wait with8B local loads per spin, then all lanes join and fence before payload access. Producer polling remains lane0-only and local-free. Changes in stack offsets do not imply a changed protocol.

Whole-kernel resources are80registers,32,832shared bytes,352stack bytes,552/384 compiler spill-store/load totals and15,455 nonNOP instructions. Baseline is80/32,832/320,496/312 and15,530. Increased aggregate spill totals describe costs outside the now local-free point loop; they do not invalidate the loop result or establish a net benefit. Registers/shared capacity is unchanged, but achieved occupancy, local-cache behavior and runtime remain unmeasured.

This is a credible native result for the corrected prototype. Arithmetic/curve/protocol qualification is separate, and the source is not selected or submitted by this review. Frozen cd3d remains preserved.
