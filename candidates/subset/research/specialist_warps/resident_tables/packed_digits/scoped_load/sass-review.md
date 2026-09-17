# Scoped packed-load native review

Complete source `cd3d11f24e6fd42459eed3ba3a431c7f952de506cb1d22dfddc9b291e4c98667` matches all recorded production files in `native-results.json`. SASS is from `<local-path-omitted>/qsb-cuda-work/compile-cwt1bsbp/build/sass.txt`; comparison is frozen packed handoff `790949f20f5d888a32ee24042b1ac4f805ed52bba009cfab474228ec5e686c2e`. The JSON and script bind exact source, native report, SASS, baseline and parser hashes. No candidate, VM or GPU changes occurred.

The scoped variant saves **52 B local reads and52 B local writes per active candidate** versus packed790. It adds104 B shared reads inside the repeated loop and one8 B packed-cell read before the guarded final add. Its producer is instruction-identical after address rebasing. The compiler still emits the next-window prefetch near the loop tail; the source scope does not establish an early hint or a shorter native register lifetime.

| Quantity | Packed790 | Scoped cd3d |
|---|---:|---:|
| Registers / shared bytes | 80 /32,832 | 80 /32,832 |
| Stack bytes/thread | 312 | 320 |
| ptxas aggregate spill stores / loads | 492 /316 | 496 /312 |
| Whole-kernel non-NOP slots | 15,500 | 15,530 |
| Repeated-loop static slots | 1,579 | 1,594 |
| Local reads/iteration/active lane | 12 B | 8 B |
| Local writes/iteration/active lane | 12 B | 8 B |
| Local reads across13 additions | 156 B | 104 B |
| Local writes across13 additions | 156 B | 104 B |
| Shared64-bit loads/iteration | 21 | 22 |
| Shared64-bit stores/iteration | 12 | 12 |

The repeated loop is0x4430–0xa7c0. R64 initializes to0, increments at0x4550, compares against13 at0xa6f0, and exits through0xa7b0 or follows the0xa7c0 backedge. Two unpredicated4-byte local loads at0x4440/0x5830 and two4-byte stores at0xa6d0/0xa6e0 execute on each active iteration. No internal branch skips them. The naive static-region×iteration budget is20,722 versus20,527 slots; the final backedge is not executed, and slot counts are not cycles.

The8-byte next-packed-limb local roundtrip in790 is removed. Scoped code loads the current packed cell at0x4490, consumes it for the current table address/sign, then reads the next hint's packed limb intoR72/R73 at0x4640. The latter remains live through register moves while field arithmetic executes. This change replaces the old next-limb local spill with additional shared access; it does not eliminate every long-lived next-hint value.

The remaining local slots0x7c and0x84 are carried field state. Their reloads feed the final two words of the eight-word `S2-Y1` subtraction at0x67a0/0x67d0, and their canonical output words are stored at the loop tail. Source/native correspondence identifies them as deferred-Y words6/7 (bits192–255). The local def-use is directly observed; the semantic names are inferred from arithmetic correspondence rather than debug metadata.

The loop has22 unpredicated `LDS.64` sites and12 `STS.64` sites, giving2,288 B shared loads and1,248 B stores per active candidate across13 iterations. Packed790 has2,184 B loads and the same stores. The new current-cell access accounts for104 B additional shared reads in the loop. A further actual `LDS.64` at0xa7e0 reads the packed final cell before the guarded window13 load. These are logical per-lane operands, not measured shared-bank transactions or latency.

The source places each next-hint call before EC arithmetic. Native code places the two `CCTL.E.PF2` instructions at0xa700/0xa710, after shared-Z and local carry stores, near the loop tail. Packed790 already emitted its hints late at0xa5c0/0xa5d0. Along a continuing ordinary iteration there are24 intervening instruction sites from scoped's last hint to the next iteration's first table load, versus17 for790. The extra decoding at loop entry contributes to that difference; it is neither a cycle estimate nor evidence of improved prefetch effectiveness. **Do not describe this result as preserving an early prefetch across the EC add.**

The full producer region0x29950–0x37630 is instruction-identical to790 after rebasing control-flow targets and return addresses by0x1e0:3,535 slots, no local load/store instructions. Producer lane selection, free-ticket wait, full-warp join and CTA fence therefore remain unchanged. Its leader atomic is0x361d0, join call0x36e10 and pre-payload fence0x373b0. Consumer ready polling also retains its single active atomic lane at0x510, unconditional warp join and CTA fence before payload access. The common helper at0x3cbb0 uses full-mask WARPSYNC.

The same80-register/32,832-byte resources preserve the existing conditional three-CTA legality calculation. Stack growth from312 to320 B does not change that particular register/shared capacity arithmetic. Actual loaded JIT resources, cache behavior, shared carveout, achieved residency and throughput remain unmeasured. The result supports a bounded local-versus-shared traffic tradeoff; aggregate496/312 spill totals are not the basis for the per-iteration conclusion.
