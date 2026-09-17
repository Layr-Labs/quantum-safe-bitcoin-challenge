# Bounded inline HM43 versus ready3ac

Reproduce saved-artifact analysis with `python3 -B candidates/subset/research/warp_root/bounded/review_native.py`. It binds exactf47ad824 source, both production/audit reports and all19 union files. Both sm89 and default builds pass. No compiler/VM/GPU action is performed by the review.

The root change has a material whole-kernel allocation cost: **the unchanged13-add point loop now reads104B and writes104B of local operands per active candidate**, versus0/0 in ready3ac. Its0x4490–0x9ee0 region has1,446slots versus1,427 (+19), with2 unpredicated4B loads and2 stores each iteration. The backedge and counter prove13 iterations, and the sole conditional CALL is the exit transfer. Two high deferred-Y words (inferred words6/7) are carried through stack offsets0x78/0x7c. Another8B is stored before the loop. Four LDG.128,22LDS.64 and12STS.64 sites per iteration are unchanged; hints remain late.

| Quantity | Ready3ac | Inline boundedf47 |
|---|---:|---:|
| Registers/thread |80|80|
| Shared bytes/CTA |32,832|32,832|
| Stack bytes/thread |352|368|
| Compiler spill stores/loads |552/384|556/344|
| Whole-kernel nonNOP slots |15,455|16,550|
| Ordinary loop slots |1,427|1,446|
| Repeated-loop local reads/writes per candidate |0/0B|104/104B|

The decreased aggregate compiler spill-load total does not contradict the observed repeated-loop regression. Aggregate spills cover other paths and must not substitute for located dynamic repetitions.

The SHA producer remains local-free but is **not** instruction-identical after relocation: it has3,536slots versus3,535, with register/load scheduling changes. Its opcode delta is+2 ISETP.NE.AND and−1 LOP3.LUT. Consumer ticket polling is identical modulo addresses/stack offsets, preserving one active atomic lane and8B local leader reads per spin. Producer polling stays leader-only and local-free. The full-warp join helper matches independently.

The HM43 matrix main region0x1af10–0x1c880 (408sites) and its outlined alternate paths0x39e90–0x3aea0 (258sites) are local-free. The main region contains24 direct32-bit shuffle sites and13 ballots; alternate paths contain39 calls into WARPSYNC/shuffle or WARPSYNC/ballot helpers. The complete root main region also includes setup, canonicalization and fallback, with local array materialization outside the matrix body. These region counts are not per-batch dynamic totals: branches choose alternate paths, inner divsteps repeat, and helper calls add work. Do not omit compiler-outlined paths when estimating root cost.

Native control explicitly compares the matrix counter with16 at0x1af10, exits on V's zero ballot at0x1c870, and loops at0x1c880. Failure reaches lane0's four shared root reloads0x1d150–180, clears the fifth word at0x1d190, calls the original scalar inverse at0x1d1e0, then publishes four root words0x1d240–270. The repeated matrix path is therefore a distinct alternative to the bounded scalar fallback, not a replacement that eliminated the fallback code.

This is a trade between root critical-path parallelism and collective/code/state costs, including renewed point-loop traffic. Equal register/shared capacity does not establish a gain; no GPU timing, achieved occupancy, physical traffic or complete dynamic root cost is available. Preserve this source as the bounded inline control. An isolated noinline experiment can test allocation coupling, with its call ABI costs measured separately.
