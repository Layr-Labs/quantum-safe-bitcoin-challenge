# Pinning rival queue: pending public experiments

Updated from `yukon submissions --all --json` on 2026-09-27 at 11:24 UTC.
These are public, untrusted notes for triage.  No pending source is copied into
the production tree before its official validator result is available.

The current promoted source is `f0e453da` / submission `54ca2f74`, at
**995,329,477 candidates/s**.  The automatic promotion floor is
**1,005,282,772 candidates/s**.  A pending ticket is useful only if its
official result clears that floor, or if it provides a clearly isolated,
reproducible near-frontier mechanism that merits a fresh same-base test.

| Ticket | Author | State | Public change | Triage decision |
|---|---|---|---|---|
| `fb1105b1` | jacklightChen | rejected | f0 GPU/carrier unchanged; a 17 GiB high-window fixed-A CPU table (largest table construction batches, highfold physical slot) selected only on hosts with at least 32 GiB | Official RTX 4090 result **968.337M/s** (138,702 hits, 1201.562s). The local verified-hit screen was 769.381M/s; retire highfold and do not port the large table. |
| `9cb9450d` | ItlaStudent | rejected | Broad composition of public mechanisms, based on the older `e892e6e5` lineage; the note does not isolate a new component | Official RTX 4090 result **990.946M/s** (141,944 hits, 1201.592s). Below f0 and floor; retire the stale broad composition. |
| `3363bd35` | i34-9 | rejected | f0 with three slots, 16-byte evict-last stores, and shortened carry/glue forms | Official RTX 4090 result was 958,974,171/s (137,295 hits). Retire the entire three-slot/state-store/carry family. |
| `8d47e745` | jrcarlos2000 | rejected | Slot-readback composite remeasurement with inherited table/field changes | Official RTX 4090 result was 953,482,942/s (136,513 hits). It missed the frontier by 4.2%; do not duplicate the composite. |
| `e4645d0c` | kongtaoxing | rejected | f0 with only `QSB_TBL_L2POL=1->2`, rebuilding the native carrier | Official RTX 4090 redraw was 958,074,582/s (137,224 hits); together with 951f5878 at 986,930,784, retire L2POL2. |
| `25b9fa7b` | nemmbot | rejected | f0 with only `QSB_S0_SHM=0->1` | Official RTX 4090 result was 985,615,829/s (141,180 hits), confirming the earlier 923M negative probe; retire S0_SHM. |
| `0f80cdd7` | ercumentyildirim | cancelled | f0 GPU/carrier plus AVX-512 IFMA52 CPU co-grinder | Cancelled before scoring. The same IFMA family already failed official fast-class validation (`1a116ba7`: 991,912,922; `2afe4491`: 963,949,786), so there is no reason to replay it. |
| `ad469eaf` | DPZZxlz | cancelled | f0 plus register-root trees, three slots, 16-byte stores, and zero-carry ALU stack | Cancelled before validation; no score or source commit. No action. |
| `8e56bf7d` | DPZZxlz | rejected | Repeat of the four-item register-root/three-slot/16-byte/ALU stack with the f0 parent and rebuilt carrier | Official RTX 4090 result **989.702M/s** (141,761 hits, 1201.551s, seed 1063227484); below f0 and floor. Retire the stack. |
| `57c97154` | i34-9 | rejected | Repeat of the three-slot, 16-byte state-store and shortened carry/glue package | Official RTX 4090 result **954.584M/s** (136,673 hits, 1201.043s). This confirms the family regression; do not copy or replay. |
| `61d034de` | ercumentyildirim | cancelled | f0 plus IFMA52 CPU co-grinder and the shortened carry-fold bundle (`QSB_ADDOFF_CUT`, `QSB_SAS2_GLUE`, `QSB_SUB_CUT`, `QSB_YOFF_Y1_CUT`) | Cancelled before validation. Its local GPU claim was only +0.181%, and the IFMA family already failed official fast-class draws; no action. |
| `04fb24eb` | jrcarlos2000 | rejected | Repeat of the slot-readback composite remeasurement | Official RTX 4090 result was 957,100,564/s (137,027 hits); retire the composite. |
| `d2679a73` | nemmbot | rejected | One f0 device cut: replace four consecutive root `uint64_t` stores with two `qsb_st_v2` vector stores | Official RTX 4090 result **964.176M/s** (138,103 hits, 1201.535s, seed 2067505468); local verified-hit screen was also −6.7%. Retire vector root stores. |
| `0714a1f9` | jrcarlos2000 | rejected | Exact f0/slot-readback composite remeasurement with a no-op tag macro; no executable change | Official RTX 4090 result **953.439M/s** (136,509 hits, 1201.043s, seed 1265232507); identity only, below f0 and floor. Never replay. |
| `ab2cab6b` | jungjipdo | cancelled | Broad public-source import with register-root, IFMA8, cyclic-field, warp-inverse and carrier changes | Cancelled before validation; no official score. Do not copy or stack. |
| `ae170f75` | i34-9 | rejected | Repeat of the three-slot, 16-byte state-store and shortened carry/glue package (`57c97154`) | Official RTX 4090 result **960.821M/s** (137,560 hits, 1200.991s, seed 492790023); confirms the family regression. Do not copy or replay. |
| `c74c763a` | Anshumancanrock | rejected | Short-carry two cofactor-tree products plus adaptive CPU co-grinder on stale register-root source `9b633d47` | Official **995.834M/s** (RTX 4090, 1201.607s, 142,646 hits; commit `5a0cbda8`); below the 100-bips floor. Do not port or redraw. |
| `5a14d381` | cefika | cancelled | `QSB_GREEN=20->24` finish partition only | Cancelled by submitter; duplicate of `2402ebc0`, which scored 958.189M/s. No action. |
| `cf70268e` | ItlaStudent | rejected | Broad public-mechanism composition based on stale `e892e6e5`, adding tuning/offload/cache/startup glue | Official **962.728M/s** (RTX 4090, 1201.504s, 137,892 hits; commit `dad4e9b9`); retire the unisolated stack. |
| `62d66afd` | dukemawex | rejected | Two independent root queues on exact f0, preserving the four-entry ring | Official **953.240M/s** (RTX 4090, 1201.021s, 136,478 hits; commit `61e7254c`); retire root-stream splitting. |
| `e1231b29` | jrcarlos2000 | rejected | Exact public slot-readback composite remeasurement | Official **939.988M/s** (RTX 4090, 1201.014s, 134,580 hits; commit `a8ff2690`); identity/stale family, never replay. |
| `90ae8092` | jungjipdo | rejected | Exact public source evaluation from stale `664daccf` lineage | Official **989.420M/s** (RTX 4090, 1201.555s, 141,721 hits; commit `eb3b5722`); identity/stale draw, no replay. |
| `65872c52` | jacklightChen | rejected | Recoverable CPU-worker budget on the promoted source | Official **966.733M/s** (RTX 4090, 1201.516s, 138,467 hits; commit `e9d2635d`); retire the controller change. |
| `482a55e6` | cefika | cancelled | f0 plus register roots, three slots, 16-byte stores, carry cuts and L2 window cap | Cancelled before validation; no official score. Broad known-negative composition. |
| `da43687f` | cefika | cancelled | Byte-identical redraw of the same register-root/three-slot/16-byte/carry-cut composition | Cancelled by submitter before validation; no official score. No action. |
| `36ff02a6` | ssalmeock | rejected | Comment-only packaging reuse of `482a55e6` | Official **925.743M/s** (RTX 4090, 1200.946s, 132,533 hits; commit `d176fda0`); identity/slow draw, no action. |
| `19d3269b` | ssalmeock | validating | Work-normalized, aligned ABBA windows for CPU co-grinder accounting; retains actual batch size per slot | Host-only accounting/controller change; await official result. Do not port or stack before validation. |
| `8df1139b` | nemmbot | validating | One isolated `widen_root_group_finish_store` cut on the live tip | No local GPU measurement; inspect only if the official result clears the floor, with no port before then. |
| `a839900a` | pochita0 | validating | Host co-grinder accounting contention reduction on f0 | Host-only controller change without local ranked GPU evidence; await official result. |
| `02c7dda3` | ItlaStudent | validating | f0 plus stale c74 register roots/CPU controller and public carry-glue cuts | Broad stale-base composition; do not copy or stack unless it unexpectedly clears the floor. |
| `76812667` | jrcarlos2000 | validating | Byte-exact replay of an older public source | Identity/noise measurement; no new mechanism to port. |
| `9813a243` | cefika | validating | Register-root + T5V short-carry tree, carry cuts and `GLV_NZ_CUT`, retaining four slots | Stale register-root composition; inspect only for an unexpected floor-clearing result. |
| `2ecfd24e` | ercumentyildirim | validating | Frontier device plus IFMA, carry-fold shortening and GLV zero-branch cut | IFMA/carry family already has official negatives; inspect only if unexpected promotion. |
| `2a4f5081` | i34-9 | validating | Repeat of the three-slot/state-store/carry-cut family | Duplicate of rejected `ae170f75` and `57c97154`; no action. |

## Terminal reconciliation — 2026-09-27 11:24 UTC

Since the previous snapshot, `65872c52` and `36ff02a6` became terminal
rejected; `da43687f` was cancelled. The earlier terminal rows
(`90ae8092`, `c74c763a`, `62d66afd`, `cf70268e`, and `e1231b29`) remain
rejected, while `ab2cab6b` and `482a55e6` remain cancelled. Full official
metrics, source refs, and mechanism decisions are in `RIVAL-REJECTED.md`.
The remaining validating queue is exactly eight tickets: `2ecfd24e`,
`2a4f5081`, `8df1139b`, `a839900a`, `02c7dda3`, `76812667`, `9813a243`,
and `19d3269b`. Do not copy any of them before terminal official results; a
promotion still requires **1,005,282,772/s** against the unchanged f0 frontier.

## Processing rule

When a ticket becomes terminal, record its official score, elapsed time,
verified hits, runner class, commit/source ref, and the exact changed files in
`RIVAL-REJECTED.md` (or in `RIVAL-PROMOTED.md` if it is accepted and promoted).
Classify it as a promotion, a fast near-miss, a slow-runner outlier, or a true
regression.  A valid score below the floor is not a reason to replay the same
archive.  Preserve the exact f0 GPU/carrier pair while the queue is pending.
