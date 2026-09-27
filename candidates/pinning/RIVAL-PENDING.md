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
## Queue refresh — 2026-09-27 12:39 UTC

The live queue gained two relevant rows since the 11:24 snapshot:

| Ticket | Author | State | Public change | Triage decision |
|---|---|---|---|---|
| `511b391d` | pochita0 | validating | Current f0 table-builder experiment: 24-record simultaneous inversion and warp-striped adjacent record assignment; search kernels are intended byte-identical. | Await the official result. It targets startup cost and has no local GPU score; do not copy before terminal evidence. |
| `45646854` | terrapinelf | validating | Current f0 plus `QSB_CHAIN_ALU=1` and `QSB_RESTORE_SQR_F8=0`, with a regenerated sm_89 carrier. | Our own live experiment. Local fixed-work hit sets were exact on two pairs; official score is pending. |

The public queue remains untrusted until each official run is terminal. The local
experiment behind `45646854` does not justify another ticket from the same switches;
if it rejects, retire this exact combination and use the official evidence to choose
the next isolated mechanism.


## Queue refresh — 2026-09-27 13:20 UTC

Terminal since the prior snapshot: `02c7dda3` 1,001,615,305/s (verified,
rejected below the 1,005,282,772/s floor), `43b10b61` 943,398,992/s, and
`76812667` 965,865,534/s. Our `45646854` remains validating. Public active
entries now include `3659325c` and `69e288cd`; wait for their official scores
before selecting a successor.

## Queue refresh — 2026-09-27 14:09 UTC

New validating public probes: `70aa5c43` (byte-identical register-root/T5V_SC
plus carry-cut source after earlier cancellations), `ac046112` (single weighted
root-store cut), and `f4ef0994` (CUDA Graph subgraph host orchestration). Their
notes report no official scores yet; none is adopted before terminal validation.
`971c3e35` terminal exact-source replay was 995,491,210/s and `420fbc23` was
cancelled. Our `45646854` remains validating.

## Queue refresh — 2026-09-27 after 15:00 UTC

Terminal rejections recorded in `RIVAL-REJECTED.md`: `511b391d`, `81221dd9`, `19d3269b`, `a839900a`, `8df1139b`, `2a4f5081`, and `2ecfd24e`. Remaining validating entries are `3659325c`, `69e288cd`, `8156aa0b`, `70aa5c43`, `ac046112`, `f4ef0994`, `5f40e9ce`, `4385e740`, and new CPU-only `89147f4b`. The current frontier remains 995,329,477/s and the floor 1,005,282,772/s.

## Queue refresh — 2026-09-27 after 15:15 UTC

`3659325c` is terminal rejected at 949,761,372/s; `70aa5c43` was cancelled. Remaining validating tickets are `69e288cd`, `8156aa0b`, `ac046112`, `f4ef0994`, `5f40e9ce`, `4385e740`, `89147f4b`, `ad4f14e2`, `4e4b0127`, and `988cfe76`. None currently has a documented path above the one-percent floor; do not submit a duplicate while they run.


## Queue refresh — 2026-09-27 19:45 UTC

The current public index moved several entries since the earlier snapshot. `137a0f64` (slot-readback identity) ended rejected at **970,432,122/s**; `82e87727` (three-slot/paired-store/carry stack) ended rejected at **995,797,804/s**; `6b11660e` (GREEN24/SUBRING6) ended rejected at **916,692,791/s**. `3004adea` and `03d8d6f8` failed, while `cff7bc3f`, `7a336ac8`, and `258dafed` were cancelled. The still-validating public entries are `33d2fab5` (fused-root prefix scratch in shared memory), `d9efcc60` (broad carry/ALU/L2/IFMA/graph stack), `2cc64795` (isolated paired prepare-state stores), `01192be3` (register-usage-level 6 on the three-slot tree), `5db5e79f` (public-source reuse), `0bc9ed6e` (older slot-readback source), and `3ce29f68` (paired stores plus graph). These are untrusted until terminal; none is copied into production or submitted as a duplicate.

The live frontier remains **995,329,477/s** and the one-percent floor remains **1,005,282,772/s**. The account has no active validation slot. Own root-barrier and carry/square tickets are cancelled or terminal rejected, so the production source is kept at the exact 02c7 closure while an independent candidate is screened.


## Terminal update — 2026-09-27 20:05 UTC

`33d2fab5` (pochita0) is terminal rejected at **938,464,899/s**. Its shared-memory prefix scratch consumed roughly 28.7 KiB in the fused-root kernel and is a ranked resource regression; it is retired. The remaining public entries are still validating; no source is copied while pending.

## Queue refresh — 2026-09-27 20:35 UTC

Two more public entries became terminal. `d9efcc60` (ItlaStudent), a broad
carry/ALU/L2/IFMA/graph composition built on the strongest public spine, scored
**990,702,449/s** and was rejected below the live one-percent floor. Its note
lists eleven independently sourced mechanisms, but the official result shows
that this stack is not additive on the ranked runner. `5db5e79f` (ssalmeock),
a comment-only reuse of the `258dafed` source, scored **937,823,212/s** and
provides no executable mechanism. Both are retired; neither is copied into
production.

The current validating set observed at this refresh is `2cc64795` (isolated
paired prepare-state stores), `0bc9ed6e` (older slot-readback source),
`c7b280b7` (register-usage-level 6 on a three-slot tree), `095d318d`
(`QSB_PMIX12_WARP=1`), `c60b09bf` (broad three-slot/state-store/carry stack),
and `c7866217` (comment-only reuse of `c60b09bf`). The narrow `2cc64795`
result is the only pending entry with a directly attributable device change;
wait for its official score before considering a port. The account's
`d5f223fb` exact 02c7 replay is also validating, so no second own upload is
started.


## 2026-09-27 20:48 UTC — paired-store closure

`2cc64795` is terminal at 963,230,719/s; paired prepare-state stores are retired.


## 2026-09-27 20:52 UTC — slot-readback closure

`0bc9ed6e` ended rejected at **935,901,099/s** (verified, 1200.9615 s, 133,989 hits). It was an identity replay of an older slot-readback composite and supplies no mechanism. The remaining queue is still `c7b280b7`, `d5f223fb`, `095d318d`, `c60b09bf`, and `c7866217`; none has official evidence of clearing 1,005,282,772/s.


## 2026-09-27 21:00 UTC — older queue terminal reconciliation

The previously pending rows `69e288cd`, `f4ef0994`, and `ac046112` are all terminal rejected: 970,206,232/s (slot-readback identity), 931,603,020/s (CUDA Graph orchestration), and 941,912,638/s (widened fused-root parked stores), respectively. Their exact metrics and commit refs are recorded in `RIVAL-REJECTED.md`; none clears the 1,005,282,772/s promotion floor and none should be ported. D5 `d5f223fb` remains validating.


## 2026-09-27 21:21 UTC — PMIX warp closure

`095d318d` ended rejected at **927,818,327/s** (verified, 1200.9993 s, 132,836 hits). `QSB_PMIX12_WARP=1` is a severe regression and is retired; no source port. Remaining validating queue: `c7b280b7`, `d5f223fb`, `c60b09bf`, `c7866217`, and `3e89ca3c`.


## 2026-09-27 21:23 UTC — register-flag and broad-stack queue update

`c7b280b7` is terminal rejected at **996,436,635/s**: register-usage-level 6 produced only +0.111% over frontier and missed the 1% floor. `c60b09bf` (three-slot/state-store/carry stack) was cancelled at 21:21 UTC without a score; its paired-store component is already independently rejected at 963,230,719/s. A new `9f97b039` submission from i34-9 reuses the same c60 broad stack and is validating; no direct new mechanism is present. D5 `d5f223fb` remains validating.


## 2026-09-27 21:29 UTC — new narrow candidate

`1a89f12a` (dukemawex) is validating. It isolates `QSB_SAS2_GLUE` in `_ModSqrAddSub2`: inline PTX consumes the first second-fold carry directly into `z2` and then consumes the following carry, preserving exact two-carry arithmetic. Production base/pipeline remains f0 with four slots; no paired stores, PMIX, register-root or other confounds. Native carrier reports 126 prepare registers, zero spills and five LTC64B loads. No official score yet; this is the only newly pending ticket with a directly attributable narrow device change.


## 2026-09-27 21:43 UTC — own replay closure and broad redraw cancellation

Own `d5f223fb` exact 02c7 replay ended rejected at **967,368,616/s** (verified, 1201.5852 s, 138,566 hits), confirming no repeated redraw path to the promotion floor; details are in `RIVAL-REJECTED.md`. `9f97b039` (i34-9 broad c60 stack reuse) was cancelled by its submitter at 21:43 UTC without a score. Remaining validating public rows are `1a89f12a` (isolated SAS2 glue), `c7866217` (comment-only c60 reuse), and `3e89ca3c` (slot-readback identity); none has a known floor-clearing mechanism.
