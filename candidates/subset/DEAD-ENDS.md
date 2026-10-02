# Subset dead ends — 2026-10-01 night session

Every entry below was measured on the dev RTX 3090 (sm_86, GLV12 9.13 GiB geometry, `QSB_LOCAL_SM86 1`) unless noted. N=4 = the fast local ABBA protocol; N=24 = ranked difficulty.

| lever | setting | measurement | verdict |
|---|---|---|---|
| `QSB_OUTER_LITK` | 1 (literal-K on the outer SHA block) | 4-round ABBA at N=4: A(C2) 318.45±0.95 vs B(D2) 319.18±0.15, delta −0.21% | DEAD solo (noise). The earlier +2.7% (319.4 vs 311) was a mid-run epoch-line read during warm-up drift — both arms read ~318-319 at steady state. |
| `QSB_PRE3_ROOT` | 1 | q-pairs −0.46% (4/4 neg), kernel census −0.60%, smokes −3.7% | DEAD (3 reads). Spills 20/24 B + forces SC_LATE. |
| Native GLV11 geometry on the dev rig | `QSB_LOCAL_SM86 0` | table 22,688,113,472 B = 21.1 GiB; OOMs solo (20.7 GiB free with desktop session) | IMPOSSIBLE on 24 GiB 3090. All N=24 dev work must ride the GLV12 9.13 GiB geometry. "GLV12 table allocation failed" is printed by BOTH geometries — the label is not diagnostic. |
| Two concurrent scored runs in one checkout | 1025+1026 | "judge or problem files modified during run" + seed mismatch REJECT | Scored runs are strictly serial; they share `benchmark-results/problem/` and `results/`. |
| `bash stat %.9Y` stamps for gpu_wrap | (rig bug) | prints `1790855154.253761415` (dot) vs python `st_mtime_ns` `1790855154253761415` (integer) — never equal | Always write stamps with python, or the harness plain-rebuilds (and on this rig, OOMs). |
| Pre-include `#define` overrides of knobs tree.cu defines | — | WRONG THEORY (retracted): the four knobs ARE `#ifndef`-guarded; wrapper defines and `-D` DO take effect. The real failure was geometry (row 3). | Keep wrapper .cu files; they are valid. |

## Confirmed-live levers (for stacking)

| lever | setting | N=24 scored (seed 1789110211) | status |
|---|---|---|---|
| `QSB_TAIL_STAGGER` | 1 (software-half FFGG/FGFG) | scored 4 pairs: F 366.78 vs C2 371.57 = +1.31%; kernel census 4x120 ABBA: F 311.27 (311.7/310.1/304.7/318.6) vs C2 317.55 (316.6/317.1/317.2/319.3) = **+2.02%**, 4/4 pairwise negative | live (committed 257e92c); census says kernel gain > end-to-end gain |
| `QSB_ROOT_LUT_SMEM` | 1 | census queued (job 1046) | unmeasured anywhere |
| `QSB_ROOT_WARP` | 2 | census queued (job 1049) | unmeasured anywhere; probe-backed (B3AP: root-on-0 stretches warps 3/7 by 5.6K cycles, 72% of blocks) |
| `QSB_TAIL_STAGGER` ladder 4 | hardware warpid phasing | census running (job 1044) | in flight |

| `QSB_R_CBANK_TAILS` | 1 | Earlier N=4 ABBA 4x120 (job 1003): A 308.60±1.52 vs B 307.65±2.06 GPU-only = −0.31% | DEAD solo at N=4 (neutral-to-negative); the queued N=24 census (1051) will confirm before removal. |

| `QSB_TAIL_STAGGER` ladder 4 (hardware warpid phasing) | s4 vs C2 | N=24 kernel census 4x120 ABBA: C2 316.18±0.92 vs s4 317.40±0.87 = +0.39%, 4/4 pairwise positive but inside the 1.5% noise band | MARGINAL (not a keeper solo). Warpid phasing ≈ software-half phasing; the s4-vs-s5 control (running) tells whether any of it is phase-diversity at all. |

## Iteration 1: smaller-CTA occupancy probes

- 128-thread tree loop unrolling: 128regs,24KiB,0spills; 1 scored pair
  383.933575 vs382.593006M/s, +0.350%, bothPASS. No adoption signal.
- 128-thread launch bound3 versus4:139regs vs128,24KiB,0spills; 1 scoredpair
  327.718132 vs388.347232M/s, **-15.612%**, bothPASS. DEAD register-headroom trade.
  Fewer residentwarps/independentCTAs overwhelmsextra registerheadroom.
- In-place20KiB pluslaunchbounds5:261.247036 vs382.686202M/s,-31.733%,
  scoredN24bothPASS. Dev96regs20KiB56Bstack4store64loadspills; native-sm89
  120Bstack36store164loadspills. DEAD:fiveCTAoccupancy doesnotpaythespillcost.
- Standalonerootwarp3(noLUT/pre3) on128CTA:374.052082vs379.337741M/s,-1.393%,
  bothscoredPASS,128regs24KiB0spillsdev+native. No adoptingsignal; no four-owner
  sweepjustified. Rootownershipschedulingalone didnotbeatcurrentbest.
- In-place20KiBfourCTA: firstscreen+4.002% did NOT survive3alternatingpairs.
  Control383.984112,379.568171,380.520677mean381.357653; inplace380.686111,
  377.243409,380.057655mean379.329058, **-0.532%**, all6scoredPASS, all3pairs
  negative. DEAD adoption; don'tcitefirstscreenaswin. Footprint20KiBactual,
  butfourCTA128regcapoffersnoextraoccupancy anddestructivebarrier erasesgain.

## Iteration 2 — distinct probes on current fill2 / 128CTA

- Wave-top register shuffles (P8/P4/P2, 3 warp barriers removed): exact tree
  audit PASS; production score376.095140 vs387.523734M/s **-2.949%**, bothPASS.
  128regs24KiB0spills native+local. No adopter; removing barriers is not a win.
- Split SAME128 windows over two64CTAs (not reduced enumeration): scored
  315.913805 vs388.134841M/s **-18.607%**, bothPASS. 128regs12KiB0spills,
  eightCTA occupancy. Doubling roots is expensive. Not adoption candidate.
- Hardware-slot staggering re-opened because software half disappears at128CTA:
  373.934565 vs371.572845M/s **+0.636%**, bothPASS; marginal, not qualified.
- Packing confirmation3pairs +1.190% but drift/mixedpairs; combination packing+
  launch4 362.726365 vs372.931444M/s **-2.736%**, bothPASS. No adoption.

- Compact31-bit root decision table: exact auditPASS; constant380.052120 vs
  392.313841M/s **-3.125%**; shared376.253106 vs386.117278M/s **-2.555%**.
  Both production scoredPASS. Notadopted.32bitdecodecostmayoffsetlatencygain;
  nexttestword+flagbyte(40bits) preservesoldPRMTdecodeandfourCTAoccupancy.

- Split40 fulltable4160B =static24640B: driver3CTAsvsbaseline4CTAs, score
  **-13.645%**(329.513712vs381.581868). SHORT704entries3520Bfits24KiB but
  **-2.719%**(380.746996vs391.389281). Both exact audit+scoredPASS. Packing
  directionclosedforcurrent128CTAunlessmateriallynewdecode/placementidea.

- Launch4 open-lead confirmation completed **+2.240%** vsfill2 over3alternating
  scoredpairsALL6PASS, first2pairs+1.08%/+0.75%, third+5.02%. Notclosed; now
  directpromoted3pairqualification running1428. Keepnotyetofficialwin.

## Iteration 3 producer screen (2026-10-02)

`QSB_FIRST_FLAT_FAST=3`: specialize eight-class mapping and replace eight
scalar first-state stores with two aligned uint4 stores. Digest's native SASS
instruction text is identical to production; native first producer uses 48
registers vs 47. Existing N24/120s scored pair verified both, 368.458047 vs
379.091389 M/s **-2.805%**. Off; not a winner. Do not confuse with prior
first-state buffer packing, which changed the digest address stride.

Launch4 job1428's reported direct-promoted label was WRONG: frozen_ab.sh ignored
BASE env, selected fill2. All six verifyPASS, mean-2.379% againstfill2. The
previous launch4 +2.24% confirmation conflicts with this repeat; the actual
promoted gate must be run with positional control and saved manifest. Script
now enforces prebuilt matching-stamp arms and writes ELF/source hashes. Never
use the 1428 pair as evidence against promoted. Job1450 rebuildingpromoted.

`QSB_OUTER_PAIR=1`: same two SHA256d outer blocks run through the existing
round-interleaved generic paired gate helper instead of solo generic SHA.
Native kernel retains128regs24KiB0spills and has16 more instructions. Existing
N24/120s scored pair verified both, 370.702353 vs383.766384M/s **-3.404%**.
Off; distinct from OUTER_LITK and startup-only ZLAB_PAIRSHA, not adopted.

`QSB_ROOT_LUT_GLOBAL=1`: identical original832 uint64 table, read-only global
cache via __ldg; no packing or shared allocation. Native128regs24KiB0spills,
14512digestinstructions, fiveLDG64 replace rootLDC64. Existing mathematical
tree audit PASS all sizes/partials,98304muls8192roots0errors. N24/120s scored
pair both verified, 370.338997 vs382.694900M/s **-3.229%**. Off; another memory
space does not fix the root bottleneck. The audit itself needs exact diagnostic
arithmetic/scale switches and256-sized arena to test every launch size.

`QSB_ROOT_PARK_B=1`: rootwarp0 ONLY temporarily storesB12u64in dead inverse
rows, restoresbeforeinv16publication; noextraallocation/occupancy. Native
128regs24KiB0spills, +48instructions,+12STS64+12LDS64. N24/120s pair both
verified,370.716988 vs390.027278M/s **-4.951%** against currentlaunch4 control.
Off. True liveness hole doesn't automatically improve root latency.

| `QSB_DEN_CROSS_IDLE` | 1, both factors in non-root warp root window | N24 seed1789110211 120s verified pair: 384.832063 vs launch4 390.754387M/s (-1.516%) | REJECT. Same128regs24KiB0spills; liveness crosses upper tree/root. Eager full relocation instead confirmed+1.46% incremental and+8.73%vs promoted. |

| `QSB_DEN_CROSS_PRE` | 3 (A-only) | N24 seed1789110211 120s verified pair: 394.256431 vs qualified both-factor 406.152518M/s (-2.929%) | REJECT. Native128regs24KiB0spills. Retain productionmode1; B-onlyneutral and A-onlynegative do not support furtherpartialrelocationadoption. |
