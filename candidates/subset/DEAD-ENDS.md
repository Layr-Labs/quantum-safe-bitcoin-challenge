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

## Iteration6 sum-numerator basis / deferred second slope

Carry(N1,N1+N2,ZZ), materialize b=sum-a after parity0 in first-slope storage.
Distinct from centered-square x2-late. Actual existing exact pair PASS on both
arms: candidate394.619697 vs current405.984212M/s, -2.799250%. Native128regs,
24576Bshared0spills unchanged ceiling. One screen, not confirmed regression;
no production adoption. QSB_K2S_SUM_BASIS default0, mismatched carrier protected.

## Iteration6 packing8 + doubled launch on centered tail

Capacity-enabling combination, not packing-only retest: first-buffer bytes
remain2GiB but descriptors/maps/groups add328.699MiB. Ranked known major
requests24446.162MiB leave129.838MiB before context; local runtime gap413.838MiB
is not ranked proof but exceeds headroom. Existing paired exactPASS:
B393.430942 vs current399.694616M/s=-1.567115%, one negative screen.
DefaultOFF; no follow-up confirmation. Packed8atcurrentlaunch remains a
possible memoryrepair, not an established speedup, if watcher shows a blocker.

## Iteration7 signed centered-square offset fusion

Default-off QSB_K2S_CENTER_FUSED=1 inserts both cc2/p1 subtractions before
square's second fold, with signed high propagation copied from inherited X3.
Native128regs24KiB0spills,14512digestinstructions (-8vs14520qualified,
same CUDA12.8 cuobjdump census; old14503 count was inconsistent).
Existing N24/120s fixedseed paired benchmark: B400.125573 vs current
A408.239940M/s, -1.987647%; B5800/5800 A5898/5898 verifiedPASS.
One negative screen, not a confirmed regression. OFF/no carrier update.
Signed-word tracking costs8instructions versusunsignedprebias continuation;
even its measured8instruction reduction didnot yield a positive score screen.
The default native image was rebuilt byte-identical to qualified9c9aab2...

## Iteration7 unsigned prebias continuation

The signed-fold cost justified a distinct emitter change: host uploads 3p-c²
as9-word QSB_U2R_C (originalc plus5offsetwords). Unsigned square first fold adds
thatpositiveoffset, subtractsp1, then uses originalunsignedsecondfold. Native
128regs24KiB0spills; same-tool digest14504vsdefault14520(-16), nonNOP14495vs
14506(-11). ExistingN24/120spairedscore399.423982vs407.542940M/s=-1.992172%,
B5803/5803 A5898/5898verifiedPASS. One screen, not confirmed regression. OFF.
Reduced static instructions, including3IMAD and4IADD3, didnot yield a speedup;
both fusionforms screennegative, so no confirmation expenditure justified.
Known unsignedtopword-wrap and inheritedrarecarry hazards remain speculative.
Do not cite purefield algebra or successful-hit verification as all-domain proof.

## Iteration9 address-only final GLV record lookahead

QSB_S3_FINAL_PREFETCH1: same final gather index, two32Bsector cache requests,
no coordinates carried. ExistingN24/120sexactpairPASSA410.106447 vs B395.756486
M/s,-3.499082%,B5726/5726verified. Native128regs24KiB0spills,2CCTL.E.PF2,
+32staticinstructions. ptxas interleaves hintsinto previousaddition;sourceorder
not guaranteednativeissueorder. One negative screen,notconfirmedregression.
OFF/noadoption; cachehintpresenceandzero-spillsarenotspeedproof.

## Iteration9 scoped SHA ordinary-gate extension

QSB_SHA_LEA_GATE1(onSHA1scopedcontrol): existingN24/120sexactbothPASS, A401.315271
vsB359.765211M/s,-10.353471%,B5206/5206hits. Native14256->14128(-128)but128reg
24KiB0spills unchanged. One thermallyinterleavednegative screen,notconfirmed
regression. OFF/norepeat; GPUHash-onlypositivelead remains separatefamily.

## Iteration9 scoped GPUHash rotate-add qualification kills initial lead

InitialcurrentvsSHA1+1.322696% exactsinglepairdidNOTqualifyagainstNEWleader.
Directnewpromoted3alternatingpairsALL6PASS: A370.998396 vsB373.782204M/s,
+0.750356%<4%bar;lastpairnegative. Native-264instructionsand128reg24KiB0spill
notproofwin. Currentproductionseparatecontemporaneousgate+6.464839%all6PASS.
Jobsserializebutinterleave,thermal/yieldcaveat: don'tclaimconfirmedregression,
justqualificationfailure. Default0;stagedcarrier/integrationPASSnotadoption.

## Iteration10 — same-stride row-half pipeline interleaving: capacity PASS, speed screen negative

New host-only QSB_FIRST_SLOT_INTERLEAVE1 shares one512B/epoch row: slot0first8
classes and slot1second8. NOT FIRST_PACK8 device-pitch change; NOT producer
scratch overlay. Frozen qualified-source paired existing N24/120s test:
355.117402M/s vs335.630321M/s **-5.487504%**, both exact published-hit PASS
(5167/5167 and4880/4880). Single negative pair; not confirmed regression.
Do not repeat qualification based on memory or native instruction identity.
NVML12244->11220MiB, **1024MiB less**,56/60constant samples during same pair.
Local device payload byte-identical. Runtimeclass/geometry guarded and keeps
full checker comparison path; no raw checker output retained by wrapper.
Default0. Ranked headroom improves accounting-only458.537->1482.537MiB; no
ranked-fit proof. bat414fe58 officialFAIL37007650774 after0.6145s/zero work,
exact stderr unavailable. Evidence lab/iter10-findings.json.

## Iteration11 runtime capacity budget (not the row-half layout)
DefaultOFF `QSB_LAUNCH_BUDGET`: after table build size coherent two-slot launches
at2M or1M with1GiBVRAMreserve. Forced1M on localGLV12 scored374.997364 vs frozen
fullcap403.627048M/s, **-7.093103%**, bothPASS5400 and5846 verifiedhits. Reject
singlepair screen; no3pairfollowup (not confirmedregression). Saves1204.080811MiB
requestedslots, observed11040MiB vs prior12244MiBfullcap. Local and native device
payload byte-identical. Official low-memory auto-selection fit not proved; raw
startupstderr missing. Avoid repeat cap-only memorytradeoff; keepingfull2Mcap
with smaller devicefirstpitch is a distinct unverified lead.

## Iteration12 legacy dual-carry spelling / isolated MAC

Realpm9 emitter QSB_YP_LEGACY_DC1:389.775892->355.265565M/s(-8.853890%),
mask3odd+top:395.751801->352.340156(-10.969412%); allfourN24scorePASS.
Native14520->14552/14560, SEL -3/-6 outweighed moves/predicateconversion.
Mask1 adds8Bstack+LDL/STL though0spillreport. BothOFF; singlepair negatives,
notconfirmedregressions. Leader8sitefusionnotportablewithoutitsMACschedule.

ImportedpublicMAC2/DC1 behind QSB_YP_MAC_PORT0, literal977 fold unchanged:
sm89 -56instructions,128regs24KiB0stack/spills. ExistingN24pair358.775324->
360.086933M/s(+0.365579%),5216/5244hitsallPASS. Notabove noisefloor/adoption
bar; remainOFF, notimprovement. No cheap carry-fusion win on centered-square
carrier. Evidence lab/iter12-findings.json; productionimagebyte-identical.

## Iteration12 post-submission exact upper-wave live-lane masking

QSB_TREE_LIVE_MASK1 trimsdiscardedproducts onwaveA8/B20/C18/D17/inv16
withoutchangingwarpcollectives/sharedrows/rootnormalization. ExistingN24
120spair386.343611->385.370820M/s(-0.251794%),5597/5580allPASS. No
positivelead; no repeatgate, OFF. Native+32instructions,5BRA/BSSY/BSYNC
predicateoverhead,128regs24KiB0stack/spills. Flat singlepairnotregressionproof.
Defaultnativebyteidenticalsubmittedcompactpitcha724399a. Leader's maskedwaves
notanindependentwin onour128CTAparticularschedule. Nextdo notsweeplanes.
