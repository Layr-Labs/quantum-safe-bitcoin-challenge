# A/B measurement noise on this rig (RTX 3090, 9950X3D host)

Verified locally on the promoted subset build (subset_local_A, sha f2004c8a), 6 interleaved
120 s runs via `candidates/subset/tests/gpu_epochs/ab.sh` (proc_run id 1001, 2026-10-01):

| arm | GPU M/s | CPU M/s | total M/s |
|-----|---------|---------|-----------|
| r1_A | 311.3 | 47.2 | 358.5 |
| r1_B | 307.9 | 43.6 | 351.5 |
| r2_A | 306.6 | 41.6 | 348.2 |
| r2_B | 306.5 | 36.3 | 342.8 |
| r3_A | 302.8 | 34.7 | 337.4 |
| r3_B | 300.9 | 30.7 | 331.6 |

Findings, both reproducible in the table:

1. **Thermal drift dominates random noise.** Total throughput falls monotonically
   358.5 -> 331.6 M/s (-7.5%) across six back-to-back runs. The CPU co-grind arm falls
   harder (47.2 -> 30.7, -35%) than the GPU arm (311 -> 301, -3.4%), which points at host
   clocks/thermals rather than the device. Random A/A scatter within one thermal state is
   only ~1.2% (GPU arm). **Any A/B that always runs A first is biased by roughly +3-5%
   toward A** — the first arm of every pair is cooler.

2. **Consequence for past A/Bs:** any historical before/after measured as "run new, then
   run old" (or vice versa, consistently) on this machine inherits the drift sign. Only
   order-alternating (ABBA) designs with cool-down gaps measure a true delta.

3. **Use `ab.sh` rounds>=4**, which alternates A,B,B,A,A,B,B,A with a 30 s cool-down
   between arms, and compare arm means; treat |delta| < 1.5% as noise.

The rig-level fix (proc_run for long jobs, never shell backgrounds that die on tool-call
return) is already in ab.sh's header comment.

## Experiment 1: QSB_R_CBANK_TAILS=1 (2026-10-01, ABBA 4x120s + 30s cooldown)
- A = promoted-default build (f2004c8a), B = -DQSB_R_CBANK_TAILS=1 (cd604cac)
- GPU-only: A 308.60 ± 1.52 M/s, B 307.65 ± 2.06 M/s → **B slower by 0.31% (neutral-to-negative)**
- Total (GPU+CPU co-grind): A 355.98, B 349.00 (CPU co-grind is too noisy to score on: sd ~16%)
- VERDICT: dead end on sm_86. The constant-bank tail-R register saving (measured on the 4090 line)
  does not translate into throughput on this rig. Do not submit.

## Experiment 2: QSB_TAIL_STAGGER=1 (+PARK128=0, forced) (2026-10-01, ABBA 4x120s)
- A = promoted default, C = -DQSB_TAIL_STAGGER=1 -DQSB_PARK128=0 (9e13cb12)
- r1: A 307.5 / C 310.8 (+1.07%) — split finish/gate with sub-partition phase diversity
- (in flight; full verdict pending)
