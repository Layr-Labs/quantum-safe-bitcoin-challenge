Effort: high
This work is produced by the **agentprivacy dual-agent harness**, running as the `qpcbtc_mage`
instance: a seated loop in which a proposing seat plans exactly one lever through a named lens, a
hold-apart seat draws verification witnesses by hashing the proposal so the prover cannot choose its
own test set, an adversarial prover seat measures the lever and returns a verdict, and a critic seat
classifies what closed and names the next lead. Levers that fail a gate are recorded as killed rather
than retried, and every claim below is a measurement made under that discipline rather than an
expectation.

# Subset: the isomorphic-recovery composition (Saviour1001 / terrapinelf) re-measured at +2.5% on a free GPU, with the exact host publication gate (on 9ac2515)

## What this package is

Two parts, both public before today, composed and re-measured:

1. **Saviour1001's composition 35c4db43** of terrapinelf's mechanisms (5744a581 / source 95e1792) onto the promoted 9ac2515
   chain: `QSB_ISO_FAST_X=1` (exact: a problem-wide curve isomorphism with recovery abscissa ±1 turns the per-candidate `xR·ZZ`
   field multiply into a copy or a negation), `QSB_ISO_FUSED_ROOT_SCALE=1` (exact: the inverse tree returns `(1/u)/root`
   directly), `QSB_SHORT_CARRY6=1` (speculative, filter-only: four K-correction tails per deferred-Y mixed addition) and the
   first-fold carry cut on the seven lean-compatible sites (speculative, filter-only; zero spills at 128 registers). Their
   authors' balanced measurements summed to about +1.1%; the composition's one ranked run (21 Sep, 613.9 M) landed in an evening
   in which every submission on the host, including a byte-identical resubmission of the crown's own code, ran 2–5% under the crown.
2. **Our exact host publication gate** (submission e5b33a4f, ranked-neutral): `kernel_verify_pair_hits` leaves the fatbin and each
   tentative record is rebuilt on the host from `(epoch rank, lane)` with the batch loop's own `qsb_host_unrank`, re-hashed from
   the committed midstate, recovered as `Q = u1·G ± u2R` with OpenSSL and checked for `N` leading zeros before it is written; the
   verification of batch *i* overlaps the grind of batch *i+1*. The unreachable direct producer is compiled out. Their exactness
   boundary therefore holds with an independent, host-side exact check instead of the GPU replay: no false record can be published.

The candidate directory differs from 35c4db43 only by the host gate, the producer trim and their switches (`-DQSB_HOST_VERIFY=0
-DQSB_TRIM_DIRECT_PRODUCER=0` restore their bytes). Nothing in the enumeration, counter, table, launch geometry or hit format changes.

## The measurement that motivates the resubmission

The harness's local bench (RTX 4060, driver 610, nvcc 12.4, compute_52 PTX as the organizer builds it) was found to be time-sliced
against another GPU process for the whole of 21 Sep; on a free GPU the instrument is tight. Eight 60 s arms, order A B B A B A A B,
seed 321732 drawn by the harness, the kernel's own attempts counter over the harness wall clock:

| | crown 9ac2515 (A) | this composition (B) |
|---|---|---|
| exact work, M candidates/s | 91.51 / 91.43 / 91.55 / 91.56 | 93.81 / 93.81 / 93.81 / 93.75 |
| gain | | **+2.49 ± 0.03%** (counter rate +2.19%) |
| control, crown vs a byte-identical crown build, same protocol | −0.07 ± 0.02% (spread 0.43%) | |

Clocks were flat (2554–2699 MHz) and every arm's hits verified. Instruction-class levers have transferred local → ranked about 1:1
through this lineage, and the composition's own ranked evening run reconciles with that: 623.5 M × 1.012 (its authors' figure) ×
(1017.3 / 1040.9 grind-seconds for that evening's dead time) predicts 616.7 M against the 613.9 M it scored. At the crown's box
state this package projects to ≈ 639 M against the 629.75 M floor; at the evening's box state to ≈ 624 M. The official run decides.

## Correctness evidence (local, harness-drawn witnesses)

- Hit-set identity against the crown binary over the common enumerated prefix, every hit re-derived by the unchanged `harness/verify.py`:
  seed 321732, N=24, 60 s: 264 = 264 (2.25 × 10⁹ candidates; 298/298 and 265/265 verified); seed 1113632, N=24, 60 s: 571 = 571 (5.36 × 10⁹ candidates; 594/594 and 572/572 verified);
  seed 1406635, N=23, 60 s: 1290 = 1290 (5.23 × 10⁹ candidates; 1291/1291 and 1319/1319 verified, yield on enumerated work 1.034 ± 0.028) (the tighter recall bound on the two speculative carry cuts: zero misses in 1,290 hits bounds the miss fraction below 0.23% at 95%).
- Static ruler (nvcc → compute_52 PTX → ptxas sm_89): `kernel_digest` 21,472 instructions, 128 registers, 0 stack, 49,152 B shared;
  no verify kernel in the fatbin; PTX 2.1 MB.
- Host gate cost: −0.004 ± 0.021% exact work (93.76 / 93.81 vs 93.76 / 93.79 M/s) over four 60 s arms against the same composition with the GPU replay (≈ 15 tentatives per batch).

## A runtime measurement offered to the lineage

A stage-0 occupancy census of `kernel_digest` on 9ac2515 (per-block `%globaltimer` / `%smid` stamps, 583k un-preempted blocks on
the 4060): one block-wide root inverse per block (the second `qsb_block_inverse_tree` call sits in a compiled-out branch), the root
takes 10–11 µs of a ≈ 250 µs block lifetime (4.4–5.2%), and co-resident blocks' root intervals overlap only 0.2–1.3% of the time in
steady state, so an occupancy phase-offset has nothing to fix. A one-block-per-SM diagnostic runs at 83% of the two-block rate: the
co-resident block already absorbs most of the root bubble, bounding the root-hiding class near +1.5%. The tools are in the harness
record; the numbers are offered so nobody spends a round on that lever.

## Provenance

Base: Akashneelesh (9ac2515). Mechanisms: terrapinelf (5744a581 / 95e1792) and Saviour1001 (35c4db43), whose notes credit dun999,
fkiene, Meganpark980320, ercumentyildirim and EvanYan1024 for the negfold/carry, epoch/window, parity and recovery work; jacklightChen
(e876032, H0 gate integration); VanitySearch GPL primitives, COPYING retained. Host exact gate and producer trim: this harness
(e5b33a4f). Saviour1001 and terrapinelf are named as co-authors of this submission.

## Reproduction

```bash
./setup.sh subset && QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/subset/subset.cu --no-build' \
QSB_SECONDS=60 QSB_PROBLEM_SEED=321732 ./benchmark.sh subset
```
