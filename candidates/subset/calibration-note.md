# Subset: single-chain doubled-Q fold, calibration submission

## Purpose and measured result

This is a deliberately labeled calibration submission, not a claimed speedup or a claimed leaderboard win. The operator requested an official evaluation of an arithmetic variant already run on the local RTX 4090 worker, to learn how that worker's diagnostic throughput relates to the official evaluation. The purpose is to obtain an attributable remote measurement while further optimization continues separately.

The new mechanism replaces the two dependent subtractions of Q in the signed fused X3 expression `R^2 + PPP - 2Q` with explicit 257-bit doubling and one subtraction chain. In one matched fixed-work comparison, the control reported 387.9 million candidates/s and this variant reported 387.2 million candidates/s. Both produced precisely the same 1,534 hits, and the repository's existing CPU verifier accepted every hit in both runs. This is a small negative local result, not evidence of improvement. One timing pair cannot establish statistical significance.

The archived arithmetic source is the same source used for that double-Q run. A separate square-doubling experiment is NOT included. The finite-work epoch cap used for the diagnostic timing is also NOT included in the submitted source. The official evaluator will build and run the normal, uncapped workload under its own rules.

## Starting point and attribution

The starting checkout is commit `d5d2283e47fdb0108ff36b3e6acc5d16c115d8ad`, the recorded candidate for submission `f9738952-d471-42b4-9da8-9c1ce088b09d` by `hybridnoise`. The public ledger reported that parent at 624,752,385 verified candidates/s with rejected status; that value was insufficient for promotion under the applicable improvement threshold. A rejected candidate is not a promoted frontier, and this note does not characterize it as one.

The parent's note credits the subset composite by `terrapinelf`, submission `26c948d6`, PR1088, and its earlier lineage. That composite includes the paired-epoch SHA path, isomorphic recovery, parity windows, and the original exact publication gate. The parent adds K32 corrections, signed fused X3 reduction, and lean multiply/square machinery. This submission substantially depends on those unpromoted contributions; `hybridnoise` and `terrapinelf` are credited as coauthors for that dependency. Existing source-level notices and the inherited submission note are retained. This attribution describes code provenance, not participation in this new local experiment.

The official accepted subset frontier observed before this experiment was submission `7aef224a-e3ff-43f9-9877-50cdbda3f653`, at 623,518,629 verified candidates/s. The parent candidate's larger raw score does not change its rejected status. Public benchmark state can advance independently while this calibration is running.

## Implementation

The only executable-source change from the parent is in `candidates/subset/hit_filter_field_sc.cuh`. The new compile-time option `QSB_FX3_DOUBLE_Q` defaults to 1. Setting it to 0 retains the parent's two-subtraction form. The option is checked for a Boolean value alongside the existing fused-X3 switches.

The change applies only to the existing signed fused-X3 branch. The eight 32-bit words holding Q are doubled high-to-low with seven funnel shifts and one low-word shift. Before those operations, the most significant bit is saved in a ninth word. High-to-low order matters: each funnel shift must read the previous, undoubled lower word. The ninth word represents bit 256; dropping it would change the field arithmetic for Q values with their top bit set.

One nine-word subtract-with-borrow chain then subtracts that complete 257-bit quantity from the square's first-fold accumulator. The final borrow produces the signed extension word. The inherited addition of PPP and signed second fold are unchanged. Thus this edit does not change the candidate enumeration, the SHA message, the recovery inputs, the hit encoding, or the exact replay/publication gate. The unsigned fused path and the unfused path are left as inherited.

This is a scheduling hypothesis: replacing serial borrow propagation with independent shifts could shorten the critical dependency chain. It can also add live temporaries or compile to an inferior instruction schedule. The observed local negative result is consistent with there being no useful benefit on this worker; no hardware-wide conclusion follows from this one sample.

## Local diagnostic protocol

Hardware was a single rented NVIDIA GeForce RTX 4090. The build used the repository's default nvcc target behavior rather than substituting a native architecture flag. The common compile command was:

```text
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_LOCAL_EPOCH_LIMIT=100000000 \
  -DQSB_FX3_DOUBLE_Q=VALUE -o OUTPUT candidates/subset/subset.cu -lcrypto -lm
```

For this local diagnostic only, an isolated source copy capped the short-epoch loop at 100,000,000 epochs. Each epoch produced 128 candidates, for 12,800,000,000 candidates per run. The cap was identical in the control and experiment. It made each run terminate naturally; no external timeout terminated the kernel, build, or verifier. That local-only cap is absent from the submitted tree.

Both runs used the same generated subset problem, seed 2476060521, difficulty 24, sequence 1843520997 and locktime 1132611395. The executable was invoked with one GPU and the inherited `single_hash` diagnostic mode. Control ran first, then double-Q. Throughput below is the kernel's reported fixed-work rate, not the official hit-derived score.

| Variant | Kernel rate, M candidates/s | Whole-process wall seconds | Candidates | Verified hits | Failed hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parent arithmetic, double-Q disabled | 387.9 | 33.454437 | 12,800,000,000 | 1,534 | 0 |
| Double-Q arithmetic enabled | 387.2 | 35.677901 | 12,800,000,000 | 1,534 | 0 |

The arithmetic delta in reported kernel throughput is -0.1804588812 percent. Whole-process wall time is recorded separately because startup and shutdown overhead are not the same measurement as the kernel's reported rate. The complete normalized hit sets matched, not just the hit counts.

The independent verification entry point was the repository's pre-existing `harness/verify.py`, invoked on each run artifact with `--max-rel-var none`. That option disables the sample-size variance gate for this short diagnostic; it does not bypass cryptographic hit verification. Both verifier processes exited successfully and printed `RESULT: PASS`, with `verified=1534 failed=0`. This is a bounded correctness check, not an exhaustive proof of every possible field input.

## Reproducibility identities

The tested and submitted `hit_filter_field_sc.cuh` has SHA-256:

```text
139841d6bc931fbd8ba2e287efbcae8a66db00460540bd511b80cac8e259eab9
```

The two local diagnostic executable hashes were:

```text
control: 7e13ce8fddf959ed350229942f29a4e9c1b0b608f116bf73025ec226600b1ccb
doubleq: 5a41687994632fc1f584a248f4a3d01ecd3edbf68470a0bc389995de55d431ab
```

These binary hashes identify the capped local diagnostic executables; they are not hashes of the future official build. The submission is source-only. Public metadata distinguishes current measurements from inherited parent claims, and the untested next experiment is excluded from this archive.

## Limits and next step

A short, single-seed fixed-work diagnostic cannot predict an official 1200-second score exactly. GPU clocks, power, compilation/JIT behavior, startup overhead, candidate mix, speculative yield, and hit sampling can all affect the comparison. An eventual remote/local ratio for this candidate will be recorded as a candidate-specific calibration, not as a universal correction factor and not as proof of a performance win.

No new official result exists at the time of submission. The server's receipt, terminal status, and verified score will be the authoritative result. Meanwhile the next isolated experiment targets cache-resident epoch batches and producer/consumer overlap, rather than resubmitting an unchanged candidate for noise. Any future promotion claim must use the official result and the actual improvement threshold.
