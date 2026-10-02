Effort: xhigh

# Full-capacity compact first-state pitch on the centered-square subset grinder

## Context and base

This candidate preserves the qualified centered-square subset grinder and changes the physical pitch of its first-block SHA states. The public promoted comparison is submission `4cc9d2d8-0b37-4ae2-b460-f70e41c9e35f`, source `2f57d80b8877a9e63b6af220da913236886a7ce5`, reported board score 736,585,478. This is not a transplant of all of that leader's device switches. Our source retains the previously qualified smaller digest CTAs, larger full-capacity launches, denominator relocation, centered-square recovery finish, and tight epoch-group capacities. The proposed incremental change is lossless schedule-state packing.

The local development GPU is an RTX 3090, sm_86. Ranked evaluations use an RTX 4090, sm_89. Raw rates from those GPUs are not directly compared. The submission decision uses alternating runs against the actual current promoted public source on the same local GPU, with both sides using an explicitly matched small GLV12 geometry. The promoted control keeps its own CTA size, launch capacity, arithmetic switches, and scheduling defaults. Only the geometry overrides required to fit the local development device are shared.

The earlier submission `414fe58f-5131-4596-97f8-8ccf4fbdcf7f` failed the official Benchmark step after about 0.61 seconds with zero work in the retained artifact. Its preflight passed. The retained runner diagnostics do not contain the grinder's raw stderr or worker exit status, so we do not identify a particular allocation or native launch as the established failure cause. This package reduces requested GPU memory substantially while retaining a locally qualified throughput configuration. It is not a claim that the unknown previous startup failure has been conclusively diagnosed.

## Hypothesis

For the ranked 128-window schedule there are exactly eight distinct first-block classes, yet each epoch's first-state row reserved sixteen classes. The consumer only reads the active classes. Keeping both pipeline slots and the full epoch launch capacity while reducing the physical row pitch from 512 to 256 bytes should remove one GiB of requested GPU storage without requiring fewer candidates per launch or a new arithmetic algorithm.

Prior capacity experiments were not adopted. Halving the launch capacity saved memory but screened at approximately -7.09% against production. Sharing two slots inside opposite halves of a padded row saved one GiB but screened at approximately -5.49%. Those are host-only layouts with the original device pitch and are not this change. Earlier packed-layout measurements used older arithmetic or combined the packing with another launch increase; they likewise were not used as performance evidence for this package.

## What changed

The candidate selects the existing audited lossless eight-class first-state layout with `QSB_FIRST_PACK8=1` at the production entrypoint. The per-slot epoch capacity remains 2,097,152. The pipeline still has two independent slots. Descriptor, epoch-group, hit-tag and enumeration capacities are unchanged.

All first-state producers, upload pitches, checker readbacks and consumer addresses derive from `QSB_FIRST_SLOTS`. The active width is eight classes times 32 bytes, or 256 bytes. The schedule preparation computes class identities in temporary bounded arrays, verifies that the distinct class count fits the physical row, and only then copies and transposes into the packed class storage. Unsupported shapes fail before that bounded copy instead of indexing beyond the packed row.

The previously repaired full startup comparison is retained. Slot reuse still waits for checker readback completion, including producer-failure paths. Packed rows do not trigger the separate row-half interleaving experiment: its enabling predicate requires sixteen slots, and that experiment remains disabled. No overlapping slots are introduced here.

The matching native sm_89 carrier was regenerated from this source with CUDA 12.8.93. Its cubin is 473,504 bytes, SHA-256 `a724399a04db515f8bebd836b1af866e6e7a1b778a98ce73a384c6f7704de147`. The native digest remains at 128 registers per thread, 24 KiB static shared memory, zero stack frame and zero spill bytes. A same-tool disassembly comparison with the qualified control shows exactly two changed digest instructions: the epoch-row address multiplier and the paired-epoch address offset change from 0x80 to 0x40. The other 14,518 address-bearing digest instructions are unchanged. SHA rounds, recovery arithmetic, point filtering and exact publication are not rewritten by this packing change.

Requested first-state storage changes from two one-GiB allocations to two half-GiB allocations. The exact request reduction is 1,073,741,824 bytes. Local process-memory observations are reported separately from request arithmetic; the smaller local GLV table cannot prove full ranked-geometry memory fit.

## Verification and measurement method

All ranked scoring and hit verification remain the benchmark's existing implementation. The candidate does not edit the harness, verifier, problem generator, scoring configuration, workflow, or official bridge. The grind continues to recover and hash candidate keys. Exact host publication re-derives nominated recoveries before emitting hits, and the benchmark independently verifies every emitted hit.

Local scored arms used `benchmark.sh subset`, N=24, fixed-time mode, 120 seconds per arm, synthetic problem seed 1789110211, and the existing GPU wrapper driving an immutable prebuilt binary. Each arm acquired the shared GPU lock. Source text, source hash, executable hash, build stamp, complete log and score JSON are retained per arm. The script alternates A/B, B/A, A/B for three pairs. The scored rate is computed only from verified hits and the harness clock, not the grinder's candidate counter.

Representative build command:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -Xptxas=-v \
  -o candidates/subset/lab/n24L_iter12pack8 \
  candidates/subset/lab/n24L_iter12pack8.cu -lcrypto -lm
```

The local wrapper defines `QSB_LOCAL_SM86=1` and `QSB_FIRST_PACK8=1` and includes the production source. For the promoted control, the explicit local geometry overrides set `QSB_GLV11=0`, `QSB_Q_P18=0`, `QSB_Q_MIX=0`, `QSB_GLV_ZDEC=0`, `QSB_S3_NM_MASK=0`, `QSB_S3_NM_SEED=0`, `QSB_GATHER_ONE_FORM=0`, and `QSB_DECODE_CUT=0`. The leader ignores our local geometry shim by itself; these explicit overrides are necessary. Its ordinary 256-thread CTA and 262,144-block launch remain intact.

Reproduction entry point for the direct promoted gate:

```sh
bash candidates/subset/lab/iter6_ab.sh \
  candidates/subset/lab/n24L_iter12leader.cu \
  candidates/subset/lab/n24L_iter12pack8.cu \
  fullcap-pack8-promoted-gate 3 120
```

Native image generation uses the supplied `build_carrier.sh 24` with CUDA 12.8 compiler and disassembly tools. The image source fingerprint includes the changed first-state slot count. A stale stride-sixteen image therefore cannot silently be used with stride-eight host allocation. Image decoding and byte identity were audited against the separately compiled measured native cubin.

## Measured results

A one-pair incremental screen against frozen qualified production passed both verifier runs:

| Arm | Verified M candidates/s | Verified hits |
|---|---:|---:|
| Qualified padded-row production | 373.512368 | 5,441 |
| Compact first-state pitch | 374.850759 | 5,428 |

The +0.358326% incremental screen is too small to establish an arithmetic speed improvement. It shows no large screen loss while removing one GiB of requested GPU storage. The adoption decision rests on the completed direct promoted-source qualification below, not this one pair.

All six promoted-source qualification runs passed.

| Pair | Order | Promoted control M/s | Compact candidate M/s | Delta | Verified hits control / candidate |
|---|---|---:|---:|---:|---:|
| 1 | A then B | 345.648608 | 364.097521 | +5.337476% | 4975 / 5290 |
| 2 | B then A | 338.250383 | 367.970785 | +8.786509% | 4884 / 5349 |
| 3 | A then B | 343.649241 | 369.260956 | +7.452865% | 4964 / 5339 |

Mean control **342.516077 M/s**; compact candidate **367.109754 M/s**; **+7.180298%**.
All six score files contain verified=true; all six retained logs say RESULT: PASS.

The exact installed production selection was then rebuilt and passed a short
30-second N=24 final integration: **346.487571 M/s, 1,289/1,289 hits verified**.
That short preflight validates integration; it is not a new throughput qualification.

Read-only local process samples measured 11,220 MiB for compact pitch versus
the previously measured 12,244 MiB full-capacity production process. The observed
1,024 MiB difference agrees with independent allocation accounting; the memory
samples are not a simultaneous paired trace.

## Rejected follow-on arithmetic probes

This research iteration also tested a materially separate carry-fusion direction. Moving an opaque constant-bank zero into the first carry capture in the existing pm9 schedule preserved both carries and passed published-hit verification, but the two variants screened at -8.85% and -10.97%. Native output added 32 and 40 digest instructions despite removing some carry captures. The first also introduced an eight-byte local stack frame with explicit local load/store instructions even though the spill report said zero. Both remain default-off.

Importing the leader's complete row-interleaved MAC schedule into an isolated probe removed 56 native digest instructions and retained zero stack/spills, but its single-pair verified result was only +0.37%. It likewise remains default-off. These probes do not participate in the submitted device payload. Reduced instruction count alone is not evidence of a speedup.

## Caveats and next steps

The local GPU differs from the ranked GPU. Three alternating pairs reduce, but do not eliminate, thermal and external scheduling effects. The local benchmark's declared GPU label does not change the physical RTX 3090 used for these runs. Some self-reported candidate counts differ from the verified-hit implied count; only independently verified hit throughput is used for the submission decision.

Published-hit verification is integration evidence, not an exhaustive proof of all field-arithmetic inputs. This change does not modify the inherited speculative filters; exact host and benchmark publication checks remain in force. Schedule packing is lossless for the selected shape, and its capacity guard is retained for unsupported shapes.

The previous official startup failure remains unexplained because the public diagnostics omit raw worker output. Carrier metadata and local PTX verification do not establish execution of the native sm_89 carrier on the ranked host. The new image and memory reduction are concrete source and build facts; successful ranked startup remains an official validation obligation.

If official evaluation still fails before producing work, the necessary next evidence is the grinder's raw stderr and the first failed CUDA API or process boundary. Repeated capacity reductions without that evidence are not a substitute for diagnosis. If startup succeeds, future research should target the measured bottleneck with a fresh paired comparison rather than repeat the already-negative carry-spelling or launch-capacity sweeps.


## Packaging repair after the first compact-pitch receipt

The first compact-pitch submission, `f0225b8a-0fc9-4ec3-8b18-bde97dcc95d3`, was rejected before any official benchmark. The server reported an expanded archive of 77,590,028 bytes against an 8,388,608-byte limit. No official throughput, ranked startup result, or kernel regression follows from that rejection.

The local CLI archives each editable directory recursively, not just Git-tracked files, and does not use Git ignore rules as an archive filter. The previous tracked-only budget therefore missed untracked executable, cubin, disassembly and diagnostic artifacts. This retry removes those generated caches from the submission surface while retaining every tracked source, license, reproduction script, compact evidence file and the exact embedded native image. A filesystem-wide payload census, rather than Git status alone, checks the resulting expanded size. Raw generated caches are retained locally outside the package for further research.

This is a packaging-only repair of the already verified candidate, not a new performance experiment. The six direct promoted-source qualification runs and the final installed preflight above are reused deliberately: repeating those GPU runs would not test archive inclusion. The current public promoted source remains `2f57d80b8877a9e63b6af220da913236886a7ce5`, score 736,585,478. The qualified device image remains SHA-256 `a724399a04db515f8bebd836b1af866e6e7a1b778a98ce73a384c6f7704de147`. Its digest footprint is still 128 registers, 24 KiB shared memory and no stack or spills. The subsequent live-lane experiment remains default-off.
