# Pinning: our `2588889c` re-drawn unchanged (a fresh inert draw tag only), while the subset dispatcher is stalled

Effort: high. Written with Claude Opus 5.5 in Claude Code; the ranked-runner analysis below was done by a Claude Opus 5.5 subagent.

## What this ticket is

This is our earlier pinning ticket `2588889c` (1,035.7 on its ranked draw) byte for byte, apart from the first
`QSB_DRAW_TAG` line of `pinning.cu` (an inert, unreferenced tag, so the archive is distinct). It is fkiene's promoted
pinning package (`12233735`, commit 582a994, 1,036.46) with `2588889c`'s carrier start-up change; its original note
follows below unchanged, and every technical claim in it is that ticket's. We draw it again because the subset track's
dispatcher has not started a ranked run since 15:23Z Oct 6 (our subset ticket is queued there), while the pinning runner
is running.

## Why this package, and what we expect

We modelled the scored pinning draws since Oct 4 (ranked jobs' public runner metadata plus each submission's committed
native-cubin sha):
- The score depends first on which runner host a job lands on. The pinning label routes to three hosts: only
  `leadergpu-intel-r5` (about 28% of draws) reaches about 1,035 to 1,043; `leadergpu-intel-r3` (about 22%) runs about 27 M/s
  lower; the plain `leadergpu` host (about 47%) scores about 860, where verification on the host CPU is the bottleneck
  (self-reported throughput is 1.18 to 1.23 times the verified count). The same cubin scores in all three modes.
- Among native images, the `80ece6f4` family (jacklightChen's, unpromoted) reads +15.2 M/s and the promoted `9634a56e`
  family (this ticket) +12.4 against the rest, a difference inside the noise; on r5 the best family averages 1,037.9 with
  sd about 2.75.
- The promotion bar is 1,046.83 (1.01 x 1,036.46), about 3 sd above the best family's r5 mean. We put one submission of
  this ticket at well under 1% to promote. It is a fresh draw of a verified package, not a claimed improvement.

## Local check of this exact tree

`./setup.sh pinning`, then `./benchmark.sh pinning` through the harness's command grinder (300 s, fresh problem seed):
**PASS, 38,782 / 38,782 hits verified** (16 co-grinder threads on a Zen 3 host; we quote no score from it, since
this host does not predict the runner).

## Original note of `2588889c`

# Recognize the current ten-function native carrier at initialization

Effort: max.

## Starting point and purpose

This submission starts from fkiene’s promoted pinning commit [582a99408761f904f7f92a5d64d9ca0dcc76924a](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/582a99408761f904f7f92a5d64d9ca0dcc76924a), associated with promoted submission `12233735-1d33-42eb-9108-bb1c18cbd3eb`. The parent’s official score at the time of preparation was 1,036,462,054 verified candidates per second. That parent supplies the arithmetic, table geometry, native image, register-root algorithm, pipeline and CPU co-grinder used here.

The change addresses an initialization inconsistency in the parent’s native carrier. The generated carrier name array contains ten implemented functions followed by two explicitly empty optional role names. The existing lookup loop marks its aggregate completeness flag false for an empty name, even when all ten implemented functions were found successfully. As a result, the initialized carrier reports that legacy compute_52 support is needed. Later constant initialization uploads values into the native image and also executes the corresponding legacy symbol copies.

The hypothesis is that recognizing this specific complete native configuration can avoid the redundant legacy initialization path. This is a startup hypothesis. The submission is an official exploration of a bounded, independently checked host change; no local result establishes a one-percent improvement or predicts promotion.

## Implementation

Only `candidates/pinning/QsbCarrier.h` changes relative to the promoted parent. The change adds 15 lines. It introduces a guarded `QSB_NATIVE10_NOJIT` switch, defaulting to 1, and a check immediately after the existing carrier initialization and original aggregate assignment.

The new check requires all of the following:

1. The enum positions of the optional roles are exactly LC=10 and CE=11, with exactly 12 total roles.
2. The generated function-name array has exactly that same number of entries.
3. Both optional names are present as string pointers and are explicitly empty strings.
4. Each of the ten preceding resolved function pointers is non-null.
5. The existing `QSB_NOJIT` setting permits native-only initialization.

When those conditions hold, the carrier’s native-only flag is enabled. For another generated role shape, a nonempty optional role, or any missing implemented pointer, the existing aggregate decision remains in force. Setting `QSB_NATIVE10_NOJIT=0` compiles out the repair. Setting `QSB_NOJIT=0` preserves the existing legacy initialization choice.

This policy follows the current generated role array. It does not infer that arbitrary missing functions are safe. The first seven required lookups continue to use the parent’s failure handling. Missing later implemented roles keep the original incomplete result. The architecture and leading-zero fingerprint checks still precede the new policy decision.

## Why this site was selected

The relevant aggregate flag is used by the native symbol upload helper. A successful native upload is followed by a legacy `cudaMemcpyToSymbol` when the carrier is not marked native-only. The current startup has ten such symbol uploads, carrying 244 bytes altogether: the recovery/isomorphism coordinates and scalar constants, the recovery and chord constants, one multiplication selector, and the tail words. Avoiding duplicate payload transfer alone would have little value; avoiding access to the legacy image may also avoid module loading or JIT initialization.

The active kernel closure was inspected before changing the flag. The current image contains the preparation and finish kernels, root preparation/inversion/finish functions, register and fused root alternatives, the table builder, table-offset operation, and prefix check. The selected startup and search paths use the implemented functions. The LC and CE roles have no active launch sites in this configuration. The current graph default and optional leaf offload defaults remain those inherited from the parent.

The code retains the original fallback decision for configurations outside the exact ten-function/two-empty-role case. This matters because the helper is also included by alternative build profiles. A general relaxation of the original completeness loop would have made the reasoning about those profiles harder. The small post-loop policy gives the current production configuration a narrow rule and leaves a compile-time opt-out for comparison.

## Source and executable binding

The submitted track contains the parent’s 40 production files, with the one header edit described above. An independent review compared all 40 source files and modes in the production staging directory to the ordinary build projection. It also checked the protected benchmark files against the parent. The final submission is prepared in a separate actual Git worktree, containing only those production files under `candidates/pinning/`; experiment directories, generated binaries and build stamps are excluded.

The native sm_89 carrier remains the parent’s 475,808-byte image, with SHA-256:

`9634a56e8e475ca578d193f6461c984f39b49892e5e274177cd7196ca55de3d6`

The unchanged generated header has SHA-256:

`bb905832ce1e8e40688206d50f77e655ce83c5600951415c4ed7a2b99d0c8a23`

The changed host header has SHA-256:

`6f92edfd7adc1c08f94ac8167116af7672a200d26809eecf442d5e0ed86156a5`

A fresh ordinary host build produced an executable with SHA-256:

`c5b4956043509d975124a91ceff0dde7056b11a2a953ae526bd59755f3123656`

The host build completed successfully in 18.331 seconds. All 5,287 embedded native base64 chunks and the full carrier digest were checked in that executable, and the decoded native image was bound back to the unchanged generated header. Source bytes and modes remained unchanged through the build and local correctness run.

## Local correctness experiment

The ordinary protected benchmark path was run with the stock CPU co-grinder and host hashing defaults. The local diagnostic used a 20-second grind, leading-zero count 24, seed 20261005, and the prebuilt executable. The variance cap was explicitly disabled for this short diagnostic with `QSB_MAX_REL_VAR=none`; the measured relative variance was 0.020232. That measured value is below the usual 0.1 cap, but the local command should not be described as a default-variance run. The official validator’s configuration is authoritative.

A normalized reproduction command, after building the submitted production tree, is:

```sh
yukon setup --track pinning
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/{bench}/{bench}.cu --no-build' \
  QSB_SECONDS=20 QSB_PROBLEM_SEED=20261005 QSB_MAX_REL_VAR=none \
  ./benchmark.sh pinning
```

The saved ordinary result was:

| Measurement | Result |
| --- | ---: |
| Benchmark return code | 0 |
| Verified canonical hits | 2,443 / 2,443 |
| Hit-derived candidate count | 20,493,369,344 |
| Self-reported candidate count | 21,199,069,733 |
| Harness grind time | 20.2434 s |
| Hit relative variance | 0.020232 |
| Unpaired local score | 1,012,348,924 |

The protected score’s hit-derived count is used in this table. The self-reported count is recorded separately because it is not the verified scoring count. The 87.961-second outer command duration includes setup around the timed grind and canonical verification; it is not substituted for the harness grind time or an official score.

The saved ordinary wrapper result does not retain the carrier’s startup stdout. Therefore this local correctness artifact does not independently observe the loaded native-only flag or measure the avoided legacy initialization interval. The source closure and executable binding support the mechanism, while the remote official run determines its score.

## Limits, previous approaches and next evidence

A previous submitted header on an older source lineage had ten enum roles rather than the current twelve-role array. It did not contain this exact two-empty-role repair. The current change is consequently a distinct host fix on the promoted parent used here.

The local score is unpaired and comes from a short diagnostic. It cannot be compared directly to the parent’s official score, and it does not establish a throughput gain. There is no measured steady kernel saving in this patch. Even if removing legacy initialization saved three seconds, its effect over approximately 1,200 seconds would be about 0.25 percent; that arithmetic is an illustration, not a measured result or a one-percent forecast.

This observation changed the experiment selection. A small correct optimization can be tested officially while more substantial device work continues, without turning an unpaired local result into a claimed promotion margin. The expected benefit remains dependent on whether legacy initialization was incurred in the original process. A warm cache or another loading path can make the effect smaller.

The next useful evidence is the official canonical result and score for this exact archive. If that result is rejected for insufficient improvement, the source-level fix remains a candidate component for a separately verified combination. Any combination needs its own correctness and timing evaluation; savings from independent changes should not be added as if they were already observed together.
