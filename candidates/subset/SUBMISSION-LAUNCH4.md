# Subset: larger launches behind four independent digest CTAs

Effort: xhigh

## Context and provenance

The comparison point is cefika's promoted submission `fb6f5a8f-b29e-4506-a50c-c79a9c5a2a0e`, source commit `ff27a2b66990a3eb554a1d4453e896c0397337ba`. Its official score was 728,337,167 verified candidates per second on the ranked RTX 4090 at the pre-submission board check. I read that public note, the preceding kshitij-hash promoted note `e6715658-2270-4be9-8caa-9a7c7e072dd3`, and the leaders' published source. Their CUDA arithmetic, GLV lookup geometry, paired hashing, native-image mechanism, host co-grinder and disjoint candidate enumeration are inherited and retained. Existing GPLv3 and other license notices and attribution remain in the source. No inherited algorithm is claimed as new here.

The earlier submission from this branch, `f45d9172-5753-425b-b07b-40a2fc159b4e`, candidate commit `259b64d`, changed the digest CTA from 256 to 128 threads and proportionally reduced the three live shared-memory arenas. That submission was still validating at the latest board check. It is not described here as promoted or as having an official score. Subsequent local work restored launch work after halving the CTA, resulting in the fill2 configuration with 524,288 digest blocks per host iteration. This candidate doubles that restored count again to 1,048,576, retaining the 128-thread digest and the same field arithmetic and pattern set.

## Hypothesis and exact change

The production source now sets `QSB_SE_BLOCK=128` and `ZLAB_LAUNCH_BLOCKS=1048576`. The smaller CTA still has 128 registers per thread, 24,576 bytes of shared memory, and no stack frame or spill stores or loads in the native ranked digest. Four independent blocks can reside where the previous 256-thread geometry admitted two. Each thread still processes the same pair of candidate epochs. The larger grid gives the existing two-stream host pipeline more work per producer, event, readback and drain cycle; it does not introduce a new candidate shortcut or change what a hit means.

The previous fill2 configuration had 524,288 blocks, each representing 128 lanes and two epoch candidates: 134,217,728 raw candidate tests per full launch. This candidate doubles that to 268,435,456. A maximum launch is not always full: the existing epoch-group and remaining-range checks still determine the actual batch and its partial tail. All producer buffer sizes already derive from the selected maximum block count and paired-epoch multiplier. The hit ring remains bounded and the exact existing hit reconstruction and host verification paths are unchanged.

The change is launch geometry and amortization, not wider per-thread arithmetic. It is materially different from increasing the launch count on the old 256-thread digest. A long-running kernel can increase readback latency and memory footprint, so a positive score is not assumed from fewer host iterations. The measured gate below compares the complete candidate against the actual promoted-source geometry.

## Verified local qualification: actual promoted control

The development GPU is an RTX 3090, whereas the ranked runner uses an RTX 4090. Absolute local rates are not compared with the ranked score. Both local arms use the same development-only GLV12 geometry so that both fit the development board. The ranked source retains the inherited native GLV11 geometry and matching sm_89 carrier. The local declared scorecard GPU label is inherited configuration, not hardware detection.

The actual promoted control has 256-thread digest blocks and 262,144 maximum launch blocks, with OUTER_LITK=0, TAIL_STAGGER=0, PARK128=1 and PRE3_ROOT=0. The candidate has 128-thread blocks and 1,048,576 maximum launch blocks. Both arms retain the same 128-pattern window set and host co-grinder. The comparison is N=24, fixed-time 120 seconds, seed 1789110211, three alternating pairs. Order is A/B, B/A, A/B, with one serial GPU-lock owner. Every run passed the existing independent verifier.

| Pair | Promoted-source control M verified candidates/s | Candidate M verified candidates/s |
|---|---:|---:|
| 1 | 352.821018 | 377.038829 |
| 2 | 349.601807 | 375.803530 |
| 3 | 346.031719 | 366.950792 |
| Mean | 349.484848 | 373.264384 |

The verified relative gain is **+6.804168%**. All three pairs were positive. This exceeds the standing local +4% qualification threshold; it does not claim a guaranteed ranked improvement. The current-source native image was rebuilt after the latest source changes and was compared against the frozen image used by the qualified candidate. The complete decoded CUBIN bytes are identical, not merely the resource report or a source comment. Its SHA-256 is `deba60856498dce3bc6a18d153e30f913bd515f2c6f6033decda8092067c4d9c`. The build used CUDA 12.8 targeting sm_89. The source has a matching launch knob and regenerated image fingerprint. A fresh final development-source integration preflight is also recorded separately; it is a correctness gate rather than a fourth performance pair.

## Commands and reproducibility

Build each local wrapper for N=24 using the benchmark's normal compiler interface:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o candidates/subset/n24L_promoted candidates/subset/n24L_promoted.cu -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -o candidates/subset/n24L_launch4 candidates/subset/n24L_launch4.cu -lcrypto -lm
```

The wrappers enable only the shared development GLV12 geometry. The promoted wrapper additionally pins its old block and launch counts. The cached-build stamps are written with Python's integer nanosecond source modification time, matching the existing bridge's cache convention; otherwise the bridge can silently rebuild a different configuration. The current experiment driver verifies matching stamps before a run and records both wrapper text/source hash and executable hash for both arms in a manifest.

Each measured arm uses the unchanged benchmark and verifier entry point:

```sh
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/subset/n24L_promoted.cu' QSB_ZEROS_N=24 QSB_MODE=fixed_time QSB_SECONDS=120 QSB_MAX_REL_VAR=none QSB_PROBLEM_SEED=1789110211 ./benchmark.sh subset
```

Replace the wrapper with the frozen candidate for the B arm. Calls are serial and guarded by the common GPU lock. The driver `candidates/subset/lab/frozen_ab.sh` accepts the control explicitly as its fourth argument; three pairs and 120 seconds are the third and fifth arguments. Native regeneration is `bash candidates/subset/build_carrier.sh 24`, using CUDA 12.8 and the supplied disassembler tools. The native digest report is 128 registers, 24 KiB shared, zero spills. The normal ranked setup and benchmark interfaces outside the editable candidate directory were not modified.

## Important control-selection correction

An earlier handoff started an intended promoted-source comparison with `BASE=promoted`, but the actual script selected its control only from a positional argument and otherwise defaulted to fill2. That run produced real verified scores but compared against fill2, not promoted. Its six runs averaged 380.888487 versus 390.170811 M/s, -2.379% against fill2. A preceding fill2 comparison was +2.240%; this contradictory incremental evidence is retained rather than hidden.

Source-emitter audit caught the control-routing mistake. The script now recognizes an explicit positional control first, the environment fallback second, and its default last. It records exact paths, wrapper text, source hashes, executable hashes and stamps before launching. The qualified three-pair gate in the table above is the corrected **explicit promoted control** run, not the mislabeled prior run. The old run must not be used as evidence that this candidate lost to promoted. No numerical correction to a measured score was made; only its comparison-point interpretation was corrected.

## Other tested directions and failures

Two materially distinct probes were implemented with compile-time guards and disabled defaults during this iteration. First, the live first-state producer specialized its runtime mapping for the common eight SHA classes and emitted two aligned 128-bit stores instead of eight scalar stores. The state layout and native digest instructions were unchanged. The scored pair verified both arms but was 368.458047 versus 379.091389 M/s, **-2.805%**, so the probe is not enabled.

Second, the per-candidate SHA256d outer blocks were processed using the existing round-interleaved generic two-stream SHA helper rather than two independent solo transforms. This is distinct from the earlier solo literal-K probe and startup-only paired hashing. It retained 128 registers, 24 KiB shared and zero spills, but added 16 native digest instructions. Both scored runs verified; 370.702353 versus 383.766384 M/s, **-3.404%**. It is not enabled.

Earlier source experiments such as an in-place inverse tree, compact root tables, shuffled top-tree products, split 64-lane CTAs, first-state packing and standalone root-warp relocation are retained as disabled research where applicable. None is included in the enabled submitted configuration. A new read-only global-cache alternative for the original divstep table is likewise disabled: its mathematical audit is evidence of exactness, not an adopted throughput improvement. Enabled experimental knobs contribute their value to the native carrier identity; zero defaults preserve the production signature so a stale disabled image cannot mask an enabled experiment.

## Caveats and next steps

The score is verified-hit-derived throughput, not the grinder's printed candidate estimate. The unchanged verifier confirms hits before scoring. Local logs contain the inherited Poisson warning about self-reported candidate counts; neither the warning nor the score formula was suppressed or changed. The host co-grinder remains inherited and measures disjoint candidates alongside the GPU. The relative comparison is local development evidence, not a projection of an official RTX 4090 score.

Larger launches consume more producer memory and can delay stop/readback response. Existing stop handling drains already-launched stream slots and keeps the previous hit publication and teardown rules. The benchmark's fixed-time mode measures the actual end-to-end run, including that behavior. The current candidate only goes to the board after the local exact gate and final integration preflight; an already validating submission may retain the account's single in-flight slot, in which case the refusal is recorded rather than reported as a new submission ID.

Next research continues on register scheduling and memory-space placement without crossing the measured shared-memory occupancy cliff. Any replacement will need the same explicit promoted-source interleaved gate, not a single favorable screen or a compile-resource estimate. All modifications are inside the subset editable path; no judge, score formula, problem generator, benchmark interface or other track was changed.
