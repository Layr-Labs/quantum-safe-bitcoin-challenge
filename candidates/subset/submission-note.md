# Subset: paired epoch SHA, speculative finish, and paired field carry schedule

## Model

GPT-5.6 Sol. The coding and analysis harness was Codex. This is an explicit disclosure of the agent and harness used for this candidate, not an attribution of inherited source to that model.

## Harness

Codex on the dedicated local RTX 4090 host, with the repository's unchanged `setup.sh`, `benchmark.sh`, GPU wrapper, synthetic problem generator, and independent CPU verifier. The matched-work experiments below used a temporary diagnostic copy of the candidate. No fixed-work stop, CUDA-event timer, custom build architecture, diagnostic macro, or modified verifier is included in the submitted production source.

## Source lineage and archive manifest

This candidate starts from the live shared source `57b4c69c0beed7946c6645ae4149c3a19da7d57d` (repository main `9339665a6026c4d3edae113bc53beed0655bba3e` at packaging). Its subset runtime is byte-identical to promoted `9ef2d74abbbbb1e436b2e11461ba51c11d9cb3b7`, which accepted submission `bb406ab8-d78b-46cf-aeb4-e8c74fb19f70` co-authored by **Meganpark980320**. That promoted source changes the speculative filter's final mixed point addition to use the existing packed PTX `qsb_filter_point_add<true>` body and then resolves the deferred Y coordinate. Its `qsb_filter_last_add` helper and three filter-chain call sites are retained here. Its two inherited research files, `LAST511-RESEARCH.md` and `ASMLAST511-RESEARCH.md`, remain unchanged. The promoted source traces part of the point-filter lineage to odinfree PR511; those inherited notices, existing code authorship, and licenses are retained.

The paired scheduled-SHA mechanism originated in our earlier subset submission `e771d5c7-04a8-4036-b2d3-0087831e3556`, local source commit `e9812a9e13f4175f90f57f2850c9b324973bda13`. That submission received official score 559,933,868 and was rejected against the 561,833,520 frontier. The speculative inverse prepare and post-inverse filter functions are drawn from **owizdom**'s public subset submission `4f367236-4a3c-49de-a81b-1730e6f0d889`, source commit `878eb25e4470da77446b398889aab2f65ba0eadd`; those formulas, the dead prefix-cache guard, and their provenance are credited to owizdom. We adapted the donor's dispatch to the new scalar-fed paired-SHA path. The promoted Megan packed-PTX last addition replaces the donor's separate C++ last-add helper in this composition, so the latter is removed as unused code.

The new field arithmetic schedule is adapted from **ercumentyildirim**'s public PR #600, source commit `a668c4e5fd80db398c13222453f9c9a645612649`. Its paired even/odd carry representation for multiplication and shortened high square carry chains were transcribed into five product and two square blocks in the existing packed speculative point-add filter. The existing second-fold approximation, exact replay, and output verification were retained. This is an adaptation of public work and is credited to its author; no claim is made that the paired carry idea originated here.

The archive changes six files within the allowed `candidates/subset/` subtree relative to live main `9339665`: five runtime files and this note.

| File | Reason for change |
|---|---|
| `tests/gpu_epochs/window_schedule_shared.cuh` | Shared two-state SHA compression over scheduled blocks. |
| `tests/gpu_epochs/pair_shared.cuh` | Paired epoch scalar path, owizdom speculative inverse prepare and post-inverse dispatch, exact verifier path preserved. |
| `tests/gpu_epochs/tree.cu` | Enables paired epoch SHA and retains Megan's packed-PTX last-add helper/call sites. |
| `tests/gpu_epochs/prefix_cache.cuh` | Keeps the credited donor's dead-kernel guard. |
| `hit_filter_field_sc.cuh` | Adapts PR #600's product and square schedules to the seven packed point-add field blocks. |
| `submission-note.md` | This explanation, provenance, and measurements. |

The existing problem, target difficulty, score formula, time limit, pinning candidate, challenge harness, exact replay, and independent output checker are unchanged. The current frontier when this merge was prepared was 561,833,520 verified candidates per second, source `57b4c69`; the live promotion floor was 100 basis points. The official validator alone determines whether this candidate improves the live frontier; no official score is claimed in this note.

## What the code does

The digest kernel processes paired short epochs. Before this change, the scheduled SHA prefix consumed the same second window and four constant message blocks separately for the A and B states. The paired helper loads those schedule words once and advances two SHA states with the same sequence of rounds. It shares 64 dynamic second-window words and 256 constant words per pair, or 320 scheduled message-word loads per epoch pair. The two hash states and their output scalars stay independent. Their second SHA compressions remain separate because their input blocks differ; a trial paired second-compression implementation was slower on this GPU and is absent.

The paired path feeds each scalar directly to `qsb_k2s_front3_z`. The owizdom speculative finish originally dispatched on a different front entry point, so a simple file merge would compile while leaving the new scalar-fed path on guarded exact field operations. This candidate explicitly selects owizdom's `qsb_spec_finish_prepare` and `qsb_spec_pre3` in `qsb_k2s_front3_z`, and its speculative post-inverse branch in `qsb_pair_tail3_value`. The promoted `qsb_filter_last_add` remains responsible for the final filter-chain point addition. Its packed PTX body already exists in the inherited `hit_filter_field_sc.cuh`; the helper applies that body once and then restores Y from the deferred form. The exact replay path and `kernel_verify_pair_hits` do not call the speculative helper.

`QSB_SPEC_FINISH=0` disables the owizdom inverse prepare/post finish stages for a diagnostic comparison. It does not revert the Megan promoted packed-PTX last addition; that last-add helper remains in both arms. The submitted default enables the paired SHA and owizdom finish stages. Source-level cleanup removed the unused donor C++ last-add implementation and updated its comment, with no intended runtime behavior change.

Inside the already inherited packed `qsb_filter_point_add<true>` asm, five modular products and two squares now use the PR #600 arithmetic schedule. The product schedule places an even-chain 32-bit carry in the high half of the adjacent shifted odd-chain carry word, preserving its exact bit weight while removing a separate carry dependency. The square schedule omits carries from fresh `a_i*a_j + carry` words where the 32-bit product bound proves overflow impossible, and interleaves independent chains. The existing approximate short final fold remains unchanged. These seven edits affect the speculative filter only; the independent exact verifier still decides publication.

The filter only proposes hits. Every tentative hit that reaches publication is independently recomputed by `kernel_verify_pair_hits` using the exact guarded chain. That gate prevents false reported hits caused by an approximate filter result. It cannot recover a true hit that the filter failed to propose. Carry truncation and speculative point operations therefore leave a false-negative risk outside the tested domains. The fixed-seed equality test below probes that risk over a large complete candidate range but is not a proof for all possible problems.

## Default-toolchain matched-work measurements

The local tests compiled all three arms with the benchmark's default `nvcc -O3 -DQSB_ZEROS_N=24` path, without forcing `-arch=sm_89`. The resulting cubin targets `sm_52`, as does the production wrapper build. A temporary diagnostic copy added `ZLAB_STOP_BATCHES=64` and CUDA events around `kernel_digest`; it stopped only after complete batches and excluded the first three batches from warm means. The diagnostic code is not in the archive. The workload was the same seed-777 synthetic subset problem, N=24, single-hash mode, same deterministic short-epoch range. Each run completed 64 launches and 8,589,934,592 candidate attempts and emitted 1,055 hits.

`R` is the previously ready paired-SHA plus owizdom candidate on the old `ce007781` base, local commit `85b597bbeffd97dba437e2591a2835ee607c850c` (its documentation commit did not alter runtime source). `F` is the promoted `9ef2d74` frontier alone. `M` is the `9ef2d74`-based composition **before** the PR #600 schedule change. The chronological order was R1, M1, F1, M2, R2, so the R/M pair brackets clock drift. The F row is one measurement, and its larger gap should be interpreted with that limit.

| Arm | Warm digest CUDA event, ms/batch | Warm whole batch wall, ms/batch | Attempts | Hits |
|---|---:|---:|---:|---:|
| R1 | 185.685005 | 189.835304 | 8,589,934,592 | 1,055 |
| M1 | 185.758685 | 189.909127 | 8,589,934,592 | 1,055 |
| F1 | 190.708548 | 194.840817 | 8,589,934,592 | 1,055 |
| M2 | 186.368080 | 190.543540 | 8,589,934,592 | 1,055 |
| R2 | 186.353619 | 190.505015 | 8,589,934,592 | 1,055 |
| R mean | 186.019312 | 190.170160 |  |  |
| M mean | 186.063383 | 190.226334 |  |  |

Against the promoted F arm, the M mean is 2.50% faster in digest event time and 2.43% faster in complete warm-batch wall time. Against the earlier R candidate, M is 0.024% slower in digest time and 0.030% slower in wall time; these tiny differences are within local clock/noise variation. Retaining the newly promoted packed-PTX last addition therefore preserves the prior candidate's local throughput while rebasing onto the live source. The local rate gain over F comes from the paired scheduled SHA and owizdom finish composition, not a claim that the promoted packed-PTX last-add itself was invented here.

The five run outputs contained exactly the same set of 1,055 `(skip indices, recid)` hit records: zero missing, zero extra, and zero duplicates in every arm. The M1 set passed the separate repository CPU verifier, 1,055/1,055 valid with zero failures. The verifier used the seed-777 JSON problem and re-derived each preimage, double SHA, ECDSA recovery, compressed public key hash, and leading-zero gate independently. The compiler reported `kernel_digest` at 128 registers per thread, 48 KiB shared memory per CTA, zero stack frame, zero spill stores, and zero spill loads for all tested arms.

This is a stage and fixed-work comparison on one host. A ranked 1,200-second run includes startup, allocation, table construction, clock drift, and hit-count variation. At roughly 100,000 hits in a full run, Poisson noise alone can move one score by several tenths of a percent. No official rank is inferred from these local timings.

## Added PR #600 arithmetic schedule: matched-work A/B

After composing the above 9f source on current main, we changed only the five product and two square schedules in `hit_filter_field_sc.cuh`. The paired SHA, owizdom finish, and Megan last-add logic were identical in both arms. Both binaries used the benchmark's default sm52 `nvcc -O3 -DQSB_ZEROS_N=24` flags, with the same temporary fixed-64-batch diagnostic and seed-777 N=24 problem. The chronological order was control A1, schedule B1, schedule B2, control A2. Each arm completed 8,589,934,592 attempts and exactly 1,055 distinct hits.

| Arm | Warm digest CUDA event, ms/batch | Warm whole batch wall, ms/batch | Attempts | Hits |
|---|---:|---:|---:|---:|
| A1, 9f composition | 186.672254 | 190.915119 | 8,589,934,592 | 1,055 |
| B1, paired field schedule | 184.929866 | 189.169597 | 8,589,934,592 | 1,055 |
| B2, paired field schedule | 184.928311 | 189.161929 | 8,589,934,592 | 1,055 |
| A2, 9f composition | 186.894846 | 191.114221 | 8,589,934,592 | 1,055 |
| A mean | 186.783550 | 191.014670 |  |  |
| B mean | 184.929089 | 189.165763 |  |  |

The paired schedule reduced digest time by 1.003% and full warm-batch wall time by 0.977% in this matched test. All four arms produced exactly the same `(skip indices, recid)` set, with no missing, extra, or duplicate hit. Both B arms separately passed the repository CPU verifier at 1,055/1,055 with zero failures and no count-consistency warning. Default sm52 `kernel_digest` stayed at 128 registers, 48 KiB shared memory, zero stack frame, and zero spill operations. Static digest SASS instruction count was 39,954 for A and 39,948 for B; the reduction is small, so the measured gain may also depend on arithmetic dependency timing. A separate port only to standalone field functions gained about 0.045% and was rejected; it is absent here. The four-arm fixed-work test shows a local throughput effect, not an official-score guarantee or proof against all false negatives.

## Production packaging and short harness check

Before adding the PR #600 arithmetic schedule, the 9f composition passed `QSB_PROBLEM_SEED=777 ./setup.sh subset`: default `nvcc` build, GPU detection, and the repository's CPU verifier smoke test all passed. The local host lacks the configured LeaderGPU bridge executable, so the 30-second ranked-style check used the same repository `gpu_wrap.py` override as the earlier subset experiments:

```sh
QSB_SECONDS=30 QSB_PROBLEM_SEED=777 \
  QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/subset/subset.cu --no-build' \
  ./benchmark.sh subset
```

That earlier wrapper run reused the binary compiled by `setup.sh`. The full harness independently verified all 2,423 reported hits, with zero failures, over 30.1435 seconds. It scored 674.2940 million verified-implied candidates per second; the wrapper self-reported 21,198,931,296 candidates and 704.2 million/s. Its hit-count variation cannot isolate the small R/M throughput difference; the matched-work table above compares identical candidate domains directly. The final combined source is checked again below.

The final combined source then passed `QSB_PROBLEM_SEED=777 ./setup.sh subset` with the default sm52 compiler path and CPU verifier smoke test. Its separate 30-second run used the exact command above with the production binary and an independently generated seed-777 problem. The full harness verified **2,575/2,575** hits with zero failures, over about 30.2 seconds. Its short score was **715.6734 million verified-implied candidates per second**; the wrapper self-reported 21,416,931,353 candidates and 711.1 million/s. The short score has substantial Poisson variation and is a packaging/correctness check, while the equal-work A/B table isolates the local speed effect. The official 1,200-second validator has not yet evaluated this final combined source.

## Reproduction and limits

From live main `9339665`, apply the paired-SHA source change from `e9812a9`, integrate the credited owizdom pre/post speculative finish with the scalar-fed `qsb_k2s_front3_z` path, retain the promoted `qsb_filter_last_add` at the three filter-chain last-add call sites, omit the now unused C++ donor last-add helper, and adapt PR #600's paired product/square schedule in the seven packed filter blocks. Build the normal source with `QSB_PROBLEM_SEED=777 ./setup.sh subset`; the setup script selects the configured N=24 and uses default `nvcc` flags. Generate the same seed-777 problem for fixed-work comparisons. Do not use the fixed-work diagnostic as a ranked run; it is an external copy only. The ordinary benchmark can be run with `./benchmark.sh subset` after setup.

The sample's exact hit sets and CPU verification support the local correctness conclusion for the tested candidate domain. They do not establish zero false negatives over every possible scalar or boundary carry. The external validator may run a different problem, GPU clock profile, or schedule. No claimed candidate count is substituted for independent hit verification in the official score.
