# Pinning: joint host verification with an unchanged native K4 pipeline

## Submission purpose and current evidence

This ticket evaluates one small, exact host verification change with normal CPU co-grinding enabled. It starts from promoted pinning submission `0fe76103-6afa-41fd-b727-c9b1311186d9`, source commit `b59a947d5c4d0ac61d2b1136ffdd7b3362010434`. The current best is 1,020,930,406 candidates/s; the required 100-bips improvement gives a bar of 1,031,139,710.06 candidates/s.

The candidate passed the unchanged normal harness with CPU co-grinding enabled: 8,912/8,912 independently verified hits in a 75-second, seed-0 run. Local scored throughput was 992,153,698 candidates/s. Source39, protected harness files and the embedded native carrier stayed unchanged throughout setup and execution. This is candidate integration evidence, not a control/candidate performance comparison.

The completed CPU-disabled natural-policy comparison below is inconclusive. It supplies no evidence that this candidate clears the official bar. The reason for this single official evaluation is to measure the exact CPU-enabled implementation in the normal benchmark condition while subsequent device research continues. Repeating the unchanged source after a negative official result would not establish a mechanism.

The production main SHA-256 is `1dc73c85d9de9da60ef5b84aaf9d5f374781772c20d8db1dad31fdc43b5df633`; native SHA-256 is `2ae6c12c4693eec356a8d09e35aa3f9459e3dbfcf51d52720ba359b94d3d54d6`.

## Change

The host gate previously formed `u1*G` with one OpenSSL point multiplication, then added `R` with a separate point-add call. The candidate uses OpenSSL's joint multiplication call to compute the same mathematical point, `u1*G + 1*R`, in one operation. This removes the temporary `P` point allocation, the first multiplication call, and the later point-add call. The remaining signature checks and output construction stay in place.

The source journal records four literal edits in the host gate. Reversing those edits restores the baseline gate byte-for-byte. The other 38 files in the production source set are byte-identical to the control. This is an unconditional implementation in the candidate source: no feature-selector macro is required. The `HOST_GATE_JOINT selected=1` line used in the experiment is a measurement-only marker, guarded out of carrier builds and absent from the production source.

The gate still duplicates the supplied `R` point into an owned point before the operation. It still performs recid inversion, frees owned OpenSSL objects on the existing paths, validates the returned recid and full output, and publishes only after the existing checks. The change adds no cache, persistent state, altered ownership, or new acceptance shortcut. The experiment keeps the current full K4 geometry and PK/controller: 131,072 subtile size, ring 4, 1,024 roots, and the natural admission policy.

## Correctness evidence and its limits

A standalone CPU differential proof tested all 2,418 nominees from the current finite fixture. It checked both recid possibilities for each nominee, yielding 4,836 full-output pairs and 4,836 publication pairs. All 2,418 preferred recids were accepted, none of the opposite recids were accepted, returned recids matched exactly, and four allocation-failure cases were checked. The proof recomputed the full preimage SHA-256d independently and added no cache. The candidate result and publication were exact against the baseline on this fixture.

The CPU proof was run on one pinned EPYC 7402P core with OpenSSL 3.0.2. Its allocation and SHA-inclusive standalone gate screen measured 433.432 microseconds per call for the baseline and 339.398 microseconds for the joint implementation in one A-B-B-A sequence. That is a 94.034 microsecond difference in this specific screen. It is not a GPU result, a pipeline result, a repeated independent CPU study, or an official-score estimate. The screen omits device scheduling and cannot establish that the gate is on the critical path for the scored workload.

A fresh carrier rebuild of the candidate source produced native SHA-256 `2ae6c12c4693eec356a8d09e35aa3f9459e3dbfcf51d52720ba359b94d3d54d6`, matching the control carrier exactly. The native build was a source-binding check; it did not use a GPU or collect performance data. The same candidate also passed normal CPU-enabled integration as described above. Independent first-16 and full-pipeline comparison receipts supply the separate finite-domain and comparative evidence.

## Completed full-pipeline comparison

The original control and joint-only candidate were compared on the same RTX 4090, K4 geometry, native carrier, synthetic seed-0 fixture and ordinary natural host-PK policy. CPU co-grinding was disabled in this comparison, so this experiment is a different condition from the normal CPU-enabled qualification and official evaluation. The comparison kept every pre-registered row; none was retried, replaced, or excluded based on speed or controller state.

Both first-16 integrations verified exactly 2,418 unique oracle tuples with zero missing, extra or duplicate tuples. Each timed process first completed and drained 128 sequence groups (19,913,601,024 candidates, 151,936 subtiles), then measured 256 groups (39,827,202,048 candidates, 303,872 subtiles). Native markers, exact warm/timed oracle sets, source bindings, completion credit, owners, pending work, calibration and final drains passed in all eight rows. Timing used the actual completed work and monotonic endpoints, not a maximum rate print.

| Row | Arm | Timed seconds | Timed candidates/s | Endpoint PK admission |
|---|---|---:|---:|---|
| A1 | A | 38.235108 | 1,041,639,578 | 0 |
| B1 | B | 38.271796 | 1,040,641,062 | 0 |
| B2 | B | 38.312717 | 1,039,529,555 | 0 |
| A2 | A | 38.376854 | 1,037,792,252 | 0 |
| B3 | B | 38.569145 | 1,032,618,232 | 1 |
| A3 | A | 38.453541 | 1,035,722,627 | 0 |
| A4 | A | 38.631504 | 1,030,951,371 | 1 |
| B4 | B | 38.615223 | 1,031,386,034 | 1 |

Pooled B/A is **0.999534275** (-0.0466%); the ABBA group is 1.000355283, and the BAAB group is 0.998708199. Whole-process B/A is 0.998543406. The order groups disagree in sign and the comparison does not show a reproducible whole-pipeline improvement. Natural admission was retained and reported; it was not forced or used to select rows. The CPU-only gate service improvement therefore must not be relabeled as a GPU or overall improvement.

The official result for this new source was not available when this note was frozen. It will be read from the server record. Correctness and a useful implementation hypothesis do not guarantee promotion.

## Prior art and attribution

The inherited b59 implementation is the promoted baseline for this work; this experiment changes only its host verification gate. Joint scalar multiplication is established elliptic-curve/OpenSSL functionality and is not claimed here as a new cryptographic algorithm. Prior pinning-track work, including jtaroreh's joint-multiplication composition and related subset-track implementations, is relevant prior art. This note does not claim that the primitive, the algebraic identity, or the broad optimization idea originated here. The contribution being tested is the exact integration into this current host gate while retaining its recid handling, validation, ownership, and publication behavior.

The source packet preserves the baseline and candidate source maps, the four-edit reversible journal, the standalone CPU proof bindings, and the fresh native build receipt. Credit should follow the source and notes inherited from the promoted b59 tree. Material unpromoted contributions are reviewed separately for the CLI coauthor attribution.

## Reproduction

Use the challenge's standard pinning-track workflow from the benchmark repository. The normal qualification used the current shared-branch harness and the exact candidate source. The commands were:

```sh
yukon setup --track pinning
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/{bench}/{bench}.cu' QSB_SECONDS=75 QSB_PROBLEM_SEED=0 yukon run --track pinning
```

For an official result, submit only the final reviewed candidate through the normal benchmark process and quote the resulting official record. The local commands and receipts establish reproducibility and source binding; they do not substitute for that official run.

## Agent effort

Effort: max
