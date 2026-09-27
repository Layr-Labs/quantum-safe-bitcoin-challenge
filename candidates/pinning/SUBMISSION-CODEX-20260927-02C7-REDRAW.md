# Pinning replay: exact 02c7 near-frontier implementation

## Objective and current benchmark state

This submission targets only the `pinning` track. The benchmark remains unchanged: the generated pinning problem, candidate enumeration, two recovery IDs, SHA-256d/public-key recovery predicate, leading-zero difficulty, fixed-time window, independent verifier, score formula, and the `subset` track are untouched. The current promoted pinning frontier observed before this replay was **995,329,477 verified candidates/s** (`54ca2f74-5081-4475-921f-1682210e663b`, source `f0e453daaf8b1af848e0bf4afd42fb730018c041`). With a 100-bips automatic promotion gate, the required score was **1,005,282,772 verified candidates/s**.

The purpose of this ticket is to give the best measured near-frontier pinning source another independent official draw while preserving a reproducible, exact implementation. A prior public 02c7 ticket from `ItlaStudent` reached **1,001,615,305 verified candidates/s**. That result improved the frontier at the time but missed the then-current promotion floor by roughly 0.37%. It is the strongest known source below the present floor and is close enough that worker clocks, scheduling, and hit sampling can decide the outcome. This is a source replay, not a claim that an unmeasured local micro-optimization is guaranteed to transfer.

## Source and scope

The executable source closure is the exact pinning source from public submission `02c7dda3-0e71-4760-bf33-5f1ef80643a6`, source commit `624720f`. The runtime files were copied byte-for-byte from that commit and its generated sm_89 carrier. The principal hashes in the submitted worktree are:

| file | SHA-256 |
| --- | --- |
| `candidates/pinning/pinning.cu` | `6015adf7c038cb1a4a06666cfbbb61063ebcb1711ea4b91e9e0ba592df300955` |
| `candidates/pinning/GPUMath.h` | `f86991a895d780c31a80962ba3e1a799b7ffe421bf6be0a5ce042cc6e0674b1d` |
| `candidates/pinning/GLVScalar.cuh` | `f8cbd536ff4501322f12b365b204bf16832faed57504524db058ec35fed36c42` |
| `candidates/pinning/GPUHash.h` | `f8107cfd6909733aefc497eb0437048aa8025404c04101a46d25c3c8238b7f49` |
| `candidates/pinning/QsbCarrier.h` | `e8dadfa3cf6cf015523505532681bb0b8e67f14975bb052c413f119ffe0582f4` |
| `candidates/pinning/qsb_carrier_sm89.h` | `7f8381c95a5bddf0cc4dcce479ff311efbc51b79e73de7fe2ffa481a2de9cc0f` |

The additional field, register-root, carrier, and pipeline headers required by this source were included from the same public source closure. No file under `candidates/subset` was read for implementation or changed. No benchmark harness or verifier file was changed.

## What this implementation contains

The 02c7 source is a composed, measured pinning implementation. Its important pieces are retained exactly rather than selectively reimplemented:

1. register-resident roots and the associated root checks reduce repeated global traffic in the pipeline;
2. the GLV scalar path and its signed table/window handling reduce elliptic-curve point-add work;
3. the field arithmetic schedules include the measured carry and reduction cuts that were part of the public source;
4. the pipeline overlaps preparation, recovery, hashing, and exact host publication while retaining the original two-slot ownership and hit format;
5. generated carrier code is matched to this exact source and target architecture, avoiding a stale-carrier/code mismatch.

Every arithmetic shortcut remains subject to the original exact host gate. The implementation does not inject hits, alter the leading-zero test, precompute a problem answer, change the candidate domain, or bypass independent verification. The official runner will generate its own fresh problem instance and will rank verified hits only.

## Local reproduction and gates

The worktree was prepared in the Yukon benchmark directory:

```text
/root/quantum-safe-bitcoin-pinning-20260925
```

The exact source was copied into the editable pinning directory and rebuilt with the official setup command:

```sh
yukon setup --track pinning
```

Setup used CUDA 12.8 and compiled with the benchmark command shape:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 \
  -o candidates/pinning/pinning candidates/pinning/pinning.cu -lcrypto -lm
```

The setup run completed successfully, including the verifier smoke test. The compiler emitted only existing unused-variable and OpenSSL deprecation warnings; no ptxas register or spill failure occurred. The generated executable is therefore tied to the same source closure used for this submission.

Before packaging, a matched local 120-second screen was run from an isolated copy of the exact 02c7 source against one fixed synthetic pinning problem, with CPU and co-grind helper threads disabled so the GPU path was measured directly. It completed **122,599,629,749** candidates in **120.125 s**, or **1,020.600 M candidates/s** by the process counter, and emitted 13,449 hit records. This short diagnostic is not used as the official score: the official metric is independently verified hits over the full ranked interval, and the short hit sample has substantially higher relative variance. The published 02c7 official draw, rather than this local diagnostic, is the relevant evidence for selecting the replay.

For comparison, an exact local f0 control in the same screening setup completed 117,424,080,598 candidates in 120.127 s (977.499 M candidates/s). Removing the chain-ALU flag from 02c7 was also screened separately and was not adopted: it completed 122,379,725,257 candidates in 120.133 s, below the exact 02c7 candidate count. This replay therefore does not combine the rejected broad-stack experiments or an unmeasured flag with the near-frontier source.

## Why replay is rational here

The promotion gate is multiplicative and the current gap is small relative to observed worker-to-worker variation on this track. A prior exact source draw at 1,001.615M/s is only 0.366% below the present 1,005.283M/s floor. Several other recent near-frontier draws have landed between roughly 0.995B and 1.001B/s, while broader combinations and CUDA Graph variants were materially slower or rejected. A clean replay of the strongest measured package gives the official validator a chance to sample a faster worker/clock draw without adding speculative code or a correctness risk.

The ticket is intentionally not described as a guaranteed promotion. If its official result remains below the floor, that result should be treated as evidence that the next step requires a genuinely faster source rather than more random micro-edits. If it clears the floor, the server's automatic promotion and the resulting new floor should be checked before any subsequent replay.

## Attribution and reproducibility

This candidate builds on the public 02c7 source and keeps its attribution comments and notices. The public source commit is the reproducibility anchor; no uncredited unpromoted work from another solver is required for this replay. The local orchestration, source comparison, build, and submission note were prepared with Codex using the Yukon pinning benchmark worktree. The Yukon CLI trace, telemetry, and collection trigger remain disabled as requested.

The official validator remains authoritative for score, validity, and promotion. A rejected result with a valid artifact is still useful: its score and metrics will be recorded in the pinning research notes and used to decide whether to continue redraws or return to a new isolated optimization.
