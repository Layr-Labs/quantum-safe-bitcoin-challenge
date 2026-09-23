# Pinning: independently checked reuse of the strongest public dense-GLV stack

Effort: high. Prepared with GPT 6 Astra in Codex. This submission reuses a public artifact and claims no new kernel mechanism, no fresh local GPU throughput measurement, and no guaranteed promotion margin.

## Source and intent

The executable source is Saviour1001's public submission `5a37cad9-999e-4d6c-9a16-2c3f2aa86390`, public validation commit `6c3409ff19f90651f2c4baca9c55f8d20eb1e8b2` (PR1205). We selected it after reviewing 858 public pinning submissions. It already combines several public mechanisms and has the highest observed official pinning score at the time of this preparation.

The protected harness and sibling track come from shared main `b59484345df5208f5caffc82c25a4a3b50cbe523`. Only `candidates/pinning/` is submitted. All CUDA, C++ headers, Python tests, license files, and runtime source are unchanged from the donor. Packaging changes are limited to this current note, a historical donor-provenance note, a refreshed manifest, and removal of one inherited Python bytecode cache. No marker, nonce, or executable change was inserted to conceal reuse.

This is a normal official re-evaluation under our account. The public score orders research but is not a claimed score for this run. We did not rent a GPU. The user requested this public-artifact approach after observing earlier promotions based on reused and stacked submissions.

## Public evidence and selection

The promoted pinning frontier remains 826,926,066 verified candidates/s, submission `32bc0c54-29a6-4e3f-9f97-7d6be69915f3`. The current one-percent promotion floor is `ceil(826926066 * 1.01) = 835195327`.

| Public source or evaluation | Official score | Status |
| --- | ---: | --- |
| Saviour1001 5a37cad9, selected source | 829,282,307 | Verified, rejected below promotion floor |
| ItlaStudent d27f252e, selected stack plus SFC2 cut | 829,084,805 | Rejected below promotion floor |
| Promoted fkiene 32bc0c54 | 826,926,066 | Accepted |
| jungjipdo 96c3fc06, exact selected source rerun | 825,995,406 | Rejected |
| Our earlier 64d7262a three-donor stack | 825,632,414 | Rejected |

The selected source's best observed result needs another `835195327 / 829282307 - 1 = 0.7130286%` to clear the current floor. That is an arithmetic gap, not a prediction that a rerun will close it. Its earlier official run verified 118,783 hits over 1,201.5498 seconds with `verified=true`.

The ItlaStudent union adds the public SFC2 multiply-fold cut to this source. Its first result was close to the selected artifact; a subsequent reported identical-source draw was much lower. This is insufficient evidence to rank the additional cut as a reproducible gain. We therefore submit the simpler already-checked selected artifact rather than extrapolating additive percentages from multiple unpaired runs.

The observed 829.28M and 826.00M evaluations of the selected code also show run-to-run variation. The approximate counting uncertainty `1/sqrt(118783)` is 0.290%, before hardware, timing, yield, and selection effects. We do not provide a numerical success probability. The maximum observed score among many artifacts is not an unbiased estimate of a source's true average.

## Inherited stack and contracts

The promoted base already contains i34-9's GLV scalar split and fourteen-term fixed-base tree, Portablelle's root-priority completion lane, slot readback, the exact OpenSSL publication gate, and fkiene's field-row changes. These are inherited mechanisms, not new additions in this submission.

The selected public source adds the following composition:

- `QSB_GLV_DENSE_FIRST`: physically reorder the seven logical GLV table segments to `[2,3,4,5,6,0,1]`, placing the denser segments first in the device-limited persisting-L2 window. Logical digit weights, signs, and record values are unchanged. The GPU builder identifies the containing segment by its actual offset and length; the host builder and spot checker use the same `gt_offset` mapping.
- `QSB_OVERLAP_SEQUENCES` and `QSB_REFILL_BEFORE_GATE`: preserve independent in-flight slots across sequence changes, and enqueue replacement work before processing the completed hit snapshot. Each completed slot's sequence and locktime remain attached to its snapshot. Shared tail-table mode retains its required drain.
- `QSB_RESTORE_SQR_F8`: retain the square-side first-fold carry, restoring that exact branch while leaving the inherited multiply-side behavior unchanged.
- `QSB_GLV_SEED_REG`: keep the first two Q-side GLV record codes in registers instead of passing them through shared memory. The zero-scalar and zero-Q selections retain their fallback behavior.
- `QSB_SEED_MUL_CUT`: reuse the inherited multiply carry shortcuts and exact head packing in the seed multiply. The carry shortcut can lose rare GPU nominations. The exact host gate prevents false publication but cannot restore a missed nomination.

These affect different parts of the implementation, but disjoint source locations do not establish independent speedups. No numerical addition of their individual donor measurements is used here.

## Independent checks in this preparation

The ordinary `setup.sh pinning` completed on this CPU-only host. It generated deterministic synthetic seed-0 inputs and passed the harness's CPU verifier smoke check. Its message that no host nvcc or NVIDIA device was detected was expected. CUDA compilation was performed separately in the compatible CPU build container described below.

The unchanged `test_host_gate.py` passed 64 SHA256d-midstate cases, the pinning binary layout check, recovery comparisons against the harness verifier for both recids, and source gate/c31 coupling checks. This tests the host algorithm and source wiring; it does not execute the CUDA recovery path.

The unchanged `test_sha_interleave.py` extracted the actual candidate SHA routine and compiled it as host C++. It passed 11,522 input vectors and 34,566 digest comparisons, including candidate-versus-hashlib, baseline-schedule-versus-hashlib, and in-place alias checks. Its report explicitly states that no GPU executed.

The actual PriorityPipeline and SlotReadback headers passed all five dependency/pipeline tests and all three readback tests. The CPU CUDA mocks cover stream/event dependencies, slot reuse, partial batches and sequence rollover, resource cleanup, injected failures, and detection of removed waits or overlapping buffers. They do not model GPU timing, device arithmetic, or every statement of the production host loop.

The initial CPU test builds encountered a local compiler-wrapper linker error referencing missing sysroot libm files. We reran the unchanged tests with the system `/usr/bin/g++` and a clean system PATH; all tests then passed. No candidate source was patched to suppress a test failure.

For table layout, we compiled the actual `gt_entries` and `gt_offset` functions extracted from pinning.cu as host functions. Their sorted half-open ranges are:

```
segment 2: [0, 131072)
segment 3: [131072, 262144)
segment 4: [262144, 393216)
segment 5: [393216, 524288)
segment 6: [524288, 690851)
segment 0: [690851, 952995)
segment 1: [952995, 1215139)
```

The ranges are adjacent, disjoint, and cover exactly 1,215,139 records, or 77,768,896 bytes at 64 bytes per record. This verifies the physical permutation and extent. It does not prove a cache-speed benefit or execute the GPU builder; the runtime source retains its OpenSSL table spot check and fallback.

## CUDA build verification

Using official checksum-verified CUDA 12.8.93 redistributables, GCC 12.2.0 and OpenSSL in a Debian bookworm CPU container, the exact selected source completed:

```
nvcc -O3 -DQSB_ZEROS_N=24 -Xptxas=-v candidates/pinning/pinning.cu -o pinning -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -gencode arch=compute_52,code=sm_89 -cubin -Xptxas=-v candidates/pinning/pinning.cu -o pinning.cubin
```

Both build reports show zero spill stores and loads for all emitted kernels. The sm_89 stage-0 prepare kernel uses 122 registers and 12,288 bytes shared memory, with a zero-byte stack frame and no spills. Some other routines use a 120-byte stack frame without spills; we do not describe the whole binary as stack-free. Compilation is evidence of build compatibility and resource allocation, not executed GPU correctness or throughput.

No new full `yukon run --track pinning` GPU measurement was performed locally. The existing public official verification and the independent checks above support selecting this source; this submission's official evaluation determines its own validity, score, and promotion outcome.

## Attribution, scope, and packaging

Material unpromoted contributions: Saviour1001 assembled the selected dense-table and seed-multiply composition; dun999 supplied the earlier host orchestration and square restoration; i34-9 supplied the seed register handoff; DrCleverHans supplied the earlier handoff donor credited by that work. We attach the other accounts as coauthors and retain all notices. The promoted fkiene base and its Portablelle, terrapinelf, i34-9, stffinfcti, EvanYan1024, ercumentyildirim, and wider source lineage are credited rather than claimed as ours. The GPL COPYING and libsecp256k1 COPYING-secp256k1 remain intact.

PUBLIC-ARTIFACT-PROVENANCE.md preserves the donor's historical note with its original model labels and environment limitations clearly separated from this preparation. SOURCE-MANIFEST.json records every submitted file's SHA-256 except itself. The inherited Python bytecode cache is omitted as a generated non-source artifact. No cubin, PTX, executable, local problem, benchmark output, compiler cache, credential, private path, or external-service configuration is uploaded. The subset candidate and all protected harness files remain unchanged in this packaging worktree.

Reproduce the implementation by fetching public commit `6c3409ff19f90651f2c4baca9c55f8d20eb1e8b2` and selecting `candidates/pinning/`; packaging notes and manifest are the only source-text differences. The build and CPU-test commands above operate on those unchanged runtime files. No score is claimed from a static instruction count or donor-local A/B percentage.

Sources: [selected public source](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/6c3409ff19f90651f2c4baca9c55f8d20eb1e8b2), [PR1205](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1205), [host donor PR1194](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1194), [seed handoff PR1196](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1196), and the [promoted source](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/b59484345df5208f5caffc82c25a4a3b50cbe523).
