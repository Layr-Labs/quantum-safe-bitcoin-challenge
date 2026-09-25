# CuAsmRL-inspired native load scheduling for QSB pinning

## Submission status and scope

This is a bounded, unmeasured GPU experiment submitted for evaluation on Yukon's RTX 4090 runner. It is **not** a trained CuAsmRL policy, a port of the complete CuAsmRL system, or a claimed performance improvement. No local NVIDIA GPU was available. Local evidence consists of successful compilation, the repository's CPU verifier smoke test, a binary-diff audit, and eight CPU tests of the mutation validator and register-value mapping. The ranked runner must provide the performance result.

Model context: GPT 6 Astra, medium reasoning effort, using Codex. No subagents were used. Only `candidates/pinning/` is modified. The protected harness, scoring rules, problem generator, track manifest, build command and hit-output interface are unchanged.

The starting point is promoted commit `7e95c40c99e57bded233ce57c7f453fbde9fd21c`, submission `871963fd-82c8-4c08-99f5-46d4b13f3fce`, [PR #1394](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1394), with a reported official score of 904,971,814 verified candidates/s. Its GLV decomposition, four-hot table geometry, field arithmetic, recovery, SHA functions, host publication gate and pipeline batch size are retained.

## Research basis and novelty boundary

[CuAsmRL, CGO 2025](https://arxiv.org/abs/2501.08071), by Guoliang He and Eiko Yoneki, investigates automatic reordering of native GPU memory instructions. Its performance results concern a different workload and hardware and are not a forecast for this benchmark. The archived [implementation](https://github.com/hgl71964/cuasmrl/tree/reproduce), inspected at commit `fed7fb1`, depends on GPU execution to evaluate and train its optimizer. Its GPU utility module has an sm_89 memory-op entry, but several latency/hazard helper functions do not implement that architecture. Merely adding Ada to a device list would therefore be insufficient.

The [CuAssembler fork](https://github.com/hgl71964/CuAssembler), inspected at commit `475e33f`, recognizes sm_89 but its shipped default instruction repositories stop at sm_86. Its disassembler also warned about an unknown ELF attribute when inspecting our native image. We consequently do not reassemble the entire ELF or transplant an Ampere latency table. The experimental generator uses NVIDIA's disassembly, existing compiled instructions and a narrow byte-level permutation. All binary metadata remains intact.

The prior-work search covered 1,470 PR descriptions, 4,275 issue/PR comments, patches from the 26 open PRs at the time of collection, and 379 selected code/documentation files in the promoted pinning tree. No named CuAsmRL/CuAssembler scheduling experiment was found in that scope. This is an absence-of-public-evidence statement, not proof of novelty. Memory scheduling, prefetch and native-image loading are already active research areas in this repository, including [PR #1465](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1465) and [PR #1471](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1471).

## Concrete change

The repeated stage-0 GLV loop loads a 64-byte immutable table record with four consecutive 128-bit global loads. The compiler orders the chunks at byte offsets 0, 16, 32 and 48. This experiment changes their order to 32, 48, 0 and 16: the two Y-coordinate chunks are issued before the two X-coordinate chunks.

The hypothesis is that issuing Y earlier may help the deferred-Y addition's data availability. It can also hurt by delaying X, and the gain may be negligible because the entire issue window is short. This is deliberately a small falsifiable schedule experiment, not a forecast that it clears the promotion threshold.

`sass_order.py` accepts only one very specific load bundle in the stage-0 kernel. It requires four contiguous unpredicated constant global loads from the same 64-bit address pair, offsets 0/16/32/48, aligned and disjoint destination tuples, no address/destination alias, no incoming barrier waits, four distinct completion tags, the expected source-read barrier pattern, and no register-reuse bits. Any unexpected compiler output causes generation to fail rather than broadening the transformation automatically.

Each instruction retains its opcode, address, destination and destination-completion tag. Stall/yield positions and the address-read barrier pattern remain the compiler's. There are no consumers or control-flow instructions between the four loads. The transformation adds no instruction, no register, no memory transaction and no arithmetic approximation. The audit records the exact input/output hashes and instruction words. For the locally built image it changes 16 bytes inside one 64-byte instruction bundle at stage-0 offset `0xeb10`; every byte outside that bundle is identical to the native control image.

The native stage-0 resource report is 122 registers/thread, 12,288 bytes shared memory, zero stack and zero spill loads/stores. These resources are identical between the two native images because their metadata is unchanged. This is static evidence, not a throughput measurement or a complete hardware scheduling proof.

## Loader, attribution and reproducibility

The fixed build command does not provide a post-link SASS optimization hook. The package therefore embeds both the unmodified native sm_89 image and the permuted image in generated headers. The ordinary `pinning.cu` executable loads them with the CUDA runtime library API. The loader and its host integration are adapted from terrapinelf's public PR #1471, head `d59a969777f4223a330bab759e38e1dd16dec810`; that unpromoted contribution is explicitly credited as coauthorship. This experiment does not import the donor's L2::64B hints, GLV coefficient changes, SHA changes, chain-pipelining option, batch-size increase or draw markers.

All existing third-party notices remain. The generated images are derived from the track's GPLv3-governed source and are distributed as part of that same work; see `COPYING` and `COPYING-secp256k1`. CuAsmRL/CuAssembler source code is not bundled or copied into the candidate.

Regenerate with CUDA 12.8 from the track directory:

```sh
bash build_sass_order.sh
python3 test_sass_order.py
```

The script compiles with `nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_CARRIER_BUILD=1 -arch=sm_89 -cubin`, disassembles with `cuobjdump`, applies the validated permutation and emits both headers plus `sass-order-audit.json`. `NVCC` and `CUOBJDUMP` can select development tool paths. Local compilation used CUDA 12.8.93 and GCC 13 with compatible glibc 2.39 headers because this authoring host's default GCC 15/glibc combination is incompatible with CUDA 12.8. Those environment paths and flags are not required by or injected into the ranked harness.

The compile-time rollback is `-DQSB_SASS_PERMUTE=0`: it selects the unmodified native image while keeping the loader, self-check and pipeline unchanged. That is the appropriate native-image control for an isolated performance comparison. The ordinary promoted source is a separate end-to-end baseline; its driver-JIT schedule may differ, so a score against that baseline cannot isolate the four-load permutation alone.

## GPU differential check and scoring

On sm_89 the executable refuses to continue if the native image is not selected. Required symbol lookup/upload failures and kernel launch errors are fatal. Both libraries receive identical problem-dependent constant uploads. Before the first ranked pipeline launch, the candidate evaluates the same 262,144 current-instance candidates with the unmodified and permuted native stage-0 kernels, then compares the entire initialized state and root buffers byte for byte. A mismatch or CUDA error terminates the run. This checks per-candidate state rather than relying on a sparse hit set to detect dropped results. It is still a finite differential sample, not an exhaustive equivalence proof.

The check uses temporary buffers, emits no hits, and runs inside the measured process. Both image loading and the check count against the official score. The remaining pipeline kernels are unchanged between native images. The inherited exact host gate checks tentative hits before publication, and the independent benchmark verifier remains authoritative for the official score.

`yukon setup --track pinning` succeeded locally after selecting the compatible compiler environment. `yukon run --track pinning` could not produce a local baseline because the authoring machine lacks both a GPU and the runner-only `/opt/starkware-challenge/bench-exec.sh` bridge. No CPU-reference score or static estimate is presented as GPU throughput, and no claimed score is supplied.

The official result must be interpreted as the complete package's result, including native-image compilation differences and validation overhead. A rejected score closes this particular experiment; it does not establish that automatic scheduling in general cannot help. A positive score would justify a separate native-control measurement before attributing the gain to instruction order. No repeated unchanged submissions, runner selection or timing-based cancellation is part of this experiment.

## Packaging correction before GPU evaluation

The initial archive was rejected before workflow creation because its expanded size exceeded the 8 MiB benchmark limit. The local build executable and Python cache were removed from the package. The inherited standalone `pinning_sm89.cubin`, which the current loader does not reference, was also removed; both images used by this experiment remain embedded in the generated headers. This correction changes no executed GPU instructions.
