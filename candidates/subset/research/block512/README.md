# 512-thread CTA experiment with 96 KiB shared memory

Status: default and candidate built; exact tree audit and 60-second candidate
run are queued behind the common GPU lock. No official submission or score.

The preceding 128-thread trial reduced residency-unit size but increased root
inverse count and regressed on the warm repeat. This experiment tests the
opposite tradeoff:512 threads per block, with four 128-window epoch pairs, so
there is one root inverse for 1,024 candidates instead of 512. There are 131,072
physical blocks per launch instead of 262,144, preserving 1,048,576 epochs and
134,217,728 candidates per full batch. Host synchronization frequency and all
point-table/epoch/first-state global-buffer capacities stay unchanged.

The consumer needs 96 KiB shared memory, beyond CUDA's default 48 KiB static
per-block allowance. It therefore uses a single opt-in dynamic region:

| Region | uint 64 offset | uint 64 length | Bytes |
|---|---:|---:|---:|
| parked first candidate finish state |0|6,144|49,152|
| immutable product-tree levels |6,144|4,096|32,768|
| downward inverse-tree levels |10,240|2,048|16,384|

The intervals are adjacent and disjoint; their sum is 98,304 bytes. The existing
product-tree indexing uses runtime n=blockDim.x, with at most 2*n product and
n inverse entries per limb. A 512-thread launch uses 128-register launch-bound
budget, just like the prior 256-thread/two-block consumer. Compilation reports
128 registers, zero stack, and zero static shared memory;98,304 dynamic bytes
are explicitly passed on every ranked short-epoch digest launch. The host
checks cudaFuncSetAttribute(MaxDynamicSharedMemorySize) and exits on failure.
No hardware capability is assumed silently. Default 256 and 128 branches retain
static allocations and zero launch-time dynamic bytes.

All field/scalar/SHA arithmetic, window choices, odd-tail activity guards,
hit tags, exact replay and host publication remain unchanged. A larger tree
changes which candidates share a denominator inverse; exact arithmetic must
produce the same per-candidate inverse. The inherited production tree uses a
speculative carry-truncated multiplier. Adversarial edge cases already failed
the 128 trial's production audit; comparison with the baseline is tracked there.
The geometry-specific audit disables QSB_SHORT_CARRY3 and the dependent parity
window fallback, allowing comparison to OpenSSL without that known speculative
confounder. Production whole-solver tests keep the inherited settings and verify
every emitted hit independently. Do not interpret an exact-mode geometry audit
as a proof that the speculative production multiplier is universally exact.

Audit covers block sizes 32,64,128,256,512 and tail counts 1,127,128,129,255,256,
257,511,512,513,8191, plus the inherited canonical multiply and cooperative
root/fallback checks. For dynamic shared memory, the audit opts in for its own
kernel and passes the identical 96 KiB launch allocation.

Reproduce with PATH=/usr/local/cuda/bin:$PATH python 3
candidates/subset/research/block512/trial.py build, then flock -x
/tmp/qsb-gpu.lock python 3 candidates/subset/research/block512/trial.py
candidate --label=-a. It uses N=24,60 seconds, seed 2026092501 and unchanged
CPU verification. The capture wrapper preserves kernel stdout while invoking
the original gpu_wrap module. All results are RTX 3090 diagnostics, not ranked
RTX 4090 scores. The production subset.cu entry point still uses 256 threads.

Main model: GPT-6 Astra, high reasoning, Codex. No unpublished source was sent
to a helper. The earlier DeepSeek V4.1 Flash [1 m] mechanical arithmetic check
used only sanitized stage timing and resource numbers; its result was verified.

## Outcome

Exact-tree geometry audit passed, including512-thread and tail cases; inspect
tree-audit-exact-result.txt. The production whole-solver trial verified all
1,569hits, but reached only233.1M/s peak diagnostic rate and218.791459M/s
hit-derived wall-clock score. This regresses versus the preceding256-thread
baseline243.8M/s peak and245.269842M/s verified score. No submission.
All production source changes from these layout experiments were reverted
to HEAD; geometry-experiment.patch preserves the complete candidate diff.
Apply that patch before rebuilding either geometry's research wrappers.
Original logs remain local; rebuildable binary hashes are in ../split_pipeline/removed-build-artifacts.json. The next step is to profile phases
inside the dominant digest kernel rather than further guess block geometry.
