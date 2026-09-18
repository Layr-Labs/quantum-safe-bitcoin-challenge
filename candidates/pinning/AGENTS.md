# Pinning iteration workflow

When pinning is selected, work only on the pinning track. All source changes, experiments, and notes
belong under `candidates/pinning/`. Treat the repository harness, verifier,
scoring rules, problem files, and the subset candidate as read-only.

User priority policy (2026-09-16): if only one track has our pending submission,
prioritize the other track. If both are pending, prepare the successor for the
track likely to finish first, using visible evaluation progress before submission
age. Age is only a heuristic; Yukon does not expose a guaranteed FIFO queue.
If neither is pending, choose the strongest near-ready substantial improvement.
Preserve pending evaluations. Switch Yukon to the selected track before editing.
The shared monitoring ledger is `candidates/subset/research/monitor_state.json`;
its workflow metadata may be updated from either track. Scheduled follow-ups
may research and prepare. User priority update (2026-09-17): as soon as our
pinning candidate is qualified and has stronger evidence than every relevant
pending pinning submission, refresh the frontier and pending set one final time
and submit it immediately without requesting another confirmation. The active
conversation authorizes that submission even when local official-GPU testing is
unavailable. Do not submit a known-invalid candidate, unchanged duplicate, or a
candidate with weaker evidence than a relevant pending submission. Never label
an expected advantage as a measured performance win.

Before changing the kernel, inspect the current candidate and the latest
promoted pinning submission. Record the baseline source commit and official
score. Preserve third-party license notices.

The user wants large improvements only. Use roughly 25% or greater verified
throughput improvement as a working research target, with larger gains preferred.
This target is a prioritization rule, not a measured result or a user-specified
numeric threshold. Pursue structural reductions in dominant work, such as
batching inversions or improving fixed-base multiplication and table traffic.
Require a concrete cost argument or profiling evidence for the expected gain.
Do not spend submissions on isolated minor launch, batch-size, or formatting
tweaks, or chase the benchmark's 1% acceptance floor. Small supporting changes
are appropriate only when necessary for a substantial improvement or correctness.
Label predicted gains separately from official measured results.

Build a coherent candidate from the strongest compatible public work, recording
the inherited gain separately from our incremental hypothesis.
For changes to projective recovery, run `python3 candidates/pinning/check_projective.py`.
This executes extracted candidate expressions using OpenSSL field arithmetic;
it does not validate CUDA compilation, PTX arithmetic, occupancy, or speed.

Local CUDA compilation is available through the ARM Linux VM described in
`research/LOCAL_CUDA.md`; use `research/compile_local.py` for source-bound
native builds and resource checks. This Mac still has no NVIDIA GPU, so those
builds do not establish GPU execution, runtime correctness or speed. Stop the
VM between compilation sessions to release its memory.
Before comparing native reports, match the complete control include hashes to
the intended source. Similar kernel names or a previous successful build do
not establish that the report describes the selected control. Treat static
instruction counts separately from dynamic work and measured throughput.

When using the CPU reference, report its run as a correctness smoke
test only. Its scorecard inherits a GPU label from configuration; that label
does not mean the candidate ran on a GPU. Never submit its throughput as a
claimed GPU score. Use the authorized Yukon submission runner for CUDA validation.

After each submission:

1. Record the submission ID, exact candidate source hash, change, tests, and
   status in `ITERATIONS.md`.
2. Inspect the final result and refresh the promoted frontier before another
   substantial experiment. Keep pending, accepted, promoted, rejected, and
   infrastructure-failed results distinct.
3. Compare the verified score with the baseline and the large-improvement target;
   report the benchmark's required 1% improvement separately as its acceptance floor.
   Read build/verification feedback before attributing a failure to performance.
4. Add a focused local check when feedback exposes a reproducible error. Record
   what was learned and the next experiment; retain or revert the candidate
   according to evidence. Do not alter the judge to improve the score.

Submission notes are public. Use exact model/harness attribution; omit secrets,
personal information, machine paths, and unsupported performance claims. Keep
these project-specific lessons local.
