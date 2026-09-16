# Pinning iteration workflow

Work only on the pinning track. All authored changes, experiments, and notes
belong under `candidates/pinning/`. Treat the repository harness, verifier,
scoring rules, problem files, and the subset candidate as read-only.

Before changing the kernel, inspect the current candidate and the latest
promoted pinning submission. Record the baseline source commit and official
score. Preserve third-party license notices.

Use one focused performance change per submission so its result is attributable.
For changes to projective recovery, run `python3 candidates/pinning/check_projective.py`.
This executes extracted candidate expressions using OpenSSL field arithmetic;
it does not validate CUDA compilation, PTX arithmetic, occupancy, or speed.

On a host without CUDA, report the CPU reference run as a correctness smoke
test only. Its scorecard inherits a GPU label from configuration; that label
does not mean the candidate ran on a GPU. Never submit its throughput as a
claimed GPU score. Use the authorized Yukon submission runner for CUDA validation.

After each submission:

1. Record the submission ID, exact candidate source hash, change, tests, and
   status in `ITERATIONS.md`.
2. Inspect the final result and refresh the promoted frontier before another
   substantial experiment. Keep pending, accepted, promoted, rejected, and
   infrastructure-failed results distinct.
3. Compare the verified score with the baseline and the required 1% improvement.
   Read build/verification feedback before attributing a failure to performance.
4. Add a focused local check when feedback exposes a reproducible error. Record
   what was learned and the next experiment; retain or revert the candidate
   according to evidence. Do not alter the judge to improve the score.

Submission notes are public. Use exact model/harness attribution; omit secrets,
personal information, machine paths, and unsupported performance claims. Keep
these project-specific lessons local.
