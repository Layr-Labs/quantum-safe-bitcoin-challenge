# Official result: rejected

Submission `fd16dfa4-4db2-4aee-ab1f-519c564a53b8` completed official RTX 4090 validation on 2026-09-24.
Its score was **500,784,234 verified candidates/s**, versus the
unchanged frontier **623,518,629** (-19.6842%).
Yukon rejected it: `score did not improve current best`.
The official metrics mark verification true and report 71,694 verified hits.

[Official evaluation](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/actions/runs/36068642101)
completed successfully and uploaded score and diagnostics. Workflow success
is not acceptance. The candidate commit is `e7ba81746f12be9c1f46f91fc5491a19f4327724`.
See `post-rejection-frontier.json` and `official-result-latest.json` for
the authoritative Yukon records, and `github-job-latest.json` for job steps.

The local 3090 improvement did not transfer to the official 4090 result.
Aggregate metrics do not establish the cause. Extra workspace traffic or
cache interference remains only a hypothesis. Do not attribute the loss
to either mechanism without relevant device measurements.

The current working candidate is a later four-stage experiment, not this
submitted three-stage version. Its local gains remain unranked. The preserved
SHA-only alternative uses a smaller workspace and passed a direct local
comparison against public HEAD, but it also has no official result. No new
performance claim is justified for either by this rejected run.

No completion marker was written. The official monitor terminated normally
after observing rejection. Any next source submission requires resolution of
the existing automatic-review refusal; the user permission question remains
unanswered. No unauthorized upload or indirect publication was attempted.
