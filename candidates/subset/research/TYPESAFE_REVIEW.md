# TypeSafe / Jev review

Reviewed 2026-09-16 after both of our tracks had pending submissions:
subset `65fb673d-5014-4ee8-866d-97d972e7b1f0` (PR 60) and
pinning `66c031c6-16e4-484e-ad4d-0e7e0ef3f38f` (PR 74).

Decision: defer integration. Jev could reduce repetitive research triage, but
there is no measured evidence that adding it would materially improve our
challenge workflow. Preserve the user's priority on substantial solver gains.
This was a documentation review; no installation, API evaluation, or project
source transfer to TypeSafe was performed.

## Capabilities and evidence

Jev returns predefined choices, scores and probabilities. It does not generate
code or written reasoning. Its documented role fits classification and routing;
Grok, Gemini and the main agent still supply open-ended proposals and source
review. [System One documentation](https://docs.typesafe.ai/concepts/system-one.md)

The vendor advertises $0.042 per million input tokens, free output and
70–500 ms requests. Its workflow comparisons use other models' probabilities
as reference answers and acknowledge favorable workload and deployment factors.
Those results do not establish CUDA research quality or kernel performance.
[Launch report](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

Schema correctness does not establish factual correctness. Reported confidence
describes the answer distribution; it is not proof that a proposed optimization
is correct or fast. [Confidence documentation](https://docs.typesafe.ai/confidence.md),
[published skill](https://github.com/typesafe-ai/skills/blob/main/skills/typesafe-ai/SKILL.md)

The autoresearch cookbook uses an external LLM to propose questions, Jev to
produce features, and CatBoost to predict wine-review scores. Its results do not
validate performance prediction from our small, interdependent set of CUDA
submissions. [Cookbook](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery.md)

## Potential project use

These are proposed uses, not tested capabilities on our data:

- Tag incoming public notes by mechanism: field arithmetic, table geometry,
  inversion, hashing, memory traffic or host preparation. Permit multiple tags.
- Separate official GPU results, author-reported GPU measurements, compilation
  or CPU checks, and unmeasured hypotheses. Read exact scores from Yukon itself.
- Compare notes with our mechanism ledger to flag potentially new ideas and
  route them for source review. A negative model judgment must not silently
  discard a potentially substantial improvement.

Our immediate bottleneck is obtaining GPU results and validating architectural
tradeoffs. Faster note classification cannot resolve that bottleneck. Submission
state, priority policy, source hashes, benchmark arithmetic and correctness
checks should continue to come from deterministic tools and direct evidence.

## Bounded evaluation if reconsidered

Use 30–50 representative public notes with independently reviewed evidence and
mechanism labels. Include copied ideas, mixed mechanisms, strong unmeasured
ideas, incompatible changes and claims contradicted by source. Keep a held-out
set; tune neither questions nor thresholds on it. Compare against the existing
triage workflow, recording missed substantial leads, false escalations, total
review time, request cost, model version and errors.

Adopt only if the evaluation shows material review savings and preserves all
identified substantial leads; a small clean sample alone is insufficient to
authorize automatic filtering. Initially use advisory ordering with full review
of uncertain cases. No dependency belongs in the scored solver or trusted
benchmark harness. An API test remains future work and requires available
TypeSafe access; existing Grok/Gemini sharing authorization is not TypeSafe
sharing authorization.
