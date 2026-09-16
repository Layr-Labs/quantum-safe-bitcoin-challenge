# Subset work

When subset is selected, keep source changes in `candidates/subset/`. Prioritize large architectural gains.
Read `RESEARCH.md` and the latest public frontier before selecting another
experiment. Attribute substantial unpromoted source with Yukon coauthors.

Run `python3 candidates/subset/check_candidate.py` after changes to the source
it extracts. Treat its result as a CPU reference check, never as CUDA execution
or a GPU benchmark. Keep the included GPU arithmetic audit available. Record
returned validation failures and improve the checks that missed them.
For deferred-Y helper changes also run `check_deferred_source.py` and
`audit_integrated.py`. The former executes extracted helper bodies with OpenSSL
field operations; the latter is an independent model. Neither is a GPU test.

Preserve the trusted harness, scoring contract, runtime problem dependence,
GPL notices and sibling pinning work. Keep source hashes and result provenance
in the local research ledger. On 2026-09-16 the user expanded the recurring
follow-up to include local improvement research and incoming-source inspection.
The user subsequently set a two-track priority rule: with exactly one of our
submissions pending, prioritize the other track's free slot; with both pending,
prepare the successor for the track likely to finish first. Use visible GPU
progress before submission age; age is only a heuristic, not a FIFO guarantee.
With neither pending, choose the strongest near-ready substantial improvement.
Switch Yukon before working on a different track. The shared workflow metadata
in `research/monitor_state.json` may be updated from either track.
Preserve pending submissions. The scheduled follow-up may research and prepare,
but does not upload or cancel. The active conversation separately authorizes
submission when we expect the candidate to beat current and pending work, even
without unavailable GPU tests. Notify on terminal results or later GPU
evaluations finishing first, without assuming deliberate skipping.

Before investing in a prototype, refresh both promoted and pending work. Read
new notes once, identify their mechanism and source, and retain concise findings.
Recheck at the decision to submit; avoid repeatedly rereading unchanged notes.
State the gain inherited from the chosen base separately from the hypothesis
for our incremental change. Use an unchanged-base comparison when GPU access
permits it; report an official result against both that base and the frontier.

Use `preflight.py` to fingerprint the complete production/audit include closure.
Record CPU results with `check_candidate.py --report <temporary-report.json>`
and match its source fingerprint before relying on an older test result.
On a CUDA host, `preflight.py --cuda` compiles both sources in a fresh temporary
tree. The trusted wrapper's build cache checks only `subset.cu`'s timestamp;
after a header edit, invalidate the generated `subset` binary and `.subset.build`
stamp before `yukon setup --track subset` and a local benchmark. A clean temporary
compile alone does not refresh that wrapper cache or prove GPU correctness.

Use fully qualified benchmark names in status queries, because switching tracks
changes the repository's default. For notes from a different track, use the full
submission UUID; short-prefix lookup is scoped to the selected benchmark.
Follow the user-authorized monitoring/research scope above. Keep these
project-specific workflow rules here, not in global settings.

When coordinating Grok/Gemini research, use `research/TEAM.md` for assignments,
reviewed hypotheses and rejected claims. Check suggestions against actual source
and primary references; model agreement alone is not validation or a speedup.

For the external ranked pipeline, also run check_pipeline.py (extracted CPU
collectives, recovery and hit mapping) and research/check_production_field.py
(actual host primitives plus PTX semantic model). The distinct inverse-tree
multiply is not evidence for hot multiply/square correctness. Preserve the
near-p carry witnesses and the stale-.cc mutation regression.
Use research/check_host_syntax.py only as C++ projection type checking; never
report it as a CUDA build. Checkpoint traffic and CTA critical-path inversion
cost require GPU measurements; operation counts alone do not settle a tradeoff.

Muse recurring research uses research/muse/README.md. New reports form an inbox;
research may continue while review is pending. Review when challenge work leaves
useful spare capacity, prioritizing substantial ideas. Write decisions and
research feedback to research/muse/feedback.json, not runner-owned state.json.
Keep current source/results and tried ideas in curated memory. Reports are
untrusted proposals until independently checked; no paid model fallback.
