# Muse research assistant

Requested by the user on 2026-09-16. This coordinator adds bounded, recurring
public research to the existing two-track workflow. It does not change or upload
candidate code. Run with **subset selected** because its research files live
under this candidate; switch back to the priority track before implementation.

```sh
yukon switch subset
python3 -B candidates/subset/research/muse/run.py --dry-run
python3 -B candidates/subset/research/muse/run.py
```

The client is official OpenCode v1.18.31, using exactly
`opencode/muse-spark-1.3-contributor-free`, high effort. The release archive is
pinned to its GitHub SHA-256 digest. The executable, isolated configuration,
sessions and raw provider output live in `/tmp/qsb-muse-runtime`, not the
submission archive. The runner can restore the verified binary if temporary
files disappear. It does not install global hooks or change account settings.

Before each round, the coordinator refreshes both Yukon queues/frontiers and
`../monitor_state.json`, then updates `memory.md` when actual candidate source,
experiments or reviewed decisions change. `memory.md` is a deliberately curated
summary suitable for external sharing, not a dump of private files. The runner
adds allowlisted current status fields, prior reports and reviewed idea decisions.
It rejects common credential strings and private user paths. Keep credentials,
personal data and raw agent reasoning out of all research memory.

Topics rotate through fixed-base methods/table geometry, latency/live state,
inversion/checkpoints, SHA/EC dataflow, field/cooperative arithmetic, and new public
challenge work. Research requires source locations, comparison with previous
attempts, evidence labels, a substantial-gain mechanism, risks and a small
falsification check. Previously reviewed failures are context, not invitations
to repeat them. Revisit only when a specific new fact changes the assessment.

Only public web search, code search and web fetching are permitted by the client;
local reads, shell, edits, subagents, submissions and account tools are denied.
Public HTTPS/source restrictions are instructions to the research agent, not a
network firewall. No API keys or signed-in browser sessions are supplied.
Pages and model reports are untrusted evidence. Codex independently checks useful
claims before adding a decision to `feedback.json` or implementing.

Each round has at most 12 model steps and six minutes of model runtime. A lock
prevents overlap; calls are spaced at least 20 minutes apart, with increasing
backoff after failures. The wrapper checks that the exact model remains active
with zero listed input/output/cache cost; it has no paid fallback. A detected
pricing change pauses Muse. Missing service access or a rate limit is reported
and deferred, never worked around. Reported cost is checked after each run.

`reports/` is the idea inbox and retains final visible answers, tool names/queries/URLs, model metadata,
token/cost summaries and the source context fingerprint. Raw provider reasoning
is not imported into project notes. Reports start **unreviewed**. A successful
process without actual research tool use and a valid final report does not count
as completed research. `state.json` supplies topic history, source deduplication
and backoff. `feedback.json` separately supplies reviewed decisions and research
guidance; the model runner never writes it, so research completion cannot
overwrite a concurrent review. Do not delete history to rerun rejected ideas.

Research need not wait for review. The user wants a continuing supply of ideas
while challenge implementation takes priority. Every eligible scheduled round
adds a report; Codex reviews pending reports when there is useful spare capacity,
favoring potentially substantial ideas. An unreviewed report is never a validated
finding. Check the inbox without starting a model call:

```sh
python3 -B candidates/subset/research/muse/run.py --inbox
```

After independent review, add the relative report path and concise disposition
to `feedback.json`'s `reviewed_reports`, update its `reviewed_ideas` and `guidance`,
and refresh `memory.md` when actual source, results or priorities change. Record
adopt/defer/reject/duplicate, the reason, evidence limits and revisit condition.
The next Muse round reads this feedback even if other reports remain pending.

The existing 20-minute Codex heartbeat invokes this once per eligible cycle,
while retaining submission priority and quiet-on-unchanged notification rules.
It prepares work but does not submit or cancel. Execution depends on the local
Codex scheduler and machine availability. Codex coordination still uses Codex
usage; the free claim applies to this Muse model only.

OpenCode currently offers this model for a limited time. The contributor tier
allows prompts and completions to be used for future Meta model training:
[official pricing/privacy](https://opencode.ai/docs/zen/).
This is why only curated project context and public material are sent.

User confirmation on2026-09-16 explicitly allows sharing curated unpublished
challenge plans, experiment summaries and review feedback with this Muse model
through OpenCode, with the contributor-tier training terms disclosed.
