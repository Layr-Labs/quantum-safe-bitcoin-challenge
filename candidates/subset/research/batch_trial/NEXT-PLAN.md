# Follow-up audit and next hypothesis (plan only)

Final delegation result: following a separate instruction to retry with a
smaller payload, Astra extracted 55 relevant lines locally and sent them as
the complete prompt with all Claude tools disabled. DeepSeek returned the
audit in one turn, 6.442 seconds, at reported USD 0.039198 (cap USD 0.10).
All default formulas, per-buffer sizes, and totals matched the audit below:
836 MiB + 512 bytes at 262144 blocks versus 104.5 MiB + 512 bytes at 32768.
This was about 84.7% cheaper than the unsuccessful tool-driven attempt and
returned a reviewable answer. Original output and metadata are preserved in
`claude-sliced-audit-result.json`.

Astra review corrected two conditional statements in that output: disabling
EPOCH_FAST alone removes only d_epoch_group, while disabling EPOCH_GROUPS
removes d_groups and d_epoch_group; neither removes d_first. Also, halving
PAIR_MUL does not halve d_groups' fixed 512-byte addition. The missing
EPOCH_GROUPS/EPOCH_FAST defaults were correctly marked uncertain because
their definitions were intentionally outside the sliced prompt. These are
both enabled in the actual source. The verified main allocation result
supports the existing plan without changing it. No additional GPU work or
production code changes were made.

Update after explicit user approval: one Claude retry passed automatic
approval and ran `deepseek-v4.1-flash[1m]` with the four source files listed
below as its only authorized read scope. It stopped after 12 turns with
`error_max_budget_usd`, before returning an audit body. The requested cap
was USD 0.25; reported cost was USD 0.25578, so the CLI cap did not provide
a strict per-request settlement ceiling. Duration was 24.092 seconds,
permission denials and web searches were both zero. Selected nonsecret
metadata is in `claude-audit-result.json`. No second retry or GPU run was
made, and there is no Claude audit result to endorse or to change the plan.
For future low-budget delegation, pre-extracted narrow source ranges would
avoid paying for repeated tool-driven discovery.

Claude Code delegation was attempted with read-only tools (`Read,Glob,Grep`),
`--permission-mode dontAsk`, and a USD 0.25 cap through the configured local
proxy. The narrow prompt requested only buffer sizes and launch counts.
The sandboxed call failed before inference when Claude could not create
`C:\Users\DELL\.claude\debug\...txt` (EPERM; reported model usage zero).
The escalated call was rejected by automatic approval review because it
would transmit repository source to an external service and write session
data outside the workspace without specific authorization for those details.
No workaround or further Claude call was attempted. The following audit is
Astra's own read-only verification, not a Claude result.

For block count B, current `QSB_PAIR_MUL=4` (`pair_shared.cuh:9`),
`QSB_FIRST_SLOTS=16` (`window_schedule_shared.cuh:5`), epoch descriptors are
64 bytes (`tree.cu:1107`), and group records are 128 bytes
(`epoch_groups.cuh:37`). Allocation sites are in `tree.cu`:

| Buffer | Line | Bytes | B=262144 | B=32768 |
|---|---:|---:|---:|---:|
| d_epochs | 2823 | 256 B | 64 MiB | 8 MiB |
| d_groups | 2828 | 1024 B + 512 | 256 MiB + 512 B | 32 MiB + 512 B |
| d_epoch_group | 2830 | 16 B | 4 MiB | 0.5 MiB |
| d_first | 2835 | 2048 B | 512 MiB | 64 MiB |

These are allocation bounds, not measured live cache working sets. In
particular, group allocation reserves a worst case and only actual groups
are populated; first-state storage reserves sixteen slots while the current
128-window selection uses eight distinct classes.

The normal batch launches five kernels: group producer (`tree.cu:3113`),
epoch producer (3120), first-state producer (3143), consumer (3144), exact
replay (3163). The bounded fallback replaces the first two with one producer.
There is one blocking hit-count-plus-record D2H (3170) and a second D2H only
when more than eight verified records exist (3178). Reducing batch size
therefore multiplies launch/readback frequency for the same candidate count.

Next hypothesis for the official RTX 4090: retain the full production batch,
but compact first-state storage from sixteen slots to eight for the
128-window path. The selected windows structurally contain exactly eight
first-block classes (`tree.cu:2794-2800`), independent of the random signature
bytes; accidental identical messages can only reduce this number. Retain
the runtime class bound check and keep the 256-window path at 64 slots.
This would reduce the d_first allocation from 512 to 256 MiB while preserving
launch frequency, candidates, and all arithmetic. It may improve address/page
locality and reduce translation pressure alongside the fixed-base point
table. It does not halve actual producer stores or guarantee a cache win:
the existing producer already writes only the eight active classes.

Proposed validation, not executed: add a reversible compile-time slot knob;
verify the selected window classes and all producer/consumer strides; compare
same-seed hit sets over a bounded equal-candidate test; then measure an
alternating baseline/candidate comparison on the official GPU or equivalent
RTX 4090. A win must exceed timing noise and retain full hit verification.
No second candidate implementation, GPU run, or submission was made here.
