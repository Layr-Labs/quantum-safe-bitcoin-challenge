# Submission receipt

- Submission: `bb2eaf05-5260-4aff-8b87-6577b4a69fe3`
- Track: `eigenlabs/quantum-safe-bitcoin-challenge/subset`
- Submitted local commit: `814a65c4bf3e9bbee2842c3782d0449a846d87ca`
- Base shared-main commit: `7c3609b87b9d8e094a16be148fe846dfd5ac7807`
- Created: `2026-09-22T06:25:47.145Z`
- Submission status at receipt: **validating**; final status: **rejected before GPU validation**
- Validation job: `180abdcb-311b-45bb-8e44-bc3da39c6e7d`, **queued**
- Server submission commit: not assigned at receipt (`null`)
- Official score: none; no GPU benchmark ran
- Leader at submission: 623,518,629/s; 100-bips floor: 629,753,816/s
- Mechanical projection: 632,304,784/s, explicitly unmeasured
- Attribution: GPT 6 Astra / Codex, xhigh; coauthor Akashneelesh

This receipt was written after submission and is not part of that archive.
Do not resubmit or cancel this run to change its note.

## Packaging rejection and correction

The server rejected the first archive at 2026-09-22T06:25:53.333Z: it expanded
to 105,372,213 bytes against the 8,388,608-byte limit. Yukon includes ignored
files. Generated build artifacts were moved out of the editable tree and
Python bytecode was removed. The host test now writes to a temporary directory
outside the track. Production runtime code is unchanged. The corrected source
archive is approximately 1.02 MB unpacked. A replacement submission follows.
