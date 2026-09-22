# Pinning release status, 2026-09-22

Prepared with GPT 6 Astra, xhigh, Codex. Base 7c3609b, leader 805,428,058/s,
100-bips floor 813,482,339/s at the latest refresh. No local NVIDIA GPU.

The release combines default-on negative-Y seeded MAC, narrow parity with exact
replay, and a dense scalar SHA producer selected by an on-device comparison of
complete pipelines. Source milestone 9e3b410. The promoted cofactor header is
restored and inactive failed research patches are excluded from the package.

Correctness: 34,176 integer MAC triples; 4,096 OpenSSL recovered keys; 111,190
finish executions; 2,712 scalar hash comparisons; 66 buffer-order cases;
9 synthetic policy scenarios. Zero unexplained mismatches. Synthetic policy
rates are not measured performance. Default native and compute52-to-sm89
builds have no spills in any ranked kernel. Fused/producer/EC-only/finish/replay
register counts are 126/40/122/62/72.

The user has explicitly asked to put up the strongest submission and continue.
This release is being packaged for that submission. Read SUBMISSION-STATUS.md
for the full receipt once uploaded. No leaderboard result is claimed before
an official score. SUBMISSION.md describes uncertainty and rollback honestly.
YUKON-SUBMISSION.md records the archive, attribution and readback procedure.
