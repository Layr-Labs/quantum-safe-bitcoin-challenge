# Pinning: closed-PR research and composition review

Review date: September 24, 2026.
Base: `1fe5a8e40008befcd917668ea9b1a23c6ee590c4`.

The full English implementation and validation note is [SUBMISSION.md](SUBMISSION.md).
This file records the closed-PR investigation. Its evidence snapshot is
[PR-REVIEW.json](PR-REVIEW.json).

## Findings

The review identified **55 closed, unmerged PRs** with an official score above
the record at evaluation time but below the required 1% improvement. Every row
has both a matching score comment and a confirmed 100-basis-point rejection
comment. Gains are recomputed against each contemporary record, not against the
challenge's original baseline or the current record.

The closest historical rejection was
[PR #743](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/743):
+0.996957%, about 23,694 candidates/s below its then-current promotion threshold.
Its multiply-tail work has already entered the current implementation lineage.

The most relevant recent candidate was
[PR #1264](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1264):
882,096,418 verified candidates/s versus 881,273,403, or +0.093389%.
It uses the current GLV12 geometry and combines lean GLV coefficient work with a
further field-multiply approximation. The composition prepared here imports
only its `GLVScalar.cuh` changes. Its official whole-package result does not
isolate the imported group's performance.

Inspection found overlap among several apparent improvements:

- #1257 reuses #1256; inspected executable files differ only in a GLV comment.
- #1249 and #1237 have the same inspected executable code apart from comments.
- #1205's slot reuse, refill ordering, register seed, and multiply work are
  already inherited by the promoted base.
- #1139's priority-root mechanism is already present; its helper matches the
  current implementation after line-ending normalization.

These are not independent improvements whose percentages can be summed.
The two additional changes developed in this session target different work:
sparse transfer of the table's existing OpenSSL samples during startup, and
reuse of the message hash and generator multiplication between exact host-gate
attempts. Their CPU checks and CUDA builds passed, but their GPU performance
has not been measured locally.

## Reused artifacts and attribution

The imported GLV file is from #1264, commit
`d6ee1d8a37ed6f8a74bba42e4878c9825b4022a7`.
The coefficient audit is from #1256, commit
`14b41b6829b5476198df76f1ed5ea0ca224656f0`.
The unpromoted donor lineage is kaankolcu / Portablelle, composed by ItlaStudent
in #1264. Public source links and detailed credit are in the English submission
note. Existing licenses remain present.

The inherited submission description referred to an older GLV14 composition.
It has been replaced with the current English note; the old text remains
available in Git history. Other inherited research reports remain historical
records and should not be treated as measurements on this composition.

## Complete inventory

Each PR link below goes directly to the threshold-rejection comment. Scores
are official historical results, not local reproductions. Short submission
identifiers and source commits can be found in the accompanying JSON.

| PR | Submission | Record at evaluation | Score | Improvement |
|---|---|---:|---:|---:|
| [#1264](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1264#issuecomment-5803524418) | `4fe6a08` | 881273403 | 882096418 | +0.093389% |
| [#1257](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1257#issuecomment-5801801556) | `6d0cd8a` | 826926066 | 830188475 | +0.394522% |
| [#1249](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1249#issuecomment-5800196184) | `3cfd54c` | 826926066 | 827827523 | +0.109013% |
| [#1237](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1237#issuecomment-5798441811) | `d27f252` | 826926066 | 829084805 | +0.261056% |
| [#1205](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1205#issuecomment-5793662188) | `5a37cad` | 826926066 | 829282307 | +0.284940% |
| [#1174](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1174#issuecomment-5789887371) | `c48d366` | 813651852 | 814278773 | +0.077050% |
| [#1160](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1160#issuecomment-5788508260) | `3a803f9` | 813651852 | 817160797 | +0.431259% |
| [#1151](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1151#issuecomment-5787622894) | `a7baa3c` | 813651852 | 815948628 | +0.282280% |
| [#1139](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1139#issuecomment-5786694254) | `59b3693` | 813651852 | 814080739 | +0.052711% |
| [#1091](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1091#issuecomment-5781177563) | `1bffc5b` | 805428058 | 810583657 | +0.640107% |
| [#1084](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1084#issuecomment-5780487304) | `0b204c4` | 805428058 | 810053330 | +0.574263% |
| [#1070](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1070#issuecomment-5779041090) | `dea1ae4` | 805428058 | 808718163 | +0.408491% |
| [#1063](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1063#issuecomment-5778283691) | `55926af` | 805428058 | 810314192 | +0.606651% |
| [#1050](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1050#issuecomment-5776026522) | `d193bed` | 805428058 | 809250751 | +0.474616% |
| [#1031](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1031#issuecomment-5774163566) | `a8ef506` | 805428058 | 805527392 | +0.012333% |
| [#1013](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1013#issuecomment-5772447491) | `3c124ec` | 805428058 | 809952202 | +0.561707% |
| [#999](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/999#issuecomment-5771600050) | `855abbe` | 805428058 | 808035987 | +0.323794% |
| [#957](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/957#issuecomment-5768581094) | `883629e` | 797446582 | 804457861 | +0.879216% |
| [#953](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/953#issuecomment-5768100266) | `ed148db` | 797446582 | 801547267 | +0.514227% |
| [#948](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/948#issuecomment-5767560846) | `4544260` | 797446582 | 801065927 | +0.453867% |
| [#927](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/927#issuecomment-5765674920) | `8740a30` | 797446582 | 804598773 | +0.896887% |
| [#919](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/919#issuecomment-5765048858) | `21d37ea` | 797446582 | 801426223 | +0.499048% |
| [#866](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/866#issuecomment-5756171442) | `f4071dd` | 789011576 | 789268199 | +0.032525% |
| [#855](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/855#issuecomment-5755472626) | `464fad5` | 789011576 | 790907535 | +0.240295% |
| [#847](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/847#issuecomment-5755126450) | `be352be` | 789011576 | 792579857 | +0.452247% |
| [#837](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/837#issuecomment-5754506019) | `32ec88c` | 789011576 | 789405152 | +0.049882% |
| [#803](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/803#issuecomment-5752763196) | `f7e4dde` | 789011576 | 792667656 | +0.463375% |
| [#772](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/772#issuecomment-5751320499) | `cbce550` | 789011576 | 791077271 | +0.261808% |
| [#765](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/765#issuecomment-5750999445) | `b0fbfb1` | 789011576 | 789394272 | +0.048503% |
| [#747](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/747#issuecomment-5750210109) | `6fe3a56` | 778624395 | 779526447 | +0.115852% |
| [#743](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/743#issuecomment-5749960911) | `960da80` | 778624395 | 786386945 | +0.996957% |
| [#719](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/719#issuecomment-5749267526) | `358b5b9` | 766671138 | 769576233 | +0.378923% |
| [#705](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/705#issuecomment-5748812809) | `1660605` | 766671138 | 768652999 | +0.258502% |
| [#700](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/700#issuecomment-5748585203) | `58005ee` | 766671138 | 773240433 | +0.856860% |
| [#686](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/686#issuecomment-5748150292) | `fcc1754` | 766671138 | 769584560 | +0.380009% |
| [#653](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/653#issuecomment-5746516675) | `0d7aac0` | 766671138 | 770009416 | +0.435425% |
| [#202](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/202#issuecomment-5721346162) | `b0c481f` | 677121678 | 680758255 | +0.537064% |
| [#185](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/185#issuecomment-5719720037) | `b7e16a4` | 677121678 | 677759287 | +0.094165% |
| [#182](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/182#issuecomment-5719393013) | `835b240` | 677121678 | 681490123 | +0.645149% |
| [#169](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/169#issuecomment-5718403280) | `406637e` | 677121678 | 681859905 | +0.699760% |
| [#168](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/168#issuecomment-5718050797) | `0227bc3` | 677121678 | 679373443 | +0.332550% |
| [#163](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/163#issuecomment-5716560836) | `bdd7052` | 667612737 | 667957345 | +0.051618% |
| [#160](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/160#issuecomment-5715740761) | `a9aae4a` | 667612737 | 668536201 | +0.138323% |
| [#158](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/158#issuecomment-5715338098) | `437038b` | 667612737 | 668744362 | +0.169503% |
| [#155](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/155#issuecomment-5714947829) | `acca605` | 667612737 | 668745396 | +0.169658% |
| [#136](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/136#issuecomment-5712908193) | `ecbfa55` | 653505529 | 656730338 | +0.493463% |
| [#129](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/129#issuecomment-5711904784) | `0c6f4c8` | 653505529 | 657885190 | +0.670180% |
| [#123](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/123#issuecomment-5711246804) | `1a22808` | 653505529 | 659859051 | +0.972222% |
| [#104](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/104#issuecomment-5709076840) | `e0f8c13` | 644546620 | 644695963 | +0.023170% |
| [#95](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/95#issuecomment-5708196423) | `7f965b4` | 644546620 | 647553774 | +0.466553% |
| [#89](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/89#issuecomment-5707561390) | `a91746c` | 644546620 | 647064073 | +0.390577% |
| [#84](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/84#issuecomment-5707103164) | `36d4266` | 644546620 | 647595523 | +0.473031% |
| [#64](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/64#issuecomment-5705250543) | `072d9b8` | 644546620 | 647007541 | +0.381807% |
| [#61](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/61#issuecomment-5704940053) | `437253c` | 644546620 | 646395221 | +0.286806% |
| [#53](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/53#issuecomment-5704278462) | `e582bda` | 644546620 | 645336311 | +0.122519% |
