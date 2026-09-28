# Pinning: overlap SHA preparation with elliptic-curve work

Model: GPT-6 Astra. Agent harness: Codex.
Base: f0e453daaf8b1af848e0bf4afd42fb730018c041; pinning is identical at checked main 46b24ebaa033fb69c7335794b54fd6a156359ec8.
This candidate is prepared locally; it has not been submitted.

The promoted prepare kernel performs SHA-256d before scalar multiplication.
This change produces that digest in a separate low-priority stream on the
finish green context while EC preparation processes earlier sub-batches.
It reuses the first two planes of the existing four-slot state ring. The
dependency is finish(previous use) -> SHA -> EC -> root -> finish. There is no
new state allocation or device-wide hot-path synchronization. An additional
32-byte write and 32-byte read per candidate are traded for overlap.
The official green allocation becomes 108 prepare / 28 finish / 8 shared SMs.

Changes: pinning.cu, QsbCarrier.h, build_carrier.sh and the regenerated
qsb_carrier_sm89.h. EC arithmetic, lookup tables, enumeration, exact host
publication gate, CPU co-grinder, hit format and judge are unchanged. The
native SM89 image is regenerated from the included source using CUDA 12.8,
N=24; cubin SHA256:
52df8f900a3198190aa50474e1d84b1557f3f0cc12f5fe920a55bc5a632b7cf0.
Original GPLv3 and third-party notices are retained.

## Evidence and limits

The available GPU is an RTX 4060 Ti 8GB. The official 21.13 GiB GLV11 table
cannot fit there. Paired local comparisons therefore use the same inherited
compact GLV12 table on both sides; baseline 30/6/shared2, candidate 28/8/shared2.
These are local fixed-work diagnostics, not official leaderboard scores.

- Initial two fresh-seed pairs: +6.8026% aggregate;
  3534/3534 reported hits pass the unchanged CPU verifier.
- Independent confirmation, four fresh-seed pairs in ABBA BAAB order,
  14,935,200,000 candidates per run: 181.955793 -> 194.431107 million completed
  candidates/s, +6.8562%. Per-pair gains:
  6.9066%, 6.8468%, 6.8447%, 6.8268%.
  All 14,360 reported hits verified; exact hit sets match within every pair.
  Timing brackets the whole process, including startup/output.
- Same 28/8/shared2 partition-only control vs candidate: +4.1463%.
  All 3,530 hits verified and paired sets equal. This isolates the producer
  structure from the SM allocation change.
- One full sequence: all 1,244,600,000 active 64-byte prepare states and every
  block root match the original prepare kernel bit-for-bit. This is baseline
  equivalence, not an independent CPU oracle for every curve point.
- Official GLV11 SHA-only kernel: 2,181,680 digests in 160 launches pass OpenSSL
  per native-carrier and static-fallback mode; tail/wrap/canary cases covered.
- Monolithic local fallback and separate CUPTI trace: 162 hits each, identical
  to the audited sequence and independently CPU verified.
- Unmodified CPU smoke passes 3/3. Unmodified GPU harness with the local table,
  normal timeout and default CPU co-grinder passes 1383/1383.
  Its configured GPU label is RTX_4090; actual test hardware is RTX 4060 Ti.
- CUPTI shows SHA/EC kernel interval overlap for 85.97%
  of the search span. This is not an SM occupancy measurement.
- Official default nvcc build succeeds. SHA 44 registers, no spill; finish 64,
  no spill; EC 128 registers and 14 KiB shared, 8-byte compiler stack/spill report.
  SASS has one store/reload pair outside the repeated point-add loop.

The complete official GLV11 runtime has NOT been measured on an RTX 4090.
Its larger table, different L2/SM balance, the added SHA launches, green-context
scheduling and EC spill can change or erase the local gain. No official
3% improvement or reward eligibility is claimed. Public submission needs the
user's explicit approval.
