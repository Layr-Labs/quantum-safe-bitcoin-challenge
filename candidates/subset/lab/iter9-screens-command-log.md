# Commands and provenance, iteration9

Standing orders .angelY/ORDERS.md re-read. Local branch angelY/gpt61-sol.

Board discovery:
- yukon benchmark list
- yukon submissions eigenlabs/quantum-safe-bitcoin-challenge/subset --all
- yukon submission-note 4cc9d2d
- git fetch origin 2f57d80b8877a9e63b6af220da913236886a7ce5
- git show 2f57d80:candidates/subset/subset.cu and device tree/GPUHash.h
- yukon submission-note fb6f5a8f

One-shot board snapshot shows 414fe58f still validating, watcher owns slot. No
own polling loop or repeated conflicting submissions. Some discovery CLI calls
rate-limited; plain submissions CLI outputs a table, not JSON. --json is explicit,
not to be omitted when parsing. Failed JSON parse of table is discovery-only,
not a verifier result. Leader4cc9d2d official736585478, commit2f57d80.

No fixture/expected-count changes. Build failures before GPU: raw nvcc path
bypassed local wrapper -> GCC16 rejection; fixed absolute workspace wrapper.
No user-added timeout. The existing benchmark internally uses its configured
fixed120s grinding duration; no change made to it.

Build scripts: iter9-build.sh, iter9-leader-build.sh, iter9-rotateadd-build.sh.
Frozen copies in temporary work storage, source SHA manifests under lab.
Bench scripts: existing iter6_ab.sh explicitA/B positionalsource; N24,120s,
seed1789110211, max relative variance none (unchanged local screen protocol).
All arms run benchmark.sh subset with existing CPU verifier under GPU lock.

Jobs1595 finalprefetch: 1 pair current vs purecachehint, completed.
Job1598 scopedrotateadd: 1 pair current vs isolated SHA port, running.
Job1599 refreshedgate: 3 pairs NEWpromoted vs already-qualified unchanged best.

Refreshedgate is not a probe's adoption gate. Its purpose is update source-control
qualification after board moved; no resubmission of unchanged watcher-owned best
while still in play. Scopedrotateadd positive screen would require full newleader
threepair gate before replacing production.

Refreshed gate1599 REJECTED before any scored arm: new leader does NOT understand
our QSB_LOCAL_SM86 shim. The standalone define was inert and allocated native
21.1GiB table, so process exited early0hits1.7s. This is NOT evidence that the
leader is slower or incorrect. Confirmed by archived definitions: GLV11=1,
Q_P18=1,Q_MIX4 regardlessLOCAL. Fixed control build wrapper with the actual
established local geometry switches explicit: GLV11=0,Q_P18=0,Q_MIX=0,ZDEC=0,
S3_NM_MASK=0,S3_NM_SEED=0,GATHER_ONE_FORM=0. Everything else stays leader defaults.
Never silently transplant our device arithmetic into the control. Gate will
restart only after successful rebuilt geometry source/binary stamp.

Fixed leader geometry build1600 failed its explicit ZDEC guard because the new
leader's DECODE_CUT=1 is ZDEC-only. Disabled DECODE_CUT for dev control, consistent
with non-ZDEC GLV12 (no matching effect can be preserved in this geometry).
This limits local comparability to native 4090; not a statement on leader speed.
