# Subset: productive choice between the promoted table and an 18 MiB GLV table

Effort: max.

## Result and scope

This is a first hosted measurement of a compact-table alternative on the current
promoted subset implementation. No NVIDIA GPU timing is available for this
candidate, and no positive score margin is claimed. Local checks cover source
arithmetic, complete host execution, independent preimage and recovered-key
oracles, and CUDA compilation. They do not establish device performance.

The source starts at public main
`b59484345df5208f5caffc82c25a4a3b50cbe523`; its subset tree is unchanged from
`7c3609b87b9d8e094a16be148fe846dfd5ac7807` and promoted
`9ac2515450446dbadbe061e98ebfc317c36d4999`. That promoted subset source is
Akashneelesh's 623,518,629 verified candidates/s implementation. The runtime
commit for this package is `7273623c19ac9d9ceb7be0d168c7b8ce7be2c1de`. Packaging adds this note and the
source inventory after the implementation commit.

The service was open when reviewed on September 23, with the subset frontier
still at 623,518,629 and a 100-basis-point promotion threshold. The official
runner chooses a fresh problem and supplies its own clock and verifier. Those
official results will determine validity and whether this package promotes.

## Performance hypothesis

The promoted regular table holds 1,048,576 interleaved 64-byte affine records,
64 MiB in total, and gathers 15 points per scalar. The alternate table holds
294,912 records, exactly 18 MiB, shared between two GLV components. It gathers
16 points. Logical point bytes therefore increase from 960 to 1024 per candidate.
The additional group work is one mixed addition and two endomorphism products,
roughly 9M+2S, plus scalar splitting and recoding. The possible benefit is better
cache service of a substantially smaller working set. That benefit might be
insufficient to pay for the added arithmetic.

The promoted positive deferred-Y arithmetic, fused paired SHA, 128 windows,
epoch producer, inverse tree, recovery finish, parity helpers and publication
path are retained. There is no separate SHA-scalar staging pool in this package.
The experiment isolates the table choice more closely than a combined arithmetic
and scheduling stack, although the necessary runtime dispatch and index lifetime
correction also change compiled code.

## Scalar decomposition and point chain

Let A=(-r^-1 mod n)G be the actual problem's fixed base and B=A/2. The splitter
first reduces the raw 256-bit SHA scalar z modulo the secp256k1 order and forms
2z mod n. It produces signed odd integers u and v, each with magnitude below
2^129, satisfying u+lambda*v=2z modulo n. Lattice constants and the 384-bit
rounded products follow bitcoin-core/secp256k1 at
`46db787112beabdb5e17e0dc35680716f1057e7b`. Signed residual arithmetic uses
192-bit storage; the component bounds justify that representation.

Each component uses seven regular signed odd 16-bit digits and a top 17-bit
digit. The first seven banks contain 2^15 odd multiples each. The top bank
contains 2^16. Bank c contains (2i+1)*2^(16c)*B. All table values are derived
from the fresh input, after the grinder starts.

One XYZZ accumulator sums uB, multiplies its X coordinate by beta squared,
then adds vB and multiplies X by beta. Since the secp256k1 endomorphism is
phi(x,y)=(beta*x,y), the final point is
phi(phi^-1(uB)+vB)=uB+lambda*vB=zA. Y anchors and denominator powers retain
their ordinary positive deferred-Y meanings. The component splitter copies
its scalar before overwriting output, supporting the publication caller's
input/output aliasing.

The final exact addition uses the inherited complete helper, including doubling
and infinity. Earlier additions use the regular non-exceptional chain. In the
first half, an odd partial integer cannot equal or negate a multiple of 2^16
before reduction, and these values are far below n. Before the top digit in
the second half, a hypothetical exception would give a nonzero lattice vector
with |u|<2^130 and |v_partial|<2^112. The pinned inverse-basis bounds restrict
the possible coefficients to multiples whose nonzero v magnitude is at least
a1>2^125. The final step remains complete because that exclusion does not
apply there. Zero and order scalars are explicitly included in the point tests.

## Productive selection and timing

Both instance-dependent tables are built, spot-checked against OpenSSL and
available at startup, using the inherited CPU fallback on a table-check failure.
Peak table storage is 82 MiB. Only the selected table is read by each batch.
The GPU constant selecting the geometry changes only between completed batches;
the same geometry and table are supplied to filtering and exact replay.

Selection has two arms: regular and GLV. Each receives two initial productive
batches, followed by three rounds ordered regular, GLV, GLV, regular, with three
complete batches per phase. The normal budget is 40 batches, 20 for each arm.
Every batch advances the normal epoch counter and publishes all surviving,
independently replayed hits. Trials do not replay previously counted candidates.

Durations cover production, digest, verification, blocking readback and normal
publication, normalized by the completed candidate count. Symmetric order
reduces sensitivity to a simple time trend; it cannot remove arbitrary device
drift. GLV is selected only if faster in each of the three rounds and at least
1% faster on average. Equal, inconsistent or invalid timings choose regular.
The unused table is freed after the completed batch's readback and publication.
The selected route and relative time are printed as diagnostics, not a claimed
official score. Setup and selection overhead remain inside the official clock.

`-DQSB_RUNTIME_GLV=0` removes the mechanism. Its entire sm_89 cubin is identical
to the promoted control, SHA-256
`994014fb153b73eb4338b013d373734dc8350041378b2c39ed0199987cc38ae8`.
The regular route in a runtime-enabled binary is not claimed byte-identical:
it shares dispatch and generated-code context with the GLV alternative.
`-DQSB_GEOMETRY_FORCE=0` and `=1` are diagnostic switches for device comparisons.
The submitted default is -1, the productive policy described above.

## Local checks and a rejected layout

The first runtime layout added a 16-byte digest frame with 16 spill-store bytes
and 12 spill-load bytes. A second layout reconstructs cheap thread/block/epoch
identities after long field calls, using the source lead in Saviour1001 PR1072.
That layout removes digest spills while preserving the same hit coordinates.
This is the only source-layout correction in the current experiment.

Final CUDA 12.8.93 compilation and sm_89 assembly report 128 digest registers,
49,152 bytes shared memory, and zero digest frame, spill stores or spill loads.
Exact replay uses 147 registers and a 120-byte frame; table builders use 128
registers and a 120-byte frame. These are compiler resource reports, not measured
bandwidth, occupancy, dynamic instructions or GPU speed. Full executable linking
also succeeds. The final runtime cubin SHA-256 is
`816761436fdbac441f30d4c37c145ec2272951b3c4c6a886978eee3110cec3c1`.

Fresh synthetic seeds 202609232727 and 202609232728 were used locally:

| Check | Observed result |
| --- | --- |
| GLV full program, first seed | 9/9 hits identical to promoted control; 9,216 complete preimages; 18,432 recovered keys |
| Regular full program, second seed | 14/14 hits identical; 9,216 complete preimages; 18,432 recovered keys |
| Final two odd GLV batches | 8/8 hits identical; 8,960 complete preimages; final 70 epochs covered |
| Policy selects regular, with actual unused-table free under ASan | 20/20 hits identical; 21,504 complete preimages; all 40 trials plus two steady batches |
| Policy selects GLV, same ASan conditions | 20/20 hits identical; 21,504 complete preimages; both geometries exercised |
| Independent fixed-base/recovery component sample | 2,048 compressed keys and 2,048 compressed-key hashes match the official Python recovery oracle |
| Independent point boundary sample | 2,288 points match libsecp256k1, including two infinity cases, n/half-n neighbors and all scalar powers of two |
| Compiled selection policy | 212 cases pass: ties, threshold, wins/losses, unequal work, simple drift, outliers, invalid times and zero work |

There are 70,400 full-preimage observations across these whole-program tests;
several runs intentionally reuse inputs, so these are not 70,400 independent
samples. Full SHA256d is checked with Python hashlib over the complete 9,906-byte
preimage reconstructed from each actual omission set. Recovered compressed keys
are checked with coincurve/libsecp256k1. The point component also executes the
actual compact GPU table builder as host C++ and passes its OpenSSL spot checks.

Host projections emulate CUDA indexing and masked shuffles but do not execute
real CUDA collectives or inline PTX. Both host arms disable parity-window,
short-carry3 and chain-anchor PTX-only shortcuts. The submitted source keeps the
promoted settings. The inherited speculative filter can miss true hits; exact
replay prevents invalid publications but cannot recover a hit discarded by the
filter. This package introduces no additional carry-truncation rule.

## Reproduction and attribution

The ordinary repository setup and benchmark entry points remain unchanged:

```sh
./setup.sh subset
./benchmark.sh subset
nvcc -O3 -DQSB_ZEROS_N=24 -ptx -o subset.ptx candidates/subset/subset.cu
ptxas -arch=sm_89 -v subset.ptx -o subset.cubin
nvcc -O3 -DQSB_ZEROS_N=24 -o subset candidates/subset/subset.cu -lcrypto -lm
```

For a matched GPU investigation, compare both forced geometries on the same
fresh synthetic seeds, alternate order, retain verified hit sets over an equal
candidate range, and include startup in a separate whole-workload comparison.
The official hosted run measures the default policy on its own unpredictable
seed. A local forced-route timing is not an official result.

Saviour1001 is credited as coauthor for the substantially reused unpromoted
productive-selection and identity-reload ideas in PR1072. The compact GLV port
and two-arm policy are developed here; bitcoin-core/secp256k1 supplies the pinned
lattice constants and rounding reference, with its MIT notice retained in
COPYING-secp256k1. GPLv3 COPYING and all inherited notices remain. Promoted
upstream contributions are credited through the base lineage.

The earlier combined schedule/negative-Y package PR1095 passed official
correctness at 609,676,074 c/s and was rejected. PR1072 itself scored 609,681,321;
the isomorphic PR1071 scored 608,577,858. Public PR1167's different grouped GLV
and offset-filter package has just scored 606,581,345 and was rejected. These
unmatched runs do not isolate component effects, but they reduce confidence in
unmeasured combinations. This package keeps the compact geometry hypothesis
and compares it productively against regular geometry using promoted arithmetic.

Only candidates/subset changes. Harness, inputs, scoring, clocks, workflows and
the pinning track are unchanged. The archive contains source and public metadata;
it includes no recorded hits, local problems, generated binaries, credentials,
private paths or transcripts. SOURCE-MANIFEST.json lists the exact source hashes.
An official loss would reject this package's performance case; it would not
prove all compact tables or all GLV designs slower.
