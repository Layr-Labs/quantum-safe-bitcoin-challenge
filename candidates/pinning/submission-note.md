# Pinning: one tail schedule for four sequences, chain and finish rewrites, and exact device cuts, on b9736ce1

This package starts from the pinning record `b9736ce1` (commit `8d07d3eb`). It changes nine things. Items 1 to 3 are
fkiene's, from `285108a0` (PR 2643).

1. A prepare block walks 32 locktimes of four sequences, one warp per sequence. One warp expands the tail block's message
   schedule (W16 to W63) once per locktime, and all four warps use it (`QSB_ASICBOOST` 1, `QSB_AB_K` 4). This is the
   observation behind Bitcoin mining's AsicBoost: the sequence enters the tail block only through the midstate, so the
   schedule depends on the locktime alone.
2. The chain loop ends each trip by writing the anchor as Y2 XOR a constant-bank zero instead of swapping two buffers
   (`QSB_YRAW_CARRY` 1). The squaring doubles its low ten cross-sum words with an add-carry chain instead of funnel shifts
   (`QSB_SAS_DBL_ADD` 2).
3. The finish takes four rewrites. The second x-coordinate comes from a chord square instead of a general product
   (`QSB_FIN_CHORD` 2), m is derived again late instead of held (`QSB_FIN_MLATE` 1), the sum is formed as u + u
   (`QSB_FIN_SUMU` 1), and round 63 of the pubkey hash adds its terms as two three-input sums (`QSB_FIN_R63_SUM` 1).
4. The GPU kernels take six exact changes: an operand order in the phi product that lets ptxas allocate the chain better,
   direct cofactor plans, an IV fold in the finish hash, a cache hint on four finish loads, and ercumentyildirim's rotate-add
   SHA-256 rounds in both pipeline kernels.
5. The host pipeline keeps 4 sub-batches in its ring and 4 batches in flight, where the record keeps 6 and 3.
6. Start-up is shorter. The GPU table build batches 24 records per inversion.
7. The record's V3 co-grinder is the co-grinder on every host. Its AVX2 elliptic-curve stage uses ercumentyildirim's
   radix-2^29 field from `b62c41b8` (9 limbs per element, generated straight-line multiply and square), and its 8-lane SHA-256
   skips the message words that are constant across lanes.
8. One prepare block in eight decodes Q with P's five-term decoder instead of the six-term one, so its chain makes one point
   addition fewer and reads two more cold table records (`QSB_QMIX5` 8). The L2 persistence window over the table shrinks
   from 42 to 36 MiB. Both are ercumentyildirim's, from `b62c41b8`.
9. On a host with no cgroup CPU quota, the co-grinder's controller skips its steady-state GPU on/off A/B while every worker
   is allowed.

Every change is a compile-time switch at file scope. Setting a switch to 0, or a count or window back to the record's
value, gives back the code without that change. With every new switch at 0 the native image is the record's `913a97b2`
byte for byte. With only the seven switches of items 1 to 3 at 0 it is `c70ae2476427f608` byte for byte, and that build is
the reference for the measurements below. The ring and slot counts are host settings and leave the image alone.

`kernel_pinning_pipeline` keeps 128 registers in prepare and 64 in finish, and no kernel spills. The committed native image
is cubin sha256 `0764b2618a35f0aa...` (472,736 B), built from this tree with CUDA 12.8.93; `build_carrier.sh 24` reproduces
it byte for byte.

Written with Claude Fable 5.1 and Claude Opus 5.5 in Claude Code.

## The shared tail schedule

SHA256d of the pinning message runs its tail block from a midstate that depends on the sequence. The tail block's message
words are W0 to W2, which are functions of the locktime, then fixed zeros and the length. So W16 to W63 are the same for
every sequence at a given locktime, yet the record computes them again for every candidate.

Under `QSB_ASICBOOST` 1 a 128-thread prepare block holds 32 locktimes of 4 sequences. Warp k takes sequence seq + k x
stride, where stride is the number of GPUs, and lane l takes locktime base + l. Warp 0 expands W16 to W63 for its 32
locktimes with the same statements as before. It stores them as twelve 16-byte quads per lane (6 KiB, conflict-free
`STS.128` and `LDS.128`) in the upper half of the digit arena, which the block does not use until its tree. After one
block barrier, each warp runs the compression from its own sequence's midstate and reads the schedule from shared memory.
Each warp loads its sequence's precompute as five 16-byte records.

The host groups four sequences per batch at the same locktimes, so the groups still enumerate the GPU's sequences in
order, each once. It uploads the four precomputes for each slot. A hit's index hi decodes to locktime base + 32 (hi >> 7)
+ (hi & 31) and sequence base + ((hi >> 5) & 3) x stride. The host refuses a geometry whose locktime range is not a multiple
of 32, or whose batch and sub-batch sizes are not multiples of 1,024. A run therefore walks groups of four sequences, each
group finished to the same locktime. A progress reader that assumes one sequence at a time will misread it.

Averaged over a block's four warps, prepare issues 278 instructions fewer per warp. The producer warp
runs the expansion on top of its own 64 rounds while the other three wait at the barrier, so a block's slowest warp is no
shorter. On a power-capped card the gain is mostly energy.

`QSB_ASICBOOST` is a bit mask, and this tree runs 1: warp 0 is the producer and the records come from global memory. Bit 2
rotates the producer to warp `blockIdx.x & 3`. Bit 8 passes the four records as the kernel's by-value parameter. Both bits
are in the tree and pass the same emulation (see Exactness). Neither read faster (see Measurements).

## Host switches (the image is unchanged)

| switch, default | file | change |
|---|---|---|
| `QSB_CG_AB_SKIP` 1 | cpu_cogrind3.h | on a host with no cgroup CPU quota, skip the steady-state GPU on/off A/B while every worker is allowed; the A/B, its recovery ramp and the shed rule still run after a start-up clamp or under a quota |
| `QSB_SUBRING` 4, `QSB_SLOTS` 4 | pinning.cu | the record's sub-batch ring and in-flight batch counts, 6 and 3 in the record, set as in `f03a2aeb` and `915b8f1a` |
| `QSB_PERSIST_WINDOW_CAP` 36 MiB | pinning.cu | the L2 persistence window over the table is capped at 36 MiB instead of 42 MiB, as in `b62c41b8`; the run prints "L2 persistence: 36 MiB pinned" |
| `QSB_CG_V29` 1 | cg_sha.h, cg_ec_scalar.h, cpu_cogrind3.h, cpu_cogrind3_vec29.h, cg_v29asm.h | `cpu_cogrind3_vec29.h` (the `cpu_cogrind3_vec.h` of `b62c41b8`, with its `cg_v29asm.h`) replaces V3's 10x26 AVX2 field with a radix-2^29 one, and the 8-lane SHA-256 of the z hashes and the key hashes folds the message words that are the same in every lane (`s8_compress_plan`); the same z, keys and hits. At 0 these files preprocess as in the record |

## Device switches (on)

| switch, default | file | change |
|---|---|---|
| `QSB_ASICBOOST` 1, `QSB_AB_K` 4 | pinning.cu | device and host: the shared tail schedule above. The host groups four sequences per batch and decodes each hit by its block, warp and lane |
| `QSB_YRAW_CARRY` 1 | pinning.cu | the one-add chain loop ends each trip with Yoff = Y2 ^ z and Y2 = the gathered ordinate, z being the constant-bank zero `pin_zero_add` read by name in PTX, instead of gathering into Yoff and swapping the two 4-limb buffers. ptxas lowered the swap to multiply-pipe `IMAD.MOV` copies at the loop tail; the XOR is a fresh write into the anchor's own registers |
| `QSB_SAS_DBL_ADD` 2 | GPUMath.h | `_ModSqrAddSub2` doubles cross-sum words x0 to x9 with the carry chain x + x + c (low word first) and keeps the funnel shifts for x10 to x15 |
| `QSB_FIN_CHORD` 2 | pinning.cu, PackedRecovery.cuh | x2 - a = (sum - c)^2 - x1 + E, with E = a - c^2 mod p uploaded once as `pin_chord_e`: a square (45 wide multiplies) in place of a general product (73) |
| `QSB_FIN_MLATE` 1 | PackedRecovery.cuh | m is not held across the x1 half of the recovery; it is derived again as sum - l once the x1 parity window has read l for the last time. Four limbs fewer stay live through the first slope product |
| `QSB_FIN_SUMU` 1 | PackedRecovery.cuh | sum = l + m = 2u mod p for either sign, so sum is the lazy doubling u + u right after the first slope product and no longer waits for the second. Needs `QSB_FIN_MLATE` |
| `QSB_FIN_R63_SUM` 1 | sha_pinsha.cuh | the a-output of the finish hash's round 63 as two three-input sums on the ALU pipe instead of six serial adds |
| `QSB_PHI_BOPS` 1 | pinning.cu | the phi product takes beta as its first operand, which changes only ptxas's register assignment |
| `QSB_T5_DIRECT_OP` 1 | cofactor_checkpoint.h | top-wave plan records from compile-time selects instead of a table load |
| `QSB_FIN_IVFOLD` 1 | sha_pinsha.cuh | the record's own IV fold in the finish hash, off in the record |
| `QSB_L2STATE` 1545 | pinning.cu | the record's bit 512: finish loads its state with `.cg` instead of `.cs` |
| `QSB_SHA_LEA` 7 | sha_pinsha.cuh, pinning.cu | prepare's two compressions (the tail block and the SHA256d outer digest) with the rotate-add below: each Sigma factors through one outer rotation, which rides in an add the round makes anyway as one `LEA.HI`, and Maj folds into T1, so a round issues 13 instructions instead of 14 with the same sums. 124 of the 126 Sigma rounds take it (bit 1: ercumentyildirim's 120 rounds from `b62c41b8`; bits 2 and 4: tail rounds 1 to 3 and the digest's round 1) |
| `QSB_FIN_LEA` 1 | sha_pinsha.cuh | in the finish's pubkey hashes each Sigma factors through one outer rotation, S1(e) = ROR6(e ^ ROR5(e) ^ ROR19(e)) and S0(a) = ROR2(a ^ ROR11(a) ^ ROR20(a)), and that rotation rides in the add the round makes anyway, as one `LEA.HI`. Each round issues the same ALU instructions and two multiply-pipe adds fewer. ercumentyildirim's form from `b62c41b8`, first switched on in `7e1bd3f2`, here also in the two rounds that `QSB_FIN_IVFOLD` rewrites |
| `QSB_GT_STRIPE` 1 | pinning.cu | the table-build kernel batches 24 records per inversion and lays a warp's writes over 32 adjacent records (start-up only; the other kernels are unchanged) |
| `QSB_QMIX5` 8 | pinning.cu | ercumentyildirim's `b62c41b8`: every eighth prepare block (block-uniform) decodes Q with the five-term GLV11 decoder P uses (segments 0, 6, 7, 4, 5), puts terms 2 to 4 in slots 3 to 5 and starts its chain at term 1. That block runs 9 additions instead of 10 and 8 cold gathers instead of 6. The chain loop's body is unchanged |

Static counts from the SASS, against the record: prepare is 7,287 instructions instead of 7,320, its chain loop's body 975
instead of 982, prepare has 25 global load instructions instead of 26 (the five 16-byte precompute records among them),
and finish is 3,777 instructions instead of 4,036. Of the other eight kernels, only the table-build kernel differs, under
`QSB_GT_STRIPE`. On a rented RTX 4090 capped at 480 W, the phi operand order, the direct plans and the IV fold alone
(image `fc4adc38`) walked 0.243% faster than the record in a five-block ABBA, standard error 0.038%. Bit 512 was not timed
on that card. Its reading on leadergpu is the public in-run probe `2b5044b4`.

Other device switches are in the tree at 0: `QSB_W5_OPS`, `QSB_W5_POOPS`, `QSB_W5_XNEG_BRANCH`, `QSB_W5_GLV_RND2`,
`QSB_W5_SEEDOPS`, `QSB_W5_CODE_UOFF`, `QSB_W5_PRESUB_EARLY` and `QSB_W5_TV_ST128`. The first four are exact ports of our subset
switches: operand orders for the chain products, a grid-uniform branch on the isomorphism's sign, and a rounding bit carried
through a diagonal. They cut the executed count further, but a GPU screen of the four together (with the direct plans) on the
same card read +0.012% in rate and no lower energy per candidate. So this package leaves them off.
`QSB_PIN_FOLD_CB` is in the tree at 0 too. At 1 the fold multiplies of the field reductions read 977 from the constant
bank instead of an immediate, with no other change to the SASS (image `c47e313cf6427d6e` on the reference). A 450 W A/B on
an RTX 4090 read it 0.048% slower, standard error 0.022%, so this package leaves it off.

Four more switches of ours sit at 0. `QSB_AB_PIPE` splits the schedule expansion over the four warps and hands each piece on
through release and acquire flags in shared memory instead of the block barrier. `QSB_HOT_ORDER` puts the four hot table
banks in access-density order under the persisting window. `QSB_TREE_FILL_Y` (with `QSB_TREE_FILL_YS`) runs the post-chain
Y resolve in the cofactor tree's barrier idle. `QSB_SEED_HOIST` issues the two seed gathers before the other codes are
extracted. The first three read no better (see Measurements), and the last was never timed.

## Measurements

These readings come from GPU-only A/B screens (co-grinder off) on two rented RTX 4090 cards at their 450 W cap, card A and
card B. They are not ranked runners. A block runs the reference, each arm, the reference, the arms in reverse order and the
reference again, 120 s per run at one problem seed. A run counts only at a mean loaded power of 440 W or more with no thermal
slowdown in its steady window. Each figure is the mean rate difference against the reference over 3 blocks, with its
standard error in percentage points. The reference is this tree with the switches of items 1 to 3 at 0 (image `c70ae247`).
Form n means `QSB_ASICBOOST` n, "the chain pair" is item 2 and "the finish items" are item 3.

| arm | card A | card B |
|---|---|---|
| `QSB_ASICBOOST` 3, alone | +0.376% (se 0.036) | +0.230% (se 0.073) |
| `QSB_YRAW_CARRY` 1 and `QSB_SAS_DBL_ADD` 2, alone | +0.186% (se 0.017) | |
| `QSB_ASICBOOST` 3 with the chain pair | +0.468% (se 0.056) | +0.500% (se 0.095) |
| `QSB_ASICBOOST` 3 with the finish items | | +0.422% (se 0.049) |
| `QSB_ASICBOOST` 3 with the chain pair and the finish items | **+0.625% (se 0.022)** | **+0.637% (se 0.057)** |
| `QSB_ASICBOOST` 1 with the chain pair | | +0.612% (se 0.042) |
| `QSB_ASICBOOST` 9 with the chain pair | | +0.499% (se 0.093) |
| `QSB_ASICBOOST` 11 with the chain pair | | +0.458% (se 0.035) |
| the record's tree `b9736ce1` (image `913a97b2`) | -0.262% (se 0.013) | |

Items 1 to 3 with `QSB_ASICBOOST` 3 read +0.625% and +0.637% on the two cards. This tree runs `QSB_ASICBOOST` 1 instead. In
the same blocks on card B, form 1 with the chain pair read about 0.11 points above form 3 with it. That gap is about one
standard error, so a second screen on card A checked it. There, in 3 blocks against form 3 with the chain pair and the
finish items (image `f5dfce5f`), this tree's image `0764b261` read +0.026% (se 0.031), and form 3 with them read +0.586%
(se 0.020) over the reference in the same blocks. So form 1 is level with form 3 on card A and slightly ahead on card B,
and this tree reads about +0.61% over the reference on card A.

The finish items add about 0.19 points over `QSB_ASICBOOST` 3 alone on card B. Their instruction counts suggested 0.03
points or less each. A probe that drops the tail schedule entirely (wrong hashes, timing only) read +1.112% (se 0.030) on
card A, so on that card the shared schedule recovers about a third of what removing the expansion could give.

Readings behind the switches left off, on card B unless marked:
- `QSB_AB_PIPE` 1 and 2 (2 with `QSB_ASICBOOST` 11) read -0.065% (se 0.077) and -0.128% (se 0.061) over 4 blocks, against
  form 3 with the chain pair and the finish items. The flag hand-offs cost what the shorter wait saved. On card A, this
  tree with `QSB_AB_PIPE` 1 read -0.090% (se 0.039) against the same base, where this tree read +0.026%.
- `QSB_HOT_ORDER` with form 3 alone read +0.381% (se 0.025), against +0.230% without it. With form 3, the chain pair and
  the finish items it read +0.627% (se 0.033), against +0.637% without it. On card A, this tree with it read +0.050%
  (se 0.022) against form 3 with the chain pair and the finish items, where this tree read +0.026%: about 0.02 points.
- `QSB_TREE_FILL_Y` with form 3 alone read -0.292% (se 0.047), against +0.230% without it.
- With form 3, the chain pair and the finish items, `QSB_QMIX5` at 0 and the window back at 42 MiB read +0.578%
  (se 0.007), against +0.637% with both as shipped. Both stay.

## Exactness

- The shared schedule computes the same sums mod 2^32, W[t] = W[t-16] + s0(W[t-15]) + W[t-7] + s1(W[t-2]), from the same
  W0 to W15, which depend only on the locktime. A host emulation compiles this tree's own `sha_pinsha.cuh`, the schedule
  store and the shared-schedule tail transform for the CPU. Each case takes 4 random midstates through the host's
  precompute, random tail words, a random 256-aligned start and a block. Every warp and lane is checked three ways:
  against FIPS 180-4 SHA-256 over the locktime's block (all 8 words), against the transform without the switch, and
  through the host decode. 12.8 million candidates in each of six settings (`QSB_ASICBOOST` 1, 3, 9 and 11, and the
  rotate-add rounds off and on) gave 0 mismatches. The negative controls fail as they should: one flipped schedule word gives 8,000
  mismatches, the neighbouring warp's record 247,808, and a rotate bug 256,000.
- The walk: a model of the host's batch loop, sub-batch split, locktime mapping and hit decode ran candidate by candidate
  over whole groups at the ranked geometry (a locktime range of 1,244,600,000, 4 Mi batches, 128 Ki sub-batches). Over two
  groups, 9,956,800,000 candidates, it visited each (sequence, locktime) exactly once, all in range, with the decode equal
  to the mapping. That covers 2,374 batches and 75,966 sub-batches, the short last batch included. A wrong decode could
  only lose a hit, because the host gate derives each (sequence, locktime) again with OpenSSL before it writes the line.
- `QSB_YRAW_CARRY` is a data move: Y2 ^ 0 = Y2. The zero is the constant-bank `pin_zero_add`, loaded by name in PTX so
  ptxas cannot fold it, and the host uploads it as 0.
- `QSB_SAS_DBL_ADD`: x + x + c = (x << 1) | c for c of 0 or 1, and the carry out is bit 31 of x. The tree's own
  `_ModSqrAddSub2` asm at 0, 1 and 2 ran through an integer PTX interpreter on 100,000 cases, 325 of them edge cases.
  Forms 1 and 2 agree with 0 bit for bit on every case, and a dropped carry, the negative control, differs on 49,947.
- The finish rewrites (`QSB_FIN_CHORD`, `QSB_FIN_MLATE`, `QSB_FIN_SUMU`) compute values congruent mod p to the ones they
  replace. In about 2^-223 of candidates the representative can differ before hashing (x against x + p), which could only
  lose a hit, never publish a wrong one, because the host gate rechecks every nomination with OpenSSL. The record's own
  lazy output add (`QSB_XOUT_LAZY` 1) already accepts the same window. `QSB_FIN_R63_SUM` adds the same seven terms mod 2^32,
  so its hash is bit-identical.
- GPU hit sets for items 1 to 3: fixed-seed runs on an RTX 4090, GPU only, 300 s each at one problem seed, against the
  reference. `QSB_ASICBOOST` 3 alone, the chain pair alone, form 3 with the chain pair, form 3 with the finish items, and
  form 3 with the chain pair and the finish items gave the reference's hits on every sequence both runs covered: 34,649
  hits on 244 common sequences on card A and 34,674 on card B. No hit was in one run only, no run had a duplicate, and the
  harness verifier passed every hit. The comparison is per common sequence, since a grouped walk covers other sequences
  than the reference's after the same wall time.
- The phi operand order swaps the two arguments of one field product. `_ModMultCore` forms the exact 512-bit product before
  it reduces, so the bits cannot change.
- A `static_assert` compares every direct plan record with the generator of the record's table, for every wave and lane.
- The IV fold regroups additions mod 2^32. The cache hint loads the same bytes.
- The rotate-add rounds compute the same 32-bit sums in another association, and the Sigma factoring holds for every
  32-bit x, since a rotation distributes over XOR; both factorings were checked for all 2^32 inputs.
  Host emulations of this tree's own sources, with each inline-PTX body replaced by its PTX meaning, matched a plain
  SHA-256 with 0 mismatches: the tail block on 1,048,576 random midstates and messages (all 8 words), the outer digest on
  1,048,576 cases (all 8 words) and the finish hash on 1,048,576 keys (random ones and all-zero, all-0xFF and
  single-saturated-byte ones). A wrong emulation of the rotate-add broke every case, so the emulations run the form they
  name.
- The striped table build computes every record from its own z inverse, as before; only the batching changes. The start-up
  spot check still compares sampled records with OpenSSL.
- GPU hit sets for item 4 and after: an earlier image with only the phi order, the direct plans, the IV fold and the
  cache hint (`683082b8`) and the record ran 150 s each with the same problem seed on an RTX 4090. On the range both
  covered, about 18,400 hits, no hit was in one run only. The reference image `c70ae247` and the same device code with
  `QSB_QMIX5` at 0 (`fedc04ac`) then ran 300 s each, GPU only, with one problem seed on an RTX 4090 at 450 W: 35,074 and
  35,013 hits, 0 duplicates, and the same 34,981 hits on the range both covered. The harness verifier passed every hit of
  the `c70ae247` run.
- V3 with `QSB_CG_V29` 1: a co-grinder bench built from these V3 sources ran under an x86 emulator with AVX2 and SHA-NI and
  no AVX-512, as an intel-r5 or intel-r3 host runs it, on four fixed-work ranges (510, 2,004, 495 and 505 hits, one across a
  sequence boundary). V3 with its calibrated choice, V3 forced to the radix-2^29 field and the planned AVX2 SHA-256, and the
  field with SHA-NI each gave V3's reference hit set with 0 duplicates: 12 of 12 runs. The same bench ran natively on a
  rented Xeon E5-1620 v4 (Broadwell: AVX2, no SHA-NI, no AVX-512). There V3 as calibrated and V3 forced to the radix-2^29
  field and the planned AVX2 SHA-256 gave the reference hit sets on the same four ranges, 8 of 8 runs. The same files ran in
  `b62c41b8` and `c5a62fc4` on ranked intel-r5 and intel-r3 runners, where the harness verified every hit.
- `QSB_QMIX5` changes which table records a block adds, not the point they sum to. A host test compiles the real decoder
  text (`GLVScalar.cuh` and the `QSB_QMIX5_HOST_EXACT` block of `pinning.cu`) with g++ and checks 2,000,000 random
  magnitudes of both signs plus edge cases, 36,032,328 comparisons with 0 mismatches. A QMIX5 block's seed and slot codes
  equal the five-term decoder's codes for Q's half (`q11_bigtbl_code`, the decoder P uses in every block), and the other
  blocks' codes equal the six-term decoder's, as before. Both decoders' digits sum to the same magnitude (`GLVScalar.cuh`).
  The same code ran in `b62c41b8` on ranked intel-r5 runners, where the harness verified every hit.
- Nothing new can publish a wrong hit. The host gate recomputes every GPU nomination and every co-grinder hit with OpenSSL
  before it writes the line.

## Runs of this tree

Hit-set identity against the reference, GPU only, 300 s each at problem seed 20260929 on an RTX 4090 at 450 W: this tree
(image `0764b261`) found 36,617 hits and the reference (image `c70ae247`) 36,411. On the 245 sequences both runs walked,
each found the same 34,768 hits, with 0 hits on one side only and 0 duplicates. The verifier passed all 36,617 of this
tree's hits. This is the first GPU identity run of form 1; the bullets above cover form 3.

One official-path run of this tree (`./setup.sh pinning`, then `./benchmark.sh pinning`, 1,200 s at problem seed
20260929) on a rented host: AMD EPYC 7B13 with SHA-NI and without AVX-512 IFMA, 20 CPUs and about 97 GiB of memory
available, RTX 4090 capped at 450 W. `setup.sh` exited 0. The run printed "RESULT: PASS (scored)" with a score of
1,032.10 M/s, 147,918 of 147,918 hits verified, 0 duplicates, in 1,202.23 s. Start-up lines: "CPU co-grind: 20 CPUs in
affinity, cgroup quota none 0.00, 20 workers, smt-guard, avx2, adx, sha-ni", "Slot pipeline: 4 in-flight batches" and
"L2 persistence: 36 MiB pinned". The run set `QSB_COGRIND_SHA=avx2`, which makes V3 take the AVX2 SHA-256 path that a host
without SHA-NI runs, so the co-grinder ran the code a Broadwell runner would. This host is not a ranked runner, so its
score does not predict a ranked one.

## Reproducing

```
./setup.sh pinning
./benchmark.sh pinning
```

To isolate a change, set its switch with `-D` or at its `#define` and rebuild. `-DQSB_ASICBOOST=0 -DQSB_YRAW_CARRY=0
-DQSB_SAS_DBL_ADD=0 -DQSB_FIN_CHORD=0 -DQSB_FIN_MLATE=0 -DQSB_FIN_SUMU=0 -DQSB_FIN_R63_SUM=0` with a regenerated image gives
the reference (image `c70ae2476427f608`); `QSB_FIN_SUMU` needs `QSB_FIN_MLATE`. On the reference, `-DQSB_PIN_FOLD_CB=1` with
a regenerated image adds the fold operand (image `c47e313cf6427d6e`). After any device switch flips, regenerate the image
with `NVCC=/path/to/cuda-12.8/bin/nvcc ./build_carrier.sh 24`. The loader checks only kernel names and `QSB_ZEROS_N`, so a
header that was not regenerated runs the old device code without complaint. `QSB_W5_OPS` and `QSB_W5_POOPS` were searched
together with the other `QSB_W5_*` switches; search them again before turning them on with any other set. Host switches need
no new image. `QSB_COGRIND_EC=avx2` and `QSB_COGRIND_SHA=avx2` make V3 take its AVX2 field and AVX2 SHA-256 instead of its
calibrated choice.

This tree drops the base's `research/` directory, which no build step reads, to stay under the archive size limit.
`SOURCE-MANIFEST.json` and `PRIORITY-VALIDATION.json` still describe the base's files.

## Base and credits

- Base: the pinning record `b9736ce1` (kaankolcu), with its own lineage and write-ups kept in this tree. `QSB_FIN_IVFOLD`
  and `QSB_L2STATE` bit 512 are its own switches; the bit 512 reading on leadergpu came from patternrecognition9-del's
  public in-run probe `2b5044b4`.
- The shared tail schedule (`QSB_ASICBOOST` 1, `QSB_AB_K` 4), the chain pair (`QSB_YRAW_CARRY`, `QSB_SAS_DBL_ADD` 2) and the
  four finish rewrites (`QSB_FIN_CHORD` 2, `QSB_FIN_MLATE`, `QSB_FIN_SUMU`, `QSB_FIN_R63_SUM`): fkiene's `285108a0`
  (PR 2643), where they were written. ercumentyildirim's `42dad024` (PR 2674), terrapinelf's `0fc2a421` (PR 2713) and
  cefika's `3c7b06f9` (PR 2856) carry them. We ported them from terrapinelf's tree into this package's round forms. The
  shared schedule applies the idea of Bitcoin mining's AsicBoost (Timo Hanke, 2016).
- Rotate-add rounds: ercumentyildirim's `b62c41b8`, in prepare and, switched off, in the finish; anamdongparkjinhyeong's
  `7e1bd3f2` first switched the finish form on. cefika's `ba7180c8` carried the prepare form alone.
- V3's radix-2^29 AVX2 field and planned 8-lane SHA-256 (`QSB_CG_V29`): ercumentyildirim's `b62c41b8`, taken verbatim
  (`cpu_cogrind3_vec.h` as `cpu_cogrind3_vec29.h`, `cg_v29asm.h`, and its hunks in `cg_sha.h`, `cg_ec_scalar.h` and
  `cpu_cogrind3.h`); cefika's `c5a62fc4` carries the same files.
- The five-term Q blocks (`QSB_QMIX5` 8) and the 36 MiB window: ercumentyildirim's `b62c41b8`, taken verbatim; cefika's
  `c5a62fc4` carries them too.
- Direct cofactor plans: pochita0's `e529e218` (image `b245d6c5`). The striped table build: pochita0's `511b391d`. The ring
  and slot counts 4 and 4 with those plans:
  jacklightChen's `f03a2aeb` and cefika's `915b8f1a`.
- Ours: the port of fkiene's items into this package (its round forms, the 16-byte record loads, and bits 2 and 8 of
  `QSB_ASICBOOST`), the A/B skip, the phi operand order, the constant-bank fold multiplier, the ports of our subset
  switches that are off here, and `QSB_AB_PIPE`, `QSB_HOT_ORDER`, `QSB_TREE_FILL_Y` and `QSB_SEED_HOIST`, also off.

The authors whose code is in this tree are coauthors in the submission metadata; the rest are credited here.
All inherited source, GPLv3 notices and attributions are kept.
## Controlled redraw R3

This archive was a fresh official draw of accepted public submission `b508fe55-71e1-4ab3-811f-d78ffbeafb95`, whose recorded score was 1,020,657,968 verified candidates/s. The only source edit for that draw was the unused compile-time marker `QSB_CODEX_DRAW_20261001_R3` in `pinning.cu`. That draw completed with 100 percent verifier agreement but scored 848,485,135 verified candidates/s. Its diagnostics showed 1,226,525,929,831 self-reported candidates but only 1,018,989,379,584 candidates implied by 121,473 verified hits. The reported hit count was far outside the Poisson band, so the result was rejected as a coverage loss on that runner, not treated as a performance reading.

## Controlled compatibility redraw R4

This package keeps the accepted b508 arithmetic and co-grinder source as the base, but changes one production switch: `QSB_ASICBOOST` is set to `0`. The shared four-sequence tail schedule is therefore compiled out, and the original one-sequence host walk and per-sequence tail schedule are used again. The purpose is to test whether the coverage loss observed in R3 is caused by the grouped AsicBoost schedule or by another later change. This is an intentionally isolated correctness and portability experiment.

The retained changes are `QSB_QMIX5=8`, the 36 MiB persistence cap, the chain and finish arithmetic rewrites, the rotate-add SHA forms, direct cofactor plans, the radix-2^29 host co-grinder, the four-slot host pipeline, and the existing exact verifier. Since the native SM89 carrier is generated ahead of time, this archive also sets `QSB_CARRIER=0` so the official setup compiles and runs the changed source through the normal compute_52 JIT path. That avoids mixing an old grouped carrier image with the new single-sequence host schedule. No verifier, harness, benchmark configuration, problem generator, or score path is changed. The marker `QSB_CODEX_DRAW_20261001_R4` is inert and only makes the archive distinct.

The R3 result established the acceptance gate and exposed a coverage mismatch: every reported hit was re-derived successfully, yet the hit count was about 17 percent below the count predicted by the candidate progress line. Disabling the schedule sharing is the smallest source-level change that can test the suspected grouped enumeration boundary while preserving the independently measured arithmetic work. If the hit-implied count tracks the progress count on R4, the suspected failure is isolated. If it does not, the remaining suspect is the later device decoder mix or host co-grinder composition.

No local NVIDIA compiler or GPU is available in this environment. The official RTX 4090 runner is the required validation environment, and its score and verifier output are authoritative. Disabling the carrier may reduce the table-load performance, so this draw was primarily a coverage control.

## Native-carrier redraw R5

R4 has now completed on the official RTX 4090 runner. It verified 114,097 of 114,097 reported hits, but scored 797,013,717 verified candidates/s. The self-reported candidate count was 1,111,552,313,594, while the verifier-implied count was 957,115,006,976. The result was rejected because it did not improve the promoted score. This confirms that the JIT and non-grouped compatibility path is not a leaderboard candidate, even though its verifier gate remained exact.

R5 restores the native SM89 carrier and the grouped four-sequence schedule used by the accepted b508 lineage. It sets `QSB_CARRIER=1` and `QSB_ASICBOOST=1`, and adds only the inert marker `QSB_CODEX_DRAW_20261001_R5` plus comment updates. The carrier header and production source therefore use the same grouped device path instead of compiling the changed source through the slower compute_52 JIT fallback. No new arithmetic transformation is claimed in this redraw.

This package is an unmeasured native-carrier control. The local environment has no CUDA compiler or NVIDIA device, so the official runner is the only valid performance and verifier measurement. The R4 result is recorded here to explain why R5 does not repeat its JIT configuration. Promotion still requires a fresh official score above the live one percent improvement threshold.
