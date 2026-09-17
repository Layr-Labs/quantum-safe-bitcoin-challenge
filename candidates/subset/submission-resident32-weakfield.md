# Draft: resident32 specialist pipeline with a bounded weak-field region

This draft adds a full-range weak-field arithmetic region to the preserved resident32 specialist pipeline. The inherited architecture uses a complete 32 MiB runtime point table, two SHA producer warps, six EC consumer warps, a six-packet ring and a packed 32-byte digit handoff. The new increment removes repeated canonical corrections inside thirteen ordinary mixed-point additions, then restores canonical coordinates before the unchanged final guard. CPU primitive/helper/complete-chain checks and native production/audit builds pass. There is no GPU execution, throughput measurement, achieved-occupancy result or claim of guaranteed first place.

**Draft status:** this source has not been submitted. The b2 shared13 evaluation, [PR179](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/179), remains pending and is preserved. The last reviewed frontier is 505.611957 million verified candidates/s. Refresh both records and the exact package receipt before upload. The lean stage passes actual full-chain and product reruns, source-bound startup validation, and note-inclusive package preflight against the eighteen-file source closure. The staged package is below 2.5 MiB expanded, within the 8 MiB limit. Neither a successful build nor preflight is a server intake receipt.

The complete eighteen-file production/audit include-closure fingerprint is:

`3ac276e408a4bb399ea805a56fc9e27831a7bbd6da8df5159f16e95f311c5912`.

The frozen snapshot is `candidates/subset/research/weak_field/ordinary_region/candidate`. An intended submission places identical production source at `candidates/subset/`. Paths below are benchmark-relative. The previous cd3d candidate, reports and public note remain preserved.

Effort: GPT 6 Astra xhigh for the coordinating implementation, with separate GPT 6 Astra high-effort helpers for bounded primitive, range-proof, point-helper, full-chain and native reviews, through Codex. Model-assisted review is supporting evidence, not a performance result. Canonical model and harness metadata are supplied separately through the CLI.

## Motivation and official comparison

Our [PR151](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/151), submission `b273a801`, officially scored **410.302626 million verified candidates/s**; all58,744 reported hits verified over1201.0169 seconds. The earlier external-inverse [PR128](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/128), source140133dd, scored **415.082530 million/s**, with59,428 verified hits over1201.0098 seconds. Both were rejected on performance. Their different architectures and supplied problems make them feedback about complete programs, not matched ablations of fusion, geometry or cache policy. Their self-reported rate/count extrapolations are not substituted for official scores, and we do not infer a unique startup or steady-state cause.

The comparison frontier at draft preparation is odinfree's promoted [PR150](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/150), submission `e00f5566-3a6a-4d89-8c7f-528f28f06625`, promotion `ac708da8739938688e8009018d12c42af046b505`, at **505.611957 million verified candidates/s**. Its official run verified72,394 hits over1201.0889 seconds. This is a whole-program reference, not a measured advantage or disadvantage for any isolated formula. The candidate does not import odinfree's PR150 source or squaring-free recovery. It keeps our checked10M/1S cubic recovery. PR150's compact-table result motivated revisiting table residence as an architectural choice; this implementation independently uses sixteen16-bit windows and a different specialist scheduling design.

The working hypothesis is that eliminating the4GiB cold table, retaining a small complete table, and buffering SHA work separately from EC work can repay the extra point additions and synchronization. Moving normalization/digit preparation to the producer is a further load-balancing hypothesis. The preserved scoped-load change addresses compiler live values; the new weak region reduces repeated field normalization work at unchanged mathematical M/S counts. None of these mechanisms creates free compute: with an ideally utilized fixed warp budget, static role partitioning alone cannot beat the combined SHA+EC work. A gain must recover actual underutilization, memory stalls or synchronization inefficiency. We have no measured idle fraction.

## Exact lineage and final change

The preserved chain of experiments is:

- b2 shared13, `b2fd8e8896270638268058a41150409d42a470ce228ee1962c8dcc76f7ef1170`: corrected shared-state thirteen-window source with early cold hints, currently represented by the pending evaluation.
- Two-SHA/six-EC ring, `32411a6a248c7ab53beba0f025450d95f31a0f2fc3da98a549456a47f93162d7`: six ring slots, subgroup synchronization and explicit leader polling; still the old large table.
- Resident32 baseline, `008f904cfb385d030c15324690bbbc5d662777c4547d8db5442b3c1b509dedfd`: sixteen16-bit windows, runtime builder/decoder and complete-table startup policy.
- Packed handoff, `790949f20f5d888a32ee24042b1ac4f805ed52bba009cfab474228ec5e686c2e`: producer normalization and sixteen packed digit cells replace raw SHA payload interpretation.
- Preserved scoped-load base, `cd3d11f24e6fd42459eed3ba3a431c7f952de506cb1d22dfddc9b291e4c98667`: only the packed EC chain changes relative to 790949. Each point load decodes its current digit freshly; next-point hint values have a short source scope; the final digit is also decoded freshly.

- This weak-region source, `3ac276e4…1c5912`: adds `qsb_weak_addsub.cuh` and `qsb_weak_product.cuh`, selects `qsb_ec192_PointAddXYZZ_shared_z_weak` inside the packed chain, and inserts four pre-final canonicalizations. Geometry, builder, policy, tree.cu scheduling/handoff, inverse, packed producer helpers, seed, final guard and recovery remain unchanged from cd3d.

The actual production call is `compact_ec192_fixed_xyzz_packed_shared`. The older raw EC192 helper remains as a reference and is not the selected specialist path. Its mere presence is not a passing test of the new chain. The field checker asserts the selected call, executes the actual packer followed by the actual packed consumer body, and now executes the actual weak host primitives inside the selected weak helper. A mutation confined to that helper’s shared ZZ update compiles and fails the independent curve oracle. Retaining the older canonical helper in the header is not evidence that it was the tested path.

## New increment: a proved weak representation between canonical boundaries

Use p=2^256−C, C=2^32+977, and U=2^256. Inside the ordinary region, every field value lies in W=[0,U), represents the correct residue modulo p, and may be noncanonical. W is not an assumed small multiple of p with spare physical bits: the caller ABI remains four uint64 limbs, corresponding to eight physical 32-bit field words. Every add/sub/product accepts the full W input range. Output aliases are supported under their documented aligned-limb contracts.

The original canonical add/sub contracts do not extend automatically to W. Simply removing every correction was a non-submittable instruction-cost probe, not a valid candidate or a strict performance bound. The corrected implementation keeps all product carry-repair folds, provides new full-range weak add/sub helpers, retains the canonical table-Y-plus-anchor sum, and normalizes X, deferred Y, ZZ and ZZZ immediately before the original final guard. Seed, final exceptional handling, inverse and cubic recovery retain canonical contracts. No incomplete point-add domain assumption is weakened.

Each ordinary addition contains seven field multiplications and two squares. Removing their canonical tails across thirteen iterations removes 117 normalization tails, at the cost of four explicit boundary normalizations plus the new weak add/sub work. One weak T+PPP add and five weak subtractions execute per ordinary iteration. The initial affine Y2+Yoff sum remains canonical because both operands come from signed canonical table points. Normalization removal changes representation and instruction work, not the chain’s 102M/30S or chain-plus-recovery 112M/31S algebraic counts.

For addition, write a+b=s+kU. The first full-width correction computes s+kC and records its overflow j. If j=1, its wrapped remainder is at most C−2, so the second +C is at most 2C−2<2^64 and can update only the low64 bits. For subtraction, write a−b=s−kU. Initial borrow implies s≥1. If the first full-width s−kC correction borrows again, the wrapped result has low64 bits at least 2^64−C+1>C, so the second −C cannot borrow into higher words. Both proofs cover every a,b∈W. The first corrections must propagate across all256 bits; the small second correction is justified by those bounds, not by a probabilistic assumption.

Concrete double-fold witnesses are add(U−1,U−1), which must return 2C−2, and sub(0,U−1), which must return U−2C+1. Omitting either second correction changes the residue. Full-width carry/borrow propagation controls also fail on adversarial boundaries. The actual PTX follows the verified carry/borrow conventions; no extra field word is retained across operations.

Since U<2p, one conditional subtraction of p canonicalizes any weak value. The implementation forms a+C and uses its overflow to select the wrapped result exactly when a≥p. Residue zero has precisely two representatives in W, 0 and p; the separate zero helper recognizes both. Although the current ordinary region avoids exceptional tests, this distinction matters whenever representation-sensitive zero/parity logic is used. All four state fields cross an actual canonicalization boundary before the unchanged guard/recovery code.

The product header retains the original corrected multiply/square integer core, including its rare overflow repair, and removes only the final canonical output correction. Actual host products and extracted PTX are checked over full-width adversarial inputs. Weak add/sub/normalize are new local arithmetic work; the product ancestry remains credited below. The header has register-only inline PTX without a volatile qualifier or memory clobber, with all input operands captured before output stores. This is a code-generation choice backed by alias tests, not a timing claim.

## Runtime table and exceptional domain

All sixteen widths are16, at bit shifts0,16,…,240. Each window contains32,768 signed-odd entries of64 bytes. Total entries are524,288 and total table storage is33,554,432 bytes. These points are rebuilt from the supplied runtime problem/base; there are no precomputed answers or fixed challenge constants.

The exact point order is `14,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13`. Window14 uses position225 and `last=false`; window15 uses position241 and is the only `last=true` window. Their affine pair seeds the deferred XYZZ state. Thirteen ordinary additions consume windows0–12, then the complete guarded final helper consumes window13. The saved Y anchor and shared ZZ/ZZZ conventions remain those of the checked source. The current-point payload is64 demand bytes; it is not128 bytes simply because the load is split into several native instructions.

For the ordinary finite path, the chain costs102M/30S, and chain plus cubic recovery costs112M/31S. Shared13's corresponding totals were81M/24S and91M/25S. Collective inversion and rare guarded cases are separate. Thus residence is purchased with21 additional multiplications and6 squares per candidate before considering scheduling. Operation counts do not establish elapsed time.

Incomplete seed/intermediate helpers require a domain argument, not only randomized point comparisons. Let n be the secp256k1 group order, U=2^256 and delta=U−n. The inherited normalization yields an odd M in[1,n] and a global sign. The top-two seed has scale B=224; the final ordinary-window boundary is H=208. Seed sum/difference are nonzero odd multiples of2^B with magnitude at mostU−2^B<n. After factoring out the common global sign and including a low prefix ending at S≤H, the intermediate sum/difference lies strictly between `2^B−2^S` and `U−2^B+2^S`. Since delta<2^B−2^H, neither can vanish modulo n. These bounds hold for every raw uint256 scalar and any valid nonzero order-n runtime base, assuming correct field and table construction.

The final helper must remain guarded. With D=65535·2^208, the complete raw-scalar exception set is `{0,n,D,n−D}`: the first pair produces opposition and the second equality. The source-bound model exhausts the32,768 possible negative odd final digits to isolate the equality case. All four scalar witnesses also execute through the actual packed chain and independent curve oracle. The proof states its preconditions; sampled tests are not described as exhaustive256-bit execution.

## Packed32-byte producer handoff

Each producer computes the original SHA256d digest, normalizes its scalar and creates sixteen16-bit cells. A cell contains a15-bit magnitude index and one sign bit. All sixteen exactly occupy the original32-byte payload, so this change does not enlarge ring packets or copy traffic. The producer preserves the consumer's original big-endian word-to-little-endian-limb wiring; those bytes now represent packed digits. `qsb_pack_scalar16` accepts in-place input/output, and `qsb_pack_digest16` preserves that wiring explicitly.

For window c, the consumer reads one64-bit packed limb, shifts out its16-bit cell, and splits index/sign. Producer construction includes the cross-limb bit, the global scalar sign and the special top-window rule. Updating an internal packed limb is safe because subsequent groups use only higher original limbs. Independent integer peeling, alias checks and explicit crossing-bit/top/sign mutations test this convention.

The inherited scoped-load access schedule has30 logical packed-limb reads per active candidate instead of16: two seed digits, fourteen hint reads, thirteen fresh ordinary-load digits and one final digit. That adds112 logical shared-read bytes. The intent is to avoid carrying the next index and sign through long field arithmetic. This is a compiler-lifetime experiment, not an algebraic reduction. Importantly, native scheduling delays the next hint until near the loop tail: short source scope does not prove an early issued prefetch or greater memory lead time.

## Two producers, six consumers and finite progress

Physical warps0/1 produce; warps2–7 own192 EC contexts. A full CTA processes six epochs, or48 packets and1536 candidates. Producer p emits `g=8*epoch+2*j+p`, j=0..3, into slot `g%6`. Consumer warp w consumes `g=6*cohort+w` from its fixed slotw. The original choice is `(g&7)*32+lane`; descriptor and hit identity use the original epoch plusg/8. This mapping preserves each candidate exactly once across cohort and epoch boundaries.

Slot s starts with `free_ticket=s` and an invalid ready ticket. A producer waits for free=g, writes all32 lanes' payload, joins the warp and publishes ready=g. A consumer waits for ready=g, copies all payload, joins the warp and publishes free=g+6 before beginning EC work. Slot parity gives one producer owner; its fixed consumer warp is stable. Publication need not be globally ordered across slots. Six slots can buffer the following EC cohort while the current cohort computes.

The64-class first-state cache has64 producer owners. Both producer warps join after class publication and, before each later epoch, after old-cache readers retire. E epochs therefore execute2E−1 producer joins; the unused final retirement is omitted because no subsequent overwrite exists. The actual54-class schedule fits the64-class capacity. The host rejects an unsupported larger first-class count before publishing to this compact cache instead of silently truncating it.

The EC region and inverse scratch share storage only within the EC subgroup. Producers' cache, ring, tickets and subgroup counters remain separate live allocations. Every EC cohort joins after field reads before the inverse overwrites that region. Inactive tail warps do not wait for nonexistent packets: they participate in the inverse and all required EC joins using identity factors. Singular recovery factors likewise use identity without abandoning collective participation. The padded256-leaf inverse has192 real leaves,64 identity leaves,669M plus one root inverse, and five internal subgroup joins.

Subgroup synchronization uses separate producer/EC generation counters, legacy shared atomics, block fences and full-warp joins. The initial setup has a full-CTA join; steady role-separated work does not use a divergent partial-CTA barrier. Ticket waiting elects lane0, followed by an unconditional full-warp memory join and an all-lane block fence before payload access. Writers/readers join before their leader releases a ticket. Both producer warps and all six consumer warp leaders must eventually be scheduled; ticket generations are finite and bounded by the six-epoch CTA. CPU cooperative tests and finite models check protocol/identity under their stated scheduling assumptions, not target weak-memory execution.

The wrapper's default virtual target must compile too. An sm89-only build is insufficient for subgroup synchronization: aligned restrictions on older targets ruled out simply applying a named partial-CTA `barrier.sync`. The retained atomic/fence/warp-join implementation compiles under both sm89 and default flags. In the earlier fixed-mailbox experiment nvcc automatically aggregated same-address atomics; indexed ring polling did not retain that transformation. Explicit leader polling was introduced only after inspecting the actual ring native code, so no source-count-only32× traffic claim is made.

## Builder and startup qualification

The resident builder uses256-element low/high capacities. For entry t, `ch=t>>15`, the odd multiplier is `m=2*(t−offset)+1`, `hi=m>>8`, and `lo=m&255`. High indices are0..255 and low indices odd1..255. H[0] is not used as an affine point. With positive high index, low/high sum and difference cannot vanish modulo n: their nonzero magnitudes are below65536, while the runtime coefficient and powers of two are invertible. The last64-byte output ends exactly at32MiB.

Each host low/high ladder occupies262,144 bytes, with the same device allocations. Existing1,048,576-entry checkpoint scratch capacity is retained; the table uses one partial chunk,2048 prepare/finish blocks and eight root groups. Table plus allocated builder device scratch totals67,895,808 bytes, excluding unrelated allocations. CUDA allocation/copy/launch/synchronization/free checks and host spot checks remain, with serialization and inspected OpenSSL return checks. This is a much smaller construction than the cold-table variants, but no startup duration has been measured.

The optional L2 policy covers the entire32MiB table. Unsupported capacity falls back to ordinary caching; unrelated/fatal errors remain visible and partial changes are rolled back. The policy is not cache pinning and does not guarantee residency. An earlier thirteen-window prototype compiled and passed curve tests yet exited because its host policy still asserted the old cold boundary. That failure exposed a validation gap. Geometry changes now execute the actual host-policy capacity/error matrix and verify its after-build call; packaging requires the source-bound startup-policy record. The current source's fresh policy check passes180 cases,30 injected failures and8 additional paths. Five negative controls include stale cold-prefix/48MiB bounds, wrong pointer and missing rollback.

## Source-bound correctness evidence

Fresh 3ac276 evidence is saved under `research/weak_field/ordinary_region/`:

- `product-results.json`: 60,540 actual host multiplies, 40,360 actual host squares, and 2,180 source-extracted PTX semantic cases for each operation. Twenty-three bounded overflow-fold cases and the stale-carry mutation are covered. These test exact residues and W bounds; they do not require canonical output.
- `helper-results.json`: 7,682 actual weak helper calls over 1,591 synthetic polynomial states, all192 owners, both deferred/exact-Y modes and six affine-input alias patterns. There are694 noncanonical output cases. Separate compiled mutations reject a dropped affine anchor, omitted shared ZZ update and missing deferred-Y boundary normalization. The last control exposes p+1 where canonical1 is required. These synthetic states deliberately reach weak boundaries that ordinary random curve points seldom reach; they are not all valid curve states.
- `chain-check/results.json`: 628 complete packed/scoped chains, 1,242 recovered keys, 12,769 recoding cases,608 actual sparse-loader cases and8,792 prefetch-address checks. All192 owners have surrounding and other-owner canaries. Actual producer packing precedes the selected consumer body, with628 alias comparisons. All13 ordinary additions execute the actual host code from the new add/sub and product headers, including the host square’s actual weak-multiply fallback. They are not replaced by OpenSSL field mocks. Canonical seed/final/recovery and the independent curve oracle still use the disclosed OpenSSL backend.
- The complete chain executes2,512 boundary normalizations, four per chain. All four exceptional raw scalars execute. The actual final helper passes159 equal,159 opposite and159 ordinary cases, with159 unguarded-doubling negatives. A projection-only mutation removing the selected weak helper’s shared ZZ update compiles, then fails the independent curve oracle at n−D with runtime base1. This demonstrates that the new path, rather than the retained canonical helper, is under test.
- `policy-host-results.json`: the actual unchanged resident policy is executed against the new eighteen-file closure, with180 cases,30 injected failures,8 additional paths and the existing five regressions. Its after-build placement is checked.
- Fresh production and audit sm89/default builds cover an eighteen-file union. The production native screen initially omitted the default-flags pass; a separate fresh `production-default-results.json` completes that check. Each native production/audit report records seventeen includes; their union covers both entry sources and both new primitive headers. The audit was compiled, not executed.

The standalone add/sub/normalize primitive receipt is under `research/weak_field/primitive/test-artifacts/results.json`, bound to header SHA256 `31f72e474a8e42ae291b71a4f7250a3bc43c489c367a9787797f74169c52c88e`, identical to this candidate. Actual host bodies and extracted PTX each pass26,048 binary operations over13,024 pairs and13,024 normalizations. Host zero and in-place normalization also pass13,024 cases;17,824 overlapping-buffer cases preserve283,392 canary words. The local extension of the existing PTX model adds288 borrow probes. Five PTX mutations and four independently compiled host mutations are detected. Full-range proofs and finite tests are separately labeled; none is GPU execution.

Inherited component evidence remains scoped to unchanged sources:

- The resident builder, tested at008f904c, exhausts all524,288 decoder inputs and validates host ladders of6,128 points per runtime base. Across three bases,4,464 affine outputs and14,649 checkpoint outputs over19 boundary/tail cases match independent OpenSSL results. Root hierarchy counts1,255,256,257,2047,2048 and canaries cover the production dispatch. The current builder headers are unchanged.
- Packed helpers tested at790949 pass145,004 raw inputs,1,015,028 overlap comparisons,2,320,064 digit extractions and1,048,576 exhaustive cell/position extractions. Wrong crossing bit, sign, top rule, group order, mask and wire halves compile and fail. Their bodies are unchanged; the prior cd3d staged digit receipt is not relabeled a fresh3ac run.
- The preserved cd3d control/handoff projection processes26,624 identities and2,642 synthetic hits across23 CTAs and twelve epoch sizes. Actual recode/pack/unpack execute against independent integer-peel identities; SHA/curve/inverse/gating numerics are mocks. Seven compiled controls detect packing, identity and tail-participation errors. `tree.cu` and the relevant helpers are byte-identical in3ac; this is an explicitly inherited control result, not a rerun of the new arithmetic inside the protocol mock.
- The unchanged producer SHA component matches7,680 digests against the original helper and hashlib. The unchanged EC192 inverse checks3,840 values,7,680 canaries, identity padding, operation counts and mutations; separate finite-generation models cover its barriers. These are header-bound tests rather than GPU collectives.

The exceptional-domain model checks22,284 scalars and356,544 digits in addition to its all-input inequalities and exhaustive final-digit equation. The new helper’s field-range contract is supplied by the separate primitive/product proofs and actual host checks. CPU protocol, arithmetic and curve tests complement one another; none is described as an end-to-end CUDA run. The composite `ordinary_region/qualification.json` binds these results and unchanged inherited components. Separately, the assembled lean stage passes actual full-chain and product reruns and note-inclusive package preflight against the same eighteen-file source closure.

## Native result: shorter ordinary loop, costs retained outside it

`ordinary_region/native-review.md` and `native-review.json` bind the exact3ac source and SASS to the preserved cd3d comparison. The actual ordinary-add loop is `0x4450–0x9d70`. It has1,427 instruction slots versus1,594 in cd3d:167 fewer, or10.48%. This is a static instruction-region change, **not a throughput or latency improvement**. The counter executes thirteen iterations; the apparent CALL at its end is the loop-exit transfer, not an omitted arithmetic callee.

The repeated loop has zero local loads/stores, eliminating cd3d’s eight read and eight write bytes per iteration, or104/104 logical operand bytes per active candidate. Point loads, packed index/sign interpretation and22 LDS.64/12 STS.64 sites per iteration remain. Logical operand/site counts are not physical transactions, cache misses or bandwidth. The source’s M/S counts are unchanged.

Four boundary normalizations are present in native code, interleaved with final arithmetic. Their identified starts are0xa2d0,0xae20,0xb0a0 and0xbc60; coordinate names are inferred from shared offsets and following uses. Normalized ZZZ words spill around0xa470–0xa6f0, outside the repeated loop. The combined boundary/final region grows86 slots, which is not an isolated normalization cost because register allocation and scheduling change too. Other consumer regions have additional local accesses. This cost must remain visible alongside the local-free loop.

| Specialist kernel quantity | Preserved cd3d | Weak3ac |
|---|---:|---:|
| Registers/thread |80|80|
| Static shared bytes/CTA |32,832|32,832|
| Stack bytes/thread |320|352|
| Aggregate compiler spill stores/loads |496/312 B|552/384 B|
| Whole-kernel non-NOP slots |15,530|15,455|
| Ordinary loop slots |1,594|1,427|
| Repeated-loop local reads/writes per candidate |104/104 B|0/0 B|

Aggregate compiler spill totals include boundary, inverse, rare-hit and other paths and must not be multiplied by thirteen. The increased totals do not negate the located loop reduction or prove a net benefit. Whole-kernel slots decrease by75 while the loop region loses167 because other regions grow.

The SHA producer is instruction-identical after relocation and remains local-free. Its full-warp helper is independently matched despite a different target displacement. Consumer polling also matches after address/stack-offset rebasing; the leader still polls and all lanes join/fence before payload access. No source-level atomic-count shortcut is used to infer native traffic.

Prefetches remain near the loop tail at0x9cd0/0x9d10. The next ordinary point load is18 intervening instruction sites away versus24 before; that does not establish helpful memory lead time or better prefetch effectiveness. No early-native-hint claim follows from the scoped source expression.

The shared arena remains24,576 EC bytes,2,048 first-state bytes,6,144 ring bytes,48 ticket bytes and16 counter bytes. The previous CUDA occupancy-header capacity calculation permits three CTAs under its supplied Ada properties at80 registers and32,832 static shared bytes. This is not measured occupancy; loaded JIT resources, reservation/allocation rounding, actual carveout and runtime scheduling remain unmeasured. Table residence, producer/consumer balance, root inversions, atomic joins, startup work and outside-loop accesses can dominate any instruction saving.

There is no GPU arithmetic-audit execution, full GPU table comparison, CUDA race test, throughput measurement or official score for3ac. Some inherited search-path API status handling remains a diagnostic limitation. The earlier zero-hit PR86 failure has no established cause and is not OOM proof. A prior expanded-archive rejection motivates the exact8MiB expanded-package check; compressed archive size alone is insufficient. A separate lean archive passes note-inclusive preflight, including the source-bound startup policy, selected subset link and unchanged trusted/sibling files. Its actual staged full-chain and product reruns also pass. Expanded contents remain below 2.5 MiB, including this note and qualification receipts; repeat the check after any further archive changes.

## Provenance and coauthors

GPL notices and `COPYING` are preserved. Promoted ancestry includes PR60, welttowelt's PR62, alvaroborras's [PR77](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/77) (promotion `d2772418e0f372767b4c59f7382d71f9142585fe`), and i34-9's [PR120](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/120) (promotion `65cd138c80bf8f9070117a991c049a9da71d1de8`). PR120's direct-digit attribution to dun999 remains. The corrected PR77 fused arithmetic/recovery ancestry is preserved; this does not describe the earlier external pipeline as the whole PR77 architecture. Promoted PR150 is cited above as the current comparison and compact-table motivation; no odinfree source is imported here.

Retain these seven Yukon coauthors for substantial unpromoted assistance/source:

- **alvaroborras**, [PR64](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/64), `a7b21d0f62e6d73b66fe820e228f8db50d504716`: adaptive builder ancestry.
- **MakiRH4**, [PR46](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/46), and **jacklightChen**, [PR53](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/53), `9274883051636def6db5add0d3ba0e02314813f0`: batched ladder/builder ancestry.
- **ercumentyildirim**, pinning submission `e2fd8093-2ba5-4d40-8f25-dabb0a4807c5`: persisting-L2 capacity observations supporting the inherited optional-policy work. The capacity observations are credited as inherited provenance; no sibling pinning implementation is added.
- **AbdelStark**, [PR138](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/138), `2dc49dc4e4083050ffc34b9be61eb027eb8c291f`: leaf-pair inverse ancestry, including our necessary64-thread CTA-join repair. The new192-owner specialization and subgroup protocol are local work; later cancellation does not remove the original credit.
- **IvanLudvig**, [PR137](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/137), `103a6adc15391e07aac210a2653f8dfd7eb4d4c0`: host-only whole-class packing. Paired-SHA and host-drain source were not imported.
- **anamdongparkjinhyeong**, [PR156](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/156), `6f6410dc7393c48c6814d016703fb404a310a3a1`: compact64-class SHA storage mechanism that materially supports this specialization. Our producer interface/protocol is new; the whole PR156 pipeline was not imported.

The new full-range weak add/sub/canonicalization helpers, their range proofs, the ordinary-region boundary and the source-executing qualification are local work in this experiment. Product code retains the corrected inherited arithmetic and its credit. Neither the native10.48% region-size change nor the inherited official scores are attributed as a GPU gain from this increment.

## Reproduction and next checks

Development-checkout commands, from the benchmark root:

```sh
SRC=candidates/subset
W="$SRC/research/weak_field"
O="$W/ordinary_region"
R="$SRC/research/specialist_warps/resident_tables"
C="$O/candidate"
python3 -B "$W/primitive/check.py"
python3 -B "$O/check_products.py" --source "$C" --output /tmp/weak-products.json
python3 -B "$O/check_helper.py" --source "$C" --output /tmp/weak-helper.json
python3 -B "$O/check_chain.py" --source "$C" --output /tmp/weak-chain
python3 -B "$O/check_chain.py" --source "$C" --output /tmp/weak-negative --mutation-weak-zz
python3 -B "$R/small32/check_policy.py" --source "$C" --report /tmp/weak-policy.json
```

The weak-ZZ mutation is expected to compile and fail the independent curve oracle. `primitive/check.py` selects the immutable primitive header, whose SHA256 must match the candidate. Do not label a fixed-snapshot run as validation of a different root. Product, helper and chain checkers accept explicit `--source` and `--output` paths; the policy checker accepts `--source` and `--report`. The chain adapter keeps the original scoped checker unchanged, selecting the actual new helper and inserting the actual primitive headers into the CPU projection. It requires the preserved extraction/proof dependencies and baseline comparison bodies.

For the lean archive with byte-identical production source at `candidates/subset`, source-root checks are:

```sh
C=candidates/subset
O="$C/research/weak_field/ordinary_region"
R="$C/research/specialist_warps/resident_tables"
python3 -B "$O/check_products.py" --source "$C" --output /tmp/weak-archive-products.json
python3 -B "$O/check_helper.py" --source "$C" --output /tmp/weak-archive-helper.json
python3 -B "$O/check_chain.py" --source "$C" --output /tmp/weak-archive-chain
python3 -B "$R/small32/check_policy.py" --source "$C" --report "$C/startup-policy-validation.json"
python3 -B "$C/preflight.py" --package-check --note "$C/submission-resident32-weakfield.md"
yukon setup --track subset
```

The stage retains the required extraction dependencies and unchanged cd3d candidate for comparison. Native reports use benchmark-relative artifact references and retain original-report SHA256 values after local-path sanitization. The staged chain, products and note-inclusive preflight have passed; rerun package checks after any further archive change rather than treating an earlier receipt as final. The product checker now binds the explicit source root. Refresh the full eighteen-file fingerprint, startup-policy report and expanded package bound after assembling the final artifact. The setup command above is a reproduction step, not a claim of GPU execution. After header changes invalidate the trusted wrapper’s generated `subset` binary and `.subset.build` stamp before relying on setup’s timestamp cache.

On a CUDA compiler host, the staged entry commands are:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 candidates/subset/subset.cu -lcrypto -lm -o /tmp/resident32-weak-production
nvcc -O3 -DQSB_ZEROS_N=24 candidates/subset/tests/gpu_epochs/tree_audit.cu -lcrypto -lm -o /tmp/resident32-weak-audit
```

Repeat with `-arch=sm_89 -Xptxas=-v,--warn-on-spills` for native resources. Both target modes are required; source-extracted PTX semantics are not an assembler or GPU test. CPU checks need Python, C++17 and OpenSSL development libraries. `review_native.py` reads saved compiler artifacts and performs no GPU execution.

Before upload, refresh the lean package’s source-bound receipts, PR179 and the frontier. Preserve that evaluation while pending. The decisive performance test is official verified throughput. With device access, compare frozen cd3d and3ac on the same problem, measuring startup and steady state separately, and account for boundary spills, cache behavior, ring waiting and achieved residency. The corrected weak region is a checked instruction/lifetime improvement hypothesis within a larger unmeasured resident32 architecture;10.48% fewer loop slots is not a promised GPU speedup.
