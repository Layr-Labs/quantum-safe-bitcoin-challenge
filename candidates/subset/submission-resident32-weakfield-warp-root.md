# Subset: resident32 weak-field specialist pipeline with a bounded cooperative root inverse

This candidate preserves the checked 32 MiB runtime table, two-SHA/six-EC ring, packed digit handoff and weak ordinary point-add region, and changes the inverse tree’s single serial root into a bounded full-warp cooperative inverse. A scalar fallback retains the original root contract if the cooperative budget is exhausted. The final source changes the cooperative helper’s declaration to `__noinline__` after the inlined control reintroduced repeated point-loop spills. The new native screen restores the thirteen-add loop to 1,427 slots and zero local loads/stores. Native review locates the root call, local-array interface, conditional context-reload site and original-root fallback. These findings are not a GPU speedup claim.

**Submission context:** our prior shared13 candidate [PR179](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/179), source `b2fd8e8896270638268058a41150409d42a470ce228ee1962c8dcc76f7ef1170`, completed its [official evaluation](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/actions/runs/35238683518) at **421,167,078 verified candidates/s** and was rejected below the current **505,611,957** frontier. All **60,300 hits verified** over **1201.0271 seconds**. The official hit-derived candidate count is505,833,062,400; its596,444,919,666 self-reported extrapolation is not substituted for the official score. No correctness failure was reported. The slot became free without cancelling any evaluation. This result reinforces the need for a substantial architecture change, but does not identify a unique cause or constitute a matched ablation of cache policy, geometry or inversion. The present candidate removes the4GiB cold table in favor of a complete32MiB table, separates SHA/EC warp roles, reduces ordinary field-normalization work and cooperates at the root. These are checked mechanisms with a credible improvement hypothesis, not a measured guarantee of beating the leader. The ready3ac scalar-root control remains preserved. Exact source, startup policy, staged host tests, native builds, public-note review and the separate note-inclusive package checks pass. No GPU throughput is claimed for this new candidate.

The exact nineteen-file production/audit include-closure fingerprint is:

`a36071b7a5b06c22d16172f8f29ec0d86eb1cd90620e3716526b40349a890047`.

The frozen source is `candidates/subset/research/warp_root/bounded/outlined/candidate`. A future submission would place identical production source at `candidates/subset/`. All paths in this note are benchmark-relative. There has been no GPU correctness execution, race test, measured occupancy, throughput run or official score for this source.

Effort: GPT 6 Astra xhigh coordinated implementation and selection work, with separate GPT 6 Astra high-effort helpers for arithmetic, range proofs, actual-source host integration, source attribution and native review, through Codex. Canonical model/harness metadata are supplied separately by the CLI. Model-assisted review is supporting evidence, not performance evidence.

## Motivation and official comparison

Our [PR151](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/151), submission `b273a801`, officially scored **410.302626 million verified candidates/s**; all58,744 reported hits verified over1201.0169 seconds. The earlier external-inverse [PR128](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/128), source140133dd, scored **415.082530 million/s**, with59,428 verified hits over1201.0098 seconds. Both were rejected on performance. Their different architectures and supplied problems make them feedback about complete programs, not matched ablations of fusion, geometry or cache policy. Their self-reported rate/count extrapolations are not substituted for official scores, and we do not infer a unique startup or steady-state cause.

The comparison frontier at draft preparation is odinfree's promoted [PR150](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/150), submission `e00f5566-3a6a-4d89-8c7f-528f28f06625`, promotion `ac708da8739938688e8009018d12c42af046b505`, at **505.611957 million verified candidates/s**. Its official run verified72,394 hits over1201.0889 seconds. This is a whole-program reference, not a measured advantage or disadvantage for any isolated formula. The candidate does not import odinfree's PR150 source or squaring-free recovery. It keeps our checked10M/1S cubic recovery. PR150's compact-table result motivated revisiting table residence as an architectural choice; this implementation independently uses sixteen16-bit windows and a different specialist scheduling design.

The working hypothesis is that eliminating the4GiB cold table, retaining a small complete table, and buffering SHA work separately from EC work can repay the extra point additions and synchronization. Moving normalization/digit preparation to the producer is a further load-balancing hypothesis. The preserved scoped-load change addresses compiler live values; the inherited weak region reduces repeated field normalization work at unchanged mathematical M/S counts. None of these mechanisms creates free compute: with an ideally utilized fixed warp budget, static role partitioning alone cannot beat the combined SHA+EC work. A gain must recover actual underutilization, memory stalls or synchronization inefficiency. We have no measured idle fraction.

## Exact change and public source boundary

The direct base is the ready weak-field source `3ac276e408a4bb399ea805a56fc9e27831a7bbd6da8df5159f16e95f311c5912`. Its resident32 architecture descends through the two/six ring, sixteen-window resident table, packed handoff and scoped-load experiments described below. None of those architectural changes is newly attributed to the warp-root increment.

Only the EC192 root region changes from 3ac: `tree_inverse.cuh` includes one additional header and replaces its scalar-only root call with a first-EC-warp call, uniform completion status and ecid0 scalar fallback. The other seventeen base include files remain byte-identical. Existing generic256, EC224 and builder inverse interfaces remain. The new helper is derived solely from AbdelStark’s [PR189](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/189), exact public head `3c3d748e7e46bbf68fcd22d179c44b231eaa15de`, submission `db248c65-44ef-4aba-b20c-c244a4cd334a`. The donor helper SHA256 is `940f4c6ad4272a71d38533123a809c8f1affc7a9e0beca959db48380f24cd1fc`. Its Jean Luc Pons/VanitySearch and GPL notices are preserved.

The donor’s packed tree, full-CTA synchronization, HM39/HM41 alternatives and hot field arithmetic were not imported. That matters: the donor tree’s collective structure is incompatible with our concurrently active producer warps. Independent CPU semantics of the donor’s actual hot PTX also exposes canonical-input carry failures. For example, multiplying p−65537 by itself, or using its selected square on that value, should yield `0x100020001`, but yields `0x1fc30`; multiplying p−1 by p−2^64 should yield `0x10000000000000000`, but yields `0xfffffffefffffc2f`. These are source-derived semantic witnesses, not GPU execution or an assertion about the donor’s eventual official result. HM43 does not call those hot products. This candidate retains the corrected GPUMath, square and full-W weak headers from 3ac.

Three isolated root snapshots are retained:

- Unbounded donor integration, `6cb9b00606f541cd0894a48308fa9771dd68efe4bb269f49515e1f8e14d4435d`, established cooperative-host integration but did not establish an unlimited signed-accumulator bound.
- Bounded inline source, `f47ad8241adac7aac14e63ad947735ada3bf3565ae60266fa7153b6744d5583a`, adds the sixteen-batch cap, boolean API and original-root fallback. Its repeated point-loop local traffic motivated the next control.
- This outlined source, `a36071b7…a890047`, changes exactly one declaration qualifier from `__forceinline__` to `__noinline__` relative to bounded f47. The entire helper body, caller, cap, fallback and eighteen other source files are byte-identical. The new helper header SHA256 is `054bc2fa8f1bd08a22d9e0eabf73dcc545170ea1db93e746f45f1008f0aae4a7`.

The qualifier change transfers arithmetic/control evidence through exact body equality. It does not transfer native resource results or constitute a fresh host run on the outlined source. Separate fresh native production/audit builds bind a360. Preparation receipts retain their historical “not qualified” creation state; subsequent evidence must be read with its own exact source binding.

## Cooperative root mechanism and finite fallback

The padded EC192 tree still performs 669 field multiplications, has 192 real leaves and 64 identity leaves, and retains five subgroup joins. Only its root inverse changes. The first EC warp consists of physical threads64–95, with relative lanes0–31. All32 lanes initialize a five-word private root, execute the helper and participate in every full-mask shuffle/ballot. ecid0 alone loads the four canonical shared-root words and later publishes them. All other EC warps wait at the existing publication join; SHA producers do not participate in this inverse.

The helper distributes U,V,R,S into four groups of eight lanes. Each group uses six live64-bit words and two guard lanes. Six words retain signed384-bit matrix intermediates; the guards prevent carry propagation between groups. U/V are initialized from p and the runtime root, R/S from zero/one. Every lane computes the same62-bit decision matrix. Ballots implement carry generation/propagation; exchanges supply neighboring words and partner state. Signed products negate all six words, then sparse m·p correction makes the coefficient update divisible by2^62. Final normalization is performed on lane0 and broadcast. Distributed state does not eliminate replicated decision arithmetic or collective overhead.

The API is `bool hm43_warp_inverse(uint64_t result[5], int lane)`. True means complete output; false means the caller’s five-word arrays remain untouched. The production input contract is canonical nonzero x∈[1,p−1], guaranteed by canonical tree multiplication and identity substitution for inactive/singular factors. Zero/noncanonical helper tests extend finite validation but are not silently included in the tree contract.

`HM43_WARP_MAX_BATCHES` defaults to16 and permits only0..16 at compile time. Its counter and outcome are warp-uniform. Success on the sixteenth matrix update finalizes normally; otherwise false is returned before any seventeenth-batch collective. Cap0 performs the initial broadcasts and returns false. No claim is made that every input converges within eight or sixteen batches.

On false, only ecid0 reloads all four words from the still-original `tree[][1]`, explicitly sets root[4]=0 and invokes the unchanged scalar `_ModInv`. It then publishes the four result words through the original path. This is not inversion of a partial cooperative state. The first EC warp reconverges before the existing subgroup join. There is no new shared allocation, altered tree padding, producer rendezvous or early exit from the surrounding collective.

## Why the finite-width arithmetic is bounded

The independent review in `research/warp_root/range-review.md` starts with U=p,V=x,R=0,S=1 and U≡Rx,V≡Sx modulo p. After t trailing-zero bits have been consumed within a batch, row L1 norms are bounded by2^t immediately after stripping. Subtraction can temporarily add a factor two, but its two odd operands produce an even difference, and the next strip absorbs that factor. The final strip reaches62 and exits before another subtraction. Terminal row norms are therefore at most2^62; transient coefficients also fit int64 and cannot reach INT64_MIN.

This argument uses the exact low-bit parity/divisibility relation. Approximate high-word comparisons may choose a different swap than exact binary GCD, so an exact-comparison convergence theorem is not imported. Matrix row application, sign correction and division keep nonnegative U,V≤p, with predivision magnitudes below2^318. Their sign checks in word4 are valid within signed320 bits.

For coefficient state M_t=max(|R_t|,|S_t|), the sparse correction gives M_(t+1)<M_t+p, hence M_t≤1+t·p. At most sixteen batches give M_t<2^260. Before division, a row product sum plus its modular correction has magnitude below2^322, inside signed384 bits; the resulting quotient fits signed320 bits. The correction constant obeys C·MM64≡1 mod2^62 for C=2^32+977, so adding m·p permits exact division and preserves the modular invariant. Guard words and six-word signed negation are part of this argument.

If V reaches zero, preserved gcd implies U=1 for a production input and R is an inverse; bounded coefficient magnitude makes final canonicalization finite. If it has not reached zero by the cap, the original-root scalar fallback retains the baseline inverse behavior. The cap closes this new accumulator-range argument without asserting a universal convergence bound. It does not prove the inherited scalar implementation anew or establish target GPU memory ordering.

## Actual-host root and fallback evidence

The bounded f47 helper receipt `research/warp_root/bounded/helper-results.json` executes the actual C++ helper through32 synchronized host lanes. Its unchanged body is the one used here. Cap16 passes1,452 arithmetic/inverse cases and51,768 checked64-bit words. Cap0 tests115 inverse fixtures: all3,680 lane returns are false and18,400 caller words stay unchanged. Cap1 uses113 roots requiring more than one batch: all3,616 lane returns are false and18,080 caller words stay unchanged. The selection model only chooses fixtures; it does not replace actual helper execution.

Full participation and equal collective counts are checked. AddressSanitizer and UndefinedBehaviorSanitizer report no errors. Five independently compiled mutants fail arithmetic/status checks: dropped carry-in, omitted sixth-word negation, guard carry leakage, false success at cap0 and premature output modification before a false return. Compilation failure or timeout is not counted as detecting those arithmetic mutants. Helper tests retain progress logs and bounded host rendezvous/process timeouts.

The actual EC192 integration in `bounded/tree-check/results.json` tests tails0,1,31,32,33,191,192. Caps16,0,1 each verify all192 outputs, giving4,032 natural inverse comparisons. Every case preserves669 tree multiplications, five EC joins, eleven tree warp joins, all192 owners and32 root callers. Cap16 completes cooperatively for these roots; cap0/1 each invoke scalar fallback. Tree products use OpenSSL, while the expected final outputs use independent Python bigint inversion.

The fallback is the actual extracted legacy `_ModInv` and its helper/control bodies, with CUDA scalar carry/borrow/mulhi operators projected using explicit uint128 semantics. Two signed coefficient shifts are written as unsigned bit-preserving host shifts; signed wrapping is retained. Before each integration executable,520 independent legacy-root cases validate this disclosed projection. It is not a pow or OpenSSL inverse substituted for the fallback. It also is not CUDA execution of the scalar assembly.

Two additional positive tests poison all five private root words after a false return at caps0/1. The actual caller reload/clear repairs that deliberately injected state, adding2,688 output comparisons; total value canaries across all five positive configurations are13,440. With that same disclosed poison, removing the four-word reload or publishing the partial root fails the inverse oracle; omitting the fifth-word clear violates the fallback input contract. The real helper leaves caller output untouched on false, so a bare no-reload/no-clear change would be observationally redundant under its current contract. These controls test defensive restoration and are not presented as natural corruption by the valid helper.

The earlier unbounded integration separately detects wrong physical-lane arguments, leader-only calls, omitted guard-lane participation and active-only root participation. Those controls retain their original6cb source binding. Current unchanged participation is source-audited; a prior mutant receipt is not relabeled a fresh outlined test.

The new stage freshly reruns the bounded-f47 helper and tree suites from the staged preserved snapshot, including caps16/0/1 and their compiled mutations. These staged reruns remain f47 algorithm evidence, not fresh outlined-host execution. The a360 preparation proves that its only f47 delta is the declaration qualifier. Thus bounded host arithmetic/cap/fallback evidence transfers through exact unchanged bodies and caller. Fresh outlined host execution and a final source-root package adapter are separate work if required by the final package. Host collectives abstract the unchanged EC barrier as a192-thread rendezvous; they do not simulate GPU scheduling, memory latency or atomic weak-memory behavior.

## Inherited weak region: a proved weak representation between canonical boundaries

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

The optional L2 policy covers the entire32MiB table. Unsupported capacity falls back to ordinary caching; unrelated/fatal errors remain visible and partial changes are rolled back. The policy is not cache pinning and does not guarantee residency. An earlier thirteen-window prototype compiled and passed curve tests yet exited because its host policy still asserted the old cold boundary. That failure exposed a validation gap. Geometry changes now execute the actual host-policy capacity/error matrix and verify its after-build call; packaging requires the source-bound startup-policy record. The unchanged resident policy passed180 cases,30 injected failures and8 additional paths at3ac. The separate a360 stage now also passes its actual policy check and source-bound startup receipt against the nineteen-file closure. Final note-inclusive package preflight remains separate. Five negative controls include stale cold-prefix/48MiB bounds, wrong pointer and missing rollback.

## Inherited field and architecture evidence

The ready3ac arithmetic qualification remains scoped to byte-identical helper/product/geometry sources. Its actual weak host products pass60,540 multiply and40,360 square cases; source-derived PTX passes2,180 cases per operation, including23 bounded fold-overflow cases and a stale-carry negative control. The add/sub primitive passes26,048 binary operations over13,024 pairs per host/PTX backend,13,024 normalizations,17,824 overlapping-buffer cases and283,392 canary words, with compiled host and PTX mutations.

The actual weak point helper passes7,682 calls over1,591 synthetic polynomial states, all192 owners, both deferred-Y modes and six affine-input alias patterns;694 cases produce noncanonical weak outputs. The complete packed chain passes628 chains,1,242 recovered keys,12,769 recodes,608 sparse-loader cases and8,792 prefetch-address checks. Four normalizations execute per chain. All four exceptional raw scalars and477 actual final-helper cases execute. A compiled mutation confined to the selected weak helper’s ZZ update fails the independent curve oracle. These remain3ac arithmetic records, transferred by exact unchanged source; the changed root receives its separate tests above.

Unchanged builder evidence includes exhaustive524,288 decoder inputs, ladders of6,128 points per runtime base,4,464 affine outputs and14,649 checkpoint outputs across three bases and19 boundary/tail cases. Packed digit evidence includes145,004 inputs,1,015,028 overlap comparisons and2,320,064 digit extractions. The preserved cd3d control/handoff projection checks26,624 identities and2,642 synthetic hits across23 CTAs; its field/inverse numerics are mocked. Producer SHA matches7,680 digests against the original helper and hashlib. These component results are not a fresh end-to-end GPU run, and the old inverse numerical result does not replace the new HM43 integration result.

## Native comparison and root-interface accounting

Fresh a360 production and arithmetic-audit builds pass both explicit sm89 and the trusted wrapper’s default compiler flags. Their union matches all nineteen closure files, including the new root header and both entry sources. The arithmetic audit was compiled, not run on a GPU. `outlined/production-native-results.json` and `outlined/audit-native-results.json` bind these builds.

The bounded inlined control f47 reintroduced eight local read and eight local write operand bytes per ordinary iteration:104/104 per candidate over thirteen iterations. Its unchanged mathematical loop grew from1,427 to1,446 native slots. This demonstrates that changing a root helper can affect allocation in a different hot region; smaller aggregate spill totals would not excuse that observed repeated traffic.

The outlined a360 screen restores the actual thirteen-add loop at0x4450–0x9d70 to1,427 slots and zero LDL/STL. Its complete instructions match ready3ac after return-target normalization, a stronger check than matching counts. This is a located native result, not a timing measurement. The saved compiler reports give:

| Specialist resource | Ready3ac | Inline f47 | Outlined a360 |
|---|---:|---:|---:|
| Registers/thread |80|80|80|
| Static shared bytes/CTA |32,832|32,832|32,832|
| Stack bytes/thread |352|368|344|
| Aggregate spill stores/loads |552/384 B|556/344 B|552/388 B|
| Parsed specialist non-NOP slots |15,455|16,550|16,642|
| Ordinary loop slots |1,427|1,446|1,427|
| Repeated-loop local reads/writes per candidate |0/0 B|104/104 B|0/0 B|

Aggregate spill and static-slot totals are not dynamic instruction/traffic counts. The source-bound saved-artifact review in `research/warp_root/bounded/outlined/native-review.json` locates the sm89 root call at0x1ab60→0x37690, reached by physical threads64–95. The caller materializes a40-byte local root array per participating lane: five zeroed64-bit words, then four leader-only input stores. This is root-stage work per192-candidate cohort, distinct from thirteen repeated additions per candidate.

The outer matrix main region0x37af0–0x39280 has378 static sites and one4-byte local context-reload site at0x38e10 inside the outer-batch region and outside the inner divstep loop. The branch at0x38e00 can skip this site, so only lanes on its participating path execute the load. Its301-site compiler outline regions have no local instructions. Exchange helpers still contain warp joins and shuffle/ballot operations. Thus the root matrix is not entirely local-memory-free, and its alternate paths, inner loops and helper calls prevent treating these site counts as total executed instructions per batch.

The native cap test is at0x37af0. Cap failure returns to the caller, where EC lane0 reloads the original shared root at0x1abf0–0x1ac20, zeros the fifth word at0x1ac30 and calls legacy `_ModInv` at0x1ac80→0x3bdc0. Success and completed fallback publish four words at0x1ace0–0x1ad10 before the existing synchronization. This confirms the actual fallback control flow; it does not assume cooperative convergence by sixteen batches.

The caller region contains36/116 static local read/write operand bytes; the entire cooperative callee and its compiler outlines contain276/312. These sum instruction-site operand widths, including mutually exclusive paths, and are not bytes dynamically executed per root or per lane. The complete local-site list is retained in the review so the restored point-loop result does not hide the root interface cost.

`__noinline__` does not allocate registers independently to the root warp, establish runtime occupancy or promise an out-of-line cost advantage. It is a compiler hint; the actual call and local placement above are inspected evidence, rather than inferred from the annotation. The source pointer parameter addresses five64-bit words, and the saved native code confirms their local materialization. The same warp-uniform call site and full-mask helper body preserve participation structurally, but default-target compilation and native inspection remain necessary.

The original3ac loop’s1,427 versus cd3d’s1,594 slots, and its removal of104/104 loop-local bytes, are inherited weak-region findings. They are not new warp-root gains. Neither that10.48% region-size reduction nor cooperative root parallelism is converted into GPU throughput. The source-matched native review also confirms the3,535-site SHA producer at0x29690–0x37370 is instruction-identical to ready3ac after relocation and remains local-free. Consumer polling matches after relocation and stack-offset normalization, with a leader atomic and8 local read bytes per leader spin; producer polling remains leader-only and local-free. These observations establish neither poll latency nor atomic cost.

The existing32,832-byte arena and80-register kernel retain the nominal resource class used in the earlier three-CTA capacity hypothesis. This is not measured occupancy; allocation rounding, shared carveout, JIT output, runtime scheduling, root collective cost, fallback frequency, table residence and producer/consumer balance remain unmeasured. There is no official score or first-place guarantee for a360.

## Provenance and coauthors

GPL notices and `COPYING` are preserved. Promoted ancestry includes PR60, welttowelt's PR62, alvaroborras's [PR77](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/77) (promotion `d2772418e0f372767b4c59f7382d71f9142585fe`), and i34-9's [PR120](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/120) (promotion `65cd138c80bf8f9070117a991c049a9da71d1de8`). PR120's direct-digit attribution to dun999 remains. The corrected PR77 fused arithmetic/recovery ancestry is preserved; this does not describe the earlier external pipeline as the whole PR77 architecture. Promoted PR150 is cited above as the current comparison and compact-table motivation; no odinfree source is imported here.

Retain these seven Yukon coauthors for substantial unpromoted assistance/source:

- **alvaroborras**, [PR64](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/64), `a7b21d0f62e6d73b66fe820e228f8db50d504716`: adaptive builder ancestry.
- **MakiRH4**, [PR46](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/46), and **jacklightChen**, [PR53](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/53), `9274883051636def6db5add0d3ba0e02314813f0`: batched ladder/builder ancestry.
- **ercumentyildirim**, pinning submission `e2fd8093-2ba5-4d40-8f25-dabb0a4807c5`: persisting-L2 capacity observations supporting the inherited optional-policy work. The capacity observations are credited as inherited provenance; no sibling pinning implementation is added.
- **AbdelStark**, [PR138](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/138), `2dc49dc4e4083050ffc34b9be61eb027eb8c291f`: leaf-pair inverse ancestry, including our necessary64-thread CTA-join repair. The 192-owner specialization and subgroup protocol are local work; later cancellation does not remove the original credit. AbdelStark also receives substantial credit for the root-only HM43 source from PR189 at head `3c3d748e7e46bbf68fcd22d179c44b231eaa15de`. The local cap, fallback, EC192 adaptation and outlining experiment retain that attribution.
- **IvanLudvig**, [PR137](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/137), `103a6adc15391e07aac210a2653f8dfd7eb4d4c0`: host-only whole-class packing. Paired-SHA and host-drain source were not imported.
- **anamdongparkjinhyeong**, [PR156](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/156), `6f6410dc7393c48c6814d016703fb404a310a3a1`: compact64-class SHA storage mechanism that materially supports this specialization. Our producer interface/protocol is new; the whole PR156 pipeline was not imported.

The inherited full-range weak add/sub/canonicalization helpers, their range proofs, the ordinary-region boundary and the source-executing qualification are local work in this experiment. Product code retains the corrected inherited arithmetic and its credit. Neither the native10.48% region-size change nor the inherited official scores are attributed as a GPU gain from this increment.

The weak-field and HM43 contributions are separate increments. No eighth coauthor is added for PR189 because AbdelStark is already one of the seven retained coauthors. The public donor’s full pipeline and defective hot arithmetic are not imported. Existing notices and COPYING remain intact.

## Reproduction and remaining selection gates

The evidence-generating commands below are benchmark-relative. Bounded helper/tree tests currently bind the preserved f47 snapshot, so they intentionally reproduce that arithmetic/control evidence:

```sh
SRC=candidates/subset
W="$SRC/research/warp_root"
B="$W/bounded"
A="$B/outlined"
python3 -B "$B/check_helper.py" --source "$B/candidate" --output /tmp/hm43-bounded-helper.json
python3 -B "$B/check_tree.py" --source "$B/candidate" --output /tmp/hm43-bounded-tree
```

Do not point a manifest-pinned f47 checker at a360 and silently bypass its mismatch. The attribute-only transfer is recorded by `outlined/prepared-source.json`: all helper body bytes and caller bytes are identical, and the other eighteen files match. A fresh outlined host adapter must recognize the declaration attribute while preserving the actual body and binding all nineteen files. This is distinct from changing the candidate for the checker.

The local evidence-audit command `python3 -B "$A/review_native.py"` reads the original saved compiler reports/SASS; it does not run a GPU or compiler. The public stage sanitizes native JSON build directories to `BUILD_TMP`, retains original-report hashes and does not archive the full temporary SASS trees. Native review/qualification scripts therefore are not promised to replay unchanged from the archive alone. A public reproduction rebuilds with the explicit compiler commands below, saves fresh disassembly, and supplies or adapts report/artifact paths to those rebuilt files. The f47 host commands above have replayed successfully from the staged archive dependencies.

The unchanged weak-chain/product checks already accept an explicit source root:

```sh
O="$SRC/research/weak_field/ordinary_region"
C="$A/candidate"
python3 -B "$O/check_products.py" --source "$C" --output /tmp/hm43-outlined-products.json
python3 -B "$O/check_helper.py" --source "$C" --output /tmp/hm43-outlined-point-helper.json
python3 -B "$O/check_chain.py" --source "$C" --output /tmp/hm43-outlined-chain
python3 -B "$O/check_chain.py" --source "$C" --output /tmp/hm43-outlined-negative --mutation-weak-zz
python3 -B "$SRC/research/specialist_warps/resident_tables/small32/check_policy.py" --source "$C" --report /tmp/hm43-outlined-policy.json
```

These are supported reproduction commands, not claims that every listed check has already been rerun against a360. The field tests execute the point chain, not the new subgroup inverse. Cap/fallback tests and source-equality transfer are disclosed separately. The weak-ZZ mutation must compile and fail the curve oracle.

For a future lean submission with this exact source at `candidates/subset`, set `C=candidates/subset` for root-capable checks, preserve required baseline/projection dependencies, and run the actual policy checker with `--report "$C/startup-policy-validation.json"` before:

```sh
python3 -B candidates/subset/preflight.py --package-check --note candidates/subset/submission-resident32-weakfield-warp-root.md
yukon setup --track subset
```

Package preflight must bind the nineteen-file source identity and startup record and count expanded bytes under the8MiB limit. It must not reuse3ac’s eighteen-file receipt. Preserve the selected subset linkage and trusted/sibling files, sanitize local native-artifact paths while retaining original-report hashes, and verify the actual uploaded source and server intake separately. Header edits require invalidating the generated `subset` binary and `.subset.build` stamp before relying on the wrapper’s timestamp cache.

Native entry commands for the final production layout are:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 candidates/subset/subset.cu -lcrypto -lm -o /tmp/resident32-hm43-production
nvcc -O3 -DQSB_ZEROS_N=24 candidates/subset/tests/gpu_epochs/tree_audit.cu -lcrypto -lm -o /tmp/resident32-hm43-audit
```

For explicit sm89 resources and saved disassembly:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -arch=sm_89 -Xptxas=-v,--warn-on-spills candidates/subset/subset.cu -lcrypto -lm -o /tmp/resident32-hm43-production-sm89
nvcc -O3 -DQSB_ZEROS_N=24 -arch=sm_89 -Xptxas=-v,--warn-on-spills candidates/subset/tests/gpu_epochs/tree_audit.cu -lcrypto -lm -o /tmp/resident32-hm43-audit-sm89
cuobjdump --dump-sass /tmp/resident32-hm43-production-sm89 > /tmp/resident32-hm43-production.sass
cuobjdump --dump-sass /tmp/resident32-hm43-audit-sm89 > /tmp/resident32-hm43-audit.sass
```

Retain compiler version, flags, source hashes and fresh artifact hashes; a rebuilt file is not expected to have an old report hash. Both target modes matter because the trusted wrapper does not specify an architecture. The host checks require Python, C++17, threads and OpenSSL development headers; they do not require or emulate a GPU.

The completed native review supports preferring a360 over the inlined-f47 control because it removes that integration’s repeated point-loop regression. The distinct note-inclusive package checks now pass. PR179 is now terminal and its official result and the unchanged frontier are recorded above; the subset slot is free. Preserve pending evaluations. The decisive comparison is official verified throughput; with GPU access, compare exact3ac and a360 on the same supplied problem and separate startup, ordinary EC work, root/fallback time, queue waiting and achieved residency. This candidate offers a bounded, checked cooperative-root hypothesis while leaving its net performance unclaimed.
