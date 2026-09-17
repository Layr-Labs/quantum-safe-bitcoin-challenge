# Subset: shared thirteen-window recovery with early cold prefetch

This submission combines a shared-state curve pipeline with thirteen windows, corrected field arithmetic, and early cold-point prefetch. Its full CPU/native and staged package checks pass; no local GPU throughput is claimed.

Our previous ranked/direct-digit submission `b273a801`, [PR151](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/151), completed its [official evaluation](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/actions/runs/35219600194) at **410302626 verified candidates/s** and was rejected below the current505611957 frontier. All58744 reported hits verified over1201.0169 seconds. This is performance feedback, not a correctness failure. The prior source had fourteen windows and kept its point state in registers, compiling to128 registers,24KiB shared andzero ranked spills. The present composite uses thirteen windows and a32KiB shared arena, compiling to80 registers; the intended three-block residency and reduced point-chain work may improve latency coverage, but actual occupancy and throughput have not been measured. It doubles cold requests from2 to4, so the shared-memory and arithmetic gains may be offset by memory stalls. The prior result weakens a geometry-only superiority argument; this remains a distinct, credible architectural experiment rather than an established improvement.

The PR151 self-reported/extrapolated candidate count was520839559844, while its official hit-derived count was492780388352. We use only the official410302626 score and do not infer a specific cause from that difference. The earlier external pipeline scored415082530; the two runs use different full programs and problems and are not a matched fusion ablation. No pending evaluation was cancelled. The slot became available after PR151's terminal result, and this checked composite was selected promptly. The corrected a03 no-prefetch alternative remains preserved for a future matched control.

Effort: xhigh for GPT 6 Astra implementation and high for GPT 6 Astra independent reviewers, through Codex. The new increment issues four cold-point prefetches, eight32-byte sector-address hints, from the already local immutable recoder M before the seed. It does not change the thirteen-window algebra, table capacity or candidate mapping. The intended benefit is more lead time for cold requests; whether it repays hint/address issue work or causes cache pressure is unmeasured. The larger shared13 architecture and its arithmetic/spill tradeoffs are inherited and described separately below.

## Exact source and selection

The complete production/audit include-closure fingerprint is:

`b2fd8e8896270638268058a41150409d42a470ce228ee1962c8dcc76f7ef1170`.

The frozen snapshot is `candidates/subset/research/geometry_frontier/shared_thirteen/policy_corrected/cold_prefetch/local_m/candidate`; an intended archive places identical production source at `candidates/subset/`. Paths are benchmark-relative. The matched no-prefetch control is corrected a03, fingerprint `a03c3c1ca5f07be955c32af76f34e5a58cc81237d7b9e4c685ee56cfe5353bda`. Relative to a03, only `tests/gpu_epochs/compact_table_device.cuh` changes, by adding the hint block inside the existing local-M scope. Its SHA256 is `cb95bfcacf64b7ac1e889a182be99c53ca7a360d9bd9c9943536fb0158beb6e1`.

The preparation receipt records the intermediate cold-prefetch screen5001cdc6 as its immediate generator base; comparing final b2fd directly to a03 leaves just the local-M hint block. The corrected L2 header remains `05339e941862efca9dc715adead307f556e17dd06d10ff31becbaa60752cb525`. The earlier startup-defective e80c snapshot, fingerprint `e80c8d9124fdd85ab38a6079ead00969d1e60d612a5f28c72f28523d28177908`, is preserved only as historical evidence/regression input.

The original composition binds regular thirteen-window geometry donor `5d17e5fad6cb9d632d7b8b4977751da574242488d428cfedfb455accaa21ca5b` and early-anchor shared architecture `e575d61ad2a0f4d9b65e8901e741c6bbe7514c1a6df6fc60118cfe537653124a`. Geometry/decoder and chain were checked independently, but both original thirteen-window snapshots retained the wrong startup assertion. They remain evidence/control snapshots, not runnable benchmark-ready alternatives. Relative to e575, a03 changes geometry, decoder, compact chain and this necessary host-policy assertion; shared helpers, final guard, arena ownership, joins, recovery, inverse and other host mechanisms remain unchanged.

CPU/native and staged qualification are complete. This composite is selected for the now-open subset slot after PR151 completed. The source-level hypothesis is deliberately narrow: issue the four cold requests before their dependent arithmetic, using local M that already exists, with no additional shared allocation or point-payload lifetime. Native production still uses80 registers and32KiB shared memory, but adds42 non-NOP instructions relative to a03. There is no measured comparison against a03. Preserve that checked base if the final review finds a regression or the incremental hypothesis is unconvincing.

The inherited shared13 composition removes one7M+2S intermediate addition at unchanged table capacity. Its original a03 loop had40 logical local-load bytes per lane over ten adds, versus484 in e575's eleven-add loop; regular controls have zero compiler spills. Those observations motivate the base, not a new prefetch gain. The completed b2fd SASS review independently confirms the same ten-add loop after target relocation; equal compiler spill totals alone were not used to establish that fact.

The current promoted comparison is odinfree's [PR150](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/150), submission `e00f5566-3a6a-4d89-8c7f-528f28f06625`, promotion `ac708da8739938688e8009018d12c42af046b505`, at **505611957 verified candidates/s**. Its [official run](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/actions/runs/35217261521) verified72394/72394 hits over1201.0889 seconds. Its source-exact squaring-free recovery removes one further square relative to our checked cubic recovery; this candidate retains that checked recovery and pursues the separate shared13 architecture. The whole-program result is not a measured one-square gain over our source. The one-percent promotion target is now510668077 integer candidates/s. PR120, promotion `65cd138c80bf8f9070117a991c049a9da71d1de8`, remains the provenance for inherited direct-digit/ranked-compilation ideas and its495193826 historical score. No source from PR150, PR156, PR157 or PR159 is imported, and no GPU advantage over the current frontier is claimed.

## New increment: four cold hints while M is local

Immediately after unchanged scalar setup and storing M into its existing shared slots, the source issues:

```cpp
for(int cold=9;cold<13;++cold){
    uint32_t cold_idx; uint64_t cold_neg;
    gt_direct_digit(M,(uint64_t)(sign<0),
                    (unsigned)mixed_shift(cold)+1u,25u,cold==12,
                    &cold_idx,&cold_neg);
    compact_prefetch_point(gTX,cold,cold_idx);
}
```

This occurs before seed point loads in the source. The address helper emits `prefetch.global.L2` for the selected point base and base+32. These are four logical points/eight sector-address hints, not eight extra demand point loads. The actual demand loads, signed Y, point order, arithmetic, shared arena, joins and outputs remain unchanged. Each hinted index comes from the exact direct-digit helper; only12 has the top-last rule. Global sign affects signed Y but not the absolute table magnitude. The unused cold_neg cannot justify changing that magnitude rule.

M is already in a local four-limb array for setup; deriving addresses here avoids new reads of the shared recoder slots for these hints and leaves no point payload live through the seed. Source shared traffic remains2976 logical bytes per candidate and table demand payload remains832 bytes. Hint traffic, effective fetch size, duplicate requests and actual residency are hardware-dependent and unmeasured. Early hints may be ignored, arrive too late, be evicted before use or add cache/issue pressure; no guaranteed overlap or pinning is asserted. The completed native review locates all eight added `CCTL.E.PF2` instructions between0xcac0 and0xcd50, before the first seed demand load at0xcd60. This establishes instruction order, not completion or cache success. The instruction is documented in the [NVIDIA PTX ISA prefetch reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch-prefetchu).

The actual-source CPU checker extends the independent prefetch oracle to all13 windows. It verifies both32-byte halves, range, exact later point index and ordering. The specific hint-only negative control changes `cold==12` to `false` while leaving all demand loads and chain arithmetic unchanged. Its generated C++ compiles successfully, then exits86 on the named window12 address mismatch. Thus the check detects a wrong hint address rather than succeeding merely because arithmetic still works or failing to compile a malformed mutation. This is address-model evidence, not GPU prefetch effectiveness.

## Thirteen windows at the same table capacity

The exact widths and shifts are:

```text
widths = [18,18,18,17,17,17,17,17,17,25,25,25,25]
shifts = [0,18,36,54,71,88,105,122,139,156,181,206,231]
order  = [9,10,11,12,0,1,2,3,4,5,6,7,8]
```

Nine small tables cover156 scalar bits in48MiB. Four cold tables cover100 bits in4GiB. Total mandatory affine table storage is4345298944 bytes, exactly the same as the fourteen-window control. Every table point depends on the supplied runtime base; no answer table is baked into the archive.

| Ordinary candidate work | Fourteen-window9c/e575 | Thirteen-window5d17/e80c/a03/b2fd |
|---|---:|---:|
| Small/cold point requests |12/2|9/4|
| Logical64-byte point loads |896B|832B|
| Two-point seed |12+13|9+10|
| Deferred intermediate additions |11|10|
| Final guarded window |11|8|
| Point chain |88M+26S|81M+24S|
| Chain plus cubic recovery |98M+27S|91M+25S|

M/S counts exclude SHA, inverse collective, table construction, exceptional fallback and memory costs. Seed9+10 costs3M+2S; the ten intermediates11,12,0..7 cost7M+2S each; the final resolved addition at8 costs8M+2S in its ordinary branch. Only top window12 uses `last=true` in the direct-digit helper. Final addition window8 is an ordinary signed digit. The inherited interleaved small-table prefetch targets0..8; the new early block additionally targets cold9..12. Signed-Y handling, magnitude indexing and size_t byte addressing are retained.

The total point-load count falls, but cold requests double. At hypothetical small/cold hit rates hs/hc, the change in logical miss bytes is `64*(-1+3*hs-2*hc)`. Perfect small hits and no cold hits would add128 bytes per candidate. This is a simple request model, not measured DRAM traffic or latency; actual transactions, reuse and overlap can change the result. Neither the48MiB allocation class nor its policy implies guaranteed L2 residency.

The bounded runtime builder and its checkpoint discipline remain. Host ladder arrays now total13MiB, plus OpenSSL objects; the thirteen-window count replaces the prior fourteen-window array sizing. Explicit temporary GPU buffers retain48497152 bytes, with bounded chunks of at most1048576 entries and no full-table CPU fallback. Temporary buffers are released before search. Geometry/decoder bytes match the checked regular13 donor, whose CPU builder checks cover4491 entries at three runtime bases,101497 decoder cases and69811 ladder points per base. That evidence does not execute the GPU builder.

The optional checked L2 policy requests48MiB for the contiguous small-table prefix only when the selected device supports the needed capacity/window. It checks API status and effective limits; unsupported configurations use ordinary caching. No physical pinning guarantee or measured policy gain is claimed. This geometry does not remove mandatory table startup cost.

## Startup defect caught before upload

The frozen e80c and5d17 sources still asserted `mixed_offset(12)*64 == 48MiB` at the start of `qsb_enable_small_l2_policy`. In thirteen-window geometry cold storage begins at9; offset12 is48MiB+3GiB. Thus the normal short-epoch path exits after table construction and before search, even on hardware that would later skip unsupported persistence. This was a real source defect, not a performance regression, CUDA compile failure or optional-policy fallback. The ready13 selection was withdrawn before upload. The fourteen-window e575 source correctly has cold boundary12 and is unaffected.

Prior chain, recoder, builder-decoder and native PASS reports retain their stated scope: none executed this actual host policy with the new geometry. Header byte preservation had incorrectly been treated as sufficient evidence for a geometry-dependent startup precondition. The regular13 exceptional-domain proof remains valid because its widths, digits, table points and point order are unchanged; it never proved startup readiness.

The corrected a03 assertion uses `mixed_offset(9)` and the matching message. The new `policy_corrected/check_policy.py` compiles the actual policy and actual geometry together in a fault-injected host CUDA-API harness. Independent width, shift, offset, total-size and first-cold-boundary assertions precede180 capacity/limit cases and30 injected-error cases. All pass for a03 and the fresh b2fd source-bound run. Running the same harness on frozen e80c with `--expect-defect` reproduces the startup abort and its original diagnostic. The checker also verifies the actual post-build call site. These are CPU host-policy tests, not GPU cache or driver execution. The runner now accepts independent `--widths` (defaulting to this13-window geometry); the same actual-geometry matrix passes180+30 cases for staged e575 with its14 widths, confirming that the retained control is unaffected.

Package validation now closes this gap explicitly. When the closure includes `l2_policy.cuh`, `preflight.py --package-check` requires a `startup-policy-validation.json` PASS record with the same complete source fingerprint **and every one of the16 include-file hashes**, actual post-build call verification, at least180 host cases and30 injected failures. Missing, incomplete or stale-source policy records are rejected. The public recipe generates this record from the actual staged root before package validation. This gate checks source-bound startup-policy evidence; it does not claim GPU execution or replace the remaining archive-size, note and stage checks.

The old public draft is preserved as `research/geometry_frontier/shared_thirteen/policy_corrected/previous-e80c-note.md`, SHA256 `b53b03af50dc2f89df2658de70cdbac07025622b08664bcc7f5f98909990d194`. Its readiness claim is superseded here; its source-bound CPU/native observations are not erased.

## Exceptional-domain argument and actual helper checks

Reordering incomplete additions requires more than random scalar tests. For valid nonzero order-n runtime table bases and correct field/table primitives, the separate `research/geometry_frontier/exception_domain.md` argument shows no seed or intermediate equal/opposite case for any uint256 scalar in this order. The final window8 guard remains necessary.

Let U=2^256, B=156, H=139 and delta=U−n, where n is the secp256k1 order. Delta has129 bits and satisfies delta<2^B and delta<2^B−2^H. The actual scalar setup produces odd1≤M≤n and sign sigma with sigma*M congruent to2k modulo n. This includes raw hashes above n and the doubling carry. Table bases incorporate the matching half-base factor.

Every partial cold sum or difference has a nonzero odd quotient after removing2^B and absolute value at mostU−2^B<n. Thus seed9+10 and subsequent cold11/12 cannot be equal/opposite. The complete cold sum telescopes to `((M>>B)|1)*2^B`. Adding hot0..7 gives sum/difference bounds strictly between0 and n because delta<2^B−2^H. Negating all digits preserves the exclusion.

At final window8 the exceptional raw scalar domain is exactly `{0,n,D,n-D}`, where `D=(2^17-1)*2^139`. Scalars0 and n yield opposite points; the other two yield equal points under opposite global signs. The inherited guarded helper returns infinity for opposite points and doubles the affine table point for equal points. The final signed digit is−131071 in the unique equal representative. The proof enumerates all65536 possible negative final digits as an additional check; finite enumeration supports that step, while the full-domain conclusion rests on the stated inequalities and recoder identities.

The independent domain model checks22853 scalars,297089 direct-versus-peeling digits,137118 cold sum/difference relations and365648 hot relations. Wrong global sign, wrong extraction start and wrong top-last flag are rejected. This model is a hash-bound Python translation, not execution of extracted CUDA C++. Separately, the actual shared chain and final-helper bodies execute through a CPU projection with independent OpenSSL curve/affine oracles, including constructed exceptions. Source equivalence binds the geometry/order and unchanged final guard to this proof. Singular recovery retains the inherited skip/identity-padding contract; the proof does not claim that skip behavior enumerates every mathematical recovery singularity.

## Shared arena and the early-anchor lifetime

The digest kernel owns one32KiB union with three typed views:

```cpp
union QsbSharedPhaseArena {
    uint32_t sha[8][256];
    uint64_t fields[4][1024];
    uint64_t inverse[4][512];
};
```

The row strides256/1024/512 are intentional and separately checked. During SHA, the first8KiB holds grouped first states. The scheduled-hash helper accepts this scratch instead of allocating another array; its body differs from the checked original only in ABI/declaration. Both kernel source branches use the scratch helper, with an original wrapper retained for other callers. Native compilation confirms32KiB total shared memory for the ranked kernel.

After every lane finishes shared first-state consumption and SHA256d, a uniform CTA join precedes any field write. Point-phase ownership is:

```text
fields[limb][tid]       = immutable recoder M
fields[limb][256+tid]   = previous signed affine Y anchor
fields[limb][512+tid]   = ZZ
fields[limb][768+tid]   = ZZZ
```

The values are volatile to express deliberate storage instead of relying on the compiler to keep long-lived register copies. Each direct digit reads only its needed current/next64-bit limbs; the absent high limb is zero. The inherited two-stage shift avoids undefined shift-by64 at aligned bit positions. Shared13 stores M before its seed digit reads, and the actual top12 extraction uses the correct last-digit rule.

The scoped-Z helper loads ZZ/ZZZ immediately before use and again at their updates. It preserves the corrected deferred-point formula. In each of ten intermediate additions, the old anchor is fully loaded into a separate copy, then cy is stored as the new per-lane anchor **before** calling the add. The helper consumes the old copy and uses cy early in its slope term, allowing cy's live range to end sooner. The helper touches only ZZ/ZZZ slots; the lane owns its anchor slot, so publishing early changes no operand or inter-lane dependency. The seed anchor remains the first cold point's y0.

Before final window8, the prior anchor and Z scales return to the original register arrays for the unchanged guard. After all field reads and recovery preparation, another uniform CTA join precedes reuse as the inverse tree. That join protects field slots belonging to other lanes against early tree writes; a warp-only join would be insufficient. Both added joins are outside lane-dependent early returns. Inactive/unusable lanes still participate and supply identity factors. Separate omission mutations reject either missing phase join in the structural audit.

The inverse helper accepts the512-column view explicitly. Its original body and original-ABI wrapper remain for builder/audit callers. There is no second shared tree beside the union. Shared placement adds2976 logical shared bytes per candidate relative to the regular solver:1088 store bytes and1888 load bytes. This is256 bytes below shared14's3232: one removed anchor round saves64 bytes and one removed ZZ/ZZZ round saves192 bytes. M extraction itself reads192 bytes and stores32 bytes in both variants. The32KiB allocation and two extra phase joins are unchanged. These counts exclude unchanged SHA/inverse accesses and global point loads; source payload counts do not establish hardware transactions or a scheduling benefit.

## Preserved recovery, arithmetic and frontend

The checked cubic recovery remains10M+1S for both recovery IDs, compared with its older10M+4S expression. It requires two additional modular additions/subtractions and one runtime host square/tripling with a32-byte constant upload. This is inherited work, not the new thirteen-window improvement. For XYZZ `(X/A,Y/B)`, B^2=A^3, runtime R=(r,s), it forms d=r*A−X and the batch factor W=B*d. The existing curve identity reconstructs both x coordinates and y parities using the one inverse. Whole chain+recovery is now91M+25S before the collective.

The repaired PR138 leaf-pair inverse still costs3n−3 products and one inverse per CTA. Its n=64 CTA join repair remains, with2/3/3/5 CTA joins for32/64/128/256 lanes respectively. The two arena joins are additional and do not reduce inversion work. Internal arithmetic remains canonical; PR120's lazy inverse representation is not imported.

Corrected PR77 hot multiply/square and their near-modulus carry regressions are preserved. The original PR120 arithmetic fails a=b=p−65537, returning0x1fc30 instead of0x100020001; those defective bodies are not imported. The inverse multiply is distinct and cannot stand in for testing the point-chain primitives. An earlier CPU projection hid a nonexistent helper overload; explicit field-call arity checks were added before integration and remain in the actual shared helper projection.

The inherited ranked shape gate requires one effective GPU, n=150/t=9,42 prefix-remainder bytes,218 tail bytes,44 suffix bytes and9906 total preimage bytes, without tile/easy/calibration modes. Unsupported ranked shapes fail explicitly; `QSB_RANKED_ONLY=0` retains generic source routes. Runtime hashing,32768-epoch launches, omission enumeration, packed hit records, two recovery IDs and host drain remain unchanged. The PR137 host-only packer preserves all256 selected windows and54/56 schedule classes. Neither paired SHA nor host-drain changes from that work are included.

## Native evidence and its limits

Fresh source-matched CUDA12.8.93 sm89 production compilation gives:

| Source | Registers/thread | Static shared | Stack | Compiler spill stores/loads | Non-NOP SASS | Repeated-add local bytes/lane |
|---|---:|---:|---:|---:|---:|---:|
|9c regular14 control|128|24KiB|120B|0/0B|14313|0|
|5d17 regular13 control|124|24KiB|120B|0/0B|14360|0|
|e575 shared14 control|80|32KiB|280B|376/276B|14747|484 over11 adds|
|a03 corrected shared13 control|80|32KiB|280B|344/256B|14761|40 over10 adds|
|b2fd local-M cold prefetch|80|32KiB|280B|344/256B|14803|40 over10 adds|

The new compiler summary matches a03's register/shared/stack/spill totals and adds42 non-NOP instructions. Separately, `local_m/sass-review.json`, produced by `review_native.py`, compares all1615 instructions in the b2fd repeated-add loop0x10910..0x16df0 against a03 after its0x2a0 relocation and verifies equality. Thus the loop retains40 local-load bytes per lane over ten iterations and zero local stores. All shared/local opcode counts for the complete kernel also match; those whole-kernel counts are supporting context, not a dynamic-frequency model.

For context, a03's ordinary loop was SASS0x10670..0x16b50, with one unpredicated4-byte `LDL` per iteration and one associated4-byte store before the loop, giving40 loop bytes or44 including that store. That result was transferred from e80c only after all25464 parsed device instruction strings and resource records matched following the host-policy-only fix. The present prefetch change does alter device code; the new b2fd placement receipt, rather than the older host-fix equivalence receipt, qualifies its relocated loop. The added hints are before the first seed demand load and outside that repeated loop.

Whole-kernel stack and spill totals include other paths. Seed, final guard, recovery, inverse and rare hit traffic are not fully frequency-classified here. Explicit inverse local arrays must not be counted as compiler spills without source/placement evidence. The ordinary regular controls have zero spills. Static non-NOP counts do not establish throughput or latency.

At80 registers and256 threads, three CTAs require61440 registers. Three32KiB allocations plus1KiB per-block reservation total99KiB against Ada's nominal100KiB shared capacity. Thus three-CTA residency is resource-eligible with sufficient shared carveout; launch bounds do not force achieved occupancy. The source does not set a preferred carveout. A larger shared carveout also reduces L1 capacity and may affect loads/spills. These are resource constraints, not measurements; see the [NVIDIA Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html).

Fresh b2fd production and arithmetic/inverse-audit translation units both compile for sm89 and official default flags. Each report's15 source hashes match the frozen b2fd snapshot and their union covers all16 production/audit closure files. The archived resource/source records are `cold_prefetch/local_m/public-native-results.json` and `public-audit-native-results.json`. These preserve all compiler records and source hashes while omitting private compiler-work paths, and include hashes of the original local reports. This is fresh b2fd compilation evidence, not reuse of a03's report. No NVIDIA GPU kernel, race sanitizer, driver JIT or official benchmark ran locally.

## Source-bound checks and unsuccessful controls

The a03 public-validation receipt remains evidence for the preserved base. Fresh b2fd reports are under `research/geometry_frontier/shared_thirteen/policy_corrected/cold_prefetch/local_m/`; `qualification.json` binds the completed CPU/native evidence and actual staged-root repeats. Selection is recorded separately from source qualification. Current exact-source evidence is:

- Actual shared-chain/helper CPU execution:628 chains,1242 recovered keys,494 sparse point loads,8164 prefetch checks and477 final-helper cases. All159 unguarded-doubling negative controls are detected. Only independent prefetch expectations expand from nine to thirteen windows; actual candidate bodies execute.
- Shared owner/canary, typed-stride, SHA/inverse/helper-equivalence and missing-phase-join checks remain. CPU arena projection executes one owner at a time; these structural tests do not establish CUDA race freedom.
- Fresh b2fd frontend/collective PASS:6144 SHA256d digests,3840 inverse values,10262 recodings,8086 unrankings and256 packed windows, with explicit thirteen widths and the exact b2fd fingerprint.
- Fresh actual host-policy PASS:180 capacity/limit cases and30 injected failures, matching b2fd's full closure. The old-e80c startup-abort regression remains historical evidence for the unchanged corrected policy.
- Hint-only top-last mutation: successful C++ projection compilation followed by exit86 and `QSB_PREFETCH_ADDRESS_MISMATCH` for window12. One recorded witness requests offset0xc3000000 instead of the later load's0x102ffffc0; actual arithmetic and load statements were untouched. The mutation test is neither a CUDA build nor an assertion that a wrong hint would change mathematical outputs.
- Fresh production/audit sm89/default PASS covering all16 closure files. Generated hint placement, repeated-loop equivalence and source-bound qualification now pass. The separate b2fd stage now passes the same exact-source chain, startup and hint-mutation checks, plus source/package checks. Builder and exception-domain evidence retain their unchanged-component scopes.

The first frontend invocation omitted `--widths` and was rejected by the checker's legacy geometry whitelist. The successful rerun used its existing explicit-width option; neither candidate nor checker changed to turn that failure into a pass. The domain model is separate from the source-executing point projection and is labeled accordingly.

Earlier launch-bound-only pressure at80 registers caused3636 logical local bytes in the fourteen-window loop. Separate M/anchor placement did not cure that cost; scoped Z placement and then early-anchor lifetime shortening reduced it. These controls motivate the combined architecture but do not prove that shared storage always wins. Regular13 remains a useful arithmetic/native control because it removes the same field addition without shared-state overhead, at124 registers and zero compiler spills. Its known startup boundary must be repaired before benchmarking.

Our older external-inverse [PR128](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/128), source140133dd, officially scored415082530 and was rejected, with all59428 hits verified over1201.0098 seconds. It is a different pipeline, not a matched ablation of this change. Public hit coordinates establish at least501663924224 completed combinations, consistent with ordinary hit noise. Extrapolating its maximum cumulative453.6M/s rate over the whole timeout overpredicts hits strongly; setup/non-search time or rate extrapolation is a supported hypothesis, not a measured unique cause. This successor retains large-table startup costs.

An earlier zero-hit PR86 failure had no established cause; it is not evidence of OOM. A prior archive failed the expanded8MiB limit despite a small compressed archive, so packaging checks expanded bytes and the exact closure. Some inherited search synchronization/copy API statuses remain unchecked, a diagnostic limitation untouched here. No source/static evidence substitutes for official verified throughput.

## Attribution

Promoted ancestry includes PR60 at451135044, welttowelt's PR62 at477182283, alvaroborras's [PR77](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/77) at488210159 (promotion `d2772418e0f372767b4c59f7382d71f9142585fe`), and i34-9's PR120 at495193826. These are historical whole-program scores, not measurements of this source. GPL notices and `COPYING` are preserved. PR120's direct-digit helper credit to dun999 remains.

Substantial unpromoted contributions retain the same six Yukon coauthors:

- **alvaroborras**, [PR64](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/64), `a7b21d0f62e6d73b66fe820e228f8db50d504716`: adaptive table-builder ancestry.
- **MakiRH4**, [PR46](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/46), and **jacklightChen**, [PR53](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/53), `9274883051636def6db5add0d3ba0e02314813f0`: batched ladder/builder ancestry.
- **ercumentyildirim**, pinning submission `e2fd8093-2ba5-4d40-8f25-dabb0a4807c5`: capacity/persisting-L2 observations motivating the48MiB geometry and independently checked optional policy. The author's1.203% result is not ours; no new sibling pinning implementation is included.
- **AbdelStark**, [PR138](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/138), `2dc49dc4e4083050ffc34b9be61eb027eb8c291f`: leaf-pair inverse with our necessary n64 CTA-join repair. Later cancellation does not remove credit.
- **IvanLudvig**, [PR137](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/137), `103a6adc15391e07aac210a2653f8dfd7eb4d4c0`: host-only whole-class packer. No paired-SHA or host-drain source from that work is present.

The increment in this draft is the early local-M cold-prefetch block. Matched-capacity thirteen-window geometry, its exceptional-domain argument, corrected startup policy and arena/early-anchor architecture are inherited from the preserved a03 base. Model-assisted hypotheses and independent reviews are not performance evidence.

## Reproduction and next checks

From the staged benchmark root, with the included checker dependencies:

```sh
SRC=candidates/subset
python3 -B "$SRC/research/geometry_frontier/shared_thirteen/policy_corrected/check_policy.py" --source "$SRC" --report "$SRC/startup-policy-validation.json"
python3 -B "$SRC/preflight.py" --package-check --note "$SRC/submission-shared-thirteen-prefetch.md"
python3 -B "$SRC/check_candidate.py" --source "$SRC" --selection packed --inverse-layout leaf-pair --widths 18,18,18,17,17,17,17,17,17,25,25,25,25 --report /tmp/shared13-frontend.json
python3 -B "$SRC/research/formula_fusion/check_source.py" --source "$SRC" --report /tmp/shared13-recovery.json
python3 -B "$SRC/research/geometry_frontier/shared_thirteen/policy_corrected/cold_prefetch/local_m/check.py" --source "$SRC" --output /tmp/shared13-prefetch-chain
python3 -B "$SRC/research/geometry_frontier/exception_domain.py"
yukon setup --track subset
```

The b2fd chain runner above accepts an explicit source/output and checks its expected fingerprint. It uses the actual selected body and an independent all13-window prefetch oracle. The older `policy_corrected/check.py` asserts a03 and must not be presented as a b2fd checker. The generic actual-policy runner accepts b2fd with the unchanged thirteen widths and writes the fresh root-bound report required by package validation.

Reproduce the hint-only negative control with an explicit b2fd source/checker and a designated report under its required research directory:

```sh
PREF="$SRC/research/geometry_frontier/shared_thirteen/policy_corrected/cold_prefetch"
python3 -B "$PREF/run_negative.py" --source "$SRC" --checker "$PREF/local_m/check.py" --report "$PREF/local_m/prefetch-negative-reproduction.json"
```

The runner creates temporary source copies/projections and changes only the top-last flag of the hint. The original source stays unchanged. Keep its referenced checker dependencies in the archive. The separate exceptional-domain checker still binds regular13's unchanged geometry proof; its unrelated old startup defect does not invalidate that math. The original shared13/a03 validation receipts do not qualify the new prefetch increment. The archived staged-repeat receipts and qualification record match this exact source. Refresh the note, source identity and expanded package receipt at upload.

CPU checks need Python, C++17 and OpenSSL development libraries. Fresh native commands from the benchmark root are `nvcc -O3 -DQSB_ZEROS_N=24 candidates/subset/subset.cu -lcrypto -lm` and `nvcc -O3 -DQSB_ZEROS_N=24 candidates/subset/tests/gpu_epochs/tree_audit.cu -lcrypto -lm`; resource screens add `-arch=sm_89 -Xptxas=-v,--warn-on-spills`. Compilation on the local ARM Linux compiler host does not execute GPU code. The trusted wrapper caches by the entry-wrapper timestamp, so invalidate generated `subset` and `.subset.build` after header edits before a fresh local setup/benchmark.

Before this upload, PR151 completed at410302626 and the frontier was refreshed to505611957; final package/source checks were repeated. Confirm server intake and preserve the new evaluation. The decisive next test is official verified throughput. With device access, the decisive matched prefetch comparison is a03 versus b2fd; broader architectural controls are regular14, regular13 and shared14 with the same supplied problem; measure startup, cold traffic, shared carveout, actual occupancy and local accesses separately. Until those measurements exist, this is a credible checked experiment with an explicit memory/arithmetic tradeoff, not a claim of measured superiority.
