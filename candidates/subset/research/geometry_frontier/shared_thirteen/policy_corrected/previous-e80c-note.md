# Subset: thirteen fixed-base windows with phase-reused shared state

**DRAFT — no official result for this source.** At preparation time our previous ranked/direct-digit submission `b273a801`, [PR151](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/151), remains validating. Preserve that evaluation and refresh its result, the queue and the promoted subset frontier before upload. This is a ready alternative to the separately preserved fourteen-window shared-state source e575, not a report that either successor has run on a GPU.

Effort: xhigh for GPT 6 Astra implementation and high for GPT 6 Astra independent reviewers, through Codex. The proposed change removes one deferred curve addition at unchanged table capacity and composes it with explicit shared-state placement. The intended benefit is lower arithmetic work and less repeated spilling while permitting three resident CTAs under the resource limits. Its chief uncertainty is replacing three small-table requests with two additional cold requests. No static count here is a GPU speedup claim.

## Exact source and selection

The complete production/audit include-closure fingerprint is:

`e80c8d9124fdd85ab38a6079ead00969d1e60d612a5f28c72f28523d28177908`.

The frozen snapshot is `candidates/subset/research/geometry_frontier/shared_thirteen/candidate`; an intended submission archive places identical production source at `candidates/subset/`. Paths in this note are benchmark-relative. The source generator binds two donors: the checked regular thirteen-window control `5d17e5fad6cb9d632d7b8b4977751da574242488d428cfedfb455accaa21ca5b` and the early-anchor shared architecture `e575d61ad2a0f4d9b65e8901e741c6bbe7514c1a6df6fc60118cfe537653124a`. Only geometry, table decoder and compact-chain headers differ from e575. The geometry and decoder come byte-exact from5d17; shared helpers, final guard, arena ownership, phase joins, recovery, inverse and host mechanisms remain unchanged.

Selection is an explicitly uncertain architectural experiment. Shared13 combines one fewer7M+2S intermediate addition with40 logical local-load bytes per lane in its ten-add loop, compared with484 bytes in e575's eleven-add loop. The unchanged regular controls have **zero compiler spills**, so this is not a spill reduction relative to those controls. The proposed benefit depends on whether more resident work and less arithmetic outweigh extra cold requests, shared traffic, joins and residual spills. Keep both regular13 and shared14 controls for interpreting the eventual result.

The last reviewed promoted comparison point is i34-9's [PR120](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/120), submission `99ce8411-5d19-453e-8a4c-9fd9bb14b445`, promotion `65cd138c80bf8f9070117a991c049a9da71d1de8`, at495193826 verified candidates/s. Its complete-program score does not isolate the direct-digit or ranked-compilation mechanisms inherited here and must be refreshed before upload. No source from PR156, PR157 or PR159 is imported.

## Thirteen windows at the same table capacity

The exact widths and shifts are:

```text
widths = [18,18,18,17,17,17,17,17,17,25,25,25,25]
shifts = [0,18,36,54,71,88,105,122,139,156,181,206,231]
order  = [9,10,11,12,0,1,2,3,4,5,6,7,8]
```

Nine small tables cover156 scalar bits in48MiB. Four cold tables cover100 bits in4GiB. Total mandatory affine table storage is4345298944 bytes, exactly the same as the fourteen-window control. Every table point depends on the supplied runtime base; no answer table is baked into the archive.

| Ordinary candidate work | Fourteen-window9c/e575 | Thirteen-window5d17/e80c |
|---|---:|---:|
| Small/cold point requests |12/2|9/4|
| Logical64-byte point loads |896B|832B|
| Two-point seed |12+13|9+10|
| Deferred intermediate additions |11|10|
| Final guarded window |11|8|
| Point chain |88M+26S|81M+24S|
| Chain plus cubic recovery |98M+27S|91M+25S|

M/S counts exclude SHA, inverse collective, table construction, exceptional fallback and memory costs. Seed9+10 costs3M+2S; the ten intermediates11,12,0..7 cost7M+2S each; the final resolved addition at8 costs8M+2S in its ordinary branch. Only top window12 uses `last=true` in the direct-digit helper. Final addition window8 is an ordinary signed digit. Small-table prefetch targets only0..8. Signed-Y handling, magnitude indexing and size_t byte addressing are retained.

The total point-load count falls, but cold requests double. At hypothetical small/cold hit rates hs/hc, the change in logical miss bytes is `64*(-1+3*hs-2*hc)`. Perfect small hits and no cold hits would add128 bytes per candidate. This is a simple request model, not measured DRAM traffic or latency; actual transactions, reuse and overlap can change the result. Neither the48MiB allocation class nor its policy implies guaranteed L2 residency.

The bounded runtime builder and its checkpoint discipline remain. Host ladder arrays now total13MiB, plus OpenSSL objects; the thirteen-window count replaces the prior fourteen-window array sizing. Explicit temporary GPU buffers retain48497152 bytes, with bounded chunks of at most1048576 entries and no full-table CPU fallback. Temporary buffers are released before search. Geometry/decoder bytes match the checked regular13 donor, whose CPU builder checks cover4491 entries at three runtime bases,101497 decoder cases and69811 ladder points per base. That evidence does not execute the GPU builder.

The optional checked L2 policy requests48MiB for the contiguous small-table prefix only when the selected device supports the needed capacity/window. It checks API status and effective limits; unsupported configurations use ordinary caching. No physical pinning guarantee or measured policy gain is claimed. This geometry does not remove mandatory table startup cost.

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

Source-matched CUDA12.8.93 sm89 compilation gives:

| Source | Registers/thread | Static shared | Stack | Compiler spill stores/loads | Non-NOP SASS | Repeated-add local bytes/lane |
|---|---:|---:|---:|---:|---:|---:|
|9c regular14 control|128|24KiB|120B|0/0B|14313|0|
|5d17 regular13 control|124|24KiB|120B|0/0B|14360|0|
|e575 shared14 control|80|32KiB|280B|376/276B|14747|484 over11 adds|
|e80c shared13|80|32KiB|280B|344/256B|14761|40 over10 adds|

The exact shared13 ordinary loop is SASS0x10670..0x16b50. Its sole local instruction is unpredicated `LDL R73,[R1+0x7c]` at0x14b40, feeding the digit-sign XOR at0x16ab0. It executes once in each of ten iterations. The associated4-byte store at0xe1c0 occurs once before the loop:44 logical bytes if that store is included. No local stores occur inside the loop. Internal branches cannot skip/repeat the load. These are logical instruction bytes, not physical memory transactions.

Whole-kernel stack and spill totals include other paths. Seed, final guard, recovery, inverse and rare hit traffic are not fully frequency-classified here. Explicit inverse local arrays must not be counted as compiler spills without source/placement evidence. The ordinary regular controls have zero spills. Static non-NOP counts do not establish throughput or latency.

At80 registers and256 threads, three CTAs require61440 registers. Three32KiB allocations plus1KiB per-block reservation total99KiB against Ada's nominal100KiB shared capacity. Thus three-CTA residency is resource-eligible with sufficient shared carveout; launch bounds do not force achieved occupancy. The source does not set a preferred carveout. A larger shared carveout also reduces L1 capacity and may affect loads/spills. These are resource constraints, not measurements; see the [NVIDIA Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html).

Fresh production and arithmetic/inverse-audit translation units compile for both sm89 and the official default architecture flags. Their verified hash union covers all16 production/audit include-closure files. No NVIDIA GPU kernel, race sanitizer, driver JIT or official benchmark ran locally.

## Source-bound checks and unsuccessful controls

`research/geometry_frontier/shared_thirteen/public-validation.json` binds the e80c closure, native compiler records and fresh staged CPU reports. The complete development review also verifies the immutable controls. Together they record:

- Actual shared-chain/helper CPU execution:628 chains,1242 recovered affine keys,494 sparse loads,5652 prefetch assertions and477 final-helper cases. All159 unguarded-doubling negative controls are detected.
- Shared arena ownership across all256 owner lanes with other-lane canaries, correct row strides, unchanged SHA/inverse/helper bodies, and mutations rejecting either omitted phase join. CPU projection executes one owner at a time; structural joins do not constitute GPU race testing.
- Fresh frontend/collective projection:6144 SHA256d digests,3840 inverse values,10262 recodings,8086 unrankings and all256 packed window selections against the independently supplied thirteen widths. Separate staging checks repeat the actual chain and explicit-width frontend against the staged production root, whose complete fingerprint remains e80c; the development evidence is not the only source binding.
- Regular-chain byte equality to the checked13 donor, shared-helper/final-guard equality to e575, and exact native production/audit closure coverage. Builder evidence is inherited by exact geometry/decoder identity and unchanged host routines, not a newly executed GPU builder test.

The first frontend invocation omitted `--widths` and was rejected by the checker's legacy geometry whitelist. The successful rerun used its existing explicit-width option; neither candidate nor checker changed to turn that failure into a pass. The domain model is separate from the source-executing point projection and is labeled accordingly.

Earlier launch-bound-only pressure at80 registers caused3636 logical local bytes in the fourteen-window loop. Separate M/anchor placement did not cure that cost; scoped Z placement and then early-anchor lifetime shortening reduced it. These controls motivate the combined architecture but do not prove that shared storage always wins. Regular13 remains a useful alternative because it removes the same field addition without shared-state overhead, at124 registers and zero compiler spills.

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

The new increment here is matched-capacity thirteen-window geometry, its exceptional-domain argument, and its composition with the previously checked arena/early-anchor architecture. Model-assisted hypotheses and independent reviews are not performance evidence.

## Reproduction and next checks

From the staged benchmark root, with the included checker dependencies:

```sh
SRC=candidates/subset
python3 -B "$SRC/preflight.py" --package-check --note "$SRC/submission-shared-thirteen.md"
python3 -B "$SRC/check_candidate.py" --source "$SRC" --selection packed --inverse-layout leaf-pair --widths 18,18,18,17,17,17,17,17,17,25,25,25,25 --report /tmp/shared13-frontend.json
python3 -B "$SRC/research/formula_fusion/check_source.py" --source "$SRC" --report /tmp/shared13-recovery.json
python3 -B "$SRC/research/geometry_frontier/shared_thirteen/check.py"
python3 -B "$SRC/research/geometry_frontier/exception_domain.py"
yukon setup --track subset
```

The successful staged-root chain run uses a path-only adaptation of the shared13 checker, with exact source-identity verification. The command above, without that adaptation, currently targets its included frozen `shared_thirteen/candidate` and writes reports alongside that snapshot; it does not accept a staged-source override. Verify that this snapshot's complete source identity equals staged SRC before attributing that run to the archive. The domain checker separately binds the regular13 donor used for the proof. Keep the donor/control snapshots and referenced projection dependencies with these commands. The development-only `validate_reports.py` checks the original compiler artifacts and immutable controls; the compact upload instead includes `public-validation.json` with source hashes, native resources and staged CPU receipts. It omits private compiler paths. Neither record creates native evidence anew.

CPU checks need Python, C++17 and OpenSSL development libraries. Fresh native commands from the benchmark root are `nvcc -O3 -DQSB_ZEROS_N=24 candidates/subset/subset.cu -lcrypto -lm` and `nvcc -O3 -DQSB_ZEROS_N=24 candidates/subset/tests/gpu_epochs/tree_audit.cu -lcrypto -lm`; resource screens add `-arch=sm_89 -Xptxas=-v,--warn-on-spills`. Compilation on the local ARM Linux compiler host does not execute GPU code. The trusted wrapper caches by the entry-wrapper timestamp, so invalidate generated `subset` and `.subset.build` after header edits before a fresh local setup/benchmark.

Before upload, refresh PR151 and the frontier, preserve any pending evaluation, verify exact e80c stage identity and expanded archive size, and confirm server intake. The decisive next test is official verified throughput. With device access, the useful matched comparisons are regular14, regular13, shared14 and shared13 with the same supplied problem; measure startup, cold traffic, shared carveout, actual occupancy and local accesses separately. Until those measurements exist, this is a credible checked experiment with an explicit memory/arithmetic tradeoff, not a claim of measured superiority.
