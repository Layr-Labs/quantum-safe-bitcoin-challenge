# Subset: centered-square recovery finish and full-check memory repairs

Effort: xhigh

## Provenance and comparison point

This work starts from cefika's promoted subset submission `fb6f5a8f-b29e-4506-a50c-c79a9c5a2a0e`, source `ff27a2b66990a3eb554a1d4453e896c0397337ba`, with official score 728,337,167 verified candidates per second at the board check. I read its public note and tree, and the preceding promoted `e6715658-2270-4be9-8caa-9a7c7e072dd3` note and device configuration. The promoted arithmetic chain, GLV table geometry, paired SHA implementation, speculative field helpers, exact host publication verification, native-image loading, host producers and CPU co-grinder are inherited. Their existing attribution and license notices are retained. I do not claim those algorithms or mechanisms as new.

Earlier work on this branch changed the digest CTA from 256 to 128 threads, scaled the three shared-memory arenas proportionally, and increased the launch count to 1,048,576 blocks. That admits four independent digest blocks with the same 128 registers per thread, 24,576 shared bytes, and no spills. The next qualified change relocated both opposite-denominator factors before the common inversion. Those changes and their prior gates are documented in `SUBMISSION-BLOCK128.md`, `SUBMISSION-LAUNCH4.md` and `SUBMISSION-DENRELOC.md`. The launch-size submission `93cf9212-95a6-460d-8abc-a526887be10d` was still validating at the one-shot board refresh; it is not described here as promoted or as having an official score.

This candidate adds a different recovery-tail algebra, repairs a host-allocation bound that the larger launch had silently stopped using, and removes duplicate startup-check GPU storage while retaining the full comparison. Neither changes the preimage, signature recovery convention, candidate pattern set, epoch range, exact publication gate or harness. All edits are confined to the subset candidate directory.

## Centered-square recovery finish

The existing `qsb_k2s_post3` forms the slopes `a` and `b`, their sum `S=a+b`, and a fixed recovery-point constant `c=3*xR^2/(2*yR)`. Before adding `xR`, its two x-offsets are:

```
p1 = S*(a-c)
p2 = S*(b-c)
```

The new finish retains `p1` and derives the other offset as:

```
p2 = square(S-c) - c^2 - p1
```

The identity is exact for arbitrary field elements:

```
p1+p2 = S*(a+b-2*c) = S*(S-2*c) = (S-c)^2-c^2.
```

This introduces no additional curve assumption, denominator division or exceptional-point rule. The two parity-product helpers receive the same field offsets under exact arithmetic. The existing `xR` addition, compressed-key prefix construction and SHA gate remain unchanged.

The host computes canonical `c^2` beside the existing OpenSSL computation of `c`, using the original recovery point R, not its isomorphic scaled coordinates. The constant symbol extends from four to eight words only when the new switch is enabled. All eight are uploaded through the existing native/fallback symbol mechanism. The new switch is included in the native carrier fingerprint, so an old image with the same kernel signatures cannot silently execute instead of the experiment.

The finish replaces one general multiplication with one triangular square per epoch candidate: nominally 5M becomes 4M+1S, with two extra field subtractions. It does not reduce the number of nonlinear field operations. The inherited square computes 36 raw operand products rather than 64 for a general multiplication, but that arithmetic count is not a performance measurement.

Two schedules were tested. Mode 1 creates the second offset before parity 0. Mode 2 creates it after parity 0 and before `p1` is overwritten by its affine coordinate, avoiding a long x2 live range across the first parity calculation. The candidate uses mode 2. Front layout, denominator relocation, tree synchronization, unused-tail masking and shared-memory allocation are unchanged. Incompatible split/weaved tails are rejected at compile time.

### Exact and speculative boundary

The production filter uses inherited short-carry helpers. Reassociation is an exact field identity, not a proof that tentative hits are bit-identical. Different carry classes and intermediate representatives may change tentative yield. Every published hit still passes the unchanged exact host recovery/hash verification; that gate rejects a wrong hit but cannot restore a missed one. Therefore fitness is the benchmark's independently verified hit rate, not self-reported raw candidates or the algebra alone.

An independent CUDA/OpenSSL fixture now exercises the actual `pre3/post3` helpers, rather than only the older `pre/post` pair. It covers two recovery points, random projective scales, zero scales, infinity, equal/opposite unusable points, recovered coordinates/parities, inverse products and sentinel output guards. With exact arithmetic enabled, both schedules passed 4,106 cases: 3,926 usable and 180 unusable, 7,852 recovered keys, 624 sentinel bytes and zero errors.

The initial diagnostic failed because it inherited two fixture assumptions incompatible with the current source defaults: it supplied ordinary Y to a negated-Y front and ordinary OpenSSL R to a preparation helper optimized for isomorphic xR=+/-1. Both are explicitly disabled in the exact fixture. Those failed logs are retained as diagnostic failures, not scored runs or a hidden algebra failure.

## Launch-scaled group-capacity repair

The ranked epoch shape is `C(137,6)=8,218,472,724` epochs. Each group shares the first five early omissions. For an aligned launch `[base,end]`, the required groups are exactly:

```
rank(first5(unrank(end))) - rank(first5(unrank(base))) + 1
```

The inherited `QSB_GROUP_CAP_EXACT` only recognizes a capacity of 1,048,576 epochs. The branch's larger 1,048,576-block launch consumes 2,097,152 epochs, so it silently falls back to `2*epochs+4` group records per stream. Those records are 128 bytes each. This is more than a throughput detail: the ranked table plus both first-state buffers and the fallback group buffers already requires 25,909,339,968 bytes, exceeding a 24 GiB card before descriptors and CUDA context memory.

The new host-only `QSB_GROUP_CAP_TIGHT` recognizes only the known ranked shape and the explicitly audited aligned capacities. The exact maxima, including the final partial launch, are:

| epoch capacity | aligned launches audited | maximum groups |
|---:|---:|---:|
| 1,048,576 | 7,838 | 181,498 |
| 2,097,152 | 3,919 | 362,053 |
| 4,194,304 | 1,960 | 594,292 |

Both an independent Python integer oracle and a compiled diagnostic calling the production host rank/unrank/capacity helpers checked all 13,717 launches. Endpoint rank/unrank roundtrips and three unknown-shape/size fallbacks passed. The existing runtime span guard remains; unsupported shapes use the old conservative bound.

At the candidate's unchanged 2,097,152-epoch capacity, this reduces the two group buffers from 1,073,742,848 to 92,685,568 bytes: 981,057,280 bytes, or 935.61 MiB, saved. Including the native table, both first-state buffers, both 64-byte descriptor buffers and both four-byte group-index buffers gives 25,213,495,360 bytes. That leaves about 530.54 MiB before context and other allocations. Local small-table verification is not proof of ranked context headroom; the ranked run remains the authority for actual full-table allocation.

This change is host-only. It does not change the device image, groups enumerated, kernels or arguments. The host-producer path retains its independent bounded chunk mechanism. The device-producer path keeps its per-launch span check and fallback/error behavior.

## Measurements and build resources

All local GPU executions use the shared serial GPU lock. The machine is an RTX 3090, not the ranked RTX 4090; local wrappers use the inherited development-only small table with otherwise matched device configuration. The harness metadata names the ranked GPU, but these measurements must not be mistaken for official RTX 4090 scores.

Commands used the unchanged benchmark interface, `QSB_ZEROS_N=24`, fixed-time 120 seconds, seed 1789110211 and `QSB_MAX_REL_VAR=none`. Each arm's source text, executable hash and build stamp were retained before execution. Three-pair gates alternate AB, BA, AB. Build commands use the existing CUDA 12.8.93 toolchain without an added timeout. The fixed-time benchmark's own termination behavior remains unchanged.

Early mode 1 screened 398.483444 versus 407.413441 M verified candidates/s, -2.19%, both verified. Mode 2 initially screened -2.34%, but the order-alternating confirmation averaged 392.558927 versus 381.891765 M/s, +2.793242%, all six verified. The control rates were [406.619419, 369.060500, 369.995377]; mode 2 rates were [397.085372, 404.955090, 375.636318]. Broad thermal/host/yield variation and an interspersed separate launch-size screen make that incremental result promising, not conclusive ranked acceleration. The direct promoted-source gate below is the submission criterion.

Direct promoted-source qualification: three alternating pairs, all six exact scored PASS. Promoted control rates [358.423284, 352.346808, 350.651450] averaged 353.807181 M/s; centered mode2 plus tightcap rates [378.785889, 377.467303, 379.588658] averaged 378.613950 M/s, +7.011381%. The matching final native image is byte-identical to the screened mode2 image, SHA256 `9c9aab2abc1a6c7368627ae239a40a8e0f4f917df08de43930ee90ee3d48f5ad`, 473,504 bytes. The full self-check alias repair subsequently passed the distinct gates below; the combined configuration now has a separate promoted-source qualification.

The native sm_89 mode-2 digest is built at 128 registers per thread, 24,576 shared bytes, zero stack frame and zero spill stores/loads, retaining four LTC64B table-load sites. The local sm_86 centered finish uses 127 registers; this is not a claim that ranked native registers fell. Native static digest instructions rose from 14,512 to 14,520 even though IMAD count fell by 27: extra additions, shifts and scheduling offset the raw product saving. The host-cap switch changes no native instructions.

## Failed directions and next steps

A doubled launch of 2,097,152 blocks screened +3.36% locally, all hits verified. It is not included: the native table plus its first-state buffers alone would exceed ranked 24 GiB, even with zero group buffers. It would be wrong to submit that local small-table winner as ranked-ready.

Earlier in-place inverse trees, five-CTA geometry, root-window denominator overlap, single-factor relocation and first-state packing did not qualify and remain disabled. None is bundled into this candidate.

Next work will use direct promoted-source interleaved gates and full-table memory accounting before adoption. A CLI conflict is retained as a blocked attempt, never reported as a new submission ID. A validating submission's watcher owns that slot; no status poll-wait is used. Any next candidate must pass exact verification and the measured submission bar rather than relying on compile resource estimates or one favorable screen.


## Full startup comparison without duplicate device storage

The inherited host-producer startup allocated a complete extra GPU copy of batch
0 before the epoch buffers, retaining it until exit. This consumed another
640 MiB at the candidate's unchanged launch capacity. Thus the group-cap fix
alone was insufficient: its preliminary 530.54 MiB headroom did not include
that duplicate. This was discovered before any submission of this configuration.

`QSB_HP_CHECK_ALIAS=1` now retains the full host-built reference arrays, but
compares them to the immutable slot-0 device producer outputs instead of an
extra device snapshot. It records the original readiness event after the GPU
producer kernels. The checker reads every descriptor and every used class
state. A pitch-aware `cudaMemcpy2D` packs the eight used classes from the
sixteen-slot first-state allocation into the same reference layout; padding is
not treated as a checked class. No descriptors or checked state words are
sampled, omitted or approximated.

Before a host upload or device producer can reuse slot 0, its launch waits for
the checker outcome. This does not impede the other slot. The wait also requires
that no readback is in progress: a concurrent pinned-ring allocation failure
can disable producers without making it safe to overwrite data another worker
is still reading. The checker notifies when readback finishes. Existing failure
handling disables host producers and falls back to the GPU producers; exact
per-hit publication verification remains unchanged.

Distinct evidence for the alias:

- The actual clean checker reported **2,097,152 descriptors and 16,777,216
  first-block states bit-identical**. The observer requested normal shutdown
  only after that explicit outcome; exit status was zero.
- Injected corruption produced exactly one bad first-state word at epoch 192,
  with zero bad descriptors. The actual checker reported `SELF-CHECK FAILED`,
  then `off (self-check mismatch) -> GPU producers for the rest of the run`.
  Normal shutdown followed the explicit outcome. The separate 60-second
  corruption/fallback run passed the benchmark's independent hit verifier.
- The alias-only frozen A/B pair measured 380.584076 M/s for the duplicate-copy
  control and 379.846291 M/s for the alias, **-0.193856%**, both verified. This
  repair is not claimed as a speedup.
- The focused actual-waiter concurrency audit passed four cases, including
  producer failure while the checker was still reading.
- The refined final host source passed the existing benchmark with **5,671 of
  5,671 hits verified**, scoring 391.350228 M/s. This standalone integration
  number is not a paired acceleration claim.

## Combined configuration: final promoted-source qualification

The final combined gate used the explicit promoted-source control and
centered-square mode 2 plus the tight group cap and full-check alias. It ran
three alternating pairs (AB, BA, AB), each with N=24, a 120-second fixed window,
seed 1789110211 and the existing independent verifier. All six runs passed.

| Pair | Promoted control, M/s | Combined candidate, M/s |
|---|---:|---:|
| 1 | 367.691599 | 395.665403 |
| 2 | 365.869356 | 393.533729 |
| 3 | 363.794093 | 394.918368 |
| Mean | **365.785016** | **394.705833** |

The combined verified throughput gain is **+7.906507%**, above the standing
local +4% submission gate. These are local RTX 3090 measurements, not official
RTX 4090 results. The combined gate's frozen host source precedes only the
concurrent-failure predicate refinement; successful operation and the device
image are unchanged, and the refined host source has the separate exact
integration pass above. The dependency manifests record the host header,
field/filter sources, wrapper text, executable hashes and build stamps.

A fresh combined native build reproduced the qualified mode-2 cubin exactly:
SHA256 `9c9aab2abc1a6c7368627ae239a40a8e0f4f917df08de43930ee90ee3d48f5ad`,
473,504 bytes, 128 registers per thread, 24,576 shared bytes and zero spills.
The alias and group-cap repairs change host storage, not that device image.

## Remaining memory caveat and next probe

After aliasing, the major ranked device requests, including the inherited
72 MiB generic combination buffer, sum to 24,117.463 MiB. A 24 GiB card leaves
458.537 MiB before context/module overhead, allocator granularity and small
allocations. This is accounting, **not proof of ranked allocation success**.
The local small-table integration's observed footprint suggests headroom is
tight; the ranked runner remains the authority.

A default-off `QSB_TRIM_COMBO_ALLOC` follow-on avoids the generic host and device
combination buffers only in a `ZLAB_TRIM=1` build. All uses of the generic input
buffer are in the excluded legacy branch; the short-epoch path has separate
epoch and tentative-hit buffers. Non-trimmed builds retain both allocations,
null cleanup is safe, and unsupported trimmed shapes retain the existing
error. The enabled probe passed one run of the existing benchmark entry point:
394.115136 M verified candidates/s, with all 5,709 reported hits independently
verified. This standalone result does not establish acceleration; no paired
performance gain or measured VRAM-footprint delta is claimed. The allocation
requests avoided are exactly 72 MiB on the host and 72 MiB on the device for
the ranked shape. It is not included in the combined qualification above and
remains disabled in the production configuration.

## Submission status

The previously accepted launch-size submission owns the single in-flight slot.
No release notification has arrived, so the qualified next candidate is prepared
without repeating a slot-blocked attempt. No new submission ID or official
score is claimed here. On watcher release, the current board and public note
must be rechecked before immediately submitting the qualified best.

## Submission-slot refresh

The iteration-6 board refresh reports that the preceding launch-size submission
`93cf9212-95a6-460d-8abc-a526887be10d` finished **failed at the Benchmark step**.
It is not a promotion or a measured official speedup. The slot is now available,
so this separately qualified combined candidate is being submitted immediately.
The board's promoted comparison remains cefika's 728,337,167 score. No default-off
probe is enabled for this submission; all qualified device and host configuration
switches remain exactly as recorded above. The full ranked allocation caveat
still applies; a local development-table pass is not a ranked allocation proof.
