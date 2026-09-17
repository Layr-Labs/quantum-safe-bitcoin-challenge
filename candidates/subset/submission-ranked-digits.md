# Subset: fused fourteen-window recovery with direct digits and a specialized frontend

Effort: xhigh for GPT 6 Astra implementation and high for GPT 6 Astra reviewers,
through Codex. No local GPU score or throughput improvement is claimed. This
candidate combines a substantial table/point-chain redesign with corrected
field arithmetic, cubic recovery, a repaired leaf-pair inverse, host schedule
packing, and two compatible mechanisms from the latest promoted frontend.

## Exact candidate and comparison

The complete production/audit include-closure fingerprint is
`9c812950d091bd60c653e98076cd676ee419410c76cf3e24053083ecbeb546ca`.
The development snapshot is `research/ranked_specialization/digits_candidate`
under the subset editable path. All paths in this note are benchmark-relative.
The archive places that exact source in `candidates/subset/`.

The latest reviewed promoted frontier is i34-9's
[PR120](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/120),
submission `99ce8411-5d19-453e-8a4c-9fd9bb14b445`, promotion
`65cd138c80bf8f9070117a991c049a9da71d1de8`, with **495193826 verified
candidates/s**. Its whole-program result does not isolate any one mechanism.
This candidate reuses its direct-digit helpers and ranked-compilation idea,
not its entire solver, measured score, packed inverse, hit buffer, launch-size
change or original hot field arithmetic.

The immediate common control is the checked combined fused source
`581e22633f7120ace8605576e64c5022ac67b93660920b100aebd5944e35f8e9`.
That control descends from corrected
[PR77](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/77),
promotion `d2772418e0f372767b4c59f7382d71f9142585fe`. PR77 scored
488210159 before our arithmetic corrections and geometry changes; neither
modified source inherits that measurement. There is no measured GPU baseline
for the immediate control or the current candidate.

Our separate external-inversion entry `c571025c`,
[PR128](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/128),
source `140133ddb7a0408750a5396ce20658a1b48591422b5974315c8b76110de3117c`,
completed official evaluation on2026-09-17 at **415082530 verified
candidates/s**, below495193826, and was rejected for insufficient score.
All59428 reported hits verified over1201.0098 seconds. This is performance
feedback, not a correctness rejection. The harness's extrapolated/self-reported
candidate count implied453.6M/s and was outside the hit-count Poisson band;
only the verified415.08M/s is the official score. That discrepancy does not by
itself identify thermal decay, dropped hits, startup cost or an arithmetic bug.

That older source imported the PR77 frontend into an external checkpoint
pipeline. The present fused candidate instead performs its inverse inside each
search CTA and allocates no external search checkpoint storage. It also adds
cubic recovery, repaired leaf pairs, host packing, direct digits and frontend
specialization. The415.08M result weakens any claim that the shared table
geometry alone must win; it is not a matched ablation of the fused successor.
No superiority is measured. The new candidate remains a credible experimental
contender on its checked architecture/resource evidence. The earlier evaluation
completed normally and was never cancelled.

## Substantial inherited architecture

The runtime fixed-base table uses widths `[17]*12 + [26]*2`. Twelve small
tables occupy 48 MiB; the two cold tables occupy 4 GiB. Mandatory affine table
storage is **4345298944 bytes**. Each candidate loads fourteen 64-byte points,
896 logical bytes, in order `[12,13,0,1,...,11]`. Starting with the two cold
points allows their independent addresses to be available before the smaller
table chain. Address-only prefetch remains on the small-window path.

The ordinary point chain costs **88M+26S**, compared with 95M+28S for the
fifteen-window compact chain. This is a source operation count, not measured
GPU work or speed. The seed uses the actual deferred affine-add helper;
intermediate additions retain the deferred Y anchor. Only the final addition
resolves the exact Y and checks exceptional equal/opposite inputs.

The table is built from the supplied problem's runtime base in 65 bounded
chunks, each at most 1048576 entries. Explicit temporary GPU buffer storage is
48497152 bytes; host ladder arrays occupy 14680064 bytes plus OpenSSL point
objects. The builder samples 23552 table bytes rather than copying the whole
table to the host. Temporary buffers are released before searching. There is
no CPU full-table fallback, embedded answer cache, fixed problem seed, optional
16 GiB geometry, or runtime geometry tuner.

The optional L2 policy requests 48 MiB only after checking the selected
device's capacity, effective limit and access-window support. Its window
covers the small-table prefix. Unsupported capacity uses ordinary caching
with the same table. A priority window does not guarantee residency or a
particular cache hit rate. Startup and JIT time count against official scoring.

The final-add guard is necessary. Random samples previously missed constructed
equal-point witnesses where the incomplete mixed formula returns a zero
denominator instead of doubling. The selected order has a bounded
recoder-domain argument for nonsingular seed/intermediate additions, while
the final helper handles equal points by doubling the affine input and
opposite points by returning infinity. This is not a claim that random tests
prove the full CUDA program. Singular recovery denominators retain the
inherited skip/identity-padding behavior.

## Cubic recovery and inverse collective

Write the valid XYZZ source point as `(X/A,Y/B)`, where `A=ZZ`, `B=ZZZ`
and `B^2=A^3`. Let the supplied recovery point be `R=(r,s)` on secp256k1.
With `d=r*A-X`, the CTA inverts `W=B*d` for usable lanes. Define

```text
h=A/W; k=B*h; alpha=s*k; beta=Y*h; c=3*r^2
center=2*alpha^2-c*k+r
cross=2*alpha*beta
x1=center-cross; x2=center+cross
y1=(alpha-beta)*(r-x1)-s
y2=s-(alpha+beta)*(r-x2)
```

The actual curve equation yields this rewrite. It replaces 10M+4S recovery
with **10M+1S**, adding two modular additions/subtractions. One host square,
tripling and a 32-byte constant upload compute `c` from the runtime point.
The existing Y parity and second recovery-ID sign convention are retained.
The formula is not an identity on arbitrary off-curve coordinate tuples.

The repaired PR138 leaf-pair collective retains `3n-3` field products and
one inverse per block. Neighbor shuffles combine leaf pairs before the shared
tree; the pair inverses expand back to individual lanes. Our repair inserts
a uniform CTA join for n=64: warp 0 writes pair inverses that warp 1 consumes,
so a warp-only join was insufficient. For 32/64/128/256 threads, CTA joins are
2/3/3/5. At ranked n=256 this removes two joins from the common predecessor.
Inactive or singular lanes participate with identity factors. Outputs remain
canonical; PR120's lazy intermediate inverse representation is not imported.

## Direct digits: removing a serial peel state

The first new change imports byte-exact PR120 helpers `gt_field_bits_v` and
`gt_direct_digit`. PR120 credits dun999's earlier direct regular-digit
mechanism. Their SHA256 values are respectively
`6aeb908a93de6d1a4e28d20f41d39838a1ee1cb96e0306826c9f3702cc27f90b` and
`c291852d21658448963484bc6974568f3dec63b79530b46143ca74b2058081fa`.

The old small-window path repeatedly shifted a four-limb recoding state and
converted each signed digit to an index/sign. If M is the original odd
representative, its state before window c is `(M >> shift(c)) | 1`.
The direct helper reads w bits starting at `shift(c)+1` and derives the same
index/sign without updating M. For the twelve small windows w=17. The last
flag is false for all twelve because window 11 is not the final scalar
window. The two existing cold digits already use direct extraction and remain
unchanged, as do point order, loads, prefetches and final exceptional handling.

This is an adaptation to our fourteen-window layout, not a copy of PR120's
fifteen-window loop. It changes only `compact_table_device.cuh` relative to
control581e; restoring two call sites reconstructs the original point-chain
body exactly. The direct-digit-only candidate has fingerprint
`4072fb75b8c0785f0abc0a7bc342ff9693bd45a6e3cba77d8c7d672ca6ee37d4`.

## Ranked frontend specialization and observed interaction

The second change adopts PR120's compile-time frontend specialization idea.
`QSB_RANKED_ONLY=1` is the default. It directly reads the block's runtime epoch
descriptor and calls the existing scheduled hash, omitting generic byte
emission, generic unranking and generic host search routes from that build.
The existing host eligibility condition must pass before table allocation:
one effective GPU, no tile path, no easy/calibration mode, n=150, t=9,
42 prefix-remainder bytes, 218 tail bytes, 44 suffix bytes and 9906 total
preimage bytes. Unsupported shapes fail explicitly. Compiling with
`-DQSB_RANKED_ONLY=0` restores the original generic kernel and search routes.

All scheduled hash, SHA256d, EC, recovery, inverse and gate expressions are
preserved. The hit record layout and host drain are unchanged. The 32768-block
launch size is unchanged. Both table construction and hash schedules still
derive from the actual input problem. This optimization is shape
specialization, not precomputation of answers.

The two changes interact in native compilation. They must not be assigned
independent additive speedups:

| Exact variant | Registers | Shared | Stack | Spill store/load | Static non-NOP |
|---|---:|---:|---:|---:|---:|
| Combined control581e | 128 | 24576 B | 336 B | 0/0 B | 29830 |
| Direct digits4072 | 128 | 24576 B | 352 B | 4/4 B | 29786 |
| Specialization88b2 | 128 | 24576 B | 136 B | 20/12 B | 14409 |
| Both9c812950 | 128 | 24576 B | 120 B | 0/0 B | 14313 |

These are CUDA12.8.93 sm89 native compiler reports, not GPU measurements.
Specialization alone adds hot recoder spill accesses inside the eleven-add
loop. SASS review places three scalar loads and two scalar stores in that
loop, plus three initial scalar stores: 232 logical bytes per candidate in
this specific build, before coalescing/cache effects. Direct digits remove
that recurrence state in the combined build, which reports zero spills.
Explicit stack accesses inside root inversion remain; stack size and compiler
spill counts are different quantities. Much of the static code reduction
comes from omitted generic paths, so it is not a halving of executed work.

## Field correctness and evidence

PR120's original hot multiply/square still fail our near-modulus carry
witnesses. For a=b=p-65537, the defective square returns `0x1fc30` instead of
`0x100020001`, differing by `2^32+977`. We preserve the corrected PR77
multiply and square headers, including final-carry repair and canonicalization.
The tree multiply is a separate primitive with separate evidence. A passing
tree multiplication test cannot validate the hot point-chain arithmetic.

Exact new-source evidence is:

- Actual helper tests: 17278 boundary/random full-uint256 scalars, 241892
  digit comparisons, 34556 unchanged cold-digit comparisons; off-by-one bit
  starts and wrong final-window flags are rejected by negative controls.
- Actual guarded point chain into cubic recovery: 628 chains and 1242
  affine-key comparisons, including 214 constructed scalar witnesses;
  477 final-guard cases and 159 rejected unguarded doubling controls.
- Actual scheduled frontend and collective projection: 6144 SHA256d,
  3840 inverse outputs, 10262 reference recodes, 8086 unrankings and all
  256 packed window selections.
- Source/control-flow audit: 140 compiled runtime-shape predicate cases;
  flag0 generic kernel token-equivalent to its input source; ranked build
  has one consumer launch site; SHA256d through recovery/gate tokens unchanged.
- Production and GPU-audit translation units compile successfully with sm89
  and the official default architecture flags, in the existing local ARM
  Linux CUDA compiler VM on this Mac.

Unchanged-component evidence from control581e includes the integrated builder
check on 13107 entries across three runtime bases; cubic recovery on 4077
finite cases/8154 keys and115 singular cases; field host/PTX carry checks;
the 64-thread inverse synchronization witness; and 180 optional L2-policy
host scenarios with30 injected failures. These component results are bound
to their original hashes; they are not fresh GPU tests of this candidate.

CPU projections use OpenSSL field/curve references, explicit host barriers
and actual extracted helper bodies. They do not execute CUDA instructions or
prove GPU scheduling. Native compilation likewise does not provide CUDA
sanitizer results, device execution, cache measurements, throughput, or a
guarantee about the runner's driver JIT. No local claimed score is attached.
Some inherited search-loop synchronization and copy return statuses remain
unchecked; startup and builder checks do not cover that runtime-diagnostic gap.

## Attribution and prior unsuccessful work

Promoted ancestry includes our PR60 at451135044, welttowelt's PR62
at477182283 and alvaroborras's PR77 at488210159. Their results describe their
whole programs. Original GPL source notices and `COPYING` are preserved.

Substantial unpromoted contributions remain credited with Yukon coauthors:

- **alvaroborras**, PR64 `a7b21d0f62e6d73b66fe820e228f8db50d504716`,
  adaptive table construction ancestry.
- **MakiRH4**, PR46, and **jacklightChen**, PR53
  `9274883051636def6db5add0d3ba0e02314813f0`, batched ladder/builder ancestry.
- **ercumentyildirim**, pinning submission
  `e2fd8093-2ba5-4d40-8f25-dabb0a4807c5`, capacity/persisting-L2 observations
  that motivated the48MiB geometry and independently checked optional policy.
  The author's measured1.203% policy improvement is not our result.
- **AbdelStark**, PR138 `2dc49dc4e4083050ffc34b9be61eb027eb8c291f`,
  leaf-pair inverse design, with our necessary n64 repair. Its later
  cancellation does not change its unpromoted contribution.
- **IvanLudvig**, PR137 `103a6adc15391e07aac210a2653f8dfd7eb4d4c0`,
  exact host-only whole-class packer. It keeps the256 windows and54/56 classes,
  reducing summed per-warp second-class groups from62 to56. No paired SHA
  or host-drain change from that entry is present here.

The PR120 direct-digit and trim contributions are promoted work and are cited
as such. Its helper's dun999 provenance is retained. Model review is a source
of hypotheses and review assistance, not independent validation or measured
performance evidence.

Prior lessons materially affected this preparation. PR86 exited early with
zero hits and no score; available diagnostics did not establish its cause.
An earlier archive exceeded the expanded8MiB limit, so every upload stage now
checks expanded bytes as well as source closure. A first cubic prototype
called a nonexistent overload that a permissive CPU stub had invented; the
corrected source and audits now check actual helper arities before projection.
Seven compound-product reduction variants were subsequently tested/compiled;
none replaced the control because their resource tradeoffs did not establish
a substantial advantage. Those bounded screens do not rule out the families.

## Reproduction and evaluation

From the staged benchmark root:

```sh
SRC=candidates/subset
python3 -B "$SRC/preflight.py" --package-check --note "$SRC/submission-ranked-digits.md"
python3 -B "$SRC/check_candidate.py" --source "$SRC" --selection packed --inverse-layout leaf-pair --report /tmp/frontend.json
python3 -B "$SRC/research/pr120_digits/check_digits.py" --source "$SRC" --report /tmp/digits.json
python3 -B "$SRC/research/pr120_digits/check_chain_recovery.py" --source "$SRC" --output /tmp/chain-recovery
python3 -B "$SRC/research/formula_fusion/check_source.py" --source "$SRC" --report /tmp/recovery.json
yukon setup --track subset
yukon run --track subset
```

The host checks require Python, C++17 and OpenSSL development libraries.
Native commands remain `nvcc -O3 -DQSB_ZEROS_N=24 subset.cu -lcrypto -lm`
and a separate build of `tests/gpu_epochs/tree_audit.cu`; sm89 resource
screens add `-arch=sm_89 -Xptxas=-v,--warn-on-spills`. On an actual CUDA host,
execute the audit and Compute Sanitizer before treating a local score as
qualified. The Mac cannot run the official GPU benchmark bridge.

Before a local rerun after header edits, invalidate the generated candidate
binary and `.subset.build` stamp: the trusted wrapper's cache keys only on
the include-wrapper timestamp. Fresh temporary builds do not update it.
Trusted harness/scoring and the sibling track remain unchanged.

The expected advantage is the substantial fused fourteen-window pipeline plus
its arithmetic/frontend refinements. Cold-table latency, large-table startup,
cache behavior, code scheduling and driver compilation remain risks. Remote
verified throughput is authoritative. Compare the eventual result with the
frontier at evaluation time; a composite score cannot isolate individual
components or justify adding their apparent gains together.
