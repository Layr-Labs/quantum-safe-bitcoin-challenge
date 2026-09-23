# Pinning: exact GLV diagonal-10 high-word screen

## 1. Initial context and goal

This submission is a narrowly isolated optimization of the QSB pinning fixed-base
decoder. The ranked workload repeatedly constructs a scalar from the candidate
SHA state, splits that scalar with the secp256k1 GLV lattice, obtains two signed
128-bit components, and uses their digits to index a large precomputed point
table. Unlike the surrounding SHA and elliptic-curve arithmetic, this GLV
coefficient screen contains a deliberate exact fast path: most inputs use a
partial high-word multiplication schedule, while a narrow interval near the
rounding boundary calls the full out-of-line product routine.

The goal was not to compose every small idea that survived static compilation.
The goal was to identify the one current-tree mechanism with both a credible
execution-cost reduction and a correctness argument strong enough to package as
an isolated candidate. The live benchmark result at packaging time was
881,273,403 verified candidates per second on an RTX 4090, from source ref
`1fe5a8e40008befcd917668ea9b1a23c6ee590c4`. The live record is moving and is
used only as context. No local or claimed score is asserted for this package.

The selected mechanism is `QSB_GLV_HIGH10_HI=1`. It removes lower-word work from
the five products on coefficient-product diagonal 10 while preserving the exact
nearest-integer GLV coefficient. The package deliberately excludes all SHA
pipe-routing changes and the donor's direct rounding-carry member. This is a
single-mechanism submission: if the ranked result is positive or negative, the
outcome is attributable to the coefficient high-word screen rather than to a
bundle.

## 2. Environment, setup, and constraints

The development host is an Apple M5 arm64 Mac with no NVIDIA GPU and no CUDA
runtime capable of executing the ranked kernel. Local M5 measurements therefore
could not establish pinning throughput. The available development path was:

1. retain the exact live challenge source in the Yukon-linked challenge clone;
2. create a dedicated linked Git worktree from the live source ref;
3. execute the coefficient control flow on the CPU with exact C++ equivalents
   for the CUDA integer operations;
4. cross-compile the complete translation unit with CUDA 12.4.131 in a Linux
   arm64 `nvidia/cuda:12.4.1-devel-ubuntu22.04` container;
5. inspect `sm_89` SASS and ptxas resource output without executing a GPU
   kernel; and
6. package only files under the challenge's editable path,
   `candidates/pinning/`.

The dedicated worktree and branch were created with:

```bash
git -C quantum-safe-bitcoin-challenge worktree add \
  -b submission-q350-high10-20260923 \
  ../q350-high10-wt \
  1fe5a8e40008befcd917668ea9b1a23c6ee590c4
```

No harness-owned file, verifier, benchmark wrapper, problem generator, or
submission API was changed. No Yukon command was used to produce the source.
The final package must still be submitted by the human account owner under the
challenge's separate approval policy.

The package was kept below the site's 8 MiB editable-path limit. A compressed
tar archive of `candidates/pinning` measured 165,170 bytes. The directory has
no prebuilt executable, cubin, PTX, credential, or external-service payload.

## 3. Prior work and live baseline

The live baseline is the promoted GLV12 source ref
`1fe5a8e40008befcd917668ea9b1a23c6ee590c4`. Its benchmark metadata reports:

- track: `pinning`;
- benchmark GPU: RTX 4090;
- verified score: 881,273,403 candidates/s;
- leading-zero requirement `N=24`;
- fixed-time elapsed time: 1,201.4928 seconds;
- 126,224 verified hits; and
- hit-relative variance: 0.002815.

The starting source already contains a six-word-per-component GLV table, an
exact high15 coefficient screen, a 128-bit residual path, the exact host hit
gate, and the surrounding pinned recovery pipeline. The optimization had to
preserve all of those mechanisms. Rebuilding the table geometry, changing the
residual representation, or mixing in an unmeasured SHA schedule would expand
the candidate beyond the available evidence.

The coefficient high-word mechanism was introduced publicly by @Portablelle in
PR #1256. That submission combined the new coefficient work with several other
donors. Its first official draw and a later comment-only replay differed by
1.689%, so neither draw could attribute an improvement to the coefficient
member. The current package re-derives the mechanism on the newer GLV12 source,
isolates the high-word member, excludes the direct rounding-carry member, and
supplies a current-source exactness test.

The executable implementation is commit
`372461625aa2a4f33ff4d8b68675093cfefbac02`. Later commits in this branch only
update package metadata and this public note; they do not change device code.
The exact final candidate commit is the commit containing this note and can be
obtained with `git rev-parse HEAD` in the prepared worktree.

## 4. Hypotheses and decisive evidence

Four current-tree mechanisms were considered before selecting the package.

### Q350 HIGH10 coefficient screen

Hypothesis: diagonal 10 contributes only a carry to later product words for the
rounding decision. Its five full 64-bit products can therefore be replaced by
five upper 32-bit products plus a widened exact fallback guard.

Decisive local evidence:

- 601,636 scalar cases per mode passed against Python big-integer rounding;
- HIGH10 on and off had zero coefficient mismatches;
- 200,000 uniform full-width inputs produced zero full-product fallbacks in
  either mode;
- the current source's two error bounds were independently reproduced as 9 and
  8 units of `2^352`;
- only stage-0 SASS changed;
- stage 0 remained at 122 registers, 12,288 shared bytes, and zero spills; and
- the two uniform coefficient fast-branch proxies shortened from 40+45 to
  29+33 instructions.

This was the only screened mechanism with an exactness proof, a current-source
fallback census, and a positive dynamic-path proxy.

### Q350 direct rounding carry

Hypothesis: replacing compare/select rounding with `add.cc.u64` and `addc.u64`
would reduce the tail of the coefficient fast path.

The current source compiled the member exactly, but the complete stage-0 body
remained 4,096 SASS instructions. The change removed comparison/select work and
emitted carry-chain replacements without a net whole-stage instruction-count
reduction. It was therefore excluded rather than bundled on the assumption that
different instruction mixes must help.

### Q300 stage-0 ALU routing

The dormant `QSB_SHA_ALU_ADD=1` switch moved 150 stage-0 `IMAD.IADD` operations
toward `IADD3` without increasing the 4,096-instruction stage-0 body. The same
switch also changed stage 2 by eight instructions, including 24 additional
`IMAD.IADD` operations and 26 `IADD3` operations. Because Q300 is framed as a
stage-0 pipe experiment, that global effect made it unsuitable for an
attribution-capable isolated package.

### Q348 stage-0 FMA routing

A current-stage-only FMA port passed 500,002 schedule cases and 8,000,032 word
comparisons, preserved resources, and left stage 2 byte-identical. However, the
pre-table phase grew from 3,147 to 3,496 instructions while moving from
2,722 ALU / 365 FMA operations to 2,467 ALU / 969 FMA operations. This matches
the old donor's static expansion, and the old-lineage official result was
sharply negative. It was excluded.

## 5. Approach selection and tradeoffs

The selected approach was to optimize only the coefficient product diagonal
that could be proved exact with a small guard adjustment. This is conservative
relative to rewriting the GLV residual path or table geometry, but it has three
advantages:

1. the candidate changes one executable header and leaves the rest of the live
   source byte-identical;
2. the existing full-product fallback remains available for uncertain rounding;
3. `-DQSB_GLV_HIGH10_HI=0` restores the base control without source patching.

The main tradeoff is a slightly wider exact fallback interval. The old guard
began four or three words below the half boundary, depending on the reciprocal.
The new interval begins nine or eight words below it, adding five possible word
values in each case. Under a uniform word-11 model, five additional values out
of `2^32` correspond to roughly `1.16e-9` extra fallback probability. The actual
word distribution is not assumed to be uniform, so this is only a scale check.
The 200,000 uniform full-width inputs produced no fallback in HIGH10, and the
adversarial guard fixtures intentionally forced both sides of the interval.

A larger bundle might have increased the nominal chance of a positive draw, but
it would also have destroyed causal attribution. Given the measured same-tree
A/A spread of about 0.4%, bundling unrelated sub-noise mechanisms would not turn
an unidentifiable draw into reliable evidence. The single exact mechanism is
more valuable even if its expected gain is only a few tenths of a percent.

## 6. Implementation and changed logic

Four files differ from the live source ref:

- `GLVScalar.cuh`: executable optimization;
- `test_glv_high10.py`: exact CPU differential regression;
- `SOURCE-MANIFEST.json`: current file sizes and SHA-256 hashes; and
- `SUBMISSION.md`: this public note.

No other file under `candidates/pinning` changes.

### Existing coefficient schedule

Before this candidate, `q9_coeff_high15` computed complete 64-bit products for
diagonal 10:

```text
(a3*b7) + (a4*b6) + (a5*b5) + (a6*b4) + (a7*b3)
```

Five products can exceed 64 bits, so the implementation explicitly counted
lost `2^64` units. It then carried the result into diagonals 11 through 14 and
used a narrow word-11 interval to decide whether the omitted low product words
could affect bit-383 rounding.

### HIGH10 implementation

The candidate computes only the upper half of each diagonal-10 product:

```cpp
const uint32_t first = q9_mulhi32(a3,b7) + q9_mulhi32(a4,b6);
carry = (uint64_t)first
      + q9_mulhi32(a5,b5)
      + q9_mulhi32(a6,b4)
      + q9_mulhi32(a7,b3);
w10 = 0;
```

For both production reciprocals, `b7+b6 < 2^32`. Therefore the first two upper
products fit in `uint32_t`; the sum of all five upper products fits in
`uint64_t`. Word 10 itself is not needed by the later coefficient words, so it
is not materialized.

All omitted low products and their carries are nonnegative. Summing every
possible omitted contribution gives the input-independent bounds:

```text
E(g1) < 9 * 2^352
E(g2) < 8 * 2^352
```

The exact fallback thresholds become `0x7ffffff7` for `g1` and
`0x7ffffff8` for `g2`. Below the threshold, the omitted amount cannot cross the
half boundary. At or above `0x80000000`, the result is already rounded upward.
A possible wrap through the end of word 11 increments the next word and removes
that round-up without reaching another half boundary under the proven error
bound. Only the intervening uncertain interval uses the unchanged full-product
fallback.

Setting `QSB_GLV_HIGH10_HI=0` compiles the exact base diagonal-10 schedule and
old guards. No error-budget or hit-gate approximation is used: the high-word
member itself is exact for every 256-bit scalar.

## 7. Exact reproduction commands

Run the coefficient differential from the candidate root:

```bash
python3 -B candidates/pinning/test_glv_high10.py
```

The test extracts `q9_high15_begin`, `q9_mulhi32`, and `q9_coeff_high15` from
the actual production header. It compiles the extracted C++ twice, once with
`-DQSB_GLV_HIGH10_HI=0` and once with `=1`, loads both shared objects with
Python `ctypes`, and compares both against:

```python
(k * reciprocal + (1 << 383)) >> 384
```

Run the inherited source regressions:

```bash
python3 -B candidates/pinning/test_carry62.py
python3 -B candidates/pinning/test_host_gate.py
python3 -B candidates/pinning/test_priority_pipeline.py
python3 -B candidates/pinning/test_slot_readback.py
python3 -B candidates/pinning/test_sha_interleave.py
```

Cross-compile the complete organizer-style source on the no-GPU host:

```bash
docker exec qsb-toolkit mkdir -p /tmp/q350-package
docker cp "$PWD/candidates/pinning/." qsb-toolkit:/tmp/q350-package/

docker exec qsb-toolkit nvcc -O3 -DQSB_ZEROS_N=24 \
  -o /tmp/q350-package-default /tmp/q350-package/pinning.cu -lcrypto -lm

docker exec qsb-toolkit nvcc -O3 -DQSB_ZEROS_N=24 \
  -DQSB_GLV_HIGH10_HI=0 \
  -o /tmp/q350-package-control /tmp/q350-package/pinning.cu -lcrypto -lm
```

Generate static `sm_89` evidence:

```bash
docker exec qsb-toolkit nvcc -O3 -DQSB_ZEROS_N=24 -arch=sm_89 \
  -Xptxas -v -o /tmp/q350-package-sm89 \
  /tmp/q350-package/pinning.cu -lcrypto -lm

docker exec qsb-toolkit nvcc -O3 -DQSB_ZEROS_N=24 \
  -DQSB_GLV_HIGH10_HI=0 -arch=sm_89 -Xptxas -v \
  -o /tmp/q350-package-control-sm89 \
  /tmp/q350-package/pinning.cu -lcrypto -lm

docker exec qsb-toolkit cuobjdump -sass /tmp/q350-package-sm89
docker exec qsb-toolkit cuobjdump -sass /tmp/q350-package-control-sm89
docker exec qsb-toolkit cuobjdump -res-usage /tmp/q350-package-sm89
docker exec qsb-toolkit cuobjdump -res-usage /tmp/q350-package-control-sm89
```

Check the editable-path size without adding generated binaries to the package:

```bash
tar -czf /tmp/qsb-q350-high10.tar.gz candidates/pinning
stat -f '%z bytes' /tmp/qsb-q350-high10.tar.gz
```

## 8. Experiments, failures, and course corrections

The investigation deliberately recorded negative and corrected results rather
than preserving only the final favorable path.

First, the current GLV constants were compared with PR #1256. The reciprocal
constants were unchanged, but the table geometry had moved to GLV12. Treating
the donor's old thresholds as self-validating would have been unsafe. The
candidate test therefore recomputes the omitted-term bounds directly from the
constants in the current production fallback.

The first extracted CPU harness failed to compile because its wrapper referenced
a reciprocal pointer before adding it to the function signature. The wrapper was
corrected to accept and copy the four reciprocal words explicitly, then the full
test was rerun. This was a harness-construction failure, not a candidate result.

A separate Q348 edge-fixture initially randomized only one side of the old/new
comparison. That was caught because the all-zero case disagreed. The fixture was
corrected to give both implementations identical input words before rerunning
the schedule test. Q348 was later excluded for instruction growth and adverse
historical evidence; the fixture correction prevented that false mismatch from
influencing this package.

The inherited `SOURCE-MANIFEST.json` contained stale hashes for current
`GPUMath.h` and `pinning.cu`, and it listed a `__pycache__` bytecode file that
was not present in the worktree. Rather than copying the stale manifest, every
listed file was checked by size and SHA-256, the nonexistent entry was removed,
and the two stale hashes were refreshed. The final manifest covers 28 real files;
only itself and this note are intentionally excluded as metadata.

Q300 was initially attractive because its intended stage-0 migration succeeded
without increasing the stage-0 body. Comparing the complete SASS showed that
the switch also changed stage 2. That discovery changed the package decision:
the global switch was excluded instead of being described as an isolated stage-0
optimization.

## 9. Correctness, build, and static results

The final exactness run passed 601,636 scalar cases in HIGH10-off mode and the
same 601,636 in HIGH10-on mode. Each mode covered both production reciprocals.
The cases included 200,000 uniform 256-bit values, all one-bit neighborhoods,
constructed coefficient half-boundaries, guard-boundary neighborhoods, and
sparse/all-high 32-bit limb patterns. There were zero oracle mismatches. Uniform
inputs produced zero full-product fallbacks. Adversarial rounding cases were
designed to enter the fallback and therefore are coverage counts, not estimates
of production frequency.

The inherited checks also passed:

- carry audit: two million X3-union random samples with zero unexpected
  differences and the documented reduced-state predicates;
- host gate: 64 SHA-256d midstate samples with recovery matching the verifier;
- priority pipeline: five dependency, slot-reuse, partial-batch, rollover, and
  error-injection tests;
- slot readback: three capacity, isolation, reuse, overlap, and error-injection
  tests; and
- SHA interleave: 34,566 digest comparisons against `hashlib`, including alias
  behavior.

Both organizer-style builds compiled. The `sm_89` resource reports were
identical between HIGH10 on and off:

- stage 0: 122 registers, 12,288 shared bytes, zero stack frame, zero spills;
- stage 2: 64 registers, zero spills; and
- every other kernel had unchanged resource output.

Only the stage-0 function changed. Stage 2 and all non-stage-0 functions were
instruction-list identical. The complete stage-0 body remained 4,096 SASS
instructions, but the two coefficient fast-branch proxies shortened from 40+45
to 29+33 instructions, a reduction of 23 instructions in the relevant uniform
path. This is a static proxy, not a measured dynamic instruction count.

## 10. Caveats and interpretation limits

No NVIDIA kernel was executed on the M5 host. This package has no local GPU hit
set, no completed-work timing, no Nsight Compute counters, and no claimed
score. The exactness test executes production control flow compiled as host
C++, but it does not model PTX scheduling, register allocation, memory behavior,
or CUDA execution.

The 23-instruction proxy does not imply a 0.56% end-to-end gain. It is 23
instructions relative to a 4,096-instruction stage-0 body and must be discounted
for the fraction of total pipeline time spent in that body. A conservative
static expectation is roughly +0.1% to +0.3% around a +0.2% center if the
current pipeline balance resembles the source model. This range is below the
approximately 0.4% same-tree A/A spread and should not be treated as an
attribution-capable prediction.

The official benchmark is a fresh-problem, hit-derived score. Runner
assignment, clocks, thermal state, and Poisson hit variation can move a single
draw by more than the expected mechanism gain. A positive draw would prove that
the exact package is valid and fast enough to score, but one sub-noise draw
would not isolate causality beyond the deliberately single-member composition.
A negative draw would be adverse evidence, but retry policy and host class must
still be read before treating it as mechanism refutation.

The live 881,273,403 record can move before this package is evaluated. The
candidate's value must therefore be judged against the record and source ref
active at submission time, not against this frozen prose snapshot.

## 11. Learning and reusable conclusions

The main reusable conclusion is that a fast path can discard known-positive
lower-word work if the omitted contribution has a proved bound and the fallback
interval is widened accordingly. Exactness and performance are separate
questions: HIGH10 is exact for all scalars, while its value depends on fallback
frequency and the actual stage-0 schedule.

A second conclusion is that frontier changes require re-derivation. The GLV
coefficient constants survived the move to GLV12, but that fact was checked
rather than assumed. A table-geometry change does not automatically invalidate
coefficient arithmetic, yet it also does not automatically preserve an old
proof or old static instruction count.

Third, static instruction totals can hide the useful path. ROUND_CC changed
SASS without reducing the complete stage-0 count, while HIGH10 preserved the
complete count but shortened the branch-dominated uniform coefficient path.
Both static facts were reported, and only the mechanism with a positive uniform
proxy was selected.

Fourth, package identity is part of correctness. The stale manifest hashes and
nonexistent bytecode entry were repaired before packaging so that the final
archive identifies the exact source bytes being reviewed. Generated compilers,
SASS dumps, and archives were kept outside the editable path.

## 12. Next steps and submission gate

The next step is a single official pinned benchmark run of the exact final
candidate commit. Before dispatch, the human owner should verify:

```bash
cd /path/to/q350-high10-wt
git status --short
git rev-parse HEAD
git diff --check 1fe5a8e40008befcd917668ea9b1a23c6ee590c4..HEAD
```

The working tree should be clean. Approval must name the final full commit and
the `pinning` track. A changed note or manifest creates a new commit and
requires fresh approval; approval for an earlier package commit does not carry
forward.

The submission command must include the public note, model, harness, and
mechanism coauthor. The account owner, not an automated agent, performs the
Yukon submission. After one result, record the official score, verified hit
count, elapsed time, problem seed, runner identity, and promotion status. Do not
request a retry or repeat run without fresh human approval. If the result is
within run-to-run noise, retain it as an inconclusive single draw rather than
claiming the predicted fraction of a percent.

## 13. Provenance and licensing

The high-word coefficient mechanism was introduced publicly by @Portablelle in
PR #1256 and is credited as a coauthor mechanism. This package independently
re-derives the current GLV12 error bounds, ports only the high-word member,
excludes the unproven composition members, and adds a focused current-source
regression. The promoted source's GPLv3, MIT, Bitcoin, and secp256k1 notices
remain unchanged. The exact file inventory and hashes are recorded in
`SOURCE-MANIFEST.json`.
