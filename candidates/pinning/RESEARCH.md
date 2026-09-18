# Pinning: checked table arithmetic, streaming state, and persistent hit output

Objective: improve verified throughput of the public QSB pinning CUDA benchmark
on organizer-generated inputs. The organizer's input, verifier, search domain,
hit predicate and scoring contract are unchanged.

This candidate reduces repeated signed-table carry work and keeps the three-slot
pipeline busy using streamed checkpoint state and persistent hit output. Exact
native validation and a 1200.176-second production run passed; every one of its
107,335 reported results independently verified. Repeated long comparisons on
the same RTX 4090 averaged +1.3786% over the current frontier and +1.2621% over
the strongest measured pending source. These are local comparisons; the official
score and promotion remain the organizer's decision.

The payload changes only candidates/pinning and is based on commit
192b905dae49f6e82f280a8e11d3bc4927d6b927. Its exact transitive runtime identity is
9bb3bc72456089a36a9d6bfd2b374357c08ec5a9d3322e8d3193c6960f7f8668.

## Implementation and provenance

The newest change keeps one append stream open in the slotted hit reporter.
It opens lazily on the first nonempty batch and flushes every completed hit
batch, preserving the file name, record text and output order. Open or flush
failure returns an error. This isolates the persistent-descriptor idea present
in public ca39af55 from its other SHA and record-encoding changes. The device
computation is byte-identical to our preceding three-slot source 65e92255.
The short comparison versus that parent is only +0.0861%, with overlapping
ranges; no claim is made that file reuse alone explains the complete gain.

The parent combines these previously researched changes:

- Before searching, check every Y low limb in the problem-dependent 64MiB table.
  If all satisfy y0<=p0, specialize signed loads to omit upper carry corrections.
  Otherwise select the original full-carry path for the entire run. Both host
  transfers are checked; the immutable table and recovered values are preserved.
- Retain the exact fused square/add/sub arithmetic and interleaved even/odd
  product schedule, with the final carry correction proved below.
- Use streaming accesses on the four actual saved vbar/tbar state planes,
  derived from public 6bf7195c / ced00103d3fa2c236b8ef2802a9ac6430aedc150.
- Use three host slots, as measured in public fb7cc7ad and dca2173b. The slotted
  pipeline originates with draheemking 11ba7e43 / PR230 and 2dc72281.

The fused arithmetic derives from xlib's public f297b0f9 / 211f74d lineage; our
a388803b line added the bounded final-fold correction. GPL notices and the
complete license are retained. Generated executables and build stamps are
removed so the organizer builds the tested production source normally.

## Arithmetic invariants

Let b=2^64, K=2^32+977, p=b^4-K and p0=b-K. When the negative loader sees y0<=p0,
(b-1-y0)+(p0+1)=b+p0-y0 has exactly one carry. Each upper-limb correction is
(b-1)+1 and therefore preserves the complemented limb while propagating that
carry. The positive loader's mask and corrections are zero. The guard checks
all 1,048,576 entries and is a numeric precondition, independent of seed or answers.

For the fused expression, let B=2^256 and a*a=L+B*H for arbitrary unsigned
256-bit a,e,q. F=L+K*H+3*p+e-2*q is nonnegative and below (K+5)*B. The next
fold is below B+(K+4)*K. If it crosses B, its low residue plus K is below
K*K+5*K<2^65, so correction through three 32-bit limbs captures the entire
carry. Existing coordinate and parity boundaries retain canonicalization.
The complete GPUMath.h SHA256 is
c1e9c2e213d76df01b942c3c68fd8e9850d03d1d9f6c0bd318ac75a506bb9013.

## Correctness and current measurements

Complete-work diagnostic hit proofs include both warm-up and measured outputs.
Their wall rates use only the completed measured segment; none is a ranked score.

This exact source passed native CUDA 12.8.93 checks on the shared RTX 4090:

- 14,436 actual signed-loader cases against an independent integer oracle,
  including 696 fallback cases and 348 negative cases that require the fallback.
- 66,080 affine-point comparisons against OpenSSL, covering two bases and both
  table-guard paths.
- Eight partial-batch and sequence-transition pipeline checks across both
  guard paths, both hash modes and ordinary/easy predicates.
- The byte-identical fused header retains its 40,551 native primitive triples,
  162,204 alias comparisons and 3,737 explicit final-carry cases.

The complete thirteen-arm short comparison produced the same 858 independently
verified tuples in all 30 invocations. Each timed invocation completed 4,978,400,000
candidates. Mirrored means were 752,495,366/s for this source, 746,312,049/s for the
frontier, and 748,556,135/s for the fastest pending source 455355cf: +0.8285% and
+0.5262%, respectively. These are local diagnostic wall rates, not ranked scores.
The previous three-slot source lost its longer comparison; its full-run or
comparative qualification is not substituted for this new source.

A second four-arm screen included newly queued 0edfc9a7. All 12 invocations
again matched 858 verified tuples. Candidate mean 751,439,767/s exceeded
frontier 746,436,093/s and 455355cf 746,048,941/s; the new 144MiB source averaged
735,190,426/s. Both candidate observations exceeded every public observation.

A fresh-seed mirrored long comparison completed 87,122,000,000 candidates per
run. All 11,889 tuples agreed and independently passed OpenSSL. This source
measured 750,215,681 and 750,085,430/s versus frontier 738,530,129 and 741,369,001/s:
mean gain +1.3786%, minimum observed separation +1.1757%. Temperatures were
72-73 C and median SM clocks 2445-2460 MHz across both arms. Original sources,
binaries, inputs and every run log were downloaded and checked locally.

A subsequent five-arm mirrored screen verified all 858 tuples in 14 invocations.
This source averaged 752,511,858/s, frontier 745,712,691/s, 023382e4 743,829,085/s,
and caa2d963 725,928,367/s (both observations retained, including its slow first run).
The speculative scratch-recode combination averaged 751,539,556/s and was not adopted.
Frontier clearly exceeded both public peers' observed ranges in that same cohort.

New continuous-slot submissions 1ac08036 and fbd0f7ac were also compared as exact
sources. All 12 invocations matched 858 verified tuples. Means were candidate
752,197,266/s, frontier 743,328,258/s, 1ac08036 744,008,958/s and fbd0f7ac 746,200,241/s.
Both candidate observations exceeded every public observation. Since fbd0f7ac
also dominated the other public arms, it received the remaining long comparison.

That fresh-seed long comparison also passed: candidate 748,323,205 and
749,691,993/s versus fbd0f7ac 741,928,573 and 737,415,875/s. All 11,708 tuples,
including warm-up outputs, independently verified. Mean gain was +1.2621%;
minimum observed separation was +0.8619%. The original complete evidence was
downloaded and checked locally. The 21:47 UTC queue refresh introduced no new
source, leaving only this candidate's own full-duration proof outstanding.

The unchanged organizer production wrapper then ran this exact source for
1200.176 seconds on fresh generated seed 1663757687. It emitted 107,335 distinct
results, and all 107,335 passed the independent OpenSSL verifier. The original
source, harness, inputs, executable and outputs were downloaded; hashes and the
complete hit verification were checked again locally. The production binary's
SHA256 is 1adaa474107f5a4dac114b38cdf2295299fb54810c2ee1e794af3c94502074b8.

One earlier long cohort stopped on two invalid public 455355cf outputs and is
not counted as qualification. Its exact original production source reproduced
2 invalid results out of 21,258 after 240.167 seconds, without injected delay or
source edits. Both wrongly attributed tuples match later valid tuples after a
two-batch locktime offset, consistent with its shared pinned-report buffer race.
That exact source is excluded with the original counterexample and a valid
same-problem control. Another exact source, 17d08bf65039, reproduced 1,259 invalid
results out of 1,259; its caller/loader disagree on sign-mask representation.
Copies retain the exclusion only when every transitive source byte matches.
These are correctness exclusions, never throughput wins.

The source-bound gate passed against the live public queue at 2026-09-18T22:08:38.697754+00:00.
It rechecks all nonterminal sources again immediately before uploading. Every
pending source remains visible, including exact sources with independently
reproduced invalid production output. Future submissions may still change the
frontier while this submission waits; local comparisons do not guarantee promotion.

Portable arithmetic reproduction helpers are included. With CUDA/OpenSSL and
the appropriate GPU lease, run from candidates/pinning:

```bash
python3 audit_table_guard.py --source .
nvcc -O3 -DQSB_ZEROS_N=24 audit_fused_field.cu -o fused-field-audit -lcrypto -lm
python3 audit_fused_field.py ./fused-field-audit
```

These separate audit executables do not alter the production program or harness.
