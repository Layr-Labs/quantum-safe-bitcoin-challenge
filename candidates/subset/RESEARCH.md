# Subset research ledger

## Current successor, 2026-09-16

The current candidate combines pending PR36 (`472b536`) with PR40 (`08a4b7b`).
The original submission and its source record below are historical, not the
current working source. Public successor note: `submission-deferred-ranked.md`.
The user authorized submission on expected superiority without unavailable GPU
testing, and restricted local work to this Mac. The research review selects
the composition as the strongest expected candidate, with no claimed score.

Current production/audit fingerprint:
`d0ccab7c66830429f8bc39dc619ad4b369883d0ea9c7929b64a569592c65512e`.
The nine-file source audit passed 28332 comparisons; the extracted deferred
point helpers passed 31920 intermediate-state comparisons across 2128 chains.
The PR36 algebra model also passed. Grok/Gemini found no blocking source defect;
unsupported compiler/performance claims from reviews were rejected. Full
provenance, test limits and source hashes are in the successor note.

Relative to our first submission this removes 14 field multiplications from
the fixed-base chain, composes two higher-level tree fusions (14 to 12 barriers),
and adds ranked template specialization. Relative to PR36 only specialization
is incremental; its benefit remains uncertain. No GPU performance is measured.
The SHA/EC split remains an isolated research generator, outside production.

Submission attempt: Yukon rejected the successor upload with `conflict` because
this account already has one in-flight subset submission (limit one). No new
submission ID was created. The older `c9a85a87` remains pending. The candidate
and public note are ready. The user chose to preserve its pending result, watch
for later submissions finishing first, and continue local improvement research
and inspection of incoming sources. The existing 20-minute follow-up was
updated accordingly. Do not cancel the current entry or make a new submission
from that follow-up. Source and score results remain unmeasured on this Mac.

Subsequent arithmetic research found a real final-carry defect in the inherited
field primitive, including canonical inputs. `research/ARITHMETIC.md` contains
the exact counterexample, regression runner and isolated two-primitive correction
(60540 multiply and 80032 square CPU calls pass). The frozen prepared candidate
has not been patched or submitted. Its earlier OpenSSL-based passes do not
validate the affected primitive. Review this finding before any future upload.
The separate 48-product Karatsuba model passes 21465 cases but has no CUDA
implementation or measured speedup yet.

## Submitted experiment

Yukon submission `c9a85a87-e2b8-4ab8-8d4f-c79d96546d03` was accepted into
the validation queue on 2026-09-16. Initial status: **validating**, score
unavailable. Model: GPT 6 Astra; harness: Codex; effort: xhigh; coauthor:
`@nullforest8200`. Comparison baseline: 129574439 verified candidates/s.
The public note is `submission-epoch-inverse.md`. This record was added after
upload and does not change the submitted source. Check status explicitly with
`yukon submissions eigenlabs/quantum-safe-bitcoin-challenge/subset`.

## Selected candidate: GPU epochs plus a four-lane inverse tree

The production entry point includes `tests/gpu_epochs/tree.cu`. All files in
the imported port are byte-identical to public candidate
`b733504088873a409baff306a2633ec63e6e5762`, except `tree_inverse.cuh`.
The new helper keeps the bottom two inverse-tree levels in four-lane groups
and uses shared memory only above them. At 256 threads it needs 4096 shared
bytes, 14 block barriers, 765 canonical multiplies and one inverse. The
imported helper needed 16384 bytes, 18 barriers, the same 765 multiplies and
one inverse. Register pressure and GPU performance remain unmeasured locally.

The main architectural gain over the production promoted baseline comes from
the imported epoch pipeline, signed XYZZ fixed-base multiplication, folded
runtime scalar coefficient, shared recovery denominators, and block inversion.
These are other solvers' contributions, not newly invented here.

## Provenance and scan

- Promoted production baseline: Meganpark980320, submission
  `b7bdbf1c-a819-4c02-ad21-660c1f16f0da`, commit
  `a040c21c12306610bf53c718304ad1c93c360b65`, 126688029 verified candidates/s.
- Pending projective subset improvement: Meganpark980320,
  `c691d3da-07dd-4372-a6bc-1d39d3d2035a`, commit
  `56fc682cd90e21764438fc67d30909bf976e35b1`. Inspected and used in an initial
  prototype; the final imported pipeline already includes deferred inversion.
- Pending compact recovery table on pinning: jacklightChen,
  `783bdbdf-d836-4444-820c-ce76c435b4b7`, commit
  `5200fbc5335f779e894db02b3da611b43f964b26`. A subset prototype combined
  this 18 MiB runtime table with the promoted SHA prefix cache. It passed
  CPU reference checks but had no GPU measurement. It was superseded before
  submission by the more advanced pipeline below; no code from that prototype
  is in the final production include closure.
- Selected unpromoted source: nullforest8200,
  `873ed724-9815-4e13-a02f-072f21e3f992`, commit
  `b733504088873a409baff306a2633ec63e6e5762`:
  <https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/b733504088873a409baff306a2633ec63e6e5762>.
  Its public note describes a port of development promotion `a33f042b`, commit
  `7c1e716`, reportedly scored at 442824991 on an RTX 4090. That historical
  score is a source author's report, not a locally reproduced result. The
  old development benchmark and repository were unavailable during this scan.
  Credit nullforest8200 as coauthor for this substantial unpromoted source.
- `TREE_INVERSE.md` is preserved historical upstream documentation. It credits
  odinfree's GPU-epoch architecture and earlier promoted arithmetic work.
  Its measurements and original 16 KiB tree describe the predecessor, not
  this new four-lane helper.

## Verification of the selected candidate

Run `python3 candidates/subset/check_candidate.py` from the repository root.
The script builds a temporary C++ library and executes extracted production
functions, with OpenSSL field arithmetic and CPU synchronization emulation.

Result on 2026-09-16: PASS for 3840 inverse outputs, 6144 complete SHA256d
digests across two fresh synthetic problems, 10262 signed recodings and 8086
combinadic unrankings. Include closure passes for nine source files covering
production and the inherited GPU audit. The inverse cases cover 32, 64, 128,
and 256 threads, all-identity and boundary values, and inactive tail lanes.
All cases have exactly one inversion and `3*n-3` multiplications per block.

The production tree source SHA256 is
`92e9b024cd191b478f5b9d9dcb327883a9bb12650afd2e57642e83165acfdac5`.
The modified inverse helper SHA256 is
`5bba7cc408e048b6c9c22fc3be063100172d0ce17712932df7ca3a7483ef7f8f`.

This Mac has no CUDA compiler or NVIDIA GPU. `yukon setup --track subset`
completed the CPU verifier smoke, while `yukon run --track subset` cannot
start the ranked grinder: the Linux bridge is unavailable and `sudo` is
blocked. No local GPU throughput, CUDA compile, or GPU sanitizer result is
claimed. Remote validation must establish correctness and performance.

## Final frontier scan

Before submission, `c691d3d` was promoted at `df765fe`, raising the production
bar to 129574439 verified candidates/s. Use this value as the submitted
comparison baseline. The pending epoch port `873ed724` remained validating.

Additional notes reviewed: `7fdc8ed` (welttowelt), `b2ede54` (fkiene),
`dfb66e9` (newjordan), `ef833e4` (pepedesigner), and `54918fb`
(anamdongparkjinhyeong). They cover deferred projective normalization,
broadcast/read-only loads, larger launches and reduced summary synchronization.
The selected architecture already contains the major mechanisms. Additional
micro-tuning from a different kernel was not imported without GPU evidence.

## Iteration rules

The follow-up workflow review is in `WORKFLOW_REVIEW.md`. The production/audit
source fingerprint remains
`5a67fc768c6d5740d765343b0084e3b8269b042e6d4c7437f818aa593449ca8e`.
The review added a clean temporary CUDA-build option and source-bound JSON
audit reports. It found that the trusted wrapper's timestamp cache ignores
included-header changes; invalidate its generated binary/stamp before local
benchmarks after such edits. Candidate CUDA code was unchanged by this review.

Prefer architectural changes with a credible large gain over the current
promoted frontier. Read pending notes and verify their source before reuse.
Keep GPU timing claims separate from CPU algebra checks and source counts.
After each returned submission, record the actual result and compare against
both the submitted baseline and the latest promotion. Fix local checks when
remote feedback reveals a missed failure. Do not alter trusted scoring files.

If this submission fails GPU synchronization or arithmetic checks, first run
the included `tree_audit.cu` on CUDA under Compute Sanitizer before tuning.
If it is correct but slow, compare register/spill counts and occupancy with
the imported 16 KiB tree. Do not assume fewer barriers guarantee a gain.

## Joint research update, 2026-09-16

Grok 4.6 and Gemini 3.8 Flash (High) completed a reviewed research round;
`research/TEAM.md` tracks the conclusions, corrections and implementation queue.
The latest status refresh promoted the upstream epoch port `873ed724` at
`cfc0d9c`, with 433346795 verified candidates/s. Our `c9a85a8` and first pinning
submission still validate. Use the stronger frontier for future decisions.

Both models recommend testing a bounded SHA/EC kernel split; no gain is measured
or guaranteed. Codex added a symbolic geometry screen that reproduces 84/26
classes and rules out large compression-count gains in its narrowly specified
family. Production CUDA source and its fingerprint remain unchanged.

## Returned result and architectural successor, 2026-09-16

Subset c9a85a87 (PR27) returned rejected at 418504460 verified candidates/s.
This is 3.425% below the promoted 433346795 epoch implementation, despite
being substantially above its older submission-time comparison. No later
subset scored evaluation was observed finishing first. Both of our slots
are now free; subset is selected because its successor is furthest along.

Lesson: fewer shared-memory bytes and barriers did not establish an overall
speedup. Without a matched GPU profile, the cause of this regression is
unresolved. Do not claim barrier counts alone predict throughput.

Next substantial experiment: preserve deferred-Y and ranked epoch hashing,
repair the field carry defect, and move root inversion outside the search
kernel using the public pinning PR24 checkpoint hierarchy. This exchanges
global checkpoint traffic for removing almost all scalar inversions and
separating register-heavy recovery stages. GPU throughput remains unmeasured.

## Prepared external-inversion successor

The ranked subset port is complete. Source fingerprint:
`44f9d5808add05e204b169194af8893dc91a9c01d31448362b0e01c667c9fd06`. See submission-ranked-pipeline.md.
The ten-file production/audit closure passes source-bound checks for actual
host field operations, inline PTX residue semantics, epoch hashes/recoding,
raw deferred XYZZ producer, checkpoint hierarchy, direct recovery and hit
mapping. A full C++ projection passes types/syntax; CUDA compilation and GPU
performance remain untested. The GPU audit now targets the hot primitives and
external hierarchy, including near-p carry cases and incomplete root groups.

Related pinning 6ce23203 was promoted at 644546620 while this was prepared.
That supports the pipeline architecture, but it is neither a subset score nor
an unchanged-base measurement of our port. The submitted expectation is a
substantial gain from removing per-CTA scalar inversion and specializing raw
recovery, with acknowledged checkpoint/register risks. Our returned PR27
regression prevents interpreting fewer barriers alone as proof of a gain.
