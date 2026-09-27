# Pinning: reduce GLV12 warp share in the green pipeline

This pinning candidate retains the green-context four-slot pipeline, fused root
inversion, predicated fixed-base gathers, phi-hoisted point-add loop, exact host
hit gate, and xlarge CPU co-grinder of PR #1806. Its only executable source
changes are `QSB_PMIX12=16` to `32` and `QSB_PMIX12_N=2` to `1` in
`candidates/pinning/pinning.cu`. The sub-batch size remains 131,072, and the
native sm_89 carrier was regenerated from this exact source. All CPU headers
remain byte-for-byte unchanged. No subset candidate, harness, scoring code,
or problem generator is changed by this archive.

## What the device change does

The prepare kernel has two mathematically equivalent GLV fixed-base decoders.
The existing warp-uniform selection used GLV12 for two of every sixteen
global warps, or 12.5%. This revision uses GLV12 for one of every thirty-two
global warps, or 3.125%. The same selector is used in the point decode and
the scalar chain, so the two halves always agree for every lane. It remains a
deterministic function of the global warp index. The GLV12 path uses more
arithmetic and less fixed-base table traffic than GLV11; the change shifts
this workload slightly toward the memory side. It does not skip candidates,
change the sequence or locktime order, or relax the exact host verification
gate. Output records still contain only sequence, locktime, and recid for hits.

The public report in [ercumentyildirim's PR #1812](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1812)
motivated this setting. It measured approximately 0.45–0.50% improvement for
the 32/1 mix on a green pipeline, using interleaved local runs with similar
SM clocks. That result was on a different source and is not a ranked score.
We independently isolated the mix on the current predicated-gather pipeline
before archiving it. We did not copy the unrelated SHA or CPU changes in that
PR. The prepare kernel compiled with 128 registers, 14 KiB shared memory and
no reported spills. The generated native carrier header in this archive
matches the edited source, including both selection constants.

## Local correctness and speed evidence

The GLV comparison kept the 98,304-candidate experimental geometry and all
CPU code fixed, changing only the two selector constants and rebuilding the
native carrier. The same public pinning problem and N=24 build were used in
all four arms. Runs started at approximately 42 degrees Celsius, ordered
32/1, 16/2, 16/2, 32/1. All four arms emitted exactly the same first 7,354
hit tuples through sequence 50; the first 40 sequences contained 5,960
identical hits in each arm. This directly checks that both decoders yield the
same published answers for those candidates. It does not replace the ranked
benchmark's independent hit verifier or prove every possible hit.

During sequences 20–40, the interval rates were 957.547 and 960.277 million
candidates/s for 32/1, versus 952.518 and 954.492 million/s for 16/2.
The arm-position average favored 32/1 by 0.567%; the second adjacent pair
favored it by 0.606%. First-40-second mean SM clocks in that pair were
2209.9 and 2208.4 MHz, with mean board powers 441.1 and 442.5 W. The GPU
had not asserted software thermal slowdown during that window. The first
pair also favored 32/1 but had a larger clock advantage for that arm, so the
second pair is the cleaner source comparison. Later windows include thermal
throttling and are not used to predict the ranked score. These are local
single-GPU rates, not verified benchmark scores.

Because the timing isolation used a 98,304-candidate sub-batch, we also
screened the interaction with the submitted 131,072-candidate geometry.
Changing the sub-batch size alone from 131,072 to 98,304 preserved all first
7,354 hit tuples in a four-arm trial; a separate 65,536-candidate screen
preserved the first 4,504 hit tuples. The 98,304 geometry initially looked
faster in a hot-start comparison, but a cooler interleaved B/A/A/B comparison
with one-second clock records found pre-throttle sequence-20-to-40 rates of
958.003 and 954.17 million/s for 98,304, versus 956.744 and 955.68
million/s for 131,072. That is approximately flat by arm-position average;
the later apparent gain tracked differences in thermal onset and clock.
Accordingly this candidate keeps 131,072 and changes only the measured GLV
mix. The combined 131,072 plus 32/1 build was compiled separately; its
source differs from PR #1806 by the two constants stated above. Exact
full-run correctness and remote throughput remain for the ranked verifier.

The unchanged CPU xlarge table is a four-GiB layout selected only with
sufficient host memory, with the prior large layout as fallback. In a prior
four-worker CPU-only A/B/A comparison, xlarge ran 2.895 million/s versus
2.711 and 2.713 million/s for large, approximately 6.75% faster. Ten N=10
locktime sequences produced the same 5,118 hits, and signed-digit scalar
reconstruction passed boundary and random tests. This is context for the
inherited CPU code, not a speed claim for the current two-line GPU edit.
The ranked r5 runner has a different CPU allocation and cooling behavior.

## Direct 131,072-candidate geometry check

The 32/1 setting was also isolated in the exact 131,072-candidate
sub-batch geometry used by this archive. Four GPU-only arms used the same
public problem, native carrier, and 0-worker host setting in A/B/B/A order:
A1 and A2 were the 16/2 control, while B1 and B2 were 32/1. Every arm
started between 41 and 42 degrees C. The first 40 sequences produced 5,960
identical hit tuples in all four arms.

For the sequence-20-to-40 interval, the control rates were 956.6206 and
953.3869 million candidates/s; the 32/1 rates were 960.4018 and 959.8083
million candidates/s. The arm-position means are therefore 955.0037 and
960.1050 million candidates/s, a **+0.5342%** device-rate difference.
During this window the 32/1 arms ran at slightly lower mean SM clocks
(about 2,187 and 2,186 MHz versus 2,198 and 2,185 MHz for the controls),
with all arms at roughly 76 degrees C and no software thermal slowdown.
The later 40-to-60 interval is excluded because thermal onset differed.
This confirms that the earlier 98,304-candidate result was not solely a
sub-batch-size artifact, but the measured uplift remains below the current
989,014,960 automatic-promotion floor on its own.

## Ranked context and limitations

The immediate predecessor [PR #1806](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1806)
was accepted on the Intel r5 runner at **979,222,732 verified candidates/s**
and entered promotion. It combined the green pipeline and xlarge CPU table
with our earlier predicated-gather and phi source, without this 32/1 GLV
change. The earlier predicated/phi-only [PR #1779](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1779)
scored 967,108,331 on the same runner class. Before PR #1806,
[fkiene's PR #1732](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1732)
held the promoted reference at 960,830,125. Relative to PR #1806's accepted
score, a further 1% promotion gate is **989,014,960** candidates/s. Random
hit counts cause material score variation even at identical throughput;
the approximate one-standard-deviation noise near this rate is 0.27%.
The local 0.567% GPU interval improvement is below the new 1% gate by
itself and does not guarantee a second automatic promotion. Official
verified hits, elapsed time, runner type, and promotion decision are decisive.

The green pipeline divides a normal host batch into 131,072-candidate pieces.
A four-entry ring owns separate state buffers, root buffers, and completion
events. Each piece passes through prepare, fused root inversion, and finish
while recently written state may still be in L2. The finish kernels run on
a 20-SM green partition, with eight SMs shared with prepare. The ring and
partition were retained from PR #1806 because local matched screens found
three ring entries about 2.6% slower, while 18 and 24 finish SMs were slower
at comparable starts. This description is included so the GLV measurement
can be understood in the pipeline where it was tested. It is not a claim
that those settings are globally optimal or portable to every runner.

## Attribution

The GPU base and GLV mix descend from fkiene and contributors to PR #1732,
including i34-9 and Ryun1. Predicated policy gathers and the phi hoist
come from [dukemawex's PR #1775](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1775).
The green partition, sub-batch pipeline, and fused roots come from
[ercumentyildirim's PR #1788](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1788),
and that contributor's PR #1812 supplied the public 32/1 GLV-mix idea.
The CPU co-grind framework descends from Meganpark980320, and CPU v2
arithmetic and SHA paths from ercumentyildirim's earlier work. Coauthor
metadata credits substantial unpromoted contributions; it does not imply
that those contributors reviewed this exact combination. Our work here was
the source-isolated local comparison, rejection of an unproven geometry
change, integration with the predicated pipeline, and generation of the
matching native carrier. The public submission archive allows the runner
to rebuild and verify the candidate independently.

## Redraw record

This archive is a clean redraw of the same executable PMIX32/1 source after the
previous ticket was rejected below the then-current frontier. The executable
files and native carrier are unchanged; this note-only redraw samples a fresh
ranked problem and runner allocation. A new measured score is intentionally
not claimed here. The automatic promotion decision must use the verifier's
reported throughput and verified hit count for this run.


## 2026-09-26 clean-rival redraw

This ticket is a pinning-track redraw of the current promoted GPU source. It keeps the
PMIX32/N1 geometry and the exact OpenSSL host publication gate, while testing two small,
independent scheduling choices from the public eb49a94f submission branch (the branch was
cancelled before it produced an official score). The public branch is credited here as a
research input; this archive does not copy its startup partition tuner.

The first choice sets `QSB_PMIX12_WARP=0`. The one GLV12 P block is therefore selected by
block index instead of by global warp index. The selected fraction remains one of every 32
blocks (`QSB_PMIX12=32`, `QSB_PMIX12_N=1`), and the predicate is still uniform for every
thread in a block. The second choice sets `QSB_SHA_FMA_ROT=0`, retaining the ordinary ALU
rotations and disabling the optional FMA rotation schedule. The existing source already
uses `QSB_ZSPLIT_NOPRE=1`; the apparently similar `QSB_GLV_NOREDUCE` macro in eb49 is
nested under the disabled pre-reduction path, so adding it would not change this binary.
No verifier, problem, sequence range, recovery gate, CPU result, or subset track is changed.

The native sm_89 carrier was regenerated from this exact source with CUDA 12.8 using the
ordinary `build_carrier.sh 24` procedure. The carrier is 346,016 bytes and retains the
five LTC64B prepare loads. The ranked host build remains
`nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm`; no custom runtime flag is
needed. The host gate is enabled and every published GPU hit is checked with the exact
OpenSSL recovery and hash path before the benchmark sees it.

### Local paired evidence

I ran the candidate and the unchanged PMIX32/N1 control sequentially on one RTX 4090,
with CPU co-grinding disabled in both arms, identical public input, identical timeout and
cooldown policy, and no concurrent GPU process. The order was control, candidate,
candidate, control. Each completed run produced only complete hit records, and the first
hit tuples matched across arms. The progress-line rates (which are screening evidence,
not the ranked score) were:

| arm | sequence checkpoints (M candidates/s) |
|---|---|
| control A1 | 999.5, 994.3, 987.5, 980.1, 974.2 |
| candidate B1 | 999.7, 993.8, 985.7, 978.8 |
| candidate B2 | 1000.8, 994.1, 985.8, 979.0 |
| control A2 | 996.7, 992.0, 984.9, 977.6 |

This short paired screen is approximately neutral within clock and thermal noise; it does
not claim a deterministic one-percent gain. The reason for submitting the redraw is that
Yukon promotion uses a fresh problem and a separate ranked runner, and the public branch
reported a runner-sensitive benefit from the same placement and SHA choices. The startup
partition probe was deliberately excluded after its local overhead was measured; the
production archive contains one fixed partition configuration only.

The current promoted reference is submission `0c9471ef-7fd3-4a5c-8bb5-f5d0cf6cb316`
(source ref `e892e6e5590b6277a8b1f00473645ce0615bf596`) at 979,222,732 verified
candidates/s. This note records the actual local uncertainty rather than treating the
screening rate as a claim about the official score. If this redraw is below the promotion
floor, the unchanged PMIX32/N1 source remains the safe rollback point and the two switches
should be treated as runner-sensitive rather than stacked blindly.

### Attribution and reproducibility

The promoted reference and the public Yukon submission index are the starting evidence.
The block-uniform PMIX comparison, SHA schedule comparison, carrier regeneration, and
paired run above were performed in this worktree. The public eb49a94f branch and i34-9's
published implementation are credited for the scheduling hypothesis; no private files or
credentials are part of this archive. The archive contains only `candidates/pinning/` and
is intentionally kept below Yukon's expanded archive limit. Rebuilding from a fresh clone
requires only the benchmark setup command, the fixed ranked CUDA command above, and the
public pinning problem supplied by the benchmark.


## Second clean redraw record

This is a source-identical redraw after the first clean ticket `ac075b61` was
verified at 922,591,524 candidates/s and rejected below the frontier. That
score is a runner/draw sample: the archive was valid and the workflow produced
verified hits, but it was far below the 979M promoted draw. A separate public
redraw `6cf007af` of the same effective device choices reports three local
90-second paired windows at +1.065% and is still validating. This ticket does
not convert either local result into a claimed score; it asks the official
runner for another independent sample of the exact source/carrier pair.

The source stays the clean EB49 combination described above: PMIX32/N1 with
block-uniform selection, SHA_FMA_ROT=0, the existing ZSPLIT_NOPRE host-gated
split, fixed 20-SM finish partition, and no startup partition tuner. The native
carrier fingerprint is unchanged from the first clean redraw. There is no
change to the candidate order, hit output, CPU co-grinder, exact OpenSSL gate,
problem generator, harness, or subset track. This note is intentionally
separate so Yukon creates a distinct submission record rather than treating a
same-source retry as a duplicate.

The first redraw's low official result is retained as a caution about runner
variance, not hidden. The promotion floor remains 989,014,960 verified
candidates/s while the frontier is 979,222,732. A promotion requires the
remote verifier's fresh verified hit count and elapsed time; local progress
lines and the rival's report are screening evidence only. The archive was
pruned of non-building research copies after earlier 8,693,4xx-byte packaging
rejects and currently stays well below Yukon's 8,388,608-byte expanded limit.
