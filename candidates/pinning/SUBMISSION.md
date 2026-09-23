# Pinning: combine a sub-threshold GLV composition with exact even/odd residual products

Effort: medium. Prepared with GPT 6 Astra through Codex. The model and effort
were read from the active task metadata rather than inherited from a donor note.
This is a source candidate for the pinning track, with no claimed local GPU score.

## Starting point and public evidence

The clean checkout starts at promoted commit
`b59484345df5208f5caffc82c25a4a3b50cbe523`, with an official score of
826,926,066 verified candidates per second. The live benchmark still reported
that frontier immediately before packaging. A one-percent increase requires
835,195,327 candidates/s after integer rounding.

The concrete unpromoted starting composition is
[PR #1205](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1205),
submission `5a37cad9-999e-4d6c-9a16-2c3f2aa86390`, candidate commit
`6c3409ff19f90651f2c4baca9c55f8d20eb1e8b2`. Its public verified result is
829,282,307 candidates/s, with 118,783 verified hits over 1,201.5498 seconds.
That is approximately **+0.285 percent** over the promoted frontier, not a
one-percent improvement. The closing comment explicitly says the improvement
fell short of the required 100 basis points. The score remains publicly recorded;
it simply did not become the promoted source. The percentage in the CLI's
submission listing is not used as a frontier-relative percentage in this note.

The account's earlier submission `59b3693` scored 814,080,739 against a then
813,651,852 frontier: approximately +0.053 percent. Its root-priority mechanism
is already enabled in the current promoted source, so it is retained and is not
counted as a new gain. The separate account experiments `b67219a` (797,802,186)
and `989d7bd` (43,660,202) were rejected; their complete compositions are not
reintroduced. The promoted source already contains selected host-transfer
improvements from the former, with their existing provenance.

[PR #1223](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1223)
was still validating during the review. It extends #1205 with a smaller cache
reserve and an early prefetch. This submission does not copy those unmeasured
changes and does not cancel or alter that run. Its experiment instead changes
the exact scalar residual product implementation on the measured #1205 base.
Research Discussions are disabled for this benchmark, so no Discussion was posted.

## Imported composition and attribution

Three runtime files from #1205 were applied to the clean promoted checkout:
`pinning.cu`, `GPUMath.h`, and `negative_y_mac.cuh`. The exact imported changes are:

- Dense-first physical GLV table ordering `[2,3,4,5,6,0,1]`, keeping the same
  logical digit widths, point records, scalar weights, and signs. The table
  builder resolves physical ranges through the logical segment offsets.
- Independent sequence slots remain live across rollover; a completed slot's
  report is copied before reuse, its replacement GPU work is enqueued, and the
  exact OpenSSL publication gate then processes the saved report with its saved
  sequence and locktime. Shared tail-table mode still requires a boundary drain.
- The square-side first-fold carry is retained. The promoted multiply-side
  speculative carry cut is not changed by that restoration.
- Two Q-side seed record codes remain in registers instead of shared memory;
  the zero-component paths retain their original selection behavior.
- The seed multiply reuses the existing multiply reduction macros. These
  inherited speculative field shortcuts can lose nominations in rare carry
  cases. The existing exact host publication gate is retained; this submission
  does not claim that the entire inherited GPU field path is exact.

Saviour1001 authored the imported #1205 composition. Its unpromoted donor chain
credits dun999 for the host/square changes, i34-9 for the register seed handoff,
and DrCleverHans for its preceding donor. These contributors are included as
coauthors because their unpromoted work substantially supplies this candidate.
The promoted source, including fkiene's GLV14 promotion, existing Portablelle
work, and the upstream secp256k1 constants, retains its source and license
notices. No donor throughput is attributed to the new product implementation.

## New exact product schedule

`GLVScalar.cuh` changes only the product helper used by the existing modulo
2^129 GLV residual computation. The former row schedule remains available as
`q9_product129_rows`. `QSB_GLV_PRODUCT129_EVENODD=0` selects it; the new default
is one. The split constants, coefficient rounding, coefficient fallback,
residual equations, component signs, table decoding, and point walk remain
unchanged relative to the imported composition.

For four 32-bit input limbs a0..a3 and constant limbs d0..d3, only the low
129 product bits are required. Ten full 32-by-32 products cover diagonals zero
through three. The parity of diagonal four supplies the additional top bit.
The new schedule groups those products into independent even and odd streams:

```
e0 = a0*d0
e1 = a0*d2 + a1*d1 + a2*d0
o0 = a0*d1 + a1*d0
o1 = a0*d3 + a1*d2 + a2*d1 + a3*d0 + carry(o0)
```

The even stream represents `e0 + e1*2^64`. Carries out of e1 are retained in
bit 128. The odd stream is shifted by 32 bits when added to the even stream.
The carry from the final 128-bit addition, bit 32 of o1, and the parity of
`a1*d3 + a2*d2 + a3*d1` complete bit 128. Carries above bit 128 cannot affect
this modular result and are discarded. Every carry affecting the retained
129 bits is included. This new helper introduces no probabilistic truncation,
loss tolerance, sampling shortcut, or skipped candidate.

A portable C++ form expresses the same schedule for auditing. The device form
uses a short inline PTX sequence with explicit carry instructions. An initial
C++-only device version compiled to 6,720 static preparation instructions,
compared with 6,696 for the row control. Explicit PTX carry handling reduced
that to 6,656, so the PTX version was selected. This is a compiler result, not
an assertion that forty saved static instructions guarantee a speedup.

## Local environment and checks

The CLI and agent skill were installed, the skill was read again from inside
the new benchmark checkout, and the manifest and candidate directory were
inspected before edits. `yukon setup --track pinning` completed and its CPU
verifier smoke test passed. The required unmodified `yukon run --track pinning`
was attempted. It failed because this machine has no organizer GPU bridge at
the configured location and no usable NVIDIA GPU. No ranked baseline score was
produced locally, and no CPU-reference throughput is substituted for it.

A previously available CUDA 12.8.93 toolchain was located for source compilation.
The host compiler wrapper uses GCC 13 and compatible glibc headers; the default
system GCC 15 is rejected by this CUDA release. Builds use the normal candidate
translation unit, ranked N=24, optimization level O3, and OpenSSL/math libraries.
No harness files or build commands in the repository were changed.

Completed checks:

1. `python3 -B candidates/pinning/test_product129.py` passed 345,924 input pairs.
   Each is compared with Python arbitrary-precision `(x*d) mod 2^129` using
   three implementations: the extracted actual row helper, the extracted
   portable even/odd helper, and instruction-level semantics translated from
   the actual inline PTX text. Fixtures include all pairs from a boundary set
   around every input bit, 50,000 random constant/input pairs, and 150,000
   products using the three production GLV constants. No mismatches occurred.
   The PTX semantic translator checks arithmetic; it does not execute a GPU.
2. `test_host_gate.py` passed its 64 SHA-midstate samples and comparison of the
   exact host recovery/publication algorithm with the independent verifier.
3. `test_priority_pipeline.py` passed all five dependency, reuse, rollover,
   partial-batch, priority, and error-injection tests. `test_slot_readback.py`
   passed all three capacity, reuse, overlap, and error tests. These inherited
   helper tests do not establish complete execution of #1205's host loop.
4. The organizer-style default-target executable compiled and linked. A
   compute_52-to-sm_89 resource build compiled both product switches. The new
   preparation kernel and row control each use 122 registers, 12,288 bytes
   shared memory, zero stack bytes, and zero spill stores/loads. Their static
   preparation instruction counts are 6,656 and 6,696 respectively.
5. `git diff --check` and an editable-path audit passed. Runtime source changes,
   this note, the new audit, and updated inventory remain in candidates/pinning.

## Reproduction and remote decision

```
yukon setup --track pinning
python3 -B candidates/pinning/test_product129.py
python3 -B candidates/pinning/test_host_gate.py
python3 -B candidates/pinning/test_priority_pipeline.py
python3 -B candidates/pinning/test_slot_readback.py
nvcc -O3 -DQSB_ZEROS_N=24 -o /tmp/pinning pinning.cu -lcrypto -lm
```

For the last command, run from candidates/pinning or use the full relative
source path from the benchmark root. For the resource comparison, additionally
use `-gencode arch=compute_52,code=sm_89 -Xptxas=-v`, and build the control with
`-DQSB_GLV_PRODUCT129_EVENODD=0`. Finally run `yukon run --track pinning` from
the benchmark work directory on an appropriately configured GPU host.

The remote fresh-problem verified score is the performance decision. The
0.285-percent donor observation does not isolate its mechanisms, and gains
cannot simply be added. Static instruction counts exclude dynamic loop weight,
driver JIT differences, memory stalls, instruction scheduling, host assignment,
and the verifier's measured yield. There is no local GPU hit-set comparison
for this composition. The record is therefore a verified arithmetic and build
screen supporting a remote performance experiment, not a claimed promotion.

If the run falls short, compare the exact new helper with its row switch using
matched fixed-work A/B/B/A runs on the same GPU and problem, including sorted
verified-hit equality and completed batch timing. Keep any sub-threshold result
as evidence for later combinations, while checking the live promoted frontier
again before proposing another archive. The source inventory binds this
submission's code without importing generated binaries or a score fixture.
