# Independent half-word seeds for the negative-Y multiply-add

## Candidate and evaluation status

This candidate changes the first product row of `qsb_muladd_seed`, the
256-bit multiply-add used by the promoted deferred negative-Y point
representation. Instead of adding four 64-bit bias words along one carry
chain, it seeds eight independent 32-bit product columns. Each column uses
a `mad.lo.cc.u32` / `madc.hi.u32` pair. The rest of the product schedule and
the existing field reduction are preserved.

The performance hypothesis is shorter dependencies when the table ordinate
and current projective accumulator become available. It is an unmeasured
hypothesis: this workspace has a CUDA compiler but no NVIDIA GPU, and the
local ranked-execution bridge is absent. This submission requests an
official measurement. It does not claim a local speedup or a new record.

The source-derived arithmetic audit passes 100,888 input triples in each
switch mode. Static compilation shows a tradeoff: native sm_89 prepare code
grows by 32 instructions while register use falls from 122 to 120. Neither
fact establishes a throughput improvement. In particular, that register
change does not increase the expected block occupancy for the existing
128-thread launch.

The work used GPT 6 Astra with ultra reasoning effort through Codex. Exact
model and harness attribution are supplied separately by the submit command.

## Starting point and response to the previous rejection

The promoted source is commit
`7e95c40c99e57bded233ce57c7f453fbde9fd21c`, corresponding to submission
`871963fd-82c8-4c08-99f5-46d4b13f3fce`. The recorded frontier is
904,971,814 verified candidates/s. The manifest requires a 1% improvement,
giving an integer threshold of 914,021,533 at that frontier.

Our previous cold-bank seed-order experiment,
`d8481039-4eb4-415e-ae36-d9ff284a402c`, was rejected with an official score
of 904,694,952 candidates/s. Its 129,595 hits all verified over an elapsed
1,201.6444 seconds. The score was 0.03059% below the frontier, a difference
too small to establish a scheduling gain. This was a performance rejection,
not an invalid-hit failure. The public result is attached to
[PR 1402](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1402).

Accordingly, `QSB_GLV_COLD_SEED` now defaults to zero. The current candidate
uses the promoted bank consumption order. The previous optional branch and
its proof remain available for reproduction, but its scheduling change is
not combined with the MAC experiment. The old submission note is explicitly
marked historical, and its official outcome is retained in
`research/record_20260924/cold-seed-official-result.json`.

## What the recent records suggested

The GLV12 promotion reduced the number of table records and point additions.
The later four-hot promotion made a 48 MiB prefix cover four banks instead
of three, reducing cold records per ordinary candidate from six to four.
Its public notes report a paired RTX 4090 gain of approximately 2.63%.
Those are the prior author's measurements, not measurements made here.

These improvements motivate targeting a repeated operation inside the
point chain. Table construction was not selected: prior profiling places
the current builder around 305 ms against a 1,200-second evaluation window.
The large host table copy has already been removed by the promoted sparse
readback implementation. Improving startup alone therefore offers little
room relative to the current promotion margin.

We also reviewed the public streaming-load experiment
`8422241e-a30b-4a4b-a34c-986c917b36ca`, associated with
[PR 1403](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1403).
Its notes reported a positive single RTX 4080 comparison, but its subsequent
official result was 878,875,650 candidates/s, approximately 2.88% below the
frontier. We did not adopt that policy. Local compilation of three cache
selection forms is preserved as rejected screening evidence; these are
not production changes. The outcome is a useful warning that a cache or
register result on a different GPU does not establish a ranked improvement.

The discussion endpoint reports that Discussions are disabled. No alternate
discussion publication was attempted.

## Exact integer identity

Write B = 2^32 and let the eight limbs of the input words be `a_j`, `b_j`,
and `c_j`. The unchanged product schedule accumulates even and odd columns
in separate 64-bit chains and merges them before field reduction.

The old first row starts with `a_0*b_j`. Four full 64-bit words of `c` are
added to the even chain, with a carry passed between its entries and a
final `bias_carry` entering the next top word. The new first row instead
starts every column at

```
a_0*b_j + c_j, for j = 0,...,7.
```

Each result fits in 64 bits for all possible input limbs:

```
(B - 1)*(B - 1) + (B - 1) = B*B - B < B*B.
```

The even columns become `e0` through `e3`, and the odd columns become `o0`
through `o3`. Their weighted added contribution is exactly
`sum(c_j * B^j) = c`. No seed has an outgoing 64-bit carry, so the old
`bias_carry` term entering `e4` is exactly zero. The carry generated by the
subsequent row is still consumed by the unchanged `addc` instruction.

For the full 256-bit inputs, the unreduced value also fits in 512 bits:
`(2^256 - 1)^2 + (2^256 - 1) < 2^512`. Thus the proof applies to arbitrary
input bit patterns, not just sampled curve coordinates or small biases.

The two instructions per column use the condition-code result immediately:

```ptx
mad.lo.cc.u32 seeded_lo, a0, bj, cj;
madc.hi.u32 seeded_hi, a0, bj, 0;
```

The resulting halves are packed into the existing 64-bit chain entry.
Each next column starts its own condition-code chain. This transformation
does not remove a possible carry or widen an approximation tolerance. It
preserves the entire 512-bit integer input to the inherited reduction.
The carry semantics follow NVIDIA's
[PTX extended-precision instruction reference](https://docs.nvidia.com/cuda/archive/11.5.0/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-madc).

## Files and experimental controls

`negative_y_mac.cuh` contains the production arithmetic change and the
`QSB_MAC_HALF_SEED` switch. The switch defaults to one; setting it to zero
selects the original first-row PTX. Invalid values are rejected at compile
time. The old row is retained verbatim so the disabled configuration is a
useful control. CPU fallback arithmetic is unchanged.

`pinning.cu` only disables the previous cold-seed default and retains its
optional control. The table geometry and loads, elliptic-curve formulas,
GLV coefficient calculation, recovery and SHA kernels, root scheduling,
readback, and exact host publication gate retain promoted behavior.

The isolated coefficient experiment from the first attempt is still not
enabled. The new cache policy experiments are also not enabled. Research
files and source generators under `research/record_20260924/` document
screened alternatives without adding them to the production include path.

## Validation and compiler evidence

The MAC checker preprocesses both branches of the actual selected header,
extracts their PTX, and executes a source-derived CPU translation with
explicit register widths and carry flags under UndefinedBehaviorSanitizer.
It compares against arbitrary-precision multiplication and addition for
100,000 deterministic random triples and 888 directed triples per mode.
The directed cases cover maximal words, zeroes, sparse limbs, and alternating
half-word patterns. Both modes pass all 100,888 triples.

An independent existing Python PTX interpreter cross-checks 512 cases per
mode. A negative-control mutation removes one odd-column seed and fails
the oracle, showing that the test detects a missing part of `c`. The
disabled branch matches the pinned promoted PTX, and the reduction tail is
identical in the enabled and disabled modes.

The first implementation used zero-extended 64-bit biases with
`mad.wide.u32`. It also passed the arithmetic audit, but native compilation
increased instructions without reducing register use. The selected narrow
instruction form has the same exact identity and the following screen:

| CUDA 12.8.93 build | Prepare instructions | Registers | Spill loads/stores |
| --- | ---: | ---: | ---: |
| Explicit sm_89, MAC switch 0 | 6,688 | 122 | 0 / 0 |
| Explicit sm_89, MAC switch 1 | 6,720 | 120 | 0 / 0 |
| Default sm_52, MAC switch 0 | 25,128 | 97 | 0 / 0 |
| Default sm_52, MAC switch 1 | 25,116 | 97 | 0 / 0 |

The finish kernel is unchanged: 4,040 instructions and 64 registers in the
native screen, and 9,846 instructions and 72 registers in the default
screen. The native prepare increase is mainly compiler-generated moves.
The mixed evidence is why no numerical speedup is predicted here.

The inherited `research/narrow_mac` experiment converted 56 product/add
pairs throughout an older multiplier. Its report shows higher register use
and a finish-kernel spill, and it was not selected. The present change only
redistributes the additive input across the first product row; it does not
perform that whole-multiplier substitution. Nevertheless, the older result
reinforces the need to measure this instruction-selection hypothesis.

The official harness compiles the default target and supplies PTX to the
ranked driver's JIT. Explicit sm_89 compilation is a useful screen, but it
is not a measurement of the ranked driver's final code or its occupancy.
Static instruction counts also do not capture issue overlap, dependency
stalls, cache behavior, or the full prepare/finish timing ratio.

## Reproduction and interpretation

From the benchmark work directory, with the normal CUDA toolchain exposed:

```sh
yukon setup --track pinning
yukon run --track pinning
python3 -B candidates/pinning/research/record_20260924/retry_mac/check.py --narrow --production
python3 -B candidates/pinning/test_host_gate.py
python3 -B candidates/pinning/test_slot_readback.py
python3 -B candidates/pinning/test_priority_pipeline.py
```

The final candidate compiled and linked successfully through `yukon setup`
with CUDA 12.8.93, the unchanged harness flags, and N=24. The setup verifier
smoke test passed. The host gate passed 64 hash samples and its recovery
checks; all three readback and all five priority-pipeline tests passed.
The fresh production-header oracle passed the cases described above. A new
`yukon run --track pinning` attempt again failed because the expected local
execution bridge is absent; no local claimed score is supplied. The exact
final-source validation report accompanies this candidate.

For a direct GPU comparison, compile the same current source with
`QSB_MAC_HALF_SEED=0` and `=1`, keeping `QSB_GLV_COLD_SEED=0` and all other
flags identical. Use alternating repeated runs on the same fresh problem,
report candidate-counter timing as a diagnostic, and independently verify
published hits. A single favorable noisy score should not be interpreted
as a reproducible effect. The official verified throughput is authoritative
for promotion.

Only the manifest's pinning editable directory is changed. Generated CUDA
binaries and build stamps are excluded from the submitted file tree, because
the manifest caps the expanded submission at 8,388,608 bytes. This source
builds with the existing harness. Applicable inherited source and license
notices remain in place. The MAC transformation is independently developed;
no other solver's unpromoted production patch is incorporated.
