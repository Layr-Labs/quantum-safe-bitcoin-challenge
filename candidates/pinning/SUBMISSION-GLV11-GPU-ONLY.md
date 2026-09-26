# Pinning GLV11 GPU redraw with adaptive allocation and no CPU co-grinder

This is a pinning-track-only attempt to move the current verified record of
948,943,797 candidates/s. The required one-percent promotion score at the time
of preparation is 958,433,235 candidates/s. The source is a GPU-only build of
the eleven-term GLV search, with an adaptive batch length to survive lower free
GPU memory. It has no claimed local improvement to the GPU kernel. Its value is
an official draw on a fresh pinning problem and runner after a related GPU draw
fell only 421,128 candidates/s short of promotion.

## Provenance and attribution

I began from the promoted pinning source associated with submission `3b423554`
and inspected the `candidates/pinning/` include closure before editing. The GPU
implementation here comes from team i34-9's public [pinning PR
#1687](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1687),
[commit `a28befc`](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/a28befcd98ce0cedde7cc6b1f1fa754c4565d310).
That unpromoted GPU source scored 951,653,014 verified candidates/s in its first
ranked run. Its note credits earlier eleven-term geometry ports by
Meganpark980320, fkiene, terrapinelf, ercumentyildirim, and the i34-9 team;
field carry rewrites by Ryun1; and contributors to the promoted base. Those
credits remain applicable. I do not claim those GPU mechanisms as original.

The previous integrated experiment also copied a CPU co-grinder from
Meganpark980320's [public source
commit](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/66c7e04c28faed9d1015e9e5708aeb6ebc361fdf).
The `cpu_cogrind.h` and `cpu_cogrind_vec.h` files remain in this archive for
provenance, with their MIT license notice, but the `QSB_COGRIND=0` compile-time
setting in `pinning.cu` excludes them from the build. The four `qcg::` host call
sites are also excluded by that condition. No CPU search runs in this version.
Meganpark980320 and i34-9 are credited as coauthors because their unpromoted
sources substantially contributed to the present candidate.

## Exact code and search scope

The production GPU source and its device arithmetic are those of PR #1687.
The two additions to its host path are the previously tested adaptive `BATCH`
selection and the explicit CPU kill switch. The GPU enumerates sequences
upward from `0x80000000` over the allowed locktime range, evaluates two
recovery flags and both compressed-public-key hash cases, and publishes only
hits accepted by the exact OpenSSL host recovery and hash gate. It writes the
normal `sequence=`, `locktime=`, `recid=` record consumed by the unchanged
pinning bridge. The table's 216 sampled records are checked against OpenSSL.

The adaptive host path queries free GPU memory after building the ~21.6 GiB
lookup table. With normal free memory it retains the donor's eight-million
candidate batch and two overlapping slots. If there is less memory, it halves
the batch until a conservative two-slot reservation fits, down to one million.
This changes host launch geometry and allocation size, not any candidate hash,
curve operation, search bounds, or exact-hit gate. The official build uses the
locked `nvcc -O3 -DQSB_ZEROS_N=24` flags. `QSB_COGRIND` is set to zero in the
source because the official build does not add custom `-D` switches.

No benchmark harness, problem, bridge, verifier, or sibling-track candidate is
part of the archive. The only edited track is `candidates/pinning/`. The earlier
unused carrier and pair-ordinate headers and a large research snapshot were
removed to keep the archive under the Yukon upload size limit. The GPU GPLv3
notice and the CPU-origin MIT notice remain in the candidate directory.

## Measurements and decisions

The required local baseline completed with 98,608 independently verified GPU
hits and a hit-derived 688,748,730 candidates/s on this thermally limited local
RTX 4090. That baseline is a correctness reference, not a ranked throughput
prediction. The GLV11 donor and the prior CPU-integrated variant both compiled
with the locked command, built the table, passed its spot check, and yielded
exactly the same 4,400 distinct GPU hit tuples over their first 30 completed
sequences as the promoted control. The ten distinct CPU hits sampled in the
integrated trial were independently re-derived with the frozen candidate-hash
and leading-zero-bit functions. The local `yukon setup --track pinning` smoke
test of that integrated variant passed. The current build removes CPU execution
at compile time; it does not change the previously checked GPU device path. A
fresh `yukon setup --track pinning` of this GPU-only build compiled with the
locked flags and passed its verifier smoke test as well.

At 30 GPU sequences after a 34 °C start, the donor reported 968.5M/s locally.
The CPU-integrated version reported 964.1M/s after a 38 °C start. Its four AVX2
workers added about 1.97M candidates/s in the first on/off window while the GPU
batch interval lost roughly 0.255%. The conditions differ, so the combined net
effect is not proven; I am testing the GPU-only choice because it avoids host
contention. Trying five CPU workers made the GPU interval worse by about 0.93%
in a later local check, despite a small CPU gain, reinforcing that decision.

An unchanged eight-million, two-slot build can fail immediately after table
creation under an extra 1,100 MiB GPU-memory hold. Under the same hold the
adaptive source chose two-million batches and completed the first 20 full
sequences with exactly the same 3,019 GPU hit tuples as the control. At normal
free memory it chose eight million. These tests establish a local fallback;
they do not identify the cause of any official runner's earlier failure.

Other GPU changes were checked separately and rejected for this submission.
A full pair-ordinate port matched all 3,019 early hit tuples but was ~0.52%
slower in a matched interval. A narrower Y-pair pipeline port matched the same
tuples but was ~0.23–0.33% slower in matched A/B checks. Removing the stage-two
bound branch lost ~0.3–0.5%; turning off SHA FMA addition lost ~1.48%. A third
GPU slot needed about another 0.5 GiB and measured ~0.11–0.28% slower in a
matched-temperature trial. Because none supplied a repeatable gain, the ranked
source retains the donor's two slots and device code.

## Ranked feedback and promotion hypothesis

The first official integrated GPU+CPU draw, `6d8e5b0a`, passed verification on
the `starkware-rtx4090-leadergpu-20260926012106-3568275` runner and scored
917,538,862. Two later `intel-r3-948331` attempts, `d840d5dc` and
`ac95a5f4`, failed within about 2.5 seconds of Benchmark and supplied zero
searched candidates. The exact benchmark problem from a failed draw started a
normal search locally; public artifacts do not preserve the CUDA process error,
so their failure cause is unknown. The adaptive host allocation was introduced
after those failures to make low-memory starts more likely to complete.

The donor GPU redraw `08eea76e` then scored **958,012,107** on the
`intel-r5-54598` runner, just **421,128 (0.044%)** below the promotion floor.
Our next integrated CPU draw `0d2f6587` passed a full official validation on
`intel-r3-948331` but scored **923,148,094** and was rejected below the current
best. This confirms that the adaptive source can complete on r3; it does not
show that CPU execution or allocation policy caused the score difference from
r5. Team i34-9's separate `4d0a3875` pinning run on r5 scored 953,389,478.
Across these runs, fresh seeds and runners have produced several million
candidates/s of variation. The GPU-only redraw is therefore a measured attempt
to remove possible CPU contention and sample another official problem. It is
not a guaranteed promotion or a claim of a faster local kernel.

## Reproduction and limitations

Run `yukon setup --track pinning` from the cloned benchmark work directory and
then the pinning benchmark through the standard bridge. The local short checks
used a frozen synthetic `pinning2.bin` problem in `single_hash` mode; they are
correctness and relative-speed evidence only. The official validator runs the
full 1,200-second search, verifies every emitted hit, and applies the current
record and promotion rule. Runner temperature, CPU contention, table setup,
batch allocation, fresh-hit count, and total process overhead all affect its
score. The public official score is the deciding result. The note and source
archive contain no token, private host path, or private problem contents.
