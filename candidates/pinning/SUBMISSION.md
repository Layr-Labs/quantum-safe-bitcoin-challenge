# Four cached GLV banks with full-table huge-page readback

## Summary and origin

This pinning candidate combines a twelve-term GLV table geometry with four small
banks in the existing persistent L2 prefix and a best-effort Linux huge-page
allocation for the complete host table-check buffer. It does not replace or
shorten the table check. All 9,803,211,584 table bytes are still copied back,
the same sample selection and OpenSSL check run, and the same CPU table builder
is retained as the fallback. Kernel hit publication, the benchmark clock, the
scoring harness, and the independent verifier are unchanged.

The production arithmetic lineage is the promoted fkiene implementation at
b59484345df5208f5caffc82c25a4a3b50cbe523. The cache-prefix GLV12 approach was
subsequently published by odinfree in promoted submission d71d3b7, commit
1ec687fa51b136882a72e1ba70f17958e90f4932. We independently implemented and tested
the cache-aware geometry on the earlier arithmetic base, then changed the bank
sizes to place four banks in the cache prefix. We did not import donor changes
to host hit processing or publication. All inherited GPLv3 and secp256k1 notices
are retained, including the accompanying license files.

At final preparation the public record was 881,273,403/s, submission 2c7a195,
commit 1fe5a8e40008befcd917668ea9b1a23c6ee590c4. Comparison of that commit and
its parent showed only removal of the first ordinary comment in GPUMath.h.
The production files contain no line-sensitive macros affected by that edit.
Our reference executable uses the executable-equivalent 1ec687f source. Its
local measured score is reported below rather than substituting the public
runner's score for a local control.

## Mechanism and tradeoff

The twelve-term decomposition uses six table banks for each signed GLV
component. It still needs eleven mixed point additions in the combined chain.
The first four physical banks occupy 48 MiB, inside the existing 50 MiB
persistent window. Compared with the three-bank prefix, the intended benefit
is reducing expected uncached table gathers from six to four per candidate.
This is a memory-access tradeoff, not a claim that adding table capacity is
universally beneficial. A larger table needs more host and device memory and
makes address translation and construction more expensive.

The lower chunk widths are [18,19,18,18,27]. Shifts are
[0,18,37,55,73,100], with top center 170559769. Logical and physical bank order
are both [0,1,2,3,4,5]. Entry counts are
[262144,262144,131072,131072,67108864,85279885], with offsets
[0,262144,524288,655360,786432,67895296]. The total is 153175181 affine records,
64 bytes each, or 9803211584 bytes. The signed-component bias is
170559770 * 2^99 - 2^17. The record index uses 28 bits, while the Y sign remains
in bit 31. The builder radix and high-ladder capacity are 16384; the maximum
high index is 10410. Table loads and the Y-offset pass retain 64-bit byte
addressing, which matters because the table is larger than 4 GiB.

The geometry alone produced a small verified improvement, but its startup
cost diluted the gain in fixed-duration measurements. Rather than remove the
readback or change the checks, we profiled the complete setup. An unscored
CUPTI capture recorded the GPU builder at 305.239 ms and the full table DtoH
transfer at 5361.990 ms. This identified host allocation/page handling as a
more promising target than further reducing builder arithmetic.

## Host allocation change

On Linux the check buffer uses posix_memalign with 2 MiB alignment and an
allocation size rounded up to that boundary. A best-effort madvise call with
MADV_HUGEPAGE requests transparent huge pages. If aligned allocation fails,
the code falls back to the original malloc allocation. If the advice is
unavailable or ineffective, the allocated buffer is still valid ordinary
memory. On other platforms the original malloc path is used. The buffer is
released with free in either case.

Only allocation capacity is rounded up; the actual CUDA transfer length
remains exactly gt_sz. Neither the number nor identity of table samples is
changed. The original allocation failure, GPU-error decision, table-check
failure, and CPU fallback control flow remain intact. No pinned allocation,
page-lock privilege, system setting, memory overcommit setting, or power-limit
change is required. Performance depends on whether the runner's kernel and
memory policy provide huge pages, so this is not a guaranteed portable speedup.

A standalone diagnostic allocated the same full-size buffer and copied a
deterministic GPU-generated pattern. Every 64-bit word was checked. Alternating
normal/huge/huge/normal order gave copy times of 5.248310, 1.283339, 1.282183,
and 5.313579 seconds, with zero bad words in every arm. Huge-page arms reported
9574400 KiB of AnonHugePages; normal arms reported zero. Allocation calls took
under 36 microseconds. Free plus smaps inspection took about 1.00-1.06 seconds
for normal memory and 0.046 seconds for huge-page-advised memory. These are
standalone diagnostic measurements, not benchmark scores.

## Correctness and code-generation checks

The production decoder, compiled on the CPU, reconstructed 200170 signed
component cases exactly. Independent rational arithmetic checked the GLV
component bound, and 100000 independently calculated splits stayed within it.
The table geometry passed an N20 integration run with 18084 verified hits.
The full native table spot check also passed before each timed run.

The organizer build command was used without an additional architecture flag:

    nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm

The measurements used CUDA 12.8.93 and one RTX 4090 at its 450 W limit.
Preparation uses 124 registers and 12 KiB shared memory; finish uses 64
registers, with no spills in either. The huge-page allocator change produced
a byte-identical sm_89 device cubin to its geometry-only parent. Thus that
last change affects the host allocation, not GPU arithmetic or scheduling.
All sixteen shipped production/license files were checked against the exact
source manifest used for the verified binary. No prebuilt binary is shipped.

## Matched quick and ranked-length results

All results below use the unchanged harness and independent verifier, N24,
fixed-duration scoring from verified hits, and a separate equal 15-second
warmup. GPU jobs were serialized. No profiler was loaded for a scored run.
The reference and candidate used the same seed within each pair; the ranked
pair used a different seed from the quick pair.

| Run | Seconds | Verified hits | Verified candidates/s |
| --- | ---: | ---: | ---: |
| Quick reference | 300.3637 | 30505 | 851948858 |
| Quick candidate | 300.3524 | 31327 | 874938566 |
| Ranked reference | 1200.8903 | 121956 | 851902211 |
| Ranked candidate | 1200.9153 | 125298 | 875228938 |

The quick matched improvement is 2.6984845%; the ranked matched improvement
is 2.7381930%. Quick seed was 1921063479 and ranked seed was 1921063487.
An earlier quick reference measured 853379113/s; the candidate was also
2.52636% above that reference. This second control helps bound local drift.
The geometry-only candidate had measured 861970722/s before the allocator
change. Its warmup table setup was 6.75 s, versus 2.82 s for the final candidate.
These setup observations come from separate warmup logs, not replacements
for the harness-owned elapsed time.

Loaded telemetry excludes the first 60 seconds and samples below 95% GPU
utilization. In the ranked pair the candidate averaged 449.2589 W at
2227.3487 MHz and 68.8237 C; the reference averaged 450.0044 W at
2189.1509 MHz and 69.6652 C. Software power capping was active in 99.9561%
and 100% of those samples respectively. The result therefore occurred under
the same nominal power cap, without raising power or locking clocks.

The final candidate binary SHA256 used for both test lengths was
048d4fe1b96f2e4ca45e5f9bb789d3f127dd0ed75650dc7d6076b833ed19d83f.
The reference binary SHA256 was
04d6fec5fd00436eb3909d9837e1a1dab2ec0101985d29d83f1643e447277d3b.
The source manifest shipped with this submission allows source identity to
be checked independently of binary identity or local toolchain paths.

## Hit-set comparison and interpretation

All emitted hits passed the independent CPU verifier. The allocator-only
candidate versus its geometry parent had 30861 identical hits in their shared
quick search prefix, with no differences and no duplicates. In the ranked
candidate/reference prefix there were 121950 common hits and five valid hits
unique to each implementation, with no duplicates. Those implementations also
differ in table geometry and inherited arithmetic paths; exact hit-set identity
against the public reference is not claimed. Net shared-prefix hit count was
equal. Individual unique hits were included in the unchanged full verification.

These are local measurements, not an official record, a prediction of the
organizer's exact score, or a guarantee of promotion. The official record was
higher than this pod's absolute candidate score, but the same record executable
was slower on this pod too. The relevant local evidence is the paired margin.
Runner assignment, clocks, thermal conditions, huge-page availability, and
hit-count variation can all change the official outcome.

## Failed approaches that informed this candidate

A geometry minimizing total table size but leaving every bank outside the
small cache prefix lost heavily; capacity alone was the wrong objective.
A block-batched table-builder inversion experiment passed native arithmetic
and full verification but barely changed setup time or throughput. The later
transfer profile explained why: the full host copy dominated the builder.
A nine-block finish experiment with explicit shared-memory recovery staging
also passed native and end-to-end checks but lost throughput despite improving
occupancy capacity and avoiding spills. It is not included here. None of
these unsuccessful changes is silently composed into the submitted binary.

## Scope, reproducibility, and attribution

Compile the supplied pinning.cu with the command above and run it through the
challenge's standard pinning interface. For a local paired reproduction, use
the unchanged run_benchmark.py with --bench pinning --N 24 --mode fixed_time,
--seconds 300 or 1200, and the paired seeds above. Use identical warmup policy,
verify every emitted hit, and compare both score and power/clock telemetry.
The package contains only production headers/source, the inherited licenses,
this note, and a production source manifest. Test artifacts, toolchains,
telemetry files, generated problems and binaries are excluded.

Credit for the cache-prefix GLV12 direction belongs to odinfree's promoted
work; the source baseline and subsequent record are linked by their commits
above. Existing credits to the field, recovery, hashing, checkpoint and host
pipeline contributors remain in the shipped source and licenses. The new
four-bank geometry and huge-page buffer were developed and tested separately,
then evaluated together as this final candidate. The submission uses GPT 6
Astra through Codex; no claim is made that the inherited arithmetic was newly
authored in this experiment. The single authorized official evaluation will
establish whether the local improvement transfers to the organizer's runner.
