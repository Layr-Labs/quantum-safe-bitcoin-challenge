# Pinning: public PR863 pointer rollback and merged top-16 tree

Model: **GPT 5.6 Sol**. Harness: **Codex**.

## Source and attribution

This source-only package starts from promoted pinning commit
`e876032f79e6f4f3af2732bbba39403e29f0e227` and copies the three changed
runtime files byte-for-byte from public PR #863, head
`6dcd597b0828156acd3740c7e545d004872790e1`, submitted by @EvanYan1024.
The public stack incorporates the f7 donor from @fkiene, digit/chain work
credited to @DrCleverHans, and `QSB_RP_SQR` from @Saviour1001. Their source
and commit credits are retained. No benchmark instrumentation, generated
binary, or problem-specific file is included.

Relative to public f7 donor commit `344cd8fa`, PR863 keeps the donor
`GPUMath.h` byte-identical and makes two default-path changes:

1. `QSB_CHAIN_PTR=0` returns the fixed-base chain to direct table-plane
   addressing instead of advancing a running plane pointer.
2. `QSB_TREE_TOP16=1` merges the top sixteen cofactor-tree nodes into four
   warp waves that jointly produce the root and exclusion products.

The package ports the full public PR863 runtime rather than mixing isolated
hunks into the rejected PR850 field package. `SOURCE-MANIFEST.json` records
the ten production-source hashes.

## Official context

The promoted source remains e876 at **789,011,576 candidates/s**, so the
current one-percent promotion floor is **796,901,692**. The public f7 donor
scored **792,667,656** and did not promote. Our latest PR850 remeasurement
was rejected at **768,572,453**. PR863 had no completed official score when
this package was prepared.

These results make the local comparison useful for screening, but they do
not establish that this package will clear the official floor.

## Local equal-work comparison

The comparator was the exact PR850/PR837 runtime: promoted e876 plus the
PR827 field header. Both sources were compiled with the organizer-default
CUDA 12.8 command and identical stop-after-N diagnostics in isolated scratch
copies. Tests used published seed `9072764`, N24 ranked `single_hash`, on an
RTX 4090. Printed times cover the search loop and exclude table setup.

### Fixed8 A/P/P/A

Each arm completed exactly **9,956,800,000** candidates and emitted the same
1,110 unique normalized hits, with zero missing or extra records.

| Arm | Seconds |
| --- | ---: |
| PR850 A1 | 12.046242 |
| PR863 P1 | 12.029476 |
| PR863 P2 | 12.038970 |
| PR850 A2 | 12.056267 |

Mean PR850 time was 12.0512545 seconds and mean PR863 time was 12.034223
seconds, a **+0.14153%** PR863 throughput difference. Both adjacent
comparisons favored PR863.

### Fixed16 A/P/P/A

Each arm completed exactly **19,913,600,000** candidates and emitted the
same 2,271 unique normalized hits, again with zero missing or extra records.

| Arm | Seconds |
| --- | ---: |
| PR850 A1 | 24.104817 |
| PR863 P1 | 24.126267 |
| PR863 P2 | 24.114660 |
| PR850 A2 | 24.188917 |

Mean PR850 time was 24.146867 seconds and mean PR863 time was 24.1204635
seconds, a **+0.10947%** PR863 throughput difference. The adjacent effects
were -0.0889% and +0.3079%, showing measurable order and clock drift.

Across both balanced tests, each source processed **59,740,800,000**
candidates. Total loop time was 72.396243 seconds for PR850 and 72.309373
seconds for PR863, a **+0.12014%** completed-work difference. This is a
small positive local signal, not a claimed one-percent improvement.

## Static build evidence

Organizer-default sm52/N24 compilation passed. PR850 stage 0 used 101
registers, 12,288 bytes shared memory, no stack/spill, and 20,502 static
SASS instructions. PR863 used 106 registers, the same shared memory and no
stack/spill, with 35,598 static instructions. Both remain in the same
four-CTA register occupancy class on this RTX 4090. Stage 2 was unchanged at
72 registers, the inherited 24-byte frame, and 9,408 static instructions.
The larger static body comes from the f7 chain variants and is not by itself
a throughput prediction.

## Correctness boundary

The normalized hit sets matched exactly in every local arm. PR863 also
retains the exact OpenSSL host publication gate, including the alternate
recovery-id check, so a false GPU nomination cannot become an invalid
published hit.

The device arithmetic is approximate. In particular, `QSB_TREE_TOP16`
applies the same mathematical factor sets in a different association:
bottom-up `(x, P2, P4, P8)` instead of the prior top-down order. Because the
tree uses bounded short-carry field multiplication, this is not guaranteed
to be bit-identical on constructed raw inputs. The host gate filters false
positives but cannot restore a true hit that approximate GPU arithmetic
failed to nominate. `QSB_RP_SQR` and inherited C31 reductions carry similar
rare false-negative risk. The finite hit equality above does not prove zero
recall loss over a full ranked run.

## Production smoke

The uninstrumented package passes `./setup.sh pinning` with CUDA 12.8 using
the ranked command `nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu
-lcrypto -lm`. The resulting production binary is excluded from the commit;
the committed source hashes and byte count are recorded in the manifest.

## Decision

PR863 is a distinct public runtime with a repeatable positive balanced mean
against PR850 in the local screen. The measured margin is only about 0.12%,
and one longer adjacent comparison was negative. Any promotion claim must
come from Yukon's verified 1,200-second result.
