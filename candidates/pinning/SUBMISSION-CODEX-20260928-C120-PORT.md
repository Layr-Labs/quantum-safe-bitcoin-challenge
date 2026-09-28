# Pinning: current-base port of the c120 three-cut package

## Scope and objective

This package targets only the `pinning` track. The `subset` track is outside the
scope of this work and no file under its candidate directory was read or edited.
The benchmark harness, verifier, score formula, candidate domain, leading-zero
predicate, and fixed-time window are unchanged. The objective is to improve the
verified candidates-per-second rate on the single RTX 4090 runner while keeping
the exact publication gate and the normal independent verifier intact.

The live frontier observed before packaging was submission
`54ca2f74-5081-4475-921f-1682210e663b`, with an official rate of
995,329,477 verified candidates/s. The corresponding 100-bips promotion gate
was 1,005,282,772 verified candidates/s. These are benchmark state observations,
not a claimed score for this archive. The official validator remains the only
source of the score and promotion decision.

## Starting point and public provenance

The starting executable closure is the current promoted 02c7-derived pinning
source already present in the worktree. Its principal pre-port hashes were the
known 02c7 values for `pinning.cu` and `qsb_carrier_sm89.h`; the source contains
the GLV12, register-root, table, pipeline, exact host-gate, and native sm_89
work inherited from the promoted lineage. The public c120 submission
`c12006c2-37eb-4c91-a8e8-bf0facdb63e8` supplied the three narrow mechanisms
ported here. Its published note describes the same three switches and credits
its preceding public work. I inspected that source independently and ported
only the measured mechanisms into the current 02c7 closure; unrelated c120 tree
changes and stale-base differences were not copied.

The official c120 draw reached 1,003,132,947 verified candidates/s and was
rejected because it was just below the then-current promotion gate. That result
made the mechanism promising but did not by itself prove that its stale base
would beat the current frontier. The port below was therefore screened against
a rebuilt current-base control before packaging.

## Exact changes in this archive

Only these device-source files were changed for the candidate, together with
the generated carrier that embeds the resulting sm_89 image:

* `candidates/pinning/GPUMath.h`
* `candidates/pinning/pinning.cu`
* `candidates/pinning/sha_pinsha.cuh`
* `candidates/pinning/qsb_carrier_sm89.h`

### 1. Presubtraction in the fused square/add/subtract

`QSB_SAS_PRESUB=1` changes `_ModSqrAddSub2`. The routine forms the undoubled
cross-product sum C, subtracts q once before the doubling, and then doubles the
borrow-propagated value. The resulting integer is the same field value as the
existing `R^2 + PPP - 2Q` path, but the post-fold q-subtractions, the +3 bias,
and the 3K correction are no longer needed in this form. The borrow is carried
through the words that can affect the result; the implementation retains a
compile-time kill switch (`QSB_SAS_PRESUB=0`) for comparison.

This is an arithmetic identity, not a change to the candidate predicate. The
existing exact host recovery and SHA gate still re-derives every published hit.
A wrong speculative point would therefore be rejected at publication; no hit
injection, domain pruning, or verifier bypass is present.

### 2. Chain statement order and operand form

`QSB_ZZ_EARLY=1` moves the `ZZ1 *= PP` statement before the fused X3 operation,
then performs the `Qy` subtraction and `ZZZ1 *= PPP`. `QSB_ZZZ_3ARG=1` spells the
last multiplication as the three-argument `_ModMult(ZZZ1, ZZZ1, PPP)` form.
Both changes preserve the exact product and field bits. They are scheduling and
register-assignment hints for ptxas; both have kill switches and are limited to
the chain code. The generated SASS is smaller in the chain loop without adding
spills.

### 3. SHA finish identity

`QSB_FIN_W8S0=1` rewrites the s0 function for the padded final message word
W8. W8 contains only the eight live bits of the compressed-key byte plus the
fixed padding bits. The rewrite is the precomputed linear identity for those
bits, so it produces the same 32-bit word as the rotation expression. The two
other finish experiments in the same public c120 note (`QSB_FIN_RASSOC` and
`QSB_FIN_IVFOLD`) remain disabled; they are not part of this candidate.

## Build and verification steps

The source was built in the Yukon benchmark worktree with the track-specific
setup command:

```sh
yukon setup --track pinning
```

Setup regenerated the synthetic pinning problem, compiled with CUDA 12.8 using
the benchmark's `nvcc -O3 -DQSB_ZEROS_N=24` shape, and passed the verifier smoke
test. The compiler emitted only unused-variable/OpenSSL deprecation warnings;
there were no build failures or register spills.

The native carrier was regenerated from the exact candidate source with the
sm_89 carrier builder and its required symbol/LTC64B checks. The resulting
carrier has five LTC64B table loads in the prepare kernel, a 128-register
stage-0 prepare kernel, and zero spills. The candidate carrier is 477,984 bytes;
the rebuilt current-base control carrier is 478,112 bytes with 126 stage-0
registers. The small size/register difference is part of the measured code
change, not a stale embedded image.

The source and carrier hashes in the submitted worktree are:

| file | SHA-256 |
| --- | --- |
| `candidates/pinning/GPUMath.h` | `97317bdb047c823107d6582a55b55a4d825a179ea7ccc34015acb4ce1b54ef2b` |
| `candidates/pinning/pinning.cu` | `1c207b505148d356bf28b41a13bded09053a3eb8861a0d16cf0569ba85f020d1` |
| `candidates/pinning/sha_pinsha.cuh` | `760b8a0b8da6affbc4f6616d36821fb2fde2fd23f05ff01367e48a052a7f61c9` |
| `candidates/pinning/qsb_carrier_sm89.h` | `117a627588d4c617e786cfe7cba7e806429156c15d121ce0d905b50063802840` |

These hashes identify this archive only. They are not a score claim.

## Matched ABBA/BABA screen

To separate code effect from clock/order drift, I built two isolated copies
from the same current 02c7 source and ran four single-process, single-GPU
180-second arms. Every arm used the same synthetic problem and seed, the same
N=24 binary, the same fixed-time harness, and no competing GPU process. A and
B alternate candidate and rebuilt control order (A1, B1, A2, B2). The runner
was idle between arms.

| arm | package | wall seconds | candidates | progress rate | raw hit records |
| --- | --- | ---: | ---: | ---: | ---: |
| A1 | three-cut candidate | 180.131 | 184,309,528,411 | 1,023.2 M/s | 19,428 |
| B1 | current-base control | 180.139 | 180,049,081,520 | 999.5 M/s | 18,177 |
| A2 | three-cut candidate | 180.134 | 182,908,134,596 | 1,015.4 M/s | 18,190 |
| B2 | current-base control | 180.119 | 180,641,307,278 | 1,002.9 M/s | 17,355 |

The candidate mean was 1,019.3 M/s and the control mean was 1,001.2 M/s, a
matched mean difference of approximately +1.8078%. The candidate won both
order positions, so the result is not explained by a single warm-first or
cool-first arm. GPU temperature and clock were monitored; no second compute
process overlapped an arm.

The raw hit records provide an additional consistency check. For each same-order
pair, every control hit was present in the corresponding candidate hit set:
B1 had 18,177 hits, all found in A1, and B2 had 17,355 hits, all found in A2.
There were no control-only hits. The cross-arm overlap is high, as expected for
an identical deterministic candidate stream with different elapsed progress.
This is a local throughput and semantic screen, not the official score: the
ranked validator uses its own worker interval, fresh problem draw, verified hit
count, and full window.

## Rejected experiments retained as guardrails

Several tempting alternatives were explicitly excluded. The direct Karatsuba
field-product port was measured at about 3.2% below its rebuilt current-base
control and is not included. A cold-L1 load qualifier and broad composite
register/carry stacks were also slower in official or local evidence. Exact
02c7 identity redraws landed below the current promotion floor, so this ticket
is not another comment-only or carrier-only replay. The two disabled SHA
reassociations were not smuggled into the source. Keeping these negatives out
of the archive is deliberate: the submitted delta is the small c120-derived
port that won the matched current-base test.

## Reproduction commands

From the benchmark work directory, a reviewer can reproduce the setup and
carrier checks with:

```sh
yukon setup --track pinning
# the setup command also compiles the ranked executable and runs its smoke test
bash candidates/pinning/build_carrier.sh 24
```

For a local diagnostic, use the benchmark's fixed-time harness on one GPU and
compare the resulting candidate count and hit records against a copy with the
three switches disabled. A local progress line must not be reported as an
official score. The only authoritative result is the Yukon validation record.

## Attribution and limits

The presubtraction, chain scheduling, and W8 s0 ideas are substantial
unpromoted work from the public c120 submission and are credited to
`@ercumentyildirim` through the submission coauthor field. The current 02c7
base is inherited public promoted work; its prior authors remain credited in
that source's public notes. This package does not claim invention of those
mechanisms. The local port, current-base comparison, carrier regeneration,
review, and packaging were performed with GPT 5.6 Sol at max effort through
Codex. Yukon trace collection, telemetry, and the collection trigger remain
disabled as requested; this note contains no credential, API key, private path,
or personal data.

The matched local gain clears the observed promotion gate with margin in this
screen, but local measurements do not guarantee the official draw. If the
validator rejects the archive below the live floor, the result should be logged
with its worker and hit statistics before choosing another mechanism. If it is
promoted, the new frontier and its new multiplicative floor must be read again
before any follow-up submission. The archive intentionally contains one
measured candidate so that the validator's result is actionable either way.
