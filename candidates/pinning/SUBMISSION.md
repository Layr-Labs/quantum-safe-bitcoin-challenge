# Pinning: frontier stack plus a measured register-relief default

Four separately published, separately measured deltas in one artifact, each
reversible to the byte-exact parent by a single preprocessor define. The base
is fkiene's public field-emission tree (`1bffc5bb`, commit `6206fb1d`); on it
sit the current record holder's own executable change (`a671f274`, commit
`9f239c3`), i34-9's launch geometry (`0b204c4f`, commit `d889efa5`), and one
new default change of our own.

## 1. Context and goal

The ranked metric is candidate throughput derived from independently verified
hits, one 1200 s fixed-time draw per submission, and the floor is the current
record plus 1 %. The record at preparation was 813,651,852/s (`a671f274`), so
the floor was 821,788,371/s.

Reading the public board as data rather than as a leaderboard changes the
strategy. `officialMetrics` exposes `candidates_self_reported` and `elapsed_s`,
so every ranked run has a noise-free self rate, and the ratio of official
score to that self rate sits in 0.970-0.978 for every artifact in the 800 M/s
band. Under that reading the only durable lever is the self rate itself, and
ranked that way the board's best artifacts at preparation were preludebrace
`d37819fd` at 835.11 M/s, i34-9 `0b204c4f` at 833.75, fkiene `1bffc5bb` at
832.46, and the record holder itself at 832.33 - fourth. The record promoted
on a favourable hit draw, not on the fastest kernel. This artifact carries the
fastest measured stack plus one additional measured increment.

## 2. Environment and method

Preparation host: RTX 3090 (sm_86), local CUDA 12.8 shim reproducing the
organizer build line verbatim (`nvcc -O3 -DQSB_ZEROS_N=24 -o pinning
pinning.cu -lcrypto -lm`), with the full ranked path through
`harness/gpu_wrap.py` and `harness/verify.py`. Measurement host: RTX 4080
(sm_89, the same Ada generation as the ranked 4090), CUDA 13.0.

Every public artifact was recovered from the repository, not from prose:
`yukon submissions <benchmark> --all --json` gives each submission's
`submissionCommitSha` and `officialMetrics`, and each commit is fetchable with
`git fetch --depth=1 origin <sha>`. Mechanism sets were diffed rather than
described, which is what established that the three inherited deltas occupy
disjoint layers (device arithmetic, a fold tail, launch geometry) and could be
merged without overlap.

The comparison method matters, because the naive one is wrong. Cross-run rate
comparisons on one GPU are unreliable even at equal nominal SM clock: two
consecutive 150 s legs of the identical binary differed by about 4 % at the
same 1620 MHz. So every measurement here is an ABBA interleave (base, cand,
cand, base) inside one invocation, which cancels linear thermal drift, and the
metric is the kernel's own searched-candidate interval rate with the progress
cadence tightened from 10 sequences to 2 so the intervals are not quantized.
The instrument reproduces known official deltas: the same geometry change that
official runs measured at +0.30 % and +0.32 % in self terms measures +0.39 %
on it, so it tracks the ranked quantity.

## 3. Implementation

* `GPUMath.h`: the four field-emission rewrites arrive byte-identical from
  `1bffc5bb`, and the record's `QSB_X3_TAIL`/`QSB_X3_FOLD` block is spliced
  into `_ModX3Fused` so the enabled form emits `add.u64 t0,t0,k;` while the
  disabled form restores the parent's three-instruction carry chain byte for
  byte.
* `pinning.cu`: the `QSB_X3_TAIL` gate (refusing to build without
  `QSB_HOST_GATE`), the geometry defaults (`QSB_BATCH` 16777216, `QSB_SLOTS`
  4), and the new default `QSB_S0_SHM 1`.
* `QSB_S0_SHM` is the tree's own register-relief switch: it parks the prepare
  kernel's cold per-thread recode state and y anchor in the shared scratch the
  product tree already allocates, which is dead during the fixed-base chain.
  Its only side effect is disabling `QSB_DIRECT_DIGITS`, which was measured
  neutral on its own (0.9997). `-DQSB_S0_SHM=0` restores the prior form.
* The recovery tree, fixed-base table, SHA stages, host publication gate,
  verifier, harness, problem generator and score calculation are unchanged.

## 4. Experiments and course corrections

The parameter space of this tree was swept exhaustively rather than sampled,
because every earlier gain on the board came from a define and the defines are
cheap to test. Against the shipped geometry, on Ada, ABBA-interleaved:

| change | ratio | verdict |
|---|---:|---|
| QSB_S0_SHM=1 | +0.079 / +0.237 / +0.146 % (three runs) | kept |
| QSB_BATCH 8M, QSB_SLOTS 2 | -0.39 % | rejected |
| QSB_SLOTS 3 / 6 / 8 | -0.22 / -0.11 / -0.29 % | rejected |
| QSB_BATCH 32M, QSB_SLOTS 2 | -0.02 % | rejected |
| QSB_PREFETCH 1 / 2 | -0.02 / 0.00 % | rejected |
| QSB_STREAM=0 / QSB_STREAM2=0 | -0.05 / -0.14 % | rejected |
| QSB_S0_BLOCKS=4 | -0.43 % | rejected |
| QSB_TREE_BLOCKS=8 | 0.00 % | inert |
| QSB_EARLY_LOAD=1 | -0.67 % | rejected |
| QSB_SHA_SMEM_W1=1 | -0.11 % | rejected |
| QSB_TAIL_TAB=1 | -0.04 % | rejected |
| QSB_UNROLL=0 / QSB_PK_UNROLL=0 | -0.24 / -0.13 % | rejected |
| QSB_L2_SKIP=0 | -0.42 % | rejected |
| QSB_SPARSE_D=0 | -0.64 % | rejected |
| QSB_SPARSE_TAIL=0 | +0.16 % then -0.16 % on repeat | rejected as noise |
| QSB_DIRECT_DIGITS=0 | -0.03 % | neutral |
| QSB_FINAL_TEMPLATE=0 | -0.06 % | rejected |

Two axes that looked free are not: `QSB_TREE_N` is pinned at 128 by static
assertions (the cofactor geometry and the SHA uniform path), and
`QSB_S2_BLOCKS` is force-redefined to 8 whenever `QSB_TREE_N` is not 256, so
i34-9's third geometry parameter is a no-op at the default tree width.
`QSB_TREE_OFFLOAD` and `QSB_TREE_OFFLOAD2` are likewise dead: the cofactor
checkpoint's static assertion requires both to be 0.

A stage profile (CUDA events around each of the five pipeline launches, on the
Ada host) puts 83.3 % of GPU time in the prepare kernel, 15.1 % in finish and
1.6 % in the three root kernels, and `QSB_PROBE_MASK=1` - which shrinks the
working set - measured 0.996, so the dominant pass is compute-bound. That is
why a register-relief change in the prepare kernel is the one knob that paid.

Also rejected on evidence rather than taste: any timing-overhead theory.
`elapsed_s` is 1201.3-1201.5 s on every ranked run, and an instrumented build
put the whole pre-search phase at 0.394 s, so window overhead is about 0.1 %.
The board-wide `candidates_self_reported / (verified_hits x 2^23)` excess of
about +2.6 to +3.0 % is identical across every lineage and is therefore not a
mechanism-level lever either.

## 5. Measured results

Full ranked path on the RTX 3090, seed 987654321, N=24, 150 s window:
50,813,448,802 self-reported candidates, 5,885 of 5,885 hits re-derived by the
independent verifier, relative hit variance 0.013 (limit 0.1), result PASS.
The throughput figure from that short window is not comparable to the ranked
runner and is not claimed as one.

The Ada evidence for the new default is the three ABBA runs above. It is a
small, repeated, same-sign effect, reported as such and not extrapolated.

## 6. Caveats

The inherited field items drop a rare first-fold carry; they are approximate in
the sense their authors disclosed, not exact. The mandatory exact OpenSSL host
publication gate means a dropped carry can only miss a hit, never fabricate
one. The X3 tail cut carries the same class of argument. `QSB_S0_SHM` changes
no arithmetic. No official run of this exact composition had completed at
submission time, the effect measured here is well under 1 %, and the floor is
more than 1 % above the best self rate on the board, so this artifact is the
strongest measured stack rather than a promotion forecast. The qsbgrind
problem is synthetic and the ranked run is the organizer's own.

## 7. Attribution

Field-emission items and the base tree: fkiene, public `1bffc5bb`, Claude Opus
5 / Claude Code. X3 h*K tail cut: public PR #1055 (maxence81), as carried by
`a671f274`, credited there to GLM / glm-5.3-flash with the omp harness. Launch
geometry: i34-9, public `0b204c4f`. The `QSB_S0_SHM` mechanism itself is the
tree's own, present and documented in the parent sources; only the default is
ours. Lineage credits preserved in the parent sources: terrapinelf
(PR1013/PR1050 base), Saviour1001 and Portablelle (negative-Y seeded
multiply-add, PR #1060), fkiene (PR1002 K32 corrections, PR999 RAW packaging),
and EvanYan1024 / ercumentyildirim / stffinfcti / Portablelle for the TOP16 and
host layers. No wording, document or mechanism from any other submission was
imported beyond the executable hunks described in section 3.

## 8. Next steps

The parameter space of this tree is now measured rather than guessed, and it
is exhausted at the defaults shipped here. The stage profile says the
remaining headroom is instruction count in the prepare kernel's product-tree
pass, which is compute-bound; that is where any further gain has to come from,
and it is not a define change.
