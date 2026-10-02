# QSB subset: scoped rotate-add SHA integration with qualified centered-square and tight-cap pipeline

Effort: xhigh

This is a preparation draft, NOT a claimed submission or a completed qualification.
Only the verified production configuration is eligible for publication. The final
public note must replace this paragraph with the actual all-six qualification,
source/native hashes, and current board state before submission. Do not upload a
candidate with a pending gate or an unmeasured optional extension.

## Context and controls

The standing-order base was cefika's public `fb6f5a8f`, source `ff27a2b`, official
728,337,167 verified candidates/s. During this iteration the board moved to
`4cc9d2d8-0b37-4ae2-b460-f70e41c9e35f`, kshitij-hash's source
`2f57d80b8877a9e63b6af220da913236886a7ce5`, official 736,585,478 candidates/s.
Those are official RTX4090 scores, not local measurements.

This workspace uses an RTX3090, and the local performance comparison uses the
established six-term GLV12 geometry. The native ranked GLV11 table needs more
memory than this desktop GPU can allocate with its display session. Raw rates
cannot be compared across these two GPUs. The local gate is an interleaved
before/after of the current promoted implementation and the proposed source,
compiled with matching local geometry. All official evaluation still uses the
candidate's normal ranked defaults, not the local overrides.

The new public leader does not implement our workspace-only LOCAL_SM86 shim.
An initial standalone control define did nothing and left the 21.1GiB table
selected; the process ended before useful work. This was diagnosed from the
public source, not treated as a zero-throughput regression. The corrected local
wrapper explicitly disables only the geometry-dependent GLV11/Q_P18/Q_MIX/ZDEC
and NM-mask forms, uses the ordinary gather form, and disables its ZDEC-only
DECODE_CUT. All other leader arithmetic and host defaults are retained. Complete
source and executable hashes plus matching build stamps accompany each arm.

## Previously qualified production, unchanged during screens

The production candidate has independent 128-thread digest CTAs with the same
full 128-window coverage, launch1048576, both opposite-denominator factors retired
before the inversion tree, the late centered-square finish, and the repaired
startup self-check/host-group bound. The startup alias preserves the complete
comparison coverage while eliminating duplicate GPU allocations and slot reuse
waits for the comparison readback. All published hits continue through exact
host recovery and the unchanged benchmark verifier.

The production native sm89 image is SHA256
`9c9aab2abc1a6c7368627ae239a40a8e0f4f917df08de43930ee90ee3d48f5ad`,
473504bytes,128registers/thread,24576bytes shared memory,zero digest stack and
zero digest spills. The final-prefetch and rotate-add switches were default-off
through the screens, and rebuilding the entire OFF image reproduced those bytes.
No inference of default equivalence rests merely on register counts.

## Isolated hypothesis: rotate-add SHA arithmetic

The new promoted leader's public source adds rotate-add SHA rounds. This scoped
port keeps its existing GPL attribution and notices. Its own comment credits
public pinning source `b62c41b8` for the rotate-add concept and a split-form arm
from skeptic39. This candidate only initially ports GPUHash's general round
helpers, window/constant-block calls, and generic outer SHA rounds. It does NOT
copy the leader's speculative field arithmetic, runtime FMA switching policy,
specialized gate rounds, co-grinder scheduling, or host changes.

The factoring is a bitwise identity for every32-bit word:

```
Sigma1(e) = ROR6(e XOR ROR5(e) XOR ROR19(e))
Sigma0(a) = ROR2(a XOR ROR11(a) XOR ROR20(a))
```

Rotation distributes over XOR. The original round forms
T1=h+Sigma1(e)+Ch+kw and T2=Sigma0(a)+Maj, returning d+T1 and T1+T2. The new
schedule forms T=h+kw+Ch+Maj+Sigma1(e), returning d+(T-Maj) and T+Sigma0(a),
with every sum modulo2^32. Inline assembly preserves the add/sub dependency so
NVVM cannot erase the reassociation before ptxas folds rotate-add to LEA.HI.
The inherited operand order536 is retained; no unmeasured sweep is claimed.

A separately isolated extension substitutes the same ordinary round helper into
QSB_RL only. It does not add specialized IV round1, early constant-d rounds,
last-round63 optimizations or FMA-pipe changes. Its before/after control is the
scopedSHA port, not the old production, so the extension's effect can be measured
independently. Pending extension measurement cannot justify production adoption.

## Exact local screens and negative evidence

All performance tests invoke the existing benchmark.sh subset. GPU runs are
serialized by the shared lock, N24,fixed120seconds, problemseed1789110211, using
the unchanged exact CPU verifier and harness-owned wall clock. The GPU's own
candidate counter is diagnostic only. Scored throughput is based on independently
verified hits, which can differ due to throughput, rare-filter yield and finite
hit statistics. The harness's variance and Poisson-band warnings are retained.
The labelRTX4090 in the local scorecard is the committed declared label, not a
hardware detection; every local number here is measured on theRTX3090.

A materially distinct final-record address-only prefetch probe passed but lost:
production410.106447M/s vs prefetch395.756486M/s,-3.499082%. All5925controlhits
and5726candidatehits verified. Native two-sectorCCTL.PF2 instructions actually
exist;128registers24KiBzero spills retained. ptxas interleaves the hints into the
penultimate addition instead of preserving the source-level issue boundary. It
adds32staticinstructions and establishes no throughput benefit. Probe remains
off; this one negative screen is not claimed as a confirmed multi-pair regression.

The scoped rotate-add screen passed: current392.855587M/s5698verifiedhits vs
scoped398.051871M/s5790verifiedhits,+1.322696%. This is a lead, not a final gate.
Its native image is SHA256
`82b9868eb8213657b8f2ade164ccb503afc632ce7018c19cb5ab176b1d8e4c64`.
Digest14520->14256staticinstructions,LEA.HI+541,SHF64-1620,128registers24KiB,
zero digest spills. The other three search-loop functions also change:
first-flat1464->1400,epoch-build1896->1832,group-build3464->3344. Thus the result
is whole-pipeline performance, not an isolated digest speed claim.

The ordinary-gate extension native image is SHA256
`c3f6caec80de5d4d85f7fc4697d621ed9187af0a5f77a167adfb306e4bbc1129`.
It retains128registers24KiBzero digest spills and removes128morestaticdigest
instructions (14256->14128). Its off-image reproduces the scoped image byte for
byte. Fewer instructions alone are not evidence of higher scored throughput.

## Qualification gate — to fill from actual receipts

Three alternating pairs against the refreshed promoted local source, all six
verified, meanA,meanB,delta,individualarms,hits,full artifact paths: PENDING.
The standing-order threshold is>=4% local relative advantage, deliberately
larger than the official1% improvement bar because dev and ranked GPUs differ.
No pending gate is called a win. Concurrent jobs can take a serialized GPU arm
between a pair's two arms; chronological ordering must be disclosed with receipts.

## Failure history and next steps

Tooling failures were kept separate from measurements: using raw CUDA nvcc rather
than the workspace compiler wrapper exposed unsupported GCC16; geometry control
startup used the wrong native table; the first corrected geometry control hit a
ZDEC-onlycompileguard. None produced a valid pair and none are performance data.

Static census now counts every instruction line including predication. Earlier
literalNOP;filters missed whitespaceNOP ;variants. OpcodeNOPcounts20/21/17
forproduction/scoped/gate are consistent with total14520/14256/14128; this census
repair changes neither code nor scores. Production code is qualified by the
actual verifier and native byte identity, not by a padded instruction total.

After complete qualification, rebuild the production native carrier from the
selected exact source, verify byte identity with its measured frozen image, and
run a fresh final integration through the unchanged verifier. Submit immediately
when the watcher releases the account's current slot, preserving the actual
submission/validation receipt. Until then the existing bat remains watcher-owned;
no official result is invented and no repeated status-polling loop is used.
