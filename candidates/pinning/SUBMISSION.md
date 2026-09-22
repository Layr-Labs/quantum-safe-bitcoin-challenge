# Pinning: carry-chain tail reductions behind the exact host gate, on the promoted 7c3609b8 line

Model: qsb-research native agent wave (loop13: PackRebase, CarryCensus; desk
lineage loops 5-11). No local NVIDIA device; the ranked 1,200 s RTX 4090 run is
the throughput measurement. This note is the complete reproducible reasoning
narrative for the composition.

## 1. Initial context and goal

The quantum-safe-bitcoin pinning benchmark ranks **verified candidates per
second** on one RTX 4090 over a harness-owned 1,200 s clock: per candidate,
`SHA256d(preimage)` -> ECDSA pubkey recovery `Q = u1*G + u2*R`, `h =
SHA256(compress(Q))`, a hit iff `leading_zero_bits(h) >= 24`, each candidate
tried at recid in {0,1}. The promoted record at packaging time was
**805,428,058/s** (`7c3609b8`, submission 22944657); the promotion floor
(`minScoreImprovementBips = 100`) was **813,482,339**. The goal of this
composition is a one-draw verified improvement over the promoted source built
exclusively out of error-budget-funded serial-chain reductions that the shipped
**exact host publication gate** (`QSB_HOST_GATE=1`) makes safe.

## 2. Environment and setup

All development happened on an Apple M5 (arm64, macOS) with **no NVIDIA GPU and
no CUDA driver**. CUDA compilation and SASS analysis used a colima VM
(macOS Virtualization.Framework, 8 CPU / 12 GB) running the arm64
`nvidia/cuda:12.4.1-devel-ubuntu22.04` image with `libssl-dev` installed
(persistent container `qsb-toolkit`). The judge-matched compile line used for
every gate below:

```bash
nvcc -O3 -DQSB_ZEROS_N=24 -arch=sm_89 -Xptxas -v \
     -o pinning candidates/pinning/pinning.cu -lcrypto -lm
```

`-arch=sm_89` matches the benchmark GPU for register/SASS analysis only — no
kernel was ever executed locally. Source trees were staged into the container
read-only; cubins (not executables) are the byte-identity comparison surface
because executable md5s embed staging paths.

## 3. Prior work and baseline

The promoted `7c3609b8` line already ships, all default-ON: PR #743 bounded
`_ModMultCore` tail truncation (+0.997% official alone), `QSB_CARRY62`
(2^-62-budget fold/split-3p shortening), the exact **host publication gate**
`QSB_HOST_GATE=1` (kernel may approximate internally; the host re-derives every
hit from `pinning2.bin` constants via OpenSSL before publishing — subset-track
architecture ported to pinning), `QSB_C31` empty second-fold tails with
one-limb K, `QSB_YOFF` offset ordinates, `QSB_SAS_Z9SUB_ALL` (PR #827 field,
z9-lane removal), `QSB_PARITY_WINDOW` (PR #885 bounded 27-cross-product parity
window), and `QSB_ISO_XR` (problem isomorphism `u^2*xR = ±1` replacing the
per-candidate `xR*ZZ` multiply with a signed limb selection). Official
negatives we did not touch: GLV/joint-comb family, exact four-wave complete
top-16, short-carry top-16 reassociation (PR #700/#705 counterexample), chain
unroll 2/3/4, L1 prefetch/bypass, fused one-grid, Karatsuba, and the pinning
SHA flags. Same-tree A/A official spread measured 0.402% — single-draw
attribution inside that band is noise.

## 4. Hypotheses and the error-budget frame

The 2^-16 cumulative divergence ledger (`eps_rows_pinning.json`,
`eps_rows_gputmath.json`) is the shared bank all truncation members draw from;
the host gate converts kernel-side approximation into host re-derivation cost
(~93 tentative hits/s at 805 M/s = milliseconds per GPU-second), so per-tail
error classes up to ~2^-22 are bankable when the divergence predicates are
*measured*, not assumed. Three members passed that bar:

- **Q337 `QSB_MUL_SFC_ELIDE`**: the `_ModMultCore` short-carry sf block
  computes `addc.u32 sfc,0,0` and feeds it into `z2`. `sfc != 0` requires the
  fused `z8*K` word to wrap, i.e. the product top limb `x15` in the top
  ~978/2^32 band — a **delta^2-class** event because P(uv > 1-d) = O(d^2) for
  uniform operands (the top-limb density is ~-ln(t), concentrated at 0).
- **Q277a `QSB_FIELD_SC_TREE`**: candidate-tree INTERNAL multiplies (block
  inverse up/down-pass, checkpoint tree products) can ride the short-carry
  body `_ModMultCore_tree`; roots and the leaf multiplies feeding `_ModInv`
  and published coordinates must stay carry-complete (`qsb_field_mul` +
  `qsb_field_normalize`), per the upstream RESEARCH.md blessing.
- **Q323b `QSB_TREE_F8DROP`**: the tree-scoped fold1 even-carry `f8` drop
  (divergence exactly `{f8 != 0}`, 3.6e-6). The unscoped drop across the 96
  per-candidate chain muls prices 2.389e-5 > 2^-16 and was **refused** —
  scoping, not greed, is what fits the ledger.

## 5. Approach selection and tradeoffs

Three alternatives were measured and rejected during the desk phase:

1. **`_ModAdd256` masked-C replacement (Q214)** — perfect CPU exactness
   (100,002,107 canonical pairs, zero mismatches), but a 25-site caller census
   found only 3 live callsites, one of them non-canonical
   (`kernel_build_gtable` -> `_PointAddSecp256k1`), and the sm_89 SASS census
   showed a **net-zero instruction delta** (SEL -12 offset by ISETP/LOP3/IMAD
   +25). The filed mechanism was abandoned on its own pre-registered bar
   ("reject if ptxas already schedules equivalent correction"). Lesson kept:
   all-caller branchless `_ModAdd256` (public PR #112) was unsafe; a
   canonical-sites-only variant is sound but valueless.
2. **`QSB_FINISH_BGAUGE` (Q274)** — a setup-proven `b=u2r_y` gauge deleting
   the packed-finish exceptional-limb predicate was implemented and then
   **dropped**: the shipped `QSB_PARITY_WINDOW` had already removed the
   full-product parity tail the gauge targeted, making the gauge
   instruction-negative (+64 insns/candidate dynamic, 66 -> 70 regs). Its
   commit stays in the lineage at default-0 as negative evidence.
3. **A blanket sfc drop across all second folds ("N3sfc")** — initially priced
   budget-busting (2^-22.1/fold) from a *uniform-x15* assumption; the parent
   analytic check (P(uv > 1-d) = 1.038e-13, 2^-43.13, with worst-case q/f8/cy
   bands; 0/5e6 simulated) and the dedicated rate harness below corrected the
   model: product-derived tails are delta^2-class. The per-use measured rate,
   not the fold count times a uniform tail, is the only admissible pricing.

## 6. Implementation and files changed

Against pristine `7c3609b8`, exactly five files differ:

- `candidates/pinning/GPUMath.h` — `QSB_MUL_SFC_ELIDE` block: in the
  SHORT_CARRY-gated `_ModMultCore` sc sf block, delete `addc.u32 sfc,0,0` and
  change `addc.cc.u32 z2,z2,sfc` to `addc.u32 z2,z2,0` (the C31 empty
  second-fold tail makes the carry write dead); `QSB_TREE_F8DROP` block with
  the `_ModMultCore_tree` transcription (programmatic copy of the live
  frontier body with the fold1 `f8` addend dropped: `addc.u32 z8,f8,w7` ->
  `addc.u32 z8,0,w7`); `#error` gates requiring `QSB_C31 && QSB_SHORT_CARRY
  && QSB_HOST_GATE`.
- `candidates/pinning/pinning.cu` — `QSB_FIELD_SC_TREE` flag block with gate
  (`QSB_FIELD_SC_TREE && !(QSB_SHORT_CARRY && QSB_HOST_GATE)` -> error;
  `QSB_TREE_F8DROP && !QSB_FIELD_SC_TREE` -> error), `qsb_field_mul_tree`
  routing, and the regenerated `_ModMultCore_tree` body (programmatic copy of
  the live 7c3609b8 body — byte-identical genesis to the body the CPU-exact
  verification covered — with the tree-scoped f8 drop).
- `candidates/pinning/PackedRecovery.cuh` — the (dropped, default-0)
  `QSB_FINISH_BGAUGE` scaffold retained as negative evidence.
- `candidates/pinning/SUBMISSION.md` — this note.
- `candidates/pinning/SOURCE-MANIFEST.json` — recomputed production hashes
  (all 12 verified against the shipped bytes), baseline moved to
  `7c3609b8`/22944657 at 805,428,058 with floor 813,482,339.

The member defaults ship at 1 (`2b394c3` flip commit) because the judge line
carries no member `-D` flags; every member remains individually revertible
(`-DQSB_MUL_SFC_ELIDE=0` etc.), and the all-off state is byte-identical to
pristine (cubin md5 check below). `QSB_ISO_XR` retired one per-leaf multiply,
so Q337's re-census on this line is 104 short-carry uses/candidate.

## 7. Exact commands

```bash
# judge-line default build (members on by default)
nvcc -O3 -DQSB_ZEROS_N=24 -arch=sm_89 -Xptxas -v \
     -o pinning candidates/pinning/pinning.cu -lcrypto -lm

# off-reversion byte-identity vs pristine 7c3609b8
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_MUL_SFC_ELIDE=0 -DQSB_FIELD_SC_TREE=0 \
     -DQSB_TREE_F8DROP=0 -arch=sm_89 -cubin -o pack_off.cubin \
     candidates/pinning/pinning.cu -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -arch=sm_89 -cubin -o pristine.cubin \
     <pristine 7c3609b8>/pinning.cu -lcrypto -lm
md5sum pack_off.cubin pristine.cubin
# -> 1d628d372fc2d97772ab70e2d1663990  (both; byte-identical)

# CPU rate harness for the sfc divergence (5e8 uniform + 2e8 delta-box + boundary states)
gcc -O2 -fsanitize=undefined -o q337_sfc_rate q337_sfc_rate.c -lgmp   # word-exact model
./q337_sfc_rate
```

## 8. Experiments, failures, and course corrections

- **sfc rate harness**: 0/5e8 uniform canonical product pairs (any
  uniform-assumption rate >= 2.3e-7 would predict 115+ events; P < 1e-24);
  delta-box conditioned 2e8 states give rho_box = 0.50049 and absolute rate
  **1.2975e-14/use** — matching the earlier independent filing to 3 digits;
  0/64 engineered boundary states. Divergence identity verified: mismatches
  occur EXACTLY where `{sfc != 0}` was forced, zero elsewhere.
- **A/B SASS compensation**: pristine vs Z9SUB_ALL=0/CARRY62=0 builds show
  ptxas compensates (hot kernel 4096 -> 4096 SASS instructions, delta 0,
  composition shifts; registers 128 vs 124). Consequence recorded: the
  residual benefit of tail truncations is carry-chain depth and registers,
  NOT op count — all instruction-count predictions here are upper bounds.
- **Register/spill gates**: judge-line default build = 7 kernels, 0 spill
  stores/loads, stage-0 **120 registers** (pristine 128 — the members lower
  pressure), stage-2 66, `kernel_build_gtable` 128, tree kernels unchanged.
- **Flag-reversibility**: 16 flag combinations preprocess cleanly; all
  `#error` gates fire under wrong stacks (e.g. TREE_F8DROP without
  FIELD_SC_TREE); flag-off cubin byte-identical to pristine.

## 9. Measured results (local evidence only)

- Composed added error budget: **3.6e-6 = 23.6% of 2^-16** (~34.9% including
  the shipped upstream spend: Z9SUB_ALL ~2.6e-6 analytic, C31 8.2e-8, PR #743
  ~6e-12). Members: Q337 1.3e-12 (104 uses), tree f8 3.6e-6 (R6).
- Composed cubin differs from pristine (`1bbd980cf22fe5f0a3397c05c23d441f`),
  SASS -208 lines total, deltas confined to the multiply/tree regions.
- Instruction model: ~-173 insns/candidate at 0.0045%/insn ruler = +0.78%
  nominal, **+0.20% discounted** — and the ruler is an upper bound in this
  ptxas regime (see the A/B compensation above).

## 10. Caveats

There is **no official ranked draw of this composition yet**; every number
above is local compile/SASS/CPU-exactness evidence. Expected effect is a
rider; the composed draw is primarily a banked official marginal for the
vehicle line. Same-tree A/A spread 0.402% binds: single-draw attribution
inside it is noise. All promoted upstream mechanisms (ISO_XR, PARITY_WINDOW,
Z9SUB_ALL, HOST_GATE, CARRY62, C31, YOFF) remain byte-for-byte upstream.

## 11. Learning and next steps

The 2^-16 error ledger is now ~83% committed across the shipped baseline plus
the desk-ready companions (`QSB_SAS_F8DROP` on `_ModSqrAddSub2`, 2.957e-6,
unclaimed upstream) — the truncation family is close to saturated, and the
K32 correction reshapes landing upstream would require re-censusing against
their forms. Next desk steps: fold the N2 member after its M5 protocol, then
re-derive the composed budget; the measured A/B residual-benefit question
(chain depth vs op count) needs one official paired draw to resolve.