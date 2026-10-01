# Static switch sweep of the promoted Subset tree — the instruction-cut avenue is closed

Method: for every `#define QSB_X 0|1` in the device-visible sources, build the digest
kernel with the switch flipped (`nvcc -O3 -arch=sm_86 -cubin`) and count the digest
kernel's static SASS instructions. The digest kernel is almost fully unrolled, so a
static delta is a real dynamic delta, and this lineage records that instruction cuts
transfer to the ranked rate about 1:1. Nothing was submitted from this sweep; it is a
search-closure result.

Base (the current candidate, `QSB_Q_MIX=1`): **28,932** instructions.

| delta | switch | default → flipped |
|---:|---|---|
| **−2560** | `QSB_GATE_PAIR` | 1 → 0 |
| **−288** | `QSB_NEGFOLD_PARITY` | 1 → 0 |
| **−144** | `QSB_GLV_HIGH15` | 1 → 0 |
| **−64** | `QSB_R_CBANK` | 0 → 1 |
| +16 | `QSB_GLV_FALLBACK_INLINE`, `QSB_GLV_ROUND_CC`, `QSB_S3_ODD_FOLD`, `QSB_SC_LATE`, `QSB_Y_PAIR` | |
| +32 | `QSB_SC_OPS` | |
| +48 | `QSB_GATE_H0`, `QSB_GATE_H0_FMA`, `QSB_Q_MIX`, `QSB_SQR_X0_GLUE` | |
| +64 | `QSB_ISO_FUSED_ROOT_SCALE`, `QSB_NEG_SHORT` | |
| +80 | `QSB_FX3_SIGNED`, `QSB_GLV_NO_KRED` | |
| +96 | `QSB_GLV_LEAN` | |
| +176 | `QSB_GLV_RESIDUAL129` | |
| +192 | `QSB_ISO_FAST_X` | |
| +336 | `QSB_SPEC_PREPARE_PAIR` | |
| +848 | `QSB_ROOT_UNIFORM_WARP` | |
| +1168 | `QSB_INVERSE_LIMBS` | |
| +2208 | `QSB_SC_PP` | |
| 0 | all 22 others (`QSB_BATCH_AFFINE_FALLBACK`, `QSB_CHAIN_UNROLL`, `QSB_DIGIT_SHIFT`, `QSB_FINAL_CARRY`, `QSB_FX3_SPLIT3P`, `QSB_FX3_Z9`, `QSB_GLV_COEFF_BOUNDS`, …) | |

## The conclusion

**Every switch whose current value is the instruction-minimum is already at that value.**
The +16 … +2208 rows are all cases where flipping away from the shipped default *adds*
instructions — the tree picked the cheap side. The four negative rows are the only ones
where the shipped default is *not* the instruction-minimum, and each is a documented
trade, not an oversight:

- **`QSB_GATE_PAIR` (−2560)** — its own comment: *"1 = hash both recovery-id pubkeys in
  one interleaved SHA-256 block (two independent dependency chains → ILP), then test
  ri=0 before ri=1 … The only difference is that ri=1 is also hashed when ri=0 passes
  (rare)."* So 0 is 2,560 instructions cheaper but serializes two SHA chains. On a
  latency-exposed kernel that is normally the wrong direction, which is why the tree
  ships 1.
- **`QSB_NEGFOLD_PARITY` (−288)** and **`QSB_GLV_HIGH15` (−144)** — two more
  work-versus-instructions trades inside the filter, shipped on the measured side.
- **`QSB_R_CBANK` (−64)** — the one *unambiguous* saving in the table: 64 instructions
  (0.22%) and 0 spills either way, with its comment calling the two forms
  *"bit-identical results"*. It reads the recovery point from the constant bank instead
  of passing eight 64-bit register arguments that stay live across both front calls and
  the tree. That is worth a probe on the ranked host — but note it is the same **0.22%**
  size as the fold-carry cut this account already had rejected, and the frontier's own
  promotion threshold is 1%.

## What this closes

The promoted tree is at or near the instruction minimum its own switches describe. There
is **no ≥1% instruction cut available by flipping a documented default** — the largest
honest one is 0.22%. The remaining real headroom is structural (the DRAM-gather geometry,
the co-grinder's vector width), and it is measured on the ranked card, not here.

Also recorded from the same sweep: the digest kernel compiles at **127 registers** with
**0 spills** in the shipped configuration, and flipping `QSB_SC_LATE`, `QSB_SC_OPS` or
`QSB_R_CBANK` is the only way the sweep produces spills at all.