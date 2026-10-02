# Split64 CTA review before benchmark

Goal: test eight 64-thread CTAs/SM (512 lanes at 128 regs) without changing
WIN3, first-state classes, GPU epoch range or CPU complementary window family.
QSB_CTA_WINDOW_SPLIT defaults OFF. Its ON geometry requires WINDOWS=128,
BLOCK=64 and ROOT_LUT_SMEM=0 (832-word async LUT does not fit the shrunken
inverse arena). Existing carrier knob string unchanged when switch OFF.

Mapping: block b covers epoch pair 2*floor(b/2), with lane = tid+64*(b mod 2).
Thus b=2q and b=2q+1 together cover every old (epoch 2q/2q+1, lane 0..127)
exactly once. Host descriptors and first-state buffers remain allocated for
2*LAUNCH_BLOCKS epochs. Digest nblk doubles while its BLOCK halves, so total
batch_pos and thread range are unchanged. Host epoch base increment and
counters remain unchanged. QSB_SC_LATE recompute uses the same block quotient.
Tags still use epoch*128+lane, and verifier's decode still divides by 128.

Odd last epoch: both fragments alias B's f0 for safe access and set hasB false;
no B hits are published. Out-of-range A blocks substitute epoch zero and set
active false. Empty/full-block return remains uniform before tree barriers.
Invalid leaf denominators retain the existing identity substitution. The tree
uses n=64 with its compile-time shared arenas bounded by SE_BLOCK=64, and
normal exact-hit gate remains unchanged. Two independent roots now invert the
two half-window products: algebraically same inverses, possibly different
speculative short-carry filtering loss; scored verifier, not self count,
determines gain. No timing-only arithmetic deletions or harness modifications.

Kernel's non-paired fallback unsupported by this experiment: production uses
PAIR_SHARED and DUAL_EPOCH_SHA; do not enable split with alternate enumeration.
Native sm89 image must be regenerated if promoted. No production config/image
changed for this test; frozen wrapper enables the experimental switch only.
