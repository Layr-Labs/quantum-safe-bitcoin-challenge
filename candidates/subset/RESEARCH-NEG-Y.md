# Subset negative deferred ordinate research

Base: promoted shared main `7c3609b87b9d8e094a16be148fe846dfd5ac7807`.
Research prepared with GPT 6 Astra, xhigh effort, in Codex, 2026-09-22.
This is an unsubmitted experiment, not a measured throughput improvement.

The new `QSB_SUBSET_NEG_Y_MAC` flag stores N=-Ycore in the speculative
deferred-Y chain. Actual Y is -N-Yoff*ZZZ. The slope numerator becomes
(Y2+Yoff)*ZZZ+N. Its 256-bit N is seeded into the first even product row,
with the final carry included in e4. The product a*b+c fits 512 bits for
all 256-bit inputs: its maximum is 2^512-2^256. The remaining product and
reduction code retains the promoted lean-carry and speculative-tail behavior.

The chain replaces the final Q-X subtraction with X-Q to produce the next
negative ordinate. The affine seed uses the same sign convention. At the
last add a single negation restores positive Ycore, then the existing final
Yoff resolution and all recovery/verification code run unchanged. The exact
replay and hit publication paths have not been modified.

Local checks completed:

* `python3 -B candidates/subset/test_negymac_ptx.py` executes actual
  preprocessed inline assembly. Both lean and old carry variants passed
  8,272 directed/random raw 512-bit MAC triples each. Both representations
  and carry variants passed 1,024 random point updates in total. 256 full
  15-addend curve chains passed against independent affine point addition.
* Native CUDA 12.6 `nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24
  -DQSB_SUBSET_NEG_Y_MAC=0|1 -Xptxas=-v ... -lcrypto -lm` succeeded.
* Both kernel_digest variants: 128 registers, 49,152 bytes shared memory,
  zero stack, zero spill stores and loads.
* The repeated point-chain loop has 1,044 static SASS instructions versus
  1,058 with the feature disabled. Whole-kernel static counts are unchanged
  at 21,376 instructions including compiler padding. These are not timings.
* `git diff --check` passed.

The first curve test failed because its test driver accidentally reused the
previous representation's final ZZ/ZZZ as the next seed. Preserving the seed
values fixed the driver; no arithmetic change was needed for that failure.

Subsequently, extracted seed/final-resolution integration checks passed 2,048
source chains and 1,024 OpenSSL point comparisons. The initial small-scalar
test pool accidentally exercised an inherited incomplete-addition collision;
the independent random-scalar pool avoids that deliberately unhandled case.
The exact publication path remains the safety gate for speculative results.
The larger independent experiment is documented in SCALAR-PRODUCER.md. The
small loop deletion alone does not justify predicting the 1% promotion floor.

The previous host/startup/SHA composition scored 616,008,463 against the
623,518,629 leader and was rejected. Its projected gain was not a local GPU
measurement. That composition is deliberately absent here.
