# Iteration 2 — register-carried top waves, pre-test audit

Hypothesis: eliminate the three shared P8/P4/P2 round-trips and warp
publication barriers in wave-top16, not the already-tested destructive tree.
Default QSB_TREE_TOP_SHFL=0 preserves the production implementation. The
fingerprint adds the knob only in enabled builds, so the existing disabled
production image still matches; enabled hosts cannot use the disabled cubin.

State transition trace:
- Wave A has P8[j] in lane j for 0<=j<8 (all lanes compute repetitions).
- Wave B cofactor lanes i<16 need x[i^8] and P8[(i&7)^4]. The first
  remains an immutable shared-level read, the second comes from source lane
  (i&7)^4. Lanes 16+j need P8[j] and P8[j+4], obtained from lane j and j+4.
- Wave C keeps its own c[i] or P4[j] and shuffles P4[(i&3)^2] from
  lane 16+((i&3)^2), or P4[j+2] from 16+2+(j&1).
- Wave D keeps d[i] or P2[0] and takes the opposing P2 from lane 16+((i&1)^1)
  or lane 17. The resulting E16 and root products are unchanged.
- All 32 lanes execute every full-mask shuffle, including lanes without useful
  products. Each source is a useful lane. All sources are captured before the
  multiplication changes r. The root divstep and its LUT accesses are unchanged.
- E16 is still written into the same inverse rows. The existing down-level32
  and CTA publication barriers remain. No down-level ever reads P8/P4/P2:
  wave-top replaces them, and the down-sweep begins at level32 (offset 2n-64).
- Inactive lanes carry real or identity products exactly as before. No zero,
  cleanup, hit publication, candidate enumeration, or error path changes.

Smallest existing device check: tests/gpu_epochs/tree_audit.cu (multiplication,
block inverses for all shipped sizes and partial counts, root/fallback edge
cases). Disable the isomorphic scale for that stand-alone mathematical oracle,
as the oracle expects 1/x rather than invu/x. Then use the unchanged benchmark.sh
and exact OpenSSL gate for the integrated candidate screen.

## Actual audit diagnostic

Initial production-default tree_audit returned exit 1 on adversarial inputs.
Unchanged control (SHFL=0) and candidate (SHFL=1), both sized for 256 lanes,
returned identical mismatch counts, including 256 wrong of 8191 at 128 lanes.
Reason: filter_tail_sc.cuh defaults SHORT_CARRY3=1; its QSB_TREE_MUL is a
speculative filter multiplier, not the exact multiply the adversarial audit
asserts. Disabled SHORT_CARRY3, the unused narrow parity specialization
(K2S_PARITY_WINDOW=0, otherwise its direct qsb_fmul references do not compile),
and both isomorphic scale controls for the test's plain inverse contract.
The pre-existing tree_audit then PASSED: all sizes 32/64/128/256, counts
1/129/255/256/257/8191, zero wrong; 8192 direct roots exact; zero bounded status
errors. Production is unchanged except SHFL and uses its normal exact hit gate.

Builds: local digest 128 registers, 24KiB, zero spills. Native sm89 prepared
in the frozen tree with -DQSB_TREE_TOP_SHFL=1; did not touch production image.
Default SHFL=0 leaves the existing carrier fingerprint byte-identical.
