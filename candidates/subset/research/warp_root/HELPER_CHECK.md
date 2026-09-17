# Actual HM43 cooperative host check

The unchanged PR189 HM43 header (`940f4c6ad4272a71d38533123a809c8f1affc7a9e0beca959db48380f24cd1fc`, author AbdelStark) passes execution under32 synchronized CPU threads. This is helper qualification, not CUDA execution, universal divstep proof, or EC192 integration qualification. No candidate, stage, VM or submission was modified.

Reproduce from the benchmark root:

```sh
python3 -B candidates/subset/research/warp_root/check_helper.py
```

The checker directly includes the immutable actual header through `HM43_HOST_ORACLE`, which uses its own __uint128 mulhi implementation. It extracts the exact AddP/SubP/constants/sign-test macros from corrected3ac and projects only their scalar PTX carry/borrow instructions, CTZ and clz into host semantics. Expected values come from independent Python bigint arithmetic. All source/support/fixture/generated-program hashes and results are in `helper-results.json`; progress and numerical diagnostics persist in `helper-artifacts/`.

The1452 positive cases contain857 six-word additions (729 exhaustive generate/propagate/kill patterns across all four groups plus128 random cases),160 six-word negations,160 signed-coefficient products,160 sparse modulus products and115 complete inverses. Each unit case uses all four state groups. Complete roots include zero, p, p+1,2^256−1, powers of two, near-p boundaries and96 deterministic random canonical roots. These additional full-width roots are tested extensions; the proposed production tree supplies canonical nonzero factors and identity padding. All32 inverse outputs are compared, totaling3680 inverse outputs and51,768 checked64-bit words across all tests.

The host support observed25,164 collectives:10,832 exchanges,11,428 ballots and2,904 harness case joins. Every participant had exactly the same collective count, and every rendezvous completed with maskffffffff. The source device wrappers use full-mask shuffles/ballots; the host API enforces their32-participant contract. Nonleader inverse input arrays deliberately contain unrelated values, testing that the initial lane0 broadcast is used.

Four executables compiled and ran with AddressSanitizer and UndefinedBehaviorSanitizer enabled. The positive run completed without diagnostics. Three source mutations also compiled and were rejected by specific numerical mismatches, with exit code2:

- Drop generated carry-in: a propagated word becomes0 instead of1.
- Leave word5 unchanged during six-word negation: the high word becomes0 instead of allones.
- Allow propagation through both guard lanes: the next group's low word becomes1 instead of0.

These failures are arithmetic mismatches, not compiler errors or generic crashes. Guard-input words are deliberately nonzero in the helper unit tests to verify group isolation independently of incidental state initialization.

`host_warp.hpp` is the reusable collective engine for the EC192 checker. Instantiate `qsb_hm43_host::Warp`; each participating host thread sets `current_warp` and relative `current_lane`0..31. Global `hm43_exchange`/`hm43_ballot` forward to that instance. The class exposes collective counts and rejects invalid shuffle source indices, duplicate participants, and inconsistent collective kinds. A rendezvous timeout logs generation, kind and arrival mask after10seconds; the driver separately has a180second process limit and persistent progress. These diagnostics distinguish host harness failures from numerical failures; neither is GPU scheduling evidence.

The EC192 integration should use the same helper on physical threads64..95 with a relative lane argument, retaining allfive subgroup joins and the existing arena ownership. That separate checker must establish its own tree outputs, identity tails, multiplication counts and mutation controls. Current helper evidence does not execute that surrounding protocol or the hot product PTX.
