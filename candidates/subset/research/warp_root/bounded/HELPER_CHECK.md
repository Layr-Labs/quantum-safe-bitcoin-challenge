# Bounded HM43 actual-host qualification

Frozen source `f47ad8241adac7aac14e63ad947735ada3bf3565ae60266fa7153b6744d5583a`, helper `675d2b4cca914de576113b1013bac320610ce968c5bc873bf5fc2caeb205dc27`, passes the actual cooperative C++ helper tests. No candidate or prior6cb source/test artifact was changed.

Run from the benchmark root:

```sh
python3 -B candidates/subset/research/warp_root/bounded/check_helper.py
```

The checker supports explicit `--source` and `--output`. It reuses the unchanged original fixture generator and frozen `host_warp.hpp`, `host_support.hpp` and `field_projection.hpp`; their hashes are checked against the original helper receipt. It directly includes each tested bounded header and records the complete current source closure.

| Run | Cases | Result |
|---|---:|---|
| Cap16, original arithmetic/inverse suite |1452|51,768 checked64-bit words; all32 lanes agree with independent expected arithmetic|
| Cap0, all inverse fixtures |115|All3680 lane returns false;18,400 caller words unchanged|
| Cap1, roots requiring more than one batch |113|All3616 lane returns false;18,080 caller words unchanged|

Cap1 selection uses the separately recorded bigint divstep model only to select inputs requiring more than one batch; it does not replace execution of the actual helper. Zero and p complete in the model's first batch and are excluded from the forced-cap1 failure fixture. Cap0 tests them too. Every failing call preserves its entire five-word caller array, including deliberately unrelated nonleader input words. Failure therefore cannot accidentally publish an intermediate or normalized partial result.

All positive runs completed with fullffffffff participation and equal collective counts for all32 host threads. Cap0 executes only the five initial broadcasts; it has no arithmetic ballots. Cap1 executes one matrix batch before uniform failure. AddressSanitizer and UndefinedBehaviorSanitizer report no errors. Progress logs persist, with10second per-collective and180second process timeout diagnostics; no timeout occurred.

Five mutants compile successfully and fail with explicit arithmetic/status diagnostics: lost carry-in, omitted sixth-word negation, carry leakage through guard lanes, returning true at cap0, and changing result[0] before cap0 returns false. The last two directly test the new status/publication contract. No mutant is counted as detected merely because compilation fails or a process times out.

`helper-results.json` binds source, checker, model-selection, support, generated C++ and fixture hashes. The actual helper uses its host mulhi branch and corrected3ac AddP/SubP macro projection; scalar carry/borrow/CTZ/clz are host semantic replacements. This is finite CPU qualification, not CUDA execution, a proof that every input finishes within16 batches, or validation of the retained scalar fallback. The separate actual EC192 tree checker must test fallback selection, root publication and its unchanged subgroup/arena joins. No native-resource or throughput claim follows here.
