# EC192 integration with actual HM43 root

`check_tree.py` passes on frozen source
`6cb9b00606f541cd0894a48308fa9771dd68efe4bb269f49515e1f8e14d4435d`.
`results.json` verifies all nineteen closure hashes before and after execution,
and separately binds the shared host-support files and inherited tree checker.

The extracted actual EC192 body executes its unchanged pair/ascent/descent/
expansion arithmetic using independent OpenSSL multiplication. Its actual root
region calls the unchanged donor HM43 header, compiled with its host-oracle
collectives and the separately qualified carry/AddP/SubP projection. `_ModInv`
is a fail-only stub: neither OpenSSL inverse nor Python pow implements the
root. Python pow is only the independent final-output oracle.

Only physical threads64..255 are launched. The first EC warp64..95 calls HM43
with relative lanes0..31; all root arrays are initialized and nonleader zeros
are checked. Tails0,1,31,32,33,191,192 substitute identity for inactive values,
while all192 participants execute the tree. Every case checks:

- 192 outputs and384 enclosing value canaries;1,344/2,688 across seven cases;
- 669 tree field multiplications and exactly one root inversion implemented by
 32 actual HM43 callers;
- five EC subgroup joins and eleven tree warp joins per participant;
- 1,536 initial limb writes, original leaves, identity padding and tree bounds;
- cross-warp read publication epochs and all32 HM43 collective lane counters.

The identity-root case executes79 exchanges and65 ballots; the other six
selected roots execute92 exchanges and78 ballots. These are host model
collective counts for these roots, not cycle or universal iteration bounds.

Five source-projection-only mutants compile and are rejected:

1. Passing physical rather than relative lane reaches an invalid HM43 exchange
   source; the actual numerical/collective code detects the error.
2. Calling only the leader leaves a one-lane collective mask and fails a
   bounded rendezvous.
3. Omitting guard lane31 leaves mask0x7fffffff and fails a bounded rendezvous.
4. Restricting root participation to active values fails on the one-active tail.
5. Removing the root publication join violates the enforced five-join contract.
   This run was rejected by join-count auditing, not claimed as a numerical
   race witness.

The rendezvous timeouts are10 seconds each with a150-second subprocess limit.
Incomplete-participation controls intentionally omit required host callers;
their failure is not evidence of a target GPU deadlock. The current algorithm's
positive cases complete. The exact subgroup barrier body is byte-identical to
3ac and projected as a192-thread host rendezvous, so this check does not newly
verify the legacy atomic/fence implementation or GPU memory ordering.

Reproduce from the benchmark root:

```
python3 -B candidates/subset/research/warp_root/check_tree.py
```

`--source` and `--output` are supported, but the source must match the frozen
nineteen-file manifest. Generated C++ and compile/run logs are retained beside
this report. This is actual-source CPU integration, not CUDA compilation,
GPU execution, throughput measurement or complete kernel protocol testing.
No candidate/stage source was edited, and no VM or submission action occurred.
