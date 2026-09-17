# Guard-only frontier chain review

The selected guard-only source `24e96b567aa1662aa941e582ca4928cf393c9fcaf22161ee3a2b6355b0827199` passes the bounded CPU checks below. Its final complete addition fixes a constructed doubling failure in exact pending PR217. Its selected multiply, square and cold-correction bodies are byte-identical to PR217 and pass a fresh, separately executed arithmetic check. Source and receipt hashes are in `chain-review.json`; no candidate was changed by this review.

## Full raw-scalar exception domain

Let `n` be the secp256k1 group order, `U=2^256`, and `s` the global recoding sign. The selected default geometry has widths `[18]+[17]*14`, shifts `S_0=0`, `S_c=17c+1` for `c>=1`, and order `0..14`. The runtime table base is the supplied nonzero group point divided by two. Actual `gt_recode_setup` first reduces raw `k` modulo `n` (one subtraction suffices because `U<2n`), computes `t=2k mod n`, and chooses the odd `M=t,s=+1` when t is odd, otherwise `M=n−t,s=−1`. Thus `sM=2k mod n`, including raw zero and raw n where `M=n`.

Immediately before window c, the unsigned odd recurrence remainder is `M_c=(M>>S_c)|1`. The accumulated signed integer prefix is

`L_c=s*(M−M_c*2^S_c)`.

For c>=1 it is odd, nonzero, and `|L_c|<2^S_c`. The next signed term is `T_c=s*d_c*2^S_c`, with odd nonzero `d_c` and `|d_c|<2^w_c`. Consequently `L_c±T_c` cannot be integer zero (odd versus even), and its absolute value is less than `2^(S_c+w_c)`. Through c=13 this is at most `2^239<n`. Hence neither equality nor opposition is possible modulo n in the two-point seed or ordinary additions c=2..13. Scaling by any nonzero runtime base preserves this statement in the prime-order group. It also excludes an earlier infinity prefix. This is an all-uint256 algebraic argument, rather than an inference from random tests.

The last window c=14 starts at bit239 and can cross n. Set `delta=U−n` and `H=U−2^239` (`0xffff800000000000000000000000000000000000000000000000000000000000`). For raw scalar H the actual odd recode is `M=U+delta−2^240`, with positive global sign. The prefix before c14 is `L=delta−2^239`; its final term is `T=H`. Therefore `L−T=−n`: the operands are the same nonzero curve point, and an incomplete add fails. Raw `n−H` reverses the sign and produces `L−T=+n`, giving a second doubling witness. Raw zero and raw n give final opposite operands (`L+T=±n`) and an infinity result.

The complete final helper tests the actual deferred-state differences `P=X2*ZZ−X` and `R=(Y2+Yoff)*ZZZ−Ycore`. Canonical field output makes their word-zero tests valid. `P!=0` uses the ordinary formula; `P=0,R!=0` returns the infinity sentinel; `P=R=0` constructs twice the affine table point. The latter must use table `Y2`, not deferred `Ycore`. The incoming point cannot be infinity by the prefix proof, and the nonzero table point cannot have y=0 in this odd prime-order group. Thus these cases cover the selected final-add domain. No weak representation is introduced in this guard-only source, and no boundary normalization is necessary.

## Executed evidence and correctness matrix

| Component | Evidence | Scope/result |
|---|---|---|
| Actual selected recoder/digits | `guard-chain-check/results.json` | 12,771 raw uint256 cases, boundaries and random values; all15 digits and scalar/sign identity checked;332,046 ordinary equal/opposite checks. |
| Actual chain, sparse signed loader and recovery bodies | Same report |316 scalar/base chains;4,740 actual loader calls;628 recovered x/parity keys against independent OpenSSL points for both recovery signs. Canonical field/inverse operations in this projection use OpenSSL. |
| Final-add regression | `pending217-chain-check/results.json` and `guard-negative-final/results.json` | Unmodified PR217 and compiled guard-removal mutant both fail the H witness after successful compilation; guarded source passes. |
| Selected production M/S host branches | `guard-field-check/results.json` |6,750 multiply and4,500 square executions including aliases against bigint. Actual selected host bodies compiled. |
| Selected device M/S and cold correction | Same field report |2,250 M plus2,250 S actual-PTX semantic cases; source-exact C++ predicates modeled separately.74 carry repairs,140 normalizations,258 prefilter false positives; canonical near-p defects from accepted PR189 are repaired. No CUDA execution. |
| Field negative controls | Same field report | Dropping carry-triggered correction fails canonical `(p−65537)^2`; removing normalization fails the separate full-uint256 `p*1` boundary. These are semantic-model controls, not compiled CUDA mutations. |
| Rejected weak screen | `chain-check/results.json`, `chain-negative-final/results.json`, `chain-negative-zz/results.json` | Preserved independently:316 chains/628 keys with1,264 boundary normalizations; both compiled mutants detected. This is not the selected guard-only implementation. |

The device arithmetic checker verifies the actual raw second fold `S=low+carry*U`, with `S<=U−1+C²`, `C=2^32+977`. A carry implies `low<C²<2^65`, so the top32 bits are zero; no carry with `low>=p` implies top32 bits all ones. The actual cheap prefilter includes both cases. Required correction adds C modulo U and yields a canonical result; false positives return unchanged. The checker models the five-output PTX operand ABI without changing executed instruction strings and uses the actual cold-helper carry chain. Selected function hashes are explicitly equal to PR217, rather than transferring results from the older accepted or our earlier arithmetic implementations.

## Limits and reproduction

This qualifies CPU source projections and bounded arithmetic semantics, not GPU execution, timing, whole-kernel synchronization or the table builder. The algebraic exception proof covers every raw uint256 scalar for the selected default15-window order; executable tests are finite. Zero/order cases check the chain's infinity sentinel, not inherited downstream rare `W=0` recovery handling. Optional nondefault geometry/direct-digit configurations are not newly qualified. Native builds, startup checks and whole-source packaging are separate parent-owned receipts.

Run from the benchmark root:

```sh
python3 -B candidates/subset/research/frontier_rebase_sep17/check_frontier_chain.py --source candidates/subset/research/frontier_rebase_sep17/guard-candidate --output /tmp/frontier-guard-chain
python3 -B candidates/subset/research/frontier_rebase_sep17/check_frontier_chain.py --source candidates/subset/research/frontier_rebase_sep17/guard-candidate --output /tmp/frontier-guard-negative --mutation unguarded-final
python3 -B candidates/subset/research/frontier_rebase_sep17/check_frontier_field.py --source candidates/subset/research/frontier_rebase_sep17/guard-candidate --output /tmp/frontier-guard-field
```

The chain checker requires `research/wide_windows/audit_support.py`; the field checker requires `research/ptx_field_model.py`; both use `preflight.py` for source identity. The checked candidate, PR217 control and rejected weak screen remain unchanged.
