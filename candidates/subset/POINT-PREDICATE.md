# Point-local predicate accumulation with exact chain replay

Parent: `66b4bfc8d20831e767fd33c066f78c833a3d8588` (chainreplay-v2). This prototype combines the ordinary deferred-Y point addition into one inline PTX statement. Each of its 11 add/multiply/square boundary checks updates one explicitly declared predicate. The predicate is materialized once at the end of the point and ORed with the incoming integer flag. The saved scalar, trial chain, complete last addition and entire exact replay remain unchanged.

The source preserves the literal hot PTX of the canonical field primitives, with local registers renamed and operands substituted. Subtraction preserves the parent's borrow/add-back operation. All 29 inline-asm constraints are checked, including 17 read/write operands. No predicate is implicitly shared between separate asm statements. All inherited source attribution remains intact.

The correctness argument is compositional: the new point helper produces the same four raw coordinates and exact sticky flag as the old helper. The existing chain wrapper therefore retains its first-divergence/replay behavior. Boundary cases trigger the original complete chain rather than accepting a rare arithmetic error. Tests below support this source argument; they are not an exhaustive domain proof.

CPU checks completed:

- 15,060 original field operations and 75,300 sticky-state checks, including cases that are wrong without replay.
- 1,432 point inputs, covering 408 natural guard cases and 1,024 clear cases, each with four distinct prior flags: 5,728 literal PTX comparisons against separate parent operations.
- An independent modular-integer point-formula oracle agrees in all 1,024 clear canonical-input cases.

The historical CPU model is unchanged. Two explicitly documented model-only opcode lowerings implement the 32-bit wrapping add and comparison/OR; additional model-only moves initialize read/write operands. Production asm and constraints are separately extracted and hash-bound. GPU differential and OpenSSL whole-chain audits are still required.

Native CUDA13 sm89 diagnostics: 128 registers, no stack or spills, 32 KiB shared, 19,192 digest instructions. Total static instructions equal chainreplay-v2; 11 comparison instructions change form, 10 SELs disappear and 10 LOP3 instructions appear. This is not a runtime improvement claim. Both production and differential-audit sources compile locally. Official CUDA12.8.93 default builds and same-host performance remain pending.

Evidence: `point-predicate-source.json`, `point-predicate-cpu-audit.json`, `point-predicate-point-cpu-audit.json`, and `checks/chain-replay-census.json`. No submission qualification.
