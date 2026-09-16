# Subset work

Keep changes in `candidates/subset/`. Prioritize large architectural gains.
Read `RESEARCH.md` and the latest public frontier before selecting another
experiment. Attribute substantial unpromoted source with Yukon coauthors.

Run `python3 candidates/subset/check_candidate.py` after changes to the source
it extracts. Treat its result as a CPU reference check, never as CUDA execution
or a GPU benchmark. Keep the included GPU arithmetic audit available. Record
returned validation failures and improve the checks that missed them.

Preserve the trusted harness, scoring contract, runtime problem dependence,
GPL notices and sibling pinning work. Keep source hashes and result provenance
in the local research ledger. The recurring result monitor does not authorize
automatic code changes or new submissions.
