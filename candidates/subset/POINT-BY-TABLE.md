# Combined point predicate and bounded table inverse

This branch takes pointpredicate `2c66cba75487f7757e105590d22089d30dd7f0b8` and replaces only its inverse header with the exact bounded BY-table header from `c43d80048ab1e2521367f0e98472d31f3fe26b46`. Both component headers and the production tree are hash-bound in `point-by-table-source.json`. All source attribution is retained, including ercumentyildirim's PR296 table/decision function.

The point component preserves exact whole-chain replay. The inverse component retains the 32-batch limit, wide signed top accumulator and independent fallback. The component CPU checks are reusable because their exact source bytes are unchanged; actual GPU checks and timing of the combination remain pending. No component timing is added arithmetically to predict a combined gain.

Native CUDA13 sm89 compilation succeeds with 128 registers, no stack/spills and 32 KiB shared. This is a compile diagnostic, not a performance result. The combined source requires both `build_by_table.py` and `build_chain_replay.py --point-predicate` device audits before any matched speed screen. Final qualification still requires >=1.5% over the strongest contemporaneous public implementation for 1200 seconds each.
