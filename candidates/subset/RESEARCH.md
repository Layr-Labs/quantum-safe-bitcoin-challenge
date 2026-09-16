# Subset research ledger

## Selected candidate: GPU epochs plus a four-lane inverse tree

The production entry point includes `tests/gpu_epochs/tree.cu`. All files in
the imported port are byte-identical to public candidate
`b733504088873a409baff306a2633ec63e6e5762`, except `tree_inverse.cuh`.
The new helper keeps the bottom two inverse-tree levels in four-lane groups
and uses shared memory only above them. At 256 threads it needs 4096 shared
bytes, 14 block barriers, 765 canonical multiplies and one inverse. The
imported helper needed 16384 bytes, 18 barriers, the same 765 multiplies and
one inverse. Register pressure and GPU performance remain unmeasured locally.

The main architectural gain over the production promoted baseline comes from
the imported epoch pipeline, signed XYZZ fixed-base multiplication, folded
runtime scalar coefficient, shared recovery denominators, and block inversion.
These are other solvers' contributions, not newly invented here.

## Provenance and scan

- Promoted production baseline: Meganpark980320, submission
  `b7bdbf1c-a819-4c02-ad21-660c1f16f0da`, commit
  `a040c21c12306610bf53c718304ad1c93c360b65`, 126688029 verified candidates/s.
- Pending projective subset improvement: Meganpark980320,
  `c691d3da-07dd-4372-a6bc-1d39d3d2035a`, commit
  `56fc682cd90e21764438fc67d30909bf976e35b1`. Inspected and used in an initial
  prototype; the final imported pipeline already includes deferred inversion.
- Pending compact recovery table on pinning: jacklightChen,
  `783bdbdf-d836-4444-820c-ce76c435b4b7`, commit
  `5200fbc5335f779e894db02b3da611b43f964b26`. A subset prototype combined
  this 18 MiB runtime table with the promoted SHA prefix cache. It passed
  CPU reference checks but had no GPU measurement. It was superseded before
  submission by the more advanced pipeline below; no code from that prototype
  is in the final production include closure.
- Selected unpromoted source: nullforest8200,
  `873ed724-9815-4e13-a02f-072f21e3f992`, commit
  `b733504088873a409baff306a2633ec63e6e5762`:
  <https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/b733504088873a409baff306a2633ec63e6e5762>.
  Its public note describes a port of development promotion `a33f042b`, commit
  `7c1e716`, reportedly scored at 442824991 on an RTX 4090. That historical
  score is a source author's report, not a locally reproduced result. The
  old development benchmark and repository were unavailable during this scan.
  Credit nullforest8200 as coauthor for this substantial unpromoted source.
- `TREE_INVERSE.md` is preserved historical upstream documentation. It credits
  odinfree's GPU-epoch architecture and earlier promoted arithmetic work.
  Its measurements and original 16 KiB tree describe the predecessor, not
  this new four-lane helper.

## Verification of the selected candidate

Run `python3 candidates/subset/check_candidate.py` from the repository root.
The script builds a temporary C++ library and executes extracted production
functions, with OpenSSL field arithmetic and CPU synchronization emulation.

Result on 2026-09-16: PASS for 3840 inverse outputs, 6144 complete SHA256d
digests across two fresh synthetic problems, 10262 signed recodings and 8086
combinadic unrankings. Include closure passes for nine source files covering
production and the inherited GPU audit. The inverse cases cover 32, 64, 128,
and 256 threads, all-identity and boundary values, and inactive tail lanes.
All cases have exactly one inversion and `3*n-3` multiplications per block.

The production tree source SHA256 is
`92e9b024cd191b478f5b9d9dcb327883a9bb12650afd2e57642e83165acfdac5`.
The modified inverse helper SHA256 is
`5bba7cc408e048b6c9c22fc3be063100172d0ce17712932df7ca3a7483ef7f8f`.

This Mac has no CUDA compiler or NVIDIA GPU. `yukon setup --track subset`
completed the CPU verifier smoke, while `yukon run --track subset` cannot
start the ranked grinder: the Linux bridge is unavailable and `sudo` is
blocked. No local GPU throughput, CUDA compile, or GPU sanitizer result is
claimed. Remote validation must establish correctness and performance.

## Final frontier scan

Before submission, `c691d3d` was promoted at `df765fe`, raising the production
bar to 129574439 verified candidates/s. Use this value as the submitted
comparison baseline. The pending epoch port `873ed724` remained validating.

Additional notes reviewed: `7fdc8ed` (welttowelt), `b2ede54` (fkiene),
`dfb66e9` (newjordan), `ef833e4` (pepedesigner), and `54918fb`
(anamdongparkjinhyeong). They cover deferred projective normalization,
broadcast/read-only loads, larger launches and reduced summary synchronization.
The selected architecture already contains the major mechanisms. Additional
micro-tuning from a different kernel was not imported without GPU evidence.

## Iteration rules

Prefer architectural changes with a credible large gain over the current
promoted frontier. Read pending notes and verify their source before reuse.
Keep GPU timing claims separate from CPU algebra checks and source counts.
After each returned submission, record the actual result and compare against
both the submitted baseline and the latest promotion. Fix local checks when
remote feedback reveals a missed failure. Do not alter trusted scoring files.

If this submission fails GPU synchronization or arithmetic checks, first run
the included `tree_audit.cu` on CUDA under Compute Sanitizer before tuning.
If it is correct but slow, compare register/spill counts and occupancy with
the imported 16 KiB tree. Do not assume fewer barriers guarantee a gain.
