# Pinning: existing chain-end ALU routing on the last submitted fused-tail candidate

Model: GPT-6 Astra
Harness: Codex

Status: **experimental 4090 evaluation candidate; no demonstrated 4090 speedup; not submitted.**

## Concrete change and attribution

The control is our submission 8f37bc98-b581-4f28-ab0b-7b55f759f03a, PR 2169, based on promoted main 8d07d3ebad41a017dfaa5906b164f883a9b59348. This candidate enables the already implemented QSB_CHAIN_ALU option by defining it to 1 in pinning.cu, then regenerates the native sm_89 carrier with CUDA 12.8.93.

The mechanism and GPL implementation are inherited from the public fkiene / ItlaStudent lineage, also used in i34-9's PR 2110. We claim no invention of that mechanism. All existing copyright notices and licenses are retained.

Terminal carry adds/subtracts receive a zero from constant memory so ptxas can route them to IADD3 instead of IMAD. The constant is already initialized to zero. This preserves bits and does not remove arithmetic carries, change the existing rare-carry policy, or weaken the exact host publication gate.

The prior first-tail SHA interleave stays. The unconfirmed second-SHA experiment, PIPE_LEA, root-store edits and register-allocation option are **not** included. Official GLV11 table, slots=3, ring=6 and 116/20/shared8 SM configuration remain unchanged. No harness, scoring, problem, specification, setup, benchmark script or Actions change is included.

## Why evaluate separately on 4090?

Static official prepare instructions: IMAD 2871→2841, IADD3 1790→1844; 128 registers and zero spills remain. The inner backward-branch region has IMAD 631→624 and 982→983 total instructions, with no local-memory accesses.

This suggests a possible reduction in multiply-pipe pressure, but additional ALU work and scheduling can offset it. Static counts do not establish improved throughput, and we have no representative 4090 run or performance-counter proof.

The 4060 Ti uses a compact table and different generated code: enabling the flag creates 8-byte prepare spills that do not exist in the official binary. Its negative result therefore cannot by itself establish a 4090 slowdown.

## Validation

- Both official and local native images rebuilt; both host binaries compiled through the unchanged harness wrapper.
- 1,048,576 boundary/random input tuples per arm and 7,340,032 field-operation outputs per arm have identical control/candidate output hashes.
- 40,960 independent OpenSSL BN random-operation checks per arm pass.
- Boundary comparison tests preservation of inherited behavior, not universal mathematical exactness of the inherited approximate field routines.
- Two full-work paired GPU comparisons, six full sequences per run: 3,454 hit records pass the original CPU verifier with exactly matching paired hit sets.
- Local performance is negative: −0.5105% and −2.1949%, pooled −1.3463%. This is a diagnostic completed-work comparison, not an official score.
- The unchanged timed harness at N=24, 25 seconds, fresh seed 1769236105 and without diagnostic/CPU-disable environment overrides passed 565/565 records. Its short-run score is not used as A/B performance evidence. Raw integration result: benchmark-results/chain-alu-latest-20260928/harness/run.json.

The unchanged best snapshot is 1,008,206,828. The prior official score was 1,009,707,243 with 144,640 verified hits. The configured 1% floor is at least 1,018,288,897. This candidate makes no claim to reach it.

## Reproduction and limitations

Rebuild the official native carrier with build_carrier.sh 24 under CUDA 12.8.93, then use the repository's unchanged pinning harness. The official table needs a 24 GB GPU; it was not timed on the local 8 GB card. Finite-work hooks are confined to separate local diagnostics and are absent from this official source.

The package preserves inherited research documents for provenance; their historical performance claims are not measurements of this candidate. SOURCE-MANIFEST.json is inherited historical metadata. CANDIDATE-MANIFEST.json records this package's actual contents.

No public commit, PR, Yukon submission or runner allocation has been made. A new public evaluation requires the owner's explicit approval.
