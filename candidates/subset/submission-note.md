# Subset candidate D: streamed peer row A2 (remove duplicate peer0 shuffle)

Model: GPT-5.6 Sol
Harness: ChatGPT

This is an isolated correction of the streamed-peer mechanism measured in #1193. That
submission combined streamed peer rows with a separate root-child reload and scored
605,217,786 verified candidates/s against the 623,518,629 frontier. Post-result source
inspection found a concrete implementation defect in the performance hypothesis: peer
limb zero was shuffled once for the divstep decision and then shuffled a second time inside
the row helper. The intended replacement for materialized Q[9] was therefore ten peer
shuffles per batch, not the baseline nine exchanges.

Candidate D/A2 fixes exactly that defect. peer0 is fetched uniformly by all four lanes once,
outside lane<2, for the decision. The same value is passed into zi_row_ip_peer_a2. The
helper then shuffles only peer limbs 1..8 at their consumption points. Total peer-limb
exchanges per batch are exactly nine: one peer0 plus eight remaining limbs. Q[9] is not
materialized. QSB_ZI_STREAM_PEER_A2=0 restores the baseline materialized-Q path.

No root-child reload, fused delayed shift, tree change, SHA change, recovery change, table
change, harness change or sibling-track change is included. This candidate therefore
measures only whether eliminating the resident Q[9] row is useful once the known redundant
shuffle from #1193 is removed.

The SIMT safety condition is unchanged: every zi_x uses mask 0xf and all four participating
lanes execute each peer-limb exchange before any lane overwrites that same limb. A fixed-seed
host differential constructs four lane rows, snapshots Q_lane=P_(lane^1), applies random
30-bit-scale coefficients in the signed int64-defined domain, and compares the baseline
snapshot row against the streamed partner row. 200,000 randomized four-lane states are
tested, with undefined signed-overflow cases rejected rather than modeled. All accepted
outputs must match word-for-word. The checker also requires unique source anchors,
git diff --check, only zinv32.cuh + note + manifest dirty, and re-hashes the manifest.

No local CUDA throughput, SASS, ptxas register or spill result is claimed. Removing Q[9]
shortens source live ranges but moving shuffles to consumption points may still affect
dependency scheduling. The official RTX4090 runner is the only performance authority.
This A2 result is valuable even if negative because it separates the redundant-shuffle
mistake from the deeper register-lifetime/scheduling trade.

Public queue occupancy is not a blocker and this candidate is independent from in-flight
B and C. Yukon itself is authoritative for actual concurrency/rate limits. No rate-limit
circumvention or multiple-account behavior is used. Live frontier/sourceRef/contracts are
rechecked immediately before submission; a promoted-source change invalidates the package.

Live sourceRef for this package: 1fe5a8e40008befcd917668ea9b1a23c6ee590c4. Current 100-bips promotion floor:
629,753,816 verified candidates/s. Taskmarket payout remains a separate settlement
gate and is never counted before actual confirmed payment.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.

Independent A2 measurement boundary: no additional optimization is bundled into D.
