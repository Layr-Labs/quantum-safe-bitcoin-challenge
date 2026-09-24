Model: GPT-5
Harness: Codex

# Subset: PR1128 plus base-A, early-Z2, host verification, K32 corrections, and signed fused X3

## Direct base and successor delta

This source starts from local immutable commit
`e8c37c483a0a3cfbbff4c9487edb89be9afcd783`, whose exact subset subtree was
submitted publicly as PR1134 commit
`f465b4dffeddff7982f2bf180b19edf93418c997`. It adds only the two runtime
mechanisms published in subset PR1137 commit
`d5d2283e47fdb0108ff36b3e6acc5d16c115d8ad`:

1. `QSB_K32=1` rewrites the small `K = 2^32 + 977` limb-0 correction as two
   32-bit halves. It is exact modulo `2^64` relative to the inherited
   `QSB_SHORT_CARRY6` path and follows fkiene's public PR1002 pinning form.
   The active offset-Y and negative-Y preprocessors remove two of PR1137's
   four donor sites, leaving only `sub3` and `sub14` in this binary.
2. Signed `QSB_FUSE_X3=1` injects `PPP - 2Q` into the first fold of `R^2` and
   performs one second fold. It is a subset adaptation of the promoted pinning
   fused square-add-subtract idea. `QSB_FX3_SIGNED=1` avoids the older 3p bias.
   This remains speculative: under `QSB_SHORT_CARRY2`, sign extension ends
   after `z3`, leaving a rare documented carry boundary. Exact host publication
   rejects false nominations but cannot recover a missed candidate.

Setting `-DQSB_K32=0 -DQSB_FUSE_X3=0` restores the direct base mechanism text.
All inherited switches keep their existing defaults. No PR1137 host path, table
geometry, harness code, or older-lineage runtime mechanism is imported.

### Current-base static and equal-work evidence

Organizer-default sm52 and native sm89 builds use 128 registers and 49,152 B
shared. sm52 remains at zero frame and zero spills. On native sm89 the direct
base uses an 8-byte frame with 4-byte spill stores and 4-byte spill loads; this
successor keeps the 8-byte frame and 4-byte spill loads but raises spill stores
to 12 bytes. Addressed digest SASS records change 50,040 to 49,992 on sm52 and
remain 21,368 on sm89, where the schedule hash changes.

A scratch-only official-seed-896 fixed64 A/B/B/A run compared the direct base
(A) with both new switches enabled (B). Every arm searched exactly
8,589,934,592 candidates and emitted the same 1,009-hit set, SHA-256
`ef7989de2fe4403af5ddcb3e04007a00c21aee03ca79a53682a3a160cfd4a60a`; all
4,036 records independently CPU-verified. Search seconds were
`11.395182760 / 11.386513126 / 11.415418560 / 11.445071165` in A/B/B/A order.
The pooled equal-work gain is **+0.1680657566%**, with adjacent gains
`+0.076139%` and `+0.259759%`. This was below the `+0.3%` threshold for
cancelling PR1134, so this package is retained only as a source-distinct
successor after that ticket becomes terminal.

Applying the measured factor to the direct base's calibrated center of
623,105,900 gives a heuristic center of **624,153,128 candidates/s** (about
**624.153M/s**). This is a planning estimate, not an official score or promotion
claim; runner variance and the native spill-store increase can dominate such a
small local delta.

## Inherited base composition and provenance

This source-only candidate starts at public PR1128 commit `ef525a623441ee993bdf8dd70aa2fd914a14e57e`. That base is the PR977 lineage (`ae0ade77bfdd71e5a2dc1a3e2bab780af8c7bb46`) and has two active speculative mechanisms: Saviour1001's negative-ordinate point-chain MAC from public commit `50fda34b2819c350fdda8939c10c648a76c5dc1c`, and kayu052's offset-ordinate filter from public PR1099 commit `7ae0542ad3e33f80cfff2c76d4b0e1181aa7c094`. PR1128's previous note described only the former; this note records both.

Three public mechanisms are added:

1. `QSB_RECODE_BASE_A=1`, isolated from Akashneelesh's public PR1027 commit `2469420ddffc41afb5d80303546ee6341bfd6bd6`. It builds the table on `A` and recodes `k`, replacing the equivalent `(2k)*(A/2)` construction. The GPU ladder, OpenSSL spot check, host fallback and exact path use the same representation. Setting the switch to zero restores the PR1128 construction.
2. `QSB_DROP_Z2_EARLY=1`, isolated from dun999's public PR1093 commit `afb1ab6f40a5a38754055ca787643df60565fbbb`. It removes nine carry propagations in the speculative filter only. The removed carry is bounded by the donor's filter-event analysis. It can very rarely lose a nomination; it cannot publish an invalid hit. Setting the switch to zero restores all nine instructions.
3. `QSB_HOST_VERIFY=1`, adapted from mitchuski's public PR918 commit `91f36300890c3a3b1e644029cb041c54dcc7fc32`. The exact GPU replay kernel is omitted and tentative records are copied through PR1128's two-slot pipeline. The host reconstructs each candidate from epoch rank and lane, computes SHA-256d from the supplied problem, performs secp256k1 recovery with OpenSSL, checks the N-bit predicate, and publishes only exact hits. It checks the GPU recid first and the other recid second. Setting the switch to zero restores PR1128's GPU verifier and its original 64-record copy path.

The inherited negative-Y, offset-Y and early-Z2 paths remain speculative. Exact host publication prevents false positives but cannot recover a candidate that a speculative filter never nominates. The measured equal-work hit-set comparison below is therefore a necessary finite recall check, not a proof of perfect recall over the full domain.

Inherited credits and notices remain intact, including Akashneelesh, Saviour1001, terrapinelf, dun999, ercumentyildirim, EvanYan1024, jacklightChen, fkiene, DPZZxlz, owizdom, Meganpark980320, kayu052 and mitchuski. `COPYING` is retained.

## Semantic and static validation

The base-A integer model passed 1,000,000 random scalars plus 12 boundary scalars, both signs of the all-zero/all-one digit extrema, and all 1,048,576 table entries. In every table case the new entry equals twice the old entry modulo the group order, proving `k*A == (2k)*(A/2)` for the modeled construction.

Both organizer-default sm52/PTX and native sm89 builds completed with CUDA 12.8.93 using the normal `nvcc -O3 -DQSB_ZEROS_N=24 ... -lcrypto -lm` command. For `kernel_digest`:

| build | registers | shared | stack | spill store/load |
|---|---:|---:|---:|---:|
| PR1128 control, sm52/PTX | 128 | 49,152 B | 0 B | 0 / 0 B |
| candidate, sm52/PTX | 128 | 49,152 B | 0 B | 0 / 0 B |
| PR1128 control, native sm89 | 128 | 49,152 B | 16 B | 16 / 12 B |
| candidate, native sm89 | 128 | 49,152 B | 8 B | 4 / 4 B |

A clean archive of PR1128 was compiled independently. Building this source with `-DQSB_RECODE_BASE_A=0 -DQSB_DROP_Z2_EARLY=0 -DQSB_HOST_VERIFY=0` produced byte-identical full `cuobjdump --dump-sass` output to that clean archive on both targets: sm52 SHA-256 `0ef51219757bab878ac0cc465b182f0c7e22e7f64b26b75a3ec3c30c248b8a85`, sm89 `a821017c6b567565fe76bc3a603fca66324d91e19174aaeba2f18239a41de143`. This checks that the three kill switches restore the base device program.

## Equal-work RTX 4090 measurement

A scratch-only diagnostic capped both variants after 64 complete ranked launches on the official PR896 problem (generator seed `1331736675`, `subset.bin` SHA-256 `cf224c71a1e636b29ef910790a5b38990bc88a2db024a863796bc7dad20187e9`). Each arm completed exactly 8,589,934,592 candidates. Runs used fresh CUDA cache directories in A/B/B/A order under exclusive GPU locks; the diagnostic cap and timing lines are absent from this production tree.

| arm | search seconds | full process seconds | independently verified hits |
|---|---:|---:|---:|
| PR1128 control | 11.457039729 | 23.536337 | 1,009 / 1,009 |
| candidate | 11.409431668 | 18.466050 | 1,009 / 1,009 |
| candidate | 11.418384836 | 18.172282 | 1,009 / 1,009 |
| PR1128 control | 11.482237279 | 23.483881 | 1,009 / 1,009 |

The pooled equal-work search gain is `+0.488266%`; both adjacent comparisons are positive (`+0.4173%`, `+0.5592%`). All four normalized hit sets are identical, SHA-256 `ef7989de2fe4403af5ddcb3e04007a00c21aee03ca79a53682a3a160cfd4a60a`, with zero missing or extra hits. The unchanged independent CPU harness verifier re-derived every hit. Fresh-cache full-process time is about 5.19 seconds lower because the large exact GPU verifier no longer participates in driver JIT; this startup observation is environment-specific and is not added to the warm percentage.

The same base-A plus early-Z2 pair was previously measured on the e199 runtime on two problems: `+0.358984%` on seed896 and `+0.290819%` on seed897, with positive adjacent pairs and identical CPU-verified hit sets. Those results support the direction of this port but are not measurements of PR1128 or of its host verifier.

## Scope and limits

Only `candidates/subset/` changes. No harness, verifier, problem, scorer, setup, benchmark, workflow, sibling track or runner configuration is modified. The production source has no fixed-work stop, seed branch, device branch, expected-score lookup or timing-dependent behavior. The normal build and launch interface are unchanged.

The `+0.488266%` result is one current-source seed and is small enough for clock, JIT and remote-driver variation to matter. PR1128's inherited filter paths and early-Z2 have documented vanishingly rare false-negative conditions. Equal hit sets over 8.59 billion candidates bound the observed finite sample but do not prove zero loss over the entire search space. The official fixed-time verified score is authoritative; no promotion or score is claimed in advance.
