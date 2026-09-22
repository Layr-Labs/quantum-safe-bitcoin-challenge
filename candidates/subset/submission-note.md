Model: GPT-6
Harness: Codex

# Subset: PR977 ranked pipeline with a negative-ordinate point-chain MAC

## Source and attribution

This source-only candidate for the subset track starts from public PR977 validation source ae0ade77bfdd71e5a2dc1a3e2bab780af8c7bb46 and adds one speculative point-chain arithmetic mechanism. PR977 was based on public Saviour1001 submission 35c4db43, and retains its isomorphic recovery, SHORT_CARRY6, seven-site first-fold top-carry cut, two-slot host pipeline, startup trim, and SHA constant folding. The current promoted subset source at preparation is 9ac2515, official best 623,518,629 verified candidates per second. PR977's own official 622,587,731 result is below that best; this candidate is an actual runtime change, not a redraw of PR977.

The negative-ordinate idea and initial implementation come from Saviour1001's public d555fd1e submission, commit 50fda34b2819c350fdda8939c10c648a76c5dc1c. Its split SHA producer is not included: only the negative-Y field mechanism was ported. The port and independent local measurement were done with GPT-6 in Codex. Credit for the inherited PR977 mechanisms remains with Saviour1001, terrapinelf, dun999, Meganpark980320, ercumentyildirim, EvanYan1024, jacklightChen, owizdom, DPZZxlz, fkiene, and original source notices. All license and attribution notices in the runtime source remain in place.

## Runtime change

The speculative secp256k1 XYZZ chain previously carried a positive ordinate core Ycore, with actual ordinate Ycore - Yoff*ZZZ. With QSB_SUBSET_NEG_Y_MAC=1, it carries N=-Ycore, so the same actual ordinate is -N - Yoff*ZZZ. The point addition needs R=(Y2+Yoff)*ZZZ - Ycore; with N, that is R=(Y2+Yoff)*ZZZ + N. In the inlined PTX, the four 64-bit limbs of N are folded into the initial columns of the f2 multiplication before reduction. A fifth 64-bit carry f2_bias is included in the next column. This moves the addition inside the multiply's existing wide-accumulator and modular-reduction path, removing a separate field subtraction after the multiplication.

The sign also reverses the later V-X3 ordinate term to X3-V wherever required. The seed path and non-inlined device fallback use the same coordinate convention. At qsb_filter_last_add, the candidate restores the ordinary positive ordinate exactly once before the inherited parity/recovery tail. Every ranked filter-chain branch reaches this wrapper. Published hit encoding, exact replay kernel, final scalar check, problem parser, table geometry, host scheduling and submission interface are unchanged. The QSB_SUBSET_NEG_Y_MAC=0 control selects previous PR977 arithmetic. The change is confined to hit_filter_field_sc.cuh and tests/gpu_epochs/tree.cu; all other runtime files are inherited from PR977.

This is speculative front-end arithmetic. Its exact arithmetic semantics are checked by differential tests, and every emitted hit still passes through the inherited exact verifier. The verifier alone would not reveal a false negative, so matched prefix hit sets are also compared. We do not infer correctness merely from a count of verified published hits.

## Build and local correctness evidence

The production binary was built with CUDA 12.8.93 and the organizer-style no-architecture command:

    nvcc -O3 -DQSB_ZEROS_N=24 -o subset candidates/subset/subset.cu -lcrypto -lm

This emits an sm_52 fatbin and PTX, which GPU2's RTX 4090 driver JITs. An sm_89 build was used only for static resource inspection, not for speed comparison with a no-architecture base. The raw-PTX audit extracted the merged hit_filter_field_sc.cuh, not only the donor. It compared QSB_SUBSET_NEG_Y_MAC=0/1 multiply-accumulator forms with both lean-carry settings over 8,272 directed and random raw MAC cases per variant. It then exercised 1,024 random curve points and 256 fifteen-addend chains: zero mismatches. A separate source-extraction audit used the merged tree.cu last-add wrapper and an OpenSSL point reference over 2,048 source chains and 1,024 OpenSSL points: zero mismatches. The audits include inherited SHORT_CARRY6=1. Local command/output paths are in the experiment ledger; neither test substitutes for the organizer verifier.

On GPU2, a direct PR977 versus this-candidate screen used the same problem, seed, N=24 and no-architecture binaries, one process at a time. At the 61-second checkpoint, both had completed 44,023,414,784 attempts and had the same 5,305 hit lines for that common work. The candidate's cumulative checkpoint rate was 725.6M/s versus 725.9M/s for PR977. A warmed four-arm promoted/composite/composite/promoted comparison on the same problem, device and 65-second timer gave 60-61-second rates 721.0/727.2/725.0/717.2 M/s respectively. The bracket means are 719.1 and 726.1 M/s, a +0.97% local rate advantage before hit-yield adjustment. Terminal attempts can differ by a batch or two because SIGTERM can arrive between drains; the matched checkpoint and common-prefix data are more informative than an isolated terminal total.

The promoted source published one valid hit that both PR977 and this candidate missed in the two paired common prefixes: indices 0,17,18,48,51,136,137,145,149, recid 0, early epoch rank 169,508,697. At prefix ends 363,759,330 and 361,681,985, promoted versus composite hit counts were 5,601/5,600 and 5,565/5,564, with zero composite-only hits. PR977 without negative-Y also missed exactly that hit, so the observed false negative is inherited from the PR977 lineage. It is a measured roughly 0.018% yield penalty on this sample, not an exact-recall claim. The official score counts verified hits and therefore incorporates such missed nominations. Local evidence records the miss in subset-pr977-inherited-miss.json for follow-up. GPU2 uses an RTX 4090 with driver 595, while the official runner is reported to use driver 580. JIT differences and score noise limit transfer from local diagnostic to official score.

An independent 120-second local harness run of the exact no-architecture source used seed 1149987937 and N=24. The harness verifier accepted all 10,343 published hits; its hit-derived score was 721,904,407 verified candidates per second. The grinder separately reported 733.2M candidate attempts per second, which is not the scored rate. The harness report recorded a 0.009833 hit-relative variance under its 0.1 limit. This one-seed local PASS is a correctness and scale check, not a promise about the official runner. Its problem file and outputs are retained under the unique GPU2 directory /tmp/harness-pr977negy-1790089824.

## Evaluation limits

The official score is based on verified candidates in the server's scored interval, not on a locally printed peak rate. The exact +1% promotion floor over 623,518,629 is 629,753,816. The local performance advantage of the PR977 lineage may be less or more in the official driver, clock and power environment. If submitted, this is an honest experiment on the new composite, not a claim of guaranteed promotion. An official rejection would still be useful evidence about this exact source, and the source and tests remain available for further changes.

Only candidates/subset files are changed. No sibling track, harness, benchmark, scoring, setup, or workflow source is changed. Yukon notes, trace, and telemetry remain disabled throughout this local work.
