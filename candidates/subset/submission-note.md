# Subset: zero-spill isomorphic recovery + fused root + SHORT_CARRY6 + seven first-fold sites

Effort: xhigh.

## Base and target

This source starts from promoted subset commit `9ac2515450446dbadbe061e98ebfc317c36d4999`, submission `7aef224a-e3ff-43f9-9877-50cdbda3f653`, which scored **623,518,629** verified candidates/s on the RTX 4090 runner. The required 100-bips score is **629,753,816**.

The exact isomorphic-recovery, fused-root, SHORT_CARRY6 and first-fold mechanisms come from public submission `5744a581` / source `95e1792`. This package composes them onto the promoted 9ac chain-loop source, then retains only the first-fold sites that compile without local memory on native sm89. No harness, benchmark, scoring, problem, hit encoding, generated binary or sibling-track file is changed.

## Exact recovery cuts

`QSB_ISO_FAST_X=1` chooses a problem-wide curve isomorphism with transformed recovery abscissa `xR' = +/-1`. It transforms the fixed-base table and recovery point consistently, so the hot per-candidate `xR*ZZ` field multiplication becomes a copy or modular negation. The post-recovery path uses the original problem point and produces the same affine result, compressed pubkey and SHA input.

The transformed preparation numerator and denominator scale by `u^9` and `u^12`. `QSB_ISO_FUSED_ROOT_SCALE=1` starts the active four-lane zinv32 coefficient at `1/u`; the inverse tree therefore returns `(1/u)/root` directly. This removes the separate once-per-CTA root-scale field multiply and its call frame. The fixed-exponent fallback applies the same scale explicitly. Both changes are exact and retain their public kill switches.

Public donor evidence covers 64 problems by 64 valid points: 8,192 recovered compressed outputs and SHA inputs matched the original curve, both transformed signs occurred, and 1,024 transformed group-law cases passed. The fused inverse path matched OpenSSL for 32,768 normal values and 4,096 forced-fallback values. Balanced same-work measurements reported **+0.246794%** for x-isomorphism and **+0.146008%** for fused root scale.

## Speculative carry cuts

`QSB_SHORT_CARRY6=1` shortens four `K=2^32+977` correction tails in each of the thirteen deferred-Y mixed additions: the folds of `Y2+Yoff`, `U2-X1`, `S2-Y1`, and `V-X3`. It keeps limb 0 and omits only a rare carry or borrow into limb 1, removing 52 propagation instructions per candidate. Public balanced fixed-work measurement reported **+0.437561%**. The public composition of the two exact cuts and SHORT_CARRY6 is **+0.832443%**.

The first Solinas fold has two carry chains. Omitting the bit above 256 bits after `r3 + 977*x14 + carry` removes their dependency. The public full nine-hot-site experiment reported **+0.380523%** across two independent N24 problems, with all four adjacent comparisons positive. Its exceptional interval is at most `977*(2^32-1)+1` top-word values per fold. This is nomination-filter arithmetic only.

Composing the complete donor with 9ac made native sm89 spill 16-byte stores and 12-byte loads. Spills are too expensive at this kernel's 128-register occupancy wall. This package therefore applies the first-fold cut only to the six compatible lean sites `f0,f2,f5,f8,f13,f15`; the default `f9` path retains the donor cut. The two pressure-peak sites `f6,f7` keep the promoted exact first fold. Reordering `f8` and `f15` to evaluate the even fold before the odd fold makes the seven-site composition compile with **zero stack and zero spills**.

Scaling only the measured first-fold component by the retained 7/9 hot sites and composing it with the independently measured +0.832443% donor factor gives a mechanical projection of **+1.130869%**, or **630,569,808/s** on the 623,518,629 base. That is **815,992/s above** the current one-percent floor. Site scaling is a model, not a ranked measurement; the official run decides.

## Correctness boundary

The isomorphism and inverse-root changes are exact. The two carry mechanisms are confined to the speculative filter and can very rarely discard a true nomination. They cannot publish a false record: `kernel_verify_pair_hits -> qsb_pair_verify_candidate` replays every nomination through the separate unchanged exact `chain_replay_field.cuh` arithmetic before output. The public donor fixed-work arms produced identical normalized hit sets for their measured domains; finite tests do not prove universal recall.

The promoted 9ac chain schedule, 128-window family, parity window, epoch producer, table geometry, exact verifier and publication path are otherwise unchanged. `QSB_CHAIN_MUL_LEAN=1` remains the promoted default; enabling its second square (`=2`) is intentionally excluded because it spills.

## Build and checks

Native build used CUDA 12.6, `nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24 -Xptxas=-v`. The ranked `kernel_digest` uses **128 registers, 49,152 bytes shared memory, one barrier, 0-byte stack, 0-byte spill stores and 0-byte spill loads**. Exact verification uses 136 registers and zero spills. Epoch producers use 47-48 registers and zero spills.

Host/source gate:

```
python3 candidates/subset/test_leader_composite.py
python3 -m py_compile candidates/subset/test_leader_composite.py
git diff --check
```

The gate checks all four default switches, the exact six lean first-fold sites, the retained f9 donor site, exact f6/f7, exact replay/publication symbols, conflict markers, and the score arithmetic. It reports projection 630,569,808 versus floor 629,753,815. `SOURCE-MANIFEST.json` records the base, donor, selected sites and SHA-256 of every changed production file.

Production hashes:

```
80625e852184ce99deee157980e5f44d4bdb758871ff30b5759941e5328c430f  hit_filter_field_sc.cuh
a1f5040c6f2ca7eab6e1719c0283ba2fca4849fffe87bbf1bea20cb164b2ee2f  tests/gpu_epochs/hm43_warp_inverse.cuh
2c86f5ee0ede0e5bc3f847cf7f2493512612e82329c6a46592347ea4877a0e00  tests/gpu_epochs/pair_shared.cuh
1ecf8fc4c44d0c2e0203ae625d18d87e67f26fd36725aee7e7a7028c3226cd56  tests/gpu_epochs/tree.cu
9e178ed45bde47e7d460ed74695e77b394b08fd34926834bd7f10daedc55b2ab  tests/gpu_epochs/tree_inverse.cuh
7e9c1ef26df306b7158c6db9edb4a1444e7f2e3bbb43a9c0cf58c96f4775c163  tests/gpu_epochs/zinv32.cuh
```

## Attribution

Promoted base: Akashneelesh. Public donor submission and composition: terrapinelf. Its retained lineage credits dun999, fkiene, Meganpark980320, ercumentyildirim and EvanYan1024 for the negfold/carry, epoch/window, parity and recovery mechanisms. This zero-spill merge, site selection, PTX instruction ordering, native compile census and packaging were performed with GPT 5.6 Sol using Codex. All inherited license and source notices remain.
