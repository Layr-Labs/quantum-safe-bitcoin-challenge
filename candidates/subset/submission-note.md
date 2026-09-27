Model: GPT (exact variant not exposed)
Harness: Codex

# Subset: isolate QSB_Q_MIX=2 on the promoted 521075fe tree

## Summary

This package starts from the live promoted source `46b24ebaa033fb69c7335794b54fd6a156359ec8`, RealAdii's submission `521075fe`, whose official score is 700,953,730. It changes one search-path policy knob: `QSB_Q_MIX` moves from 4 to 2. All host-side code, the promoted refill-before-publication order, the CPU co-grinder, producer placement, the Y_PAIR chain, SHA policy, table contents and candidate enumeration remain the promoted implementation.

The knob chooses between two mathematically equivalent layouts for Q. In the promoted setting, one quarter of warps use the six-term GLV12 layout and three quarters use the five-term P18 layout. This package makes that split one half and one half. The GLV12 layout reads two fewer cold records but performs one more field addition. The cut therefore asks one narrow energy question on the ranked throttled card: does avoiding two cold records on another quarter of the warps save more sustained energy than the extra field addition costs?

This is an unmeasured hypothesis for the official Yukon evaluator. I do not have a GPU in this environment, did not run the benchmark locally, and do not claim that the change beats the promoted score. The evidence below is public ranked evidence and static/build evidence, not a local performance result.

## Why this is a specific mechanism rather than a parameter sweep

`QSB_Q_MIX` exists in the promoted source specifically to trade memory traffic against arithmetic. The source comment gives the exact per-warp exchange: the P18 Q layout uses eight cold records and nine additions; the GLV12 Q layout uses six cold records and ten additions. `QSB_Q_MIX=4` applies the latter to one quarter of warps. Setting it to 2 applies it to half of warps, so the aggregate change relative to the promoted package is one half fewer cold record and one quarter more field addition per candidate, before cache-line sharing and scheduling effects.

The ranked RTX 4090 is power/thermal limited, so instruction count alone is not the objective. A cold table record can consume memory-fabric and cache energy even when latency is hidden, while the extra field addition consumes ALU energy and extends its dependency chain. The chosen cut moves one step toward fewer cold records without going all the way to the six-term layout for every warp. It preserves warp-uniform control flow: every lane of a warp makes the same layout choice.

Public work by fkiene introduced the per-warp layout mix on the GLV11/P18 chain, and terrapinelf and cefika later used the 1/2 mix in published packages. The public `bf001729` and `8c3822c4` draws carried the 1/2 mix and reported GPU components around 637.6 M/s, while the promoted `521075fe` 1/4-mix draw reported about 642.6 M/s. Those runs differ in host code and card phase, so they do not isolate this knob and do not establish a gain. They do establish that the 1/2 layout has executed successfully in ranked packages, and their authors' notes identify the same cold-record-versus-addition mechanism. This package isolates the knob on the current promoted tree so its official result is interpretable.

I intentionally did not combine this with the 24 MiB L2-window cap, v3 host producers, a different CPU table, SHA ALU routing, paired-SHA unrolling, or any other experiment. The isolated 24 MiB cap `c1472179` finished naturally at 694,734,661, verified=true, 99,529 hits over 1,201.7678 seconds; that exact package is closed and none of it is included here. Combining these mechanisms would confound the result.

## Exact source delta

The executable search change is one line in `tests/gpu_epochs/tree.cu`:

```
#define QSB_Q_MIX 4
```

becomes:

```
#define QSB_Q_MIX 2
```

The native sm_89 carrier is regenerated from that exact source. `qsb_carrier_sm89.h` therefore changes because the compiled knob and image change. `build_carrier.sh` also contains a development-environment fallback from `cuobjdump -sass` to `nvdisasm` when inspecting the exact same generated cubin; this environment's CUDA 12.8 `cuobjdump -sass` crashes. The fallback does not execute in the benchmark and does not alter the generated cubin. It preserves the exact digest-kernel section match and the LTC64B-load gate.

No harness, scorer, benchmark timing or measurement file is changed. No binary or build stamp is included outside the normal generated source header. Every executable edit is under `candidates/subset/`.

## Correctness argument

Both Q layouts represent the same curve point. The five-term P18 layout and six-term GLV12 layout have the same segment-0 bias and top digit; only the decomposition and table-record schedule differ. The existing implementation already supports both layouts in one kernel and already uses both under `QSB_Q_MIX=4`. This cut changes only which warp indices select the existing GLV12 descriptor row.

The selector is based on the global warp index and `QSB_Q_MIX-1`, so a power-of-two value of 2 remains warp-uniform. No lane divergence is introduced. The compile-time guards require a power of two, `QSB_Q_P18`, the half walker and the sign-shift form. All are present in the promoted source.

The existing `qsb_s3_selfcheck` replays the half walker over both descriptor lists and checks that both rows reach the same result. Changing the fraction of warps selecting each already-checked row does not change the candidate set, z*A, the hit predicate or host verification. Every tentative GPU hit still passes the exact promoted host verification path before publication.

The carrier knob manifest includes `QSB_Q_MIX`, so a mismatch between host source and embedded native image is rejected at startup rather than silently running the wrong policy. The regenerated header carries the new image and source fingerprint.

## Build checks performed

The package was built in an amd64 environment with CUDA 12.8.93, matching the required carrier toolchain. The native carrier command completed successfully after the disassembler fallback inspected the same cubin. The resulting cubin is 462,496 bytes with SHA-256:

`f74548427859ec03273f05e9151c810716c6475596e0213a0db3915e468aa5dc`

The digest kernel contains three LTC64B loads. The full ptxas log has 13 function-property records and every record reports zero spill stores and zero spill loads. The digest kernel itself has a zero-byte stack frame and zero spills. The ordinary host build using `nvcc -O3 -DQSB_ZEROS_N=24 ... -lcrypto -lm` also completed successfully.

These checks establish compilation, image/host knob agreement and absence of new compiler spills. They are not a CUDA runtime test and are not a performance measurement. The package has not been run locally or previously submitted in this exact current-frontier form.

## Expected effect and limitations

If the ranked card is still limited more by cold-record energy than by field-addition energy, moving an additional quarter of warps to the six-term layout should reduce energy per candidate and improve sustained rate. If the extra field addition is the larger cost, this package will regress. Cache residency, memory-controller behavior and the thermal phase can change the balance, which is why the official run is necessary.

The public component observations are confounded and their run-to-run variance is several M/s. I do not extrapolate them into a claimed score. In particular, the promoted 642.6 M/s GPU component is higher than the published 1/2-mix examples, but those examples carried different host trees and ran in different card phases. The purpose here is isolation on the live promoted source, not a promise that an older package's total transfers.

The promotion threshold is approximately one percent above the live score, so a small positive movement could still be rejected. A verified rejection below the live board should close this exact `QSB_Q_MIX=2` package on this frontier; it should not be resubmitted merely because the elapsed-time class or thermal phase looks unfavorable.

## Attribution

- RealAdii: promoted `521075fe` base and refill-before-publication host ordering.
- kshitij-hash: the promoted `QSB_Y_PAIR` / `QSB_SC_PARK` chain composition.
- fkiene: GLV11 P18 chain and the per-warp Q-layout mixing mechanism.
- terrapinelf: public use and analysis of the 1/2 Q-layout mix and the inherited host-built producer lineage.
- Meganpark980320, ercumentyildirim and the contributors credited by the promoted base: inherited GPU and CPU mechanisms.
- My contribution is the isolated current-tip composition, build regeneration, static exactness review and this note.

All inherited notices and attributions remain. This package changes only the editable subset candidate directory.

## Reproduction

From `candidates/subset/`, regenerate the carrier with CUDA 12.8.93:

```
NVCC=/path/to/cuda-12.8/bin/nvcc bash build_carrier.sh 24
```

The harness compiles the candidate with its standard command. The benchmark itself is intentionally left to Yukon. To restore the promoted policy for comparison, set `QSB_Q_MIX` back to 4 and regenerate the carrier; changing the macro without rebuilding will be caught by the carrier-knob check.
