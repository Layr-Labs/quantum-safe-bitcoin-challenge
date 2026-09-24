# Native-only host carrier with unchanged subset device code

Selected for official evaluation after the valid GLV12 result. No official GPU result exists for this carrier at packaging.

The fixed build now compiles the shared subset host setup, search and publication code without candidate CUDA kernel or device-global definitions. Typed host descriptors launch the separately loaded native module. A missing, corrupt or incompatible native payload terminates the program; this carrier has no source fallback.

The device payload is exactly the one packaged with the evaluated native-isolation baseline: seven kernels, twenty device globals, the original fifteen-term/64 MiB table geometry, the same speculative filter and exact verifier, and the same 128-window candidate family. Its 933,536-byte sm89 cubin is byte-identical, SHA256 `994014fb153b73eb4338b013d373734dc8350041378b2c39ed0199987cc38ae8`. The regenerated PTX is also identical, SHA256 `8599b133fe9a27ffc33651db33e5dcdec4385955bc831907762a6430c7b674f5`. No arithmetic, seed handling, candidate mapping, launch policy or output format is changed.

## What this comparison can establish

The earlier native-isolation submission `7311c0aa` was valid at 608,538,767 verified candidates/s with 87,134 verified hits over 1201.128028308 seconds. It did not improve the 623,518,629 frontier. Its executable retained source CUDA images and allowed whole-pipeline fallback when the adjacent native file was unavailable. Official output did not expose selected-module diagnostics or stage times. Neither its score nor the presence of those images establishes source fallback, JIT cost or a loading defect.

This carrier removes that fallback and excludes candidate device code from its fixed-command executable. A later valid fresh-instance run would therefore provide stronger source/binary control-flow evidence that the native payload executed. It would not retrospectively identify the earlier run's selected route or attribute a score difference to JIT. There is no measured GPU result or throughput improvement for this preparation.

NVCC still emits a header-only sm52 PTX container, two metadata-only sm52 ELF containers and a generic RegisterFatBinary/End constructor. Actual dumps contain no candidate entries, device functions, global storage or candidate function/global registration calls. This is not a claim of zero PTX containers, zero fatbin registration or zero CUDA startup work.

## Source organization and runtime contract

An explicit carrier/full-source role is selected identically in both nvcc passes. Narrow guards retain one copy of the original host bodies and defaults; shared epoch/group POD definitions preserve their wire layouts. The full device source remains included for explicit regeneration. No generated host search body, post-link patch, opaque host executable or external build step is used.

Seven readable typed kernel descriptors preserve every declared argument type and conversion before argument addresses are packed, including all forty digest parameters. Thirteen typed global descriptors retain declared-size and bounds checks. Fourteen transfer sites, including both reads of SHA256 K, use the selected module. Lookup, transfer, ABI and launch failures are checked. The existing late 32 KiB stack request, stream 0, table construction, timing, replay and publication order remain unchanged. Carrier-only guards reject incompatible geometry and mapping flags.

The carrier requires the ranked sm89/N24 configuration and CUDA 12.8-or-later headers. It locates `native_sm89.cubin` beside its executable through `/proc/self/exe`, verifies exact length and SHA256, and resolves all required handles before instance initialization. The image contains generic CUDA computation and constants; fresh problem data still initializes the module and constructs the table at runtime. No benchmark instance or hits are packaged.

The unchanged fixed build is:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm
```

`-DQSB_NATIVE_MODULE=0` explicitly selects the retained full-source regeneration role; it is not a runtime fallback available to the default carrier. With CUDA 12.8.93, `regenerate_native.py` compiles the original default compute52 PTX frontend and sm89 assembler recipe, and stops before installing outputs unless the exact baseline cubin is reproduced. Build outputs stay outside the candidate. `native_module.md` documents the complete command and container option.

## Validation

The exact fixed-command build passed. Both author regeneration and independent regeneration from an isolated source copy reproduced the baseline cubin and PTX; the complete generated header, manifest and payload matched. Actual cubin metadata confirms all seven parameter ordinal/offset/width layouts and twenty global sizes. Source-type assertions cover distinctions that binary widths alone cannot establish. All twenty-five conservative source hashes and eleven-carrier/twenty-two-full-source dependency closures were verified.

The portable mock suite passed 27 process-isolated runtime cases and four expected unsupported-build failures. It checks seven source signatures, thirteen global types, POD sizes/alignment/offsets, fourteen typed launches and both K reads. Actual schedule helpers exercise ten transfer sites; four matching main-upload bindings complete the fourteen-site inventory. It does not execute the complete main function. A separate literal configuration-guard test passed 37 preprocessing cases: baseline acceptance, 18 carrier rejections and 18 full-source bypasses; these are not 37 full candidate builds.

Independent source review confirms the 84,192-byte host tail and earlier host helper bodies remain unchanged. Independent actual-executable inspection confirms the empty compiler containers and absence of candidate registration described above. Both architecture and validator reviews found no remaining must-fix within this bounded source, ABI, mock and compile-only scope. No local GPU was available; native driver loading, fresh-instance correctness and throughput still require official evaluation.

Portable host checks, from the candidate directory:

```sh
python3 tests/native_runtime/check.py --docker qsb-cuda
python3 tests/native_runtime/check_config.py
```

The first command uses an existing CUDA development container mounted as documented in the test README; it mocks CUDA rather than executing GPU arithmetic.

Implementation and independent review use Codex with GPT-6 Astra at high reasoning effort, coordinated through Herdr. This preserves the promoted subset arithmetic and its existing attribution/license notices; the new work is the host compilation seam, typed native-only adapters and their validation.


## Provenance and current experiment status

The direct device source is the promoted subset `9ac2515450446dbadbe061e98ebfc317c36d4999`
(submission `7aef224a-e3ff-43f9-9877-50cdbda3f653`). The direct packaging reference
is native control `7311c0aa-424a-4ad5-8197-31ad8db50a50`, official commit
`caed355d6ed9d150b4d82e4947b9854eb47eb1e6`. Its native payload is preserved exactly.
The canceled earlier native ticket `4284f910` had no GPU evaluation. The separate
minimum GLV10 ticket `aa4d949d` was valid at 439,741,553/s; none of its geometry,
arithmetic, coverage or builder changes is imported here. GLV12 ticket `466ce264`
completed valid at 552,910,868/s with 79,167/79,167 verified hits over
1201.099432425 seconds. Its official source is
`22a3e330b2cb8429657e772085eee662d0fff6b3`; rejection was score-only.
The independent source/rank/collector audit found no concrete blocker: all
79,167 identities roundtrip the literal source mapping, and all 128 family-A
suffixes appear. The completed-prefix lower bound is 663,975,100,416 candidates;
this does not establish perfect recall, stage timings or native selection.

This carrier is the next isolated loading-control comparison after that result.
Its unchanged baseline arithmetic separates this question from the GLV table
tradeoff. The GLV12 score does not identify a cache, loading or JIT cause, and no
GLV12 mechanism is imported. The campaign target of 997,629,806.4/s remains
unachieved; this carrier carries no predicted throughput or target claim.

Credit remains with the inherited subset contributors, including Akashneelesh,
Meganpark980320, dun999, ercumentyildirim, EvanYan1024, jacklightChen,
Saviour1001, odinfree, DPZZxlz, fkiene, owizdom, jrcarlos2000 and terrapinelf.
terrapinelf's public a75cf15a cold-compilation research helped motivate the
earlier native isolation; its local measurements are not proof of ranked JIT cost.
No compaction or hot arithmetic change from that work is bundled here.
All inherited source and license notices remain present.
