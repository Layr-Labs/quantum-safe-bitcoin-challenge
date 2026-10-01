# SUBSET: ff27 GPU lineage with three CPU inverse-boundary edits

This is one exploratory SUBSET submission. The package starts from the promoted ff27 implementation and changes only three CPU inverse-boundary operations in CpuGrindSubset.h. No local or official gain is claimed. The source inventory below describes this package. The ordinary build and a bounded actual-path correctness check completed. No official score or promotion is represented here as a result of this package.

## Base and attribution

The promoted base is cefika's submission fb6f5a8f-b29e-4506-a50c-c79a9c5a2a0e, source [ff27a2b66990a3eb554a1d4453e896c0397337ba](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/ff27a2b66990a3eb554a1d4453e896c0397337ba). Its retained GPU wrapper, tree source, helper headers, producer source and embedded native image are the base of this package.

The three CPU edits are attributed to jacklightChen's public submission b1c5e58e-210d-4353-addf-edc4474b8f33, immutable [CPU donor source843c82817bf2de27e7794bb6317d1d223d6fa1fd](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/843c82817bf2de27e7794bb6317d1d223d6fa1fd). Only the selected three CPU boundary edits are included. The package contains the selected CPU header with Git blob82418e20049889bb4210fd4e7fd98953e4152a6b.

The base promotion records acknowledgments to kshitij-hash, terrapinelf, i34-9, ercumentyildirim, HyeokxC, jacklightChen, fkiene, kaankolcu, newjordan and Meganpark980320. Those names are retained here as base provenance. The submitter is i34-9; this submission has no coauthors. The existing VanitySearch notices and candidate license text are preserved.

## Included CPU changes

The candidate includes three host arithmetic-boundary edits in CpuGrindSubset.h. The corresponding source controls are enabled in this package. The existing scalar inverse, vector arithmetic helpers, ordinary point operations and key-message interface remain in the header. No new arithmetic helper or worker interface is introduced.

| Candidate code surface | Package declaration |
|---|---|
| Host arithmetic header | Contains the selected three boundary edits. |
| Ordinary CPU worker | Retains the existing worker entry point and table interface. |
| Final CPU key-message stage | Retains the existing message-storage interface. |
| CPU output writer | Retains the existing output record format. |
| CPU feature selection | Uses the inherited feature-detection and selection code. |

CpuGrindSubset.h is the sole changed candidate code file. Attribution comments and the controls belong to the same header. The public note is the only other changed file. Both changes are within the SUBSET editable directory.

The package preserves its existing distinction between vector arithmetic, scalar arithmetic, key-message preparation and candidate publication. It does not add a separate executable, a dynamically loaded host library or a new runtime dependency. The selected CPU source is supplied directly to the existing candidate build.

## Preserved GPU and runtime source

| Surface | Package contents |
|---|---|
| Entry wrapper | subset.cu is byte-identical to ff27. |
| GPU tree | tests/gpu_epochs/tree.cu is byte-identical to ff27; QSB_Q_MIX remains4. |
| SHA addition selector | QSB_SHA_FMA_ADD remains0 in the entry wrapper. |
| Paired GPU helpers | pair_shared.cuh and the field/hash helper headers are inherited unchanged. |
| Native carrier | qsb_carrier_sm89.h and QsbCarrier.h are byte-identical to ff27. |
| Host producer | host_producers_v3.h is inherited unchanged. |
| Host routing | Existing launch, polling, worker selection and table-allocation source is inherited unchanged. |
| Candidate publication | Existing GPU collection and CPU output publication source is retained. |

The inherited native payload is473376 bytes. Its decoded SHA256 is e0c0897f799baf81df92f777f89adb4b351cf224df4a6a6c6d8a8cabf1631fea. This is a packaged-image identity declaration; it does not declare that a runtime selected or executed that image.

The CPU header SHA256 is ce744e34e89adea5db16d3af942321a193a7c768bebfb78a7d94ff58194a6ba7. The unchanged entry-wrapper Git blob is11c397b08579c491a70c4000f8c0e489e267caaa, and the unchanged tree Git blob is803fb4a4157b98a609c1e5acf8efd82c0f04257f. These identifiers describe source content independently of the final submission commit.

## Scope and packaging declarations

| Item | Final source declaration |
|---|---|
| Track | SUBSET only, under schemaVersion2 of qsb-grind-benchmark. |
| Editable candidate directory | candidates/subset, as listed by benchmark.json. |
| Candidate code delta | CpuGrindSubset.h only; the selected controls default enabled. |
| Public documentation | This note records source scope and attribution. |
| Additional CPU selections | No unrelated CPU arithmetic, layout or hashing edit is added. |
| Additional GPU selections | No GPU code, image, geometry or runtime-selection edit is added. |
| Protected benchmark surfaces | Manifest, harness, verifier, workflows, setup and benchmark scripts are unchanged. |
| Separate track | candidates/pinning is unchanged. |
| External inputs | No extra service, download step, credential, private fixture or precomputed hit file is added. |
| Candidate interface | Existing arguments, input formats, recovery-ID handling and output record format are retained. |
| Licenses and notices | Existing candidate COPYING and third-party notices are retained. |
| Submission authorship | Solo submission; donor attribution is recorded in this note. |
| Archive inventory and size | Checked against the track archive limits before upload. |

The retained source includes its existing carrier-selection checks and fallback implementation. The final note and candidate source were checked for credential markers, and the note uses the permitted public signature. This CPU edit does not replace those checks or modify the fallback GPU code. The input problem and acceptance predicate remain supplied by the unchanged benchmark. No score file, verification record or result artifact is represented here as a result of this package.

## Evidence declaration

| Evidence class | Current declaration |
|---|---|
| Source comparison | The candidate code delta and inherited files are identified above. |
| Compilation | Ordinary track build completed with the locked CUDA toolchain and existing build options. |
| Runtime engagement | The inherited native image and existing vector CPU/key-hash route were observed in a bounded normal run. |
| Full-record validation | All collected candidate records passed the independent cryptographic verifier. The retained CPU oracle includes matching raw outputs and a detected injected fault. |
| Local gain | No local gain claim. |
| Official result | No official score, promotion or gain claim. |

zarar@1337 <3 🎲