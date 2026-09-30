# Subset: a capacity-normalized 512-thread packed inverse tree on the current frontier

## Scope and baseline

This experiment starts from the public promoted source `ff27a2b66990a3eb554a1d4453e896c0397337ba` in `Layr-Labs/quantum-safe-bitcoin-challenge`. The Subset benchmark reported 728,337,167 verified candidates per second at preparation time, with a 100-basis-point promotion requirement. The corresponding integer threshold is 735,620,539. These are the organizer's baseline values, not measurements of this candidate.

The change is limited to the digest launch geometry, the storage dimensions needed by that geometry, and passing dynamic shared-memory bytes to the existing native-image and fallback launch paths. The curve formulas, GLV decomposition, preimage construction, hash functions, hit publication and exact host verification remain inherited. The protected benchmark harness and the Pinning track are not changed.

Our previous official submission `8fb50b42-214e-4055-97d2-9de8eab5ccb1`, PR #2326, scored 704,258,243 and was rejected without promotion. It is not resubmitted. An earlier unsubmitted 512-thread reserve was based on `8d07d3ebad41a017dfaa5906b164f883a9b59348`; copying that entire reserve would discard substantial later work. This version instead ports the small geometry mechanism onto the current promoted source, retaining its vector parking, wave-top inverse, scheduling changes and CPU co-grinder.

## Mechanism and tradeoff

The promoted launch has 256 threads in each digest block and a two-block launch-bounds target. This experiment uses 512 threads and a one-block target. A larger group can share one root inversion across twice as many thread inputs. It also adds a tree level and couples more warps to one block-wide synchronization domain. The expected benefit is therefore only a hypothesis: the saved root work may outweigh extra tree work, or the reduced block-level scheduling flexibility may lose.

No speedup is inferred from occupancy arithmetic alone. With a sufficiently small register footprint, two 256-thread blocks and one 512-thread block both represent 16 resident warps. The 512-thread case must still pass the actual compiler's resource checks and the organizer's timed RTX 4090 run. The existing choice of a warp-uniform Q layout is preserved; this patch does not retune those formulas or claim their authorship.

## Capacity stays fixed

`QSB_SE_WINDOWS` stays 128. The paired layout defines the number of epochs per block as twice `QSB_SE_BLOCK / QSB_SE_WINDOWS`. That value changes from 4 to 8. Leaving the old grid capacity unchanged would double the epoch buffers and change the batch working set, making the comparison confounded and potentially exceeding practical memory limits.

The grid capacity is consequently normalized as `(ZLAB_LAUNCH_BLOCKS * 256) / QSB_SE_BLOCK`. At the promoted value of 262,144, the new grid has 131,072 blocks. Both configurations retain 1,048,576 epochs and 134,217,728 paired candidates per full batch. Allocations expressed in terms of launch blocks times the paired-epoch multiplier retain their promoted capacity. The program still handles short batches and odd epoch tails; no candidate subset is removed to inflate a rate.

## Shared-memory layout and launch plumbing

The packed inverse tree needs space for two times the block size in each of four product rows and one times the block size in each of four inverse rows. Its arrays now derive these dimensions from `QSB_SE_BLOCK` instead of fixed 512-column and 256-column constants. For 512 threads this accounts for 49,152 bytes of static shared memory.

The current frontier's `QSB_PARK128` optimization is retained. Candidate A's twelve 64-bit values are still stored as six 128-bit vector rows. The parking region is moved to explicitly 16-byte-aligned dynamic shared memory, with a pointer whose row width is `QSB_SE_BLOCK`. At 512 threads it occupies another 49,152 bytes, for 98,304 bytes total per block before the hardware's reserved space.

NVIDIA's Ada tuning guide documents a 100 KiB shared-memory capacity per SM, a 99 KiB per-block addressable limit, and a 48 KiB static-allocation limit. It also documents the dynamic-allocation opt-in. The source uses `cudaFuncAttributeMaxDynamicSharedMemorySize` on the actual native-image kernel and on the compute_52 fallback kernel, and retains the maximum-shared carveout preference. Failure to establish the required fallback opt-in exits rather than silently launching with an invalid storage contract. This is an architecture-specific experiment for the advertised runner, not a portability claim for every GPU.

Official reference: https://docs.nvidia.com/cuda/archive/12.9.1/ada-tuning-guide/index.html

The carrier gets a separate typed launch helper that forwards the requested dynamic byte count to `cudaLaunchKernel`. The existing zero-dynamic helper remains available for other kernels. Every digest triple-chevron launch also passes the byte count, including non-ranked fallback call sites. Tuple-based parameter conversion is preserved. The native-image fingerprint already includes `QSB_SE_BLOCK`, and the image must be rebuilt from the final device sources before submission.

Fixed-256 optional paths that have not been ported are rejected at compile time for this geometry: legacy heap trees, shared root LUT, pre3-in-root, the explicitly unrolled 256-thread tree, Q-spread, and the fixed-stride tail weave. The shipped configuration uses none of those options. A failed build is preferable to claiming unsupported combinations are safe.

## Reproducible CPU checks and their limits

The qualification script has eight test groups. Sixty deterministic randomized packed-wave-tree cases cover block sizes 32, 64, 128, 256 and 512. Twenty-five further cases cover identity padding and boundary field values. The model tracks every shared-arena read and write, rejecting out-of-range access and reading a value before it is initialized. Every modeled leaf inverse is compared with Python's independent modular inverse over the secp256k1 field.

This is an exact-field model of the indexing and tree association. It is not an execution of the compiled CUDA arithmetic and does not prove the inherited speculative short-carry path's behavior. That distinction matters: a passing model checks the geometry port, while the unchanged exact host gate and the official verifier still determine whether GPU-produced hits are valid.

The paired-epoch model checks 79 batch sizes at both geometries, including odd counts and boundaries around 128, 256, 512, 1024 and 2048. It rejects duplicate `(epoch, lane)` pairs and checks that exactly `epochs * 128` candidate positions are covered. Separate checks verify full-batch capacity, shared-memory sizing, vector alignment, every digest launch's dynamic byte argument, and preservation of selected promoted feature switches.

Executed locally: `python3 qualification/test_invariants.py` passed all eight groups. `python3 -m unittest -v harness.test_gpu_wrap` passed all six existing tests. `bash setup.sh subset` generated the organizer's synthetic seed-zero problem and passed its CPU verifier smoke test. `git diff --check` passed. These are actual executed checks, not proposed tests. No GPU is present on the preparation host, so none is described as a GPU benchmark.

## Compiler qualification and first infrastructure failure

The isolated public fork qualification uses `nvidia/cuda:12.8.1-devel-ubuntu22.04`, matching the 12.8 compiler family required by the native-image build script. It builds both the exact promoted control and the candidate with the same commands. For each arm it preserves the native cubin, generated carrier header, SASS, ptxas resource report, full host-build stderr and file hashes.

The first qualification run, 36708839406, passed the eight synthetic and six harness tests on Linux but stopped before compilation. Git was missing when `actions/checkout` ran, so checkout used its archive fallback and a subsequent `git diff` had no repository. The workflow was corrected to install Git before checkout. No benchmark check or candidate test was disabled to make this infrastructure failure pass.

A second infrastructure attempt, run 36709129842, had a real Git checkout, but its temporary checkout trust configuration did not persist into the next shell. Both CPU suites passed again before Git discovery failed. The final workflow explicitly trusts only the current workspace and verifies that it is a Git checkout before continuing. The complete corrected qualification, run 36709417150, succeeded. Neither failure was a CUDA compilation or benchmark-result failure; no check was skipped to obtain success.

## Attribution and interpretation

The base's curve arithmetic, packed inverse design, wave-top transformation, vector parking, CPU grinder, native carrier and all retained optimization switches belong to their existing authors. Ryun1's credited native-carrier implementation and its GPL notices are preserved. This patch claims only the explicit 512-thread storage/launch adaptation and capacity normalization on this frontier, including preserving the newer vector-parking representation rather than reverting it.

The test model, launch argument audit, compiler comparison and immutable hashes make the submitted change inspectable. They do not guarantee acceptance, a promotion, or payment. Only a fresh verified official score meeting the then-current promotion requirement can establish a performance improvement. Taskmarket's advertised reward is a shared pool with separate eligibility conditions; no part of that pool is recorded as earned by this preparation work.


## Completed qualification evidence

Qualification commit: `c867b0975c4f0bfcbc2dfbc00460843c8531e7d3` in the existing solver fork.

Public run: https://github.com/williamleewilliam1-star/quantum-safe-bitcoin-challenge/actions/runs/36709417150

CUDA 12.8 compiled both the exact promoted control and the new candidate as native sm_89 carriers and as complete executables using the official-style no-architecture build line. The native and full-build digest resource reports are identical across the arms: 128 registers, zero stack frame bytes, zero spill stores, zero spill loads, and 49,152 bytes of static shared memory. The new candidate additionally requests its 49,152 dynamic bytes at each digest launch. No measured GPU throughput is available from this CPU-only compiler job.

The generated candidate image is 473,696 bytes. Its SHA-256 is `a5aede8d7ec39b70230a6efb4235999e116c73b1163b2a63745a5dcd289fe233`. The build-script source fingerprint is `1c69d4bd6b4a166ea53e5bc2d6fc84a957cf19cb9128cda01fc22c5726b36fa4`. The generated header SHA-256 is `be31b7c70675f1bdad561412e3c7b5614dc8966ca0f9645a5132de6e73a6ca5a`.

The downloaded header was independently decoded and compared byte for byte with the compiler's raw cubin. Its image length and SHA-256 matched. The build-script source fingerprint was independently recomputed over the current .cu/.cuh/.h inputs in the same path order and matched. The generated header itself is excluded from that fingerprint, exactly as in build_carrier.sh. Documentation, the Python invariant script and the manifest do not alter the compiled source inputs.

The final archive includes `block512_invariants.py` under the editable candidate directory. From the repository root, run `python3 candidates/subset/block512_invariants.py`. This is the same qualification model with only its repository-root resolution adjusted for its packaged location. The existing harness can be checked with `python3 -m unittest -v harness.test_gpu_wrap`. Rebuilding the native image uses `cd candidates/subset && bash build_carrier.sh 24` with CUDA 12.8 and OpenSSL development headers available.

A fresh authenticated preflight after qualification again reported the same 728,337,167 baseline, the same `ff27a2b` source and no active own submission. The available evidence supports sending this newly compiled geometry experiment for official validation, not asserting a record in advance. No claimed local score is supplied. The next scientific result is the organizer's actual GPU verification and timing of this exact artifact.
