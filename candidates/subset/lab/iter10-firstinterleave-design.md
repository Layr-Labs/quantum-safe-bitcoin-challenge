# Iteration 10: first-state slot interleave — design review before verifier

Official 414fe58 failed Benchmark with 0 work, 0.49s grinder self time, 0.614531435s bridge wall time. Retrieved failed workflow log and existing official artifacts once (not polling). Artifacts omit raw stdout/stderr; exact startup failure is unknown. Ranked major allocation requests total 24,117.463 MiB, leaving 458.537 MiB nominal before runtime, a plausible capacity failure. This is not an established OOM location.

New host-only knob QSB_FIRST_SLOT_INTERLEAVE=1, default OFF, changes allocation of slot1 only. Native kernel and fingerprint unchanged. Gate requires the existing 128-window, 16-slot, eight-class configuration and producer-scratch disabled. Otherwise independent allocation remains. It is materially different from firstslots8 (which changes device row pitch) and prior producer lifetime scratch alias (which saves only group/map storage).

Implementation traces:
* window_schedule_shared.cuh qsb_prepare_window_schedule assigns first_class IDs in [0,first_distinct); validates first_distinct <= QSB_FIRST_SLOTS before launch. Gate demands exactly eight classes.
* kernel_build_first_flat writes e*16*8+c*8 for c in [0,8). Digest scheduled-window functions read the same class mapping at e*16*8. Set slot1 pointer to slot0+64 uint32 words. Active byte ranges at epoch e: slot0 [512e,512e+256), slot1 [512e+256,512e+512). These are disjoint within and between every epoch, including last partial batch. Last slot1 word ends at allocated arena boundary.
* qhp::upload uses cudaMemcpy2DAsync width=ncls*32=256, pitch=QSB_FIRST_SLOTS*32=512 with per-piece offsets using this pitch. Slot1 shift remains 256 for every piece. No full-pitch write.
* enqueue_check_copy + check_thread compare exactly ncls*32 at each pitched row. Existing startup alias remains slot0 only. Descriptors remain separate per slot, so wait_check_reuse continues to protect slot0 on both normal and failed producer paths while slot1 modifies only its own row half.
* All stream, upload, completion/readback, slot reuse and CPU publication transitions unchanged. Distinct descriptors/group maps/hit buffers remain independently owned. GPU fallback writes only its classes too.
* Allocation error of slot1 descriptors propagates unchanged; no shifted pointer assigned until descriptor allocation succeeds. Other shapes allocate independent first arrays, including 256 windows and pack8; producer scratch opts out to prevent its bulk group/map writes aliasing the other slot.
* Teardown currently exits without individually freeing slot first arrays; no added free/double-free. Future explicit teardown must free d_first owner once, not slot1 interior pointer.
* Empty epoch condition is handled by unchanged launch loop; width/offset gate involves constants and no division. Allocation multiplication identical to already allocated slot0, so no new overflow expression.

Smallest relevant pre-existing exact verifier: lab/iter6_ab.sh current frozen vs interleave, one pair, N24 120s, seed1789110211. Existing benchmark.sh subset invokes harness/run_benchmark.py; each GPU run uses flock. Full build and stamps recorded. This is capacity/correctness screen, not speed qualification; do not automatically repeat the prior negative SHA directions or stack broader fixtures after PASS.
