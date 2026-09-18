#!/usr/bin/env python3
"""Source-shape binder: deepen slotted host pipeline to QSB_SLOTS=3 on STREAM tip."""
from pathlib import Path

src = Path(__file__).with_name("pinning.cu").read_text()
checks = [
    ("#define QSB_SLOTS 3", "three in-flight host slots"),
    ("#define QSB_SLOTPIPE 1", "slotpipe retained"),
    ("#define QSB_STREAM 1", "tip streaming operators retained"),
    ("#define QSB_L2_SKIP 1", "tip L2 skip retained"),
    ("#define QSB_DIRECT_DIGITS 1", "tip direct digits retained"),
    ("#define QSB_SPARSE_TAIL 1", "tip sparse tail retained"),
    ("#define QSB_SYM_FINISH 1", "tip symmetric finish retained"),
    ("#define QSB_PREFETCH 0", "prefetch left off (keeps direct digits)"),
    ("#define QSB_STREAM_PARM , cudaStream_t st", "launch takes a stream"),
    ("<<<blocks0,QSB_S0_THREADS QSB_STREAM_ARG>>>", "prepare launches on slot stream"),
    ("<<<blocks2,QSB_S2_THREADS QSB_STREAM_ARG>>>", "finish launches on slot stream"),
    ("d_pipeline_state[QSB_SLOTS]", "per-slot state buffers"),
    ("d_pipeline_roots[QSB_SLOTS]", "per-slot root buffers"),
    ("d_super_roots[QSB_SLOTS]", "per-slot super-roots"),
    ("d_root_checkpoint[QSB_SLOTS]", "per-slot root checkpoints"),
    ("cudaStreamCreateWithFlags(&slot_stream[s], cudaStreamNonBlocking)", "nonblocking streams"),
    ("cudaEventSynchronize(slot_done[s])", "slot reuse waits on event"),
    ("cudaMemcpyAsync(h_hit_cnt + s, d_hit_cnt_s[s]", "async hit counter readback"),
    ("batch_no % (uint64_t)QSB_SLOTS", "round-robin slotting"),
    ("Slot pipeline: %d in-flight batches", "slot count printed at startup"),
]
failed = []
for token, label in checks:
    if token not in src:
        failed.append(label)
for ban, label in [
    ("#define QSB_RESOLVE_LAST 1", "resolve-last must stay off"),
    ("#define QSB_FUSE_SQRADDSUB2 1", "fuse must stay default-off"),
    ("#define QSB_SLOTS 2", "must not keep tip two-slot default"),
]:
    if ban in src:
        failed.append(label)
# Hot search loop must wait on per-slot events, not a device-wide sync.
loop_start = src.find("uint64_t batch_no = 0;")
if loop_start < 0:
    failed.append("batch loop marker")
else:
    loop = src[loop_start:]
    if "cudaDeviceSynchronize" in loop:
        failed.append("no device sync inside batch loop")
    if "cudaEventSynchronize(slot_done[s])" not in loop:
        failed.append("event sync inside batch loop")
if failed:
    raise SystemExit("FAIL: " + ", ".join(failed))
print("PASS: slots-deepen shape; %d binders" % len(checks))
