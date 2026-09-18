#!/usr/bin/env python3
"""Source-shape binder for the slotted host pipeline on tip ce0aff4e."""
from pathlib import Path

src = Path(__file__).with_name("pinning.cu").read_text()
checks = [
    ("#define QSB_SLOTS 2", "default two slots"),
    ("cudaStream_t st", "launch takes a stream"),
    ("<<<blocks0,QSB_S0_THREADS,0,st>>>", "prepare launches on st"),
    ("<<<blocks2,QSB_S2_THREADS,0,st>>>", "finish launches on st"),
    ("d_pipeline_state[QSB_SLOTS]", "per-slot state buffers"),
    ("d_pipeline_roots[QSB_SLOTS]", "per-slot root buffers"),
    ("d_super_roots[QSB_SLOTS]", "per-slot super-roots"),
    ("d_root_checkpoint[QSB_SLOTS]", "per-slot root checkpoints"),
    ("cudaStreamCreateWithFlags(&slot_stream[s], cudaStreamNonBlocking)", "nonblocking streams"),
    ("cudaEventSynchronize(slot_done[s])", "slot reuse waits on event"),
    ("cudaMemcpyAsync(h_hit_cnt + s, d_hit_cnt[s]", "async hit counter readback"),
    ("batch_no % QSB_SLOTS", "round-robin slotting"),
    ("#define QSB_COFACTOR 1", "tip cofactor retained"),
    ("#define QSB_L2_SKIP 1", "tip L2 skip retained"),
    ("#define QSB_DIRDIG 1", "tip direct digits retained"),
    ("#define QSB_SQFREE 1", "tip squaring-free tail retained"),
]
failed = []
for token, label in checks:
    if token not in src:
        failed.append(label)
slotted = src[src.find("Slotted host pipeline") :]
if "cudaDeviceSynchronize" in slotted:
    failed.append("no device sync in slotted region")
if "uint32_t *d_hit_cnt," in slotted:
    failed.append("no single-stream hit buffer in slotted region")
if failed:
    raise SystemExit("FAIL: " + ", ".join(failed))
print("PASS: slotted host pipeline shape; %d binders" % len(checks))
