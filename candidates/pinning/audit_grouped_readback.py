#!/usr/bin/env python3
"""Source-shape checks for host-drain + side-stream hit D2H."""
from pathlib import Path
src = (Path(__file__).resolve().parent / "pinning.cu").read_text()
inner = src.split("Search all safe locktimes for this sequence")[1].split(
    "Progress every 10 sequences"
)[0]
assert "cudaDeviceSynchronize()" not in inner
assert "cudaMemsetAsync(d_hit_cnt[hit_slot], 0, 4, 0)" in inner
assert "pinning_drain_hits_stream" in src
assert "cudaStreamNonBlocking" in src
assert "pinning_upload_seq_midstate" not in src
assert "pinning_fast_graph_launch" not in src
assert 'fprintf(f, "sequence=%u\\nlocktime=%u\\nhash_choice=%d\\nrecid=%d\\n"' in src
assert "printf(\"  seq=0x%08X lt=%u hc=%d recid=%d\\n\"" not in inner
print("PASS: side-stream hit drain source shape")
