# Vector streaming first-state transport

Parent: our packed-producer composition4881a7d on ercumentyildirim PR407. Only the existing first-state stores and loads change to two aligned128-bit PTX streaming accesses each. CUDA allocations, 2048-byte epoch strides,32-byte class strides and the16-byte second vector offset guarantee the required alignment. The32-byte state layout and all SHA/field arithmetic are unchanged. Producer and consumer kernels remain ordered in one stream, with no concurrent writes to a consumed plane.

The intent is to reduce retention of temporary first-state data beside the reusable fixed-base table. This is a performance hypothesis, not a measured improvement. NVIDIA documents .cs as an evict-first cache hint: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators
