// N=24 scored-path arm "tu": QSB_TREE_UNROLL 1 — the tree's up and down loops
// written with the block size as the compile-time 256 of every kernel_digest
// launch, fully unrolled: the per-pass loop control all eight warps run goes,
// row offsets become immediates (tree_inverse.cuh:208; pinning's QSB_POST_GLUE
// bits 8+128 write the levels the same way). Documented bit-identical.
// Never measured on this tree.
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 1
#define QSB_TREE_UNROLL 1
#include "subset.cu"
