// SPDX-License-Identifier: GPL-3.0-only
// Research-only checkpoint inverse tree for a layer-blocked affine chain.
// Adapted from the pinned qsb_block_*_checkpoint implementation in pinning.cu.
// Include after qsb_field_mul, qsb_field_mul_sc and qsb_field_normalize.
#pragma once

#ifndef QSB_AFFINE_TREE_EXACT
#define QSB_AFFINE_TREE_EXACT 1
#endif
#if QSB_AFFINE_TREE_EXACT != 0 && QSB_AFFINE_TREE_EXACT != 1
#error QSB_AFFINE_TREE_EXACT must be 0 or 1
#endif

__device__ __forceinline__ void qsb_affine_tree_mul(
    uint64_t out[5], uint64_t a[5], uint64_t b[5]) {
#if QSB_AFFINE_TREE_EXACT
    // A rare incorrect product in a shared inverse poisons a whole subtree.
    // Keep carry-complete arithmetic as the default for this experiment.
    qsb_field_mul(out, a, b);
#else
    // An explicit performance experiment only: retains the inherited field-SC
    // approximation and needs a separate recall/error audit before deployment.
    qsb_field_mul_sc(out, a, b);
#endif
}

// On-chip variant for a resident worker CTA. The caller owns products, which
// must remain immutable while an inversion service handles the published root.
// Only lane zero receives root[0..3]. Other lanes may continue to CTA-local
// work, but may not overwrite this scratch until finish_shared has completed.
template<int N>
__device__ __forceinline__ void qsb_affine_tree_prepare_shared(
    const uint64_t value[4], uint64_t root[4],
    uint64_t (*products)[2*N]) {
    static_assert(N >= 32 && N <= 256 && !(N & (N-1)),
                  "affine tree requires power-of-two N in [32,256]");
    const int tid = (int)threadIdx.x;
    #pragma unroll
    for (int k=0; k<4; ++k) products[k][tid] = value[k];
    __syncthreads();
    int offset = 0;
    #pragma unroll 1
    for (int count=N; count>1; count>>=1) {
        const int half = count>>1;
        if (tid < half) {
            uint64_t a[5], b[5], product[5];
            #pragma unroll
            for (int k=0; k<4; ++k) {
                a[k] = products[k][offset+tid];
                b[k] = products[k][offset+half+tid];
            }
            a[4] = b[4] = 0;
            qsb_affine_tree_mul(product, a, b);
            #pragma unroll
            for (int k=0; k<4; ++k)
                products[k][offset+count+tid] = product[k];
        }
        offset += count;
        if (count > 2) {
            if (half > 32) __syncthreads();
            else __syncwarp();
        }
    }
    if (tid == 0) {
        #pragma unroll
        for (int k=0; k<4; ++k) root[k] = products[k][2*N-2];
        qsb_field_normalize(root);
    }
}

// Only lane zero's root_inverse is read. The inverse service MUST return 1/root,
// without the production recovery pipeline's ISO weighting. All N lanes call
// this helper after the service reply is visible to lane zero. The initial
// block barrier publishes the root inverse and also waits for every producer
// of the retained tree. Shared requirement: 4*(2N+N)*8 bytes = 12 KiB at N=128.
template<int N>
__device__ __forceinline__ void qsb_affine_tree_finish_shared(
    uint64_t out[4], const uint64_t root_inverse[4],
    uint64_t (*products)[2*N], uint64_t (*inverses)[N]) {
    static_assert(N >= 32 && N <= 256 && !(N & (N-1)),
                  "affine tree requires power-of-two N in [32,256]");
    const int tid = (int)threadIdx.x;
    if (tid == 0) {
        #pragma unroll
        for (int k=0; k<4; ++k) inverses[k][N-2] = root_inverse[k];
    }
    __syncthreads();
    int offset = 2*N-4;
    #pragma unroll 1
    for (int count=2; count<N; count<<=1) {
        const int half = count>>1;
        if (tid < count) {
            uint64_t parent[5], sibling[5], child[5];
            #pragma unroll
            for (int k=0; k<4; ++k) {
                parent[k] = inverses[k][offset+count-N+(tid&(half-1))];
                sibling[k] = products[k][offset+(tid^half)];
            }
            parent[4] = sibling[4] = 0;
            qsb_affine_tree_mul(child, parent, sibling);
            #pragma unroll
            for (int k=0; k<4; ++k) inverses[k][offset-N+tid] = child[k];
        }
        offset -= count<<1;
        if (count >= 32) __syncthreads();
        else __syncwarp();
    }
    uint64_t parent[5], sibling[5], leaf_inverse[5];
    #pragma unroll
    for (int k=0; k<4; ++k) {
        parent[k] = inverses[k][tid&(N/2-1)];
        sibling[k] = products[k][tid^(N/2)];
    }
    parent[4] = sibling[4] = 0;
    qsb_affine_tree_mul(leaf_inverse, parent, sibling);
    qsb_field_normalize(leaf_inverse);
    #pragma unroll
    for (int k=0; k<4; ++k) out[k] = leaf_inverse[k];
}

// Each participating block has exactly N threads and N nonzero effective
// leaves. Callers replace inactive or singular leaves by 1 and keep their own
// validity masks. Checkpoint allocation: blocks * 4 * (N-2) uint64_t words.
// Roots are four words per block. A separate hierarchy must replace roots by
// their UNWEIGHTED inverses before finish; the production ISO-scaled inverse
// kernel is not interchangeable with that operation.
//
// Packed product levels use (offset,count)=(0,N),(N,N/2),...,(2N-2,1).
// Only nodes N through 2N-3 are checkpointed: no leaves and no root.
// Prepare costs N-1 field multiplications. All threads must call the helper.
template<int N>
__device__ __forceinline__ void qsb_affine_tree_prepare(
    const uint64_t value[4], uint64_t *roots, uint64_t *checkpoint) {
    static_assert(N >= 32 && N <= 256 && !(N & (N-1)),
                  "affine tree requires power-of-two N in [32,256]");
    __shared__ uint64_t products[4][2*N];
    const int tid = (int)threadIdx.x;
    const size_t block_base = (size_t)blockIdx.x * 4u * (N-2);
    #pragma unroll
    for (int k=0; k<4; ++k) products[k][tid] = value[k];
    __syncthreads();

    int offset = 0;
    #pragma unroll 1
    for (int count=N; count>1; count>>=1) {
        const int half = count>>1;
        if (tid < half) {
            uint64_t a[5], b[5], product[5];
            #pragma unroll
            for (int k=0; k<4; ++k) {
                a[k] = products[k][offset+tid];
                b[k] = products[k][offset+half+tid];
            }
            a[4] = b[4] = 0;
            qsb_affine_tree_mul(product, a, b);
            const int node = offset+count+tid;
            #pragma unroll
            for (int k=0; k<4; ++k) {
                products[k][node] = product[k];
                if (node < 2*N-2)
                    checkpoint[block_base+(size_t)k*(N-2)+node-N] = product[k];
            }
        }
        offset += count;
        if (count > 2) {
            // Once producers fit in warp zero, all remaining readers do too.
            if (half > 32) __syncthreads();
            else __syncwarp();
        }
    }
    if (tid == 0) {
        uint64_t root[5];
        #pragma unroll
        for (int k=0; k<4; ++k) root[k] = products[k][2*N-2];
        root[4] = 0;
        qsb_field_normalize(root); // _ModInv requires a canonical input.
        #pragma unroll
        for (int k=0; k<4; ++k) roots[(size_t)blockIdx.x*4u+k] = root[k];
    }
}

// Each lane supplies the SAME effective leaf used in prepare, restored from
// state or recomputed exactly. The input may contain any congruent raw value
// in [0,2^256); output is canonical. Four-word input/output arrays are valid.
// Finish costs 2N-2 multiplications, including the final N leaf products.
// Thus the pair costs 3N-3, rather than forming cofactors and multiplying each
// leaf by the root inverse in another pass (which adds approximately N).
template<int N>
__device__ __forceinline__ void qsb_affine_tree_finish(
    uint64_t value[4], const uint64_t *root_inverses,
    const uint64_t *checkpoint) {
    static_assert(N >= 32 && N <= 256 && !(N & (N-1)),
                  "affine tree requires power-of-two N in [32,256]");
    __shared__ uint64_t products[4][2*N];
    __shared__ uint64_t inverses[4][N];
    const int tid = (int)threadIdx.x;
    const size_t block_base = (size_t)blockIdx.x * 4u * (N-2);
    #pragma unroll
    for (int k=0; k<4; ++k) {
        products[k][tid] = value[k];
        if (tid < N-2)
            products[k][N+tid] = checkpoint[block_base+(size_t)k*(N-2)+tid];
        if (tid == 0)
            inverses[k][N-2] = root_inverses[(size_t)blockIdx.x*4u+k];
    }
    __syncthreads();

    int offset = 2*N-4;
    #pragma unroll 1
    for (int count=2; count<N; count<<=1) {
        const int half = count>>1;
        if (tid < count) {
            uint64_t parent[5], sibling[5], child[5];
            #pragma unroll
            for (int k=0; k<4; ++k) {
                parent[k] = inverses[k][offset+count-N+(tid&(half-1))];
                sibling[k] = products[k][offset+(tid^half)];
            }
            parent[4] = sibling[4] = 0;
            qsb_affine_tree_mul(child, parent, sibling);
            #pragma unroll
            for (int k=0; k<4; ++k) inverses[k][offset-N+tid] = child[k];
        }
        offset -= count<<1;
        // At count==32 the NEXT level first crosses the warp boundary.
        if (count >= 32) __syncthreads();
        else __syncwarp();
    }

    uint64_t parent[5], sibling[5], leaf_inverse[5];
    #pragma unroll
    for (int k=0; k<4; ++k) {
        parent[k] = inverses[k][tid&(N/2-1)];
        sibling[k] = products[k][tid^(N/2)];
    }
    parent[4] = sibling[4] = 0;
    qsb_affine_tree_mul(leaf_inverse, parent, sibling);
    qsb_field_normalize(leaf_inverse);
    #pragma unroll
    for (int k=0; k<4; ++k) value[k] = leaf_inverse[k];
}
