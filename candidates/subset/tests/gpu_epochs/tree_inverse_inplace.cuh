// Experimental packed batch inverse. Same field contract as tree_inverse.cuh.
// Supported blocks: powers of two from 32 through 256; inactive lanes supply 1.
#pragma once
__device__ __forceinline__ void qsb_block_inverse_tree(uint64_t *value) {
    __shared__ uint64_t products[4][512];
    const int tid = threadIdx.x, n = blockDim.x;
    #pragma unroll
    for (int k = 0; k < 4; k++) products[k][tid] = value[k];
    __syncthreads();

    int offset = 0;
    #pragma unroll 1
    for (int count = n; count > 2; count >>= 1) {
        const int half = count >> 1;
        if (tid < half) {
            uint64_t a[5], b[5], out[5];
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                a[k] = products[k][offset + tid];
                b[k] = products[k][offset + half + tid];
            }
            a[4] = b[4] = 0;
            qsb_field_mul_raw(out, a, b);
            #pragma unroll
            for (int k = 0; k < 4; k++) products[k][offset + count + tid] = out[k];
        }
        offset += count;
        if (half > 32) __syncthreads(); else __syncwarp();
    }

    // Both root children belong to lane zero. Load both before overwriting.
    if (tid == 0) {
        uint64_t a[5], b[5], root[5];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            a[k] = products[k][offset];
            b[k] = products[k][offset + 1];
        }
        a[4] = b[4] = 0;
        qsb_field_mul_raw(root, a, b);
        qsb_field_normalize(root);
        _ModInv(root);
        root[4] = 0;
        qsb_field_mul_raw(a, root, a);
        qsb_field_mul_raw(b, root, b);
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            products[k][offset] = b[k];
            products[k][offset + 1] = a[k];
        }
    }
    __syncwarp();

    offset -= 4;
    #pragma unroll 1
    for (int count = 4; count < n; count <<= 1) {
        const int half = count >> 1;
        if (tid < half) {
            uint64_t parent[5], left[5], right[5];
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                parent[k] = products[k][offset + count + tid];
                left[k] = products[k][offset + tid];
                right[k] = products[k][offset + half + tid];
            }
            parent[4] = left[4] = right[4] = 0;
            // A thread owns the entire sibling pair. No other thread reads
            // either old child at this level, so in-place writes are safe.
            qsb_field_mul_raw(left, parent, left);
            qsb_field_mul_raw(right, parent, right);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                products[k][offset + tid] = right[k];
                products[k][offset + half + tid] = left[k];
            }
        }
        offset -= count << 1;
        // Next internal level has count readers; the final leaf stage has n.
        if (count > 32 || (count << 1) == n) __syncthreads(); else __syncwarp();
    }

    const int half = n >> 1;
    uint64_t parent[5], sibling[5];
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        parent[k] = products[k][n + (tid & (half - 1))];
        sibling[k] = products[k][tid ^ half];
    }
    parent[4] = sibling[4] = 0;
    qsb_field_mul_raw(value, parent, sibling);
    value[4] = 0;
}
