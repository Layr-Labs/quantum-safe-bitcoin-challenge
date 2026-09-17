// One work-efficient binary product tree per block. The caller supplies a
// power-of-two block size at most 256 and identity factors for inactive lanes.
// The leaf pair is reduced with a warp shuffle before entering the shared tree;
// the inverse pair is expanded directly back to the two lanes at the end. This
// removes the two outer block barriers without carrying leaf state across the
// expensive root inverse.
#pragma once
__device__ __forceinline__ void qsb_block_inverse_tree_scratch(uint64_t *value, uint64_t tree[4][512]){
    const int tid=threadIdx.x,n=blockDim.x,lane=tid&31;

    // Retain the original leaves for the final expansion. Publishing them is
    // warp-local; the pair producer reads only its immediate lane neighbor.
    #pragma unroll
    for(int k=0;k<4;k++)tree[k][n+tid]=value[k];
    uint64_t sibling[5]={0,0,0,0,0};
    #pragma unroll
    for(int k=0;k<4;k++)
        sibling[k]=__shfl_sync(0xffffffffu,(unsigned long long)value[k],lane^1);
    if((lane&1)==0){
        uint64_t pair[5]={0,0,0,0,0};
        qsb_field_mul(pair,value,sibling);
        #pragma unroll
        for(int k=0;k<4;k++)tree[k][(n>>1)+(tid>>1)]=pair[k];
    }
    __syncthreads();

    #pragma unroll 1
    for(int width=n>>2;width>0;width>>=1){
        if(tid<width){
            int node=width+tid;
            uint64_t a[5]={0,0,0,0,0},b[5]={0,0,0,0,0};
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=tree[k][2*node];b[k]=tree[k][2*node+1];}
            qsb_field_mul(a,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)tree[k][node]=a[k];
        }
        if(width>32)__syncthreads();else __syncwarp();
    }
    if(tid==0){
        uint64_t root[5]={0,0,0,0,0};
        #pragma unroll
        for(int k=0;k<4;k++)root[k]=tree[k][1];
        _ModInv(root);
        #pragma unroll
        for(int k=0;k<4;k++)tree[k][1]=root[k];
    }
    __syncthreads();
    #pragma unroll 1
    for(int width=1;width<(n>>1);width<<=1){
        if(tid<width){
            int node=width+tid;
            uint64_t parent[5]={0,0,0,0,0},left[5]={0,0,0,0,0},right[5]={0,0,0,0,0};
            #pragma unroll
            for(int k=0;k<4;k++){
                parent[k]=tree[k][node];
                left[k]=tree[k][2*node];right[k]=tree[k][2*node+1];
            }
            qsb_field_mul(right,parent,right);qsb_field_mul(left,parent,left);
            #pragma unroll
            for(int k=0;k<4;k++){
                tree[k][2*node]=right[k];tree[k][2*node+1]=left[k];
            }
        }
        if((width<<1)>32)__syncthreads();else __syncwarp();
    }

    // At n=64, warp 0 produced pair inverses consumed by warp 1.
    if(n==64)__syncthreads();

    // tree[n/2 + tid/2] is the inverse of this lane pair. Multiplying it by
    // the neighboring original leaf gives this lane's inverse directly.
    uint64_t pair_inverse[5]={0,0,0,0,0};
    #pragma unroll
    for(int k=0;k<4;k++){
        pair_inverse[k]=tree[k][(n>>1)+(tid>>1)];
        sibling[k]=tree[k][n+(tid^1)];
    }
    qsb_field_mul(value,pair_inverse,sibling);
}

// Builder/audit callers keep their original local-shared wrapper.
__device__ __forceinline__ void qsb_block_inverse_tree(uint64_t *value){
    __shared__ uint64_t tree[4][512];
    qsb_block_inverse_tree_scratch(value,tree);
}
