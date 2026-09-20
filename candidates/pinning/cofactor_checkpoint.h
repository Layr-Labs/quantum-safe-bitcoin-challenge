// Dependency-scoped barrier mechanism follows Calcutatatoraa95b1b9;
// applied here to the distinct public cofactor exclusion traversal.
// Public cofactor collective: tekkac, submission31e98e47, commit554fa24c.
#pragma once

// The caller supplies nonzero effective leaves (identity for unusable lanes).
// Preserve immutable products and accumulate exclusion products separately.
// All N lanes participate in every barrier; one block publishes one raw root.
template<int N> __device__ __forceinline__ void qsb_cofactor_prepare(
    uint64_t *value,uint64_t *roots,uint64_t (*products)[2*N],uint64_t (*excluded)[N]) {
    static_assert(N>=2 && !(N&(N-1)),"power-of-two tree");
    int tid=threadIdx.x;
    #pragma unroll
    for(int k=0;k<4;k++)products[k][tid]=value[k];
    __syncthreads();
    int offset=0;
    #pragma unroll 1
    for(int count=N;count>(N>=16?2:1);count>>=1) {
        int half=count>>1;
        if(tid<half) {
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=products[k][offset+tid];b[k]=products[k][offset+half+tid];}
            a[4]=b[4]=0;qsb_field_mul(out,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)products[k][offset+count+tid]=out[k];
        }
        offset+=count;
        if(count>2){if(half>32)__syncthreads();else __syncwarp();}
    }
    // Root and four exclusions use exactly the original ordered operands.
    // Lanes 0..4 share a warp; the barrier publishes all four exclusions.
    if(N>=16) {
        if(tid<5) {
            uint64_t a[5],b[5],out[5];
            const int ia=tid<4 ? 2*N-4+((tid&1)^1) : 2*N-4;
            const int ib=tid<4 ? 2*N-8+(tid^2) : 2*N-3;
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=products[k][ia];b[k]=products[k][ib];}
            a[4]=b[4]=0;qsb_field_mul(out,a,b);
            if(tid<4) {
                #pragma unroll
                for(int k=0;k<4;k++)excluded[k][N-8+tid]=out[k];
            } else {
                #pragma unroll
                for(int k=0;k<4;k++)roots[(size_t)blockIdx.x*4+k]=out[k];
            }
        }
    } else if(tid==0) {
        #pragma unroll
        for(int k=0;k<4;k++) {
            roots[(size_t)blockIdx.x*4+k]=products[k][2*N-2];
            excluded[k][N-2]=k==0?1:0;
        }
    }
    __syncwarp();
    offset=N>=16?2*N-16:2*N-4;
    #pragma unroll 1
    for(int count=N>=16?8:2;count<N;count<<=1) {
        int half=count>>1;
        if(tid<count) {
            uint64_t parent[5],sibling[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++) {
                parent[k]=excluded[k][offset+count-N+(tid&(half-1))];
                sibling[k]=products[k][offset+(tid^half)];
            }
            parent[4]=sibling[4]=0;
            if(count==2){Load256(out,sibling);}else{qsb_field_mul(out,parent,sibling);}
            #pragma unroll
            for(int k=0;k<4;k++)excluded[k][offset-N+tid]=out[k];
        }
        offset-=count<<1;
        if((count<<1)>32)__syncthreads();else __syncwarp();
    }
    uint64_t parent[5],sibling[5];
    #pragma unroll
    for(int k=0;k<4;k++) {
        parent[k]=excluded[k][tid&(N/2-1)];
        sibling[k]=products[k][tid^(N/2)];
    }
    parent[4]=sibling[4]=0;
    if(N==2){Load256(value,sibling);}else{qsb_field_mul(value,parent,sibling);}
    value[4]=0;
}
