// Dependency-scoped barrier mechanism follows Calcutatatoraa95b1b9;
// applied here to the distinct public cofactor exclusion traversal.
// Public cofactor collective: tekkac, submission31e98e47, commit554fa24c.
#pragma once

// Register-carried operands across the collective (all three are exact index
// identities, see the block comments): the left operand a thread reads at level
// L+1 is the node that same thread wrote at level L.
#ifndef QSB_TREE_UP_CARRY
#define QSB_TREE_UP_CARRY 1    /* up-sweep left operand from the register it was just written from */
#endif
#ifndef QSB_TREE_DOWN_CARRY
#define QSB_TREE_DOWN_CARRY 1  /* down-sweep parent from the register it was just written from */
#endif
#ifndef QSB_TREE_LEAF_CARRY
#define QSB_TREE_LEAF_CARRY 1  /* final leaf parent likewise (requires QSB_TREE_DOWN_CARRY) */
#endif

// The caller supplies nonzero effective leaves (identity for unusable lanes).
// Preserve immutable products and accumulate exclusion products separately.
// All N lanes participate in every barrier; one block publishes one raw root.
template<int N> __device__ __forceinline__ void qsb_cofactor_prepare(
    uint64_t *value,uint64_t *roots,uint64_t (*products)[2*N],uint64_t (*excluded)[N]) {
    static_assert(N>=16 && !(N&(N-1)),"power-of-two tree");
    int tid=threadIdx.x;
    #pragma unroll
    for(int k=0;k<4;k++)products[k][tid]=value[k];
    __syncthreads();
    int offset=0;
#if QSB_TREE_UP_CARRY
    /* Level L writes products[offset_L+count_L+tid] and level L+1 reads
     * products[offset_{L+1}+tid] with offset_{L+1}=offset_L+count_L: the same
     * word, by the same thread, since half shrinks monotonically and an active
     * thread never becomes inactive and then active again. Level 0's left
     * operand is products[tid], which is this thread's own value (stored four
     * lines above). So the left operand is always a register this thread just
     * held; the shared store stays, because the *partner* still reads it. */
    uint64_t carry[5];
    #pragma unroll
    for(int k=0;k<4;k++)carry[k]=value[k];
#endif
    #pragma unroll 1
#if QSB_TREE_TOP2
    for(int count=N;count>2;count>>=1) {   /* stop below the root: the top pair is merged into the down-sweep */
#else
    for(int count=N;count>1;count>>=1) {
#endif
        int half=count>>1;
        if(tid<half) {
            uint64_t a[5],b[5],out[5];
#if QSB_TREE_UP_CARRY
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=carry[k];b[k]=products[k][offset+half+tid];}
#else
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=products[k][offset+tid];b[k]=products[k][offset+half+tid];}
#endif
            a[4]=b[4]=0;qsb_field_mul_sc(out,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)products[k][offset+count+tid]=out[k];
#if QSB_TREE_UP_CARRY
            #pragma unroll
            for(int k=0;k<4;k++)carry[k]=out[k];
#endif
        }
        offset+=count;
        if(count>2){if(half>32)__syncthreads();else __syncwarp();}
    }
#if QSB_TREE_DOWN_CARRY
    uint64_t dcarry[4];
#endif
#if QSB_TREE_TOP2
    /* P12: the top pair n0=products[2N-4], n1=products[2N-3] needs no separate root level
     * and no copy level: one warp-multiply gives the root n0*n1 (lane 4) together with the
     * four excluded products of the level below, E(c)=E(parent)*sibling with E(n0)=n1 and
     * E(n1)=n0 (lanes 0..3). Same operands in the same order as the two levels it replaces,
     * so the root and every excluded product are bit-identical. */
    if(tid<5) {
        uint64_t a[5],b[5],out[5];
        const int ia=tid<4 ? 2*N-4+((tid&1)^1) : 2*N-4;
        const int ib=tid<4 ? 2*N-8+(tid^2) : 2*N-3;
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=products[k][ia];b[k]=products[k][ib];}
        a[4]=b[4]=0;qsb_field_mul_sc(out,a,b);
        if(tid<4) {
            #pragma unroll
            for(int k=0;k<4;k++)excluded[k][N-8+tid]=out[k];
        } else {
            #pragma unroll
            for(int k=0;k<4;k++)roots[(size_t)blockIdx.x*4+k]=out[k];
        }
    }
    __syncwarp();
    offset=2*N-16;
    #pragma unroll 1
    for(int count=8;count<N;count<<=1) {
#else
    if(tid==0) {
        #pragma unroll
        for(int k=0;k<4;k++) {
            roots[(size_t)blockIdx.x*4+k]=products[k][2*N-2];
            excluded[k][N-2]=k==0?1:0;
        }
    }
    __syncwarp();
    offset=2*N-4;
    #pragma unroll 1
    for(int count=2;count<N;count<<=1) {
#endif
        int half=count>>1;
        if(tid<count) {
            uint64_t parent[5],sibling[5],out[5];
#if QSB_TREE_DOWN_CARRY
            /* Level L (width count) has thread tid<count write
             * excluded[offset_L-N+tid]; level L+1 has thread tid read
             * excluded[offset_{L+1}+count_{L+1}-N+(tid&(count_L-1))] with
             * offset_{L+1}=offset_L-2*count_L and count_{L+1}=2*count_L, i.e.
             * excluded[offset_L-N+tid] for tid<count_L -- the same word, same
             * thread. Threads count_L<=tid<count_{L+1} are new at this level
             * and take the shared path. The store stays for those readers. */
            const bool from_reg=(tid<half);
            #pragma unroll
            for(int k=0;k<4;k++) {
                parent[k]=from_reg?dcarry[k]:excluded[k][offset+count-N+(tid&(half-1))];
                sibling[k]=products[k][offset+(tid^half)];
            }
#else
            #pragma unroll
            for(int k=0;k<4;k++) {
                parent[k]=excluded[k][offset+count-N+(tid&(half-1))];
                sibling[k]=products[k][offset+(tid^half)];
            }
#endif
            parent[4]=sibling[4]=0;
            if(count==2){Load256(out,sibling);}else{qsb_field_mul_sc(out,parent,sibling);}
            #pragma unroll
            for(int k=0;k<4;k++)excluded[k][offset-N+tid]=out[k];
#if QSB_TREE_DOWN_CARRY
            #pragma unroll
            for(int k=0;k<4;k++)dcarry[k]=out[k];
#endif
        }
        offset-=count<<1;
        if((count<<1)>32)__syncthreads();else __syncwarp();
    }
    uint64_t parent[5],sibling[5];
#if QSB_TREE_DOWN_CARRY && QSB_TREE_LEAF_CARRY
    /* The last down-sweep level has width N/2 and offset N, so thread tid<N/2
     * wrote excluded[tid]; the leaf step reads excluded[tid&(N/2-1)], which is
     * excluded[tid] for exactly those threads. The upper half was never active
     * in that level and keeps the shared read. */
    #pragma unroll
    for(int k=0;k<4;k++) {
        parent[k]=(tid<N/2)?dcarry[k]:excluded[k][tid&(N/2-1)];
        sibling[k]=products[k][tid^(N/2)];
    }
#else
    #pragma unroll
    for(int k=0;k<4;k++) {
        parent[k]=excluded[k][tid&(N/2-1)];
        sibling[k]=products[k][tid^(N/2)];
    }
#endif
    parent[4]=sibling[4]=0;
    if(N==2){Load256(value,sibling);}else{qsb_field_mul_sc(value,parent,sibling);}
    value[4]=0;
}
