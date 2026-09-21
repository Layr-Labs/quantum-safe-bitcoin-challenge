// Dependency-scoped barrier mechanism follows Calcutatatoraa95b1b9;
// applied here to the distinct public cofactor exclusion traversal.
// Public cofactor collective: tekkac, submission31e98e47, commit554fa24c.
#pragma once

#ifndef QSB_TREE_LEAF_KEEP
#define QSB_TREE_LEAF_KEEP 1 /* the lower half keeps its partner leaf for the exclusion finish */
#endif
#if QSB_TREE_LEAF_KEEP && !(QSB_TREE_SEED_REG && QSB_TREE_DOWN_REG)
#error "the retained partner leaf is the peeled seed level's right operand"
#endif

/* QSB_TREE_LEVEL_NAMED: both traversal loops have compile-time trip counts (the up-sweep
 * runs from N/2 down to 4, the exclusion sweep from 8 up to N/2), so naming the levels
 * instead of rolling them turns every `offset`, `half` and `count` into a literal: the
 * shared addresses become immediate-offset LDS/STS, the loop counter and its two
 * comparisons disappear, and the per-level barrier choice (__syncthreads above 32 active
 * lanes, __syncwarp below, none at the top of the up-sweep) resolves at compile time
 * instead of being predicated at run time. The body, its operands, their order, the
 * destination slots and every barrier are exactly the ones the rolled form executed, so
 * the published products, the block root and the excluded products are bit-identical;
 * only loop control and address arithmetic are removed. -DQSB_TREE_LEVEL_NAMED=0
 * restores the rolled traversal with its runtime address chain. */
#ifndef QSB_TREE_LEVEL_NAMED
#define QSB_TREE_LEVEL_NAMED 1
#endif

/* QSB_TOP16_ACC_REG: the merged top's third and fourth waves read their left
 * operand out of the slot the same lane wrote one wave earlier, so the value
 * is already in a register and the shared load is redundant.
 *
 *   wave B, lane t<16   writes excluded[e+t];   wave C, lane t<16  reads excluded[e+t]
 *   wave B, lane 16..19 writes products[p4+t-16]; wave C, lane 16,17 read products[p4+t-16]
 *   wave C, lane t<16   writes excluded[e+t];   wave D, lane t<16  reads excluded[e+t]
 *   wave C, lane 16     writes products[p8];    wave D, lane 16    reads products[p8]
 *
 * Both index maps are the identity in `tid` over the lanes that stay active
 * (C runs lanes 0..17, D lanes 0..16), and no other lane and no other wave
 * writes those slots between the store and the load: wave C only ever writes
 * excluded[e+0..15] and products[p8+0..1], wave D only excluded[e+0..15] and
 * the block root. So the left operand of C is exactly this lane's wave-B
 * product and the left operand of D exactly its wave-C product, and carrying
 * it in a register removes four shared loads per active lane in each of the
 * two waves. The right operands still come from shared memory across lanes,
 * so both intra-warp barriers stay where they are, and every publication is
 * still performed: the exclusion arena and the two half roots hold the same
 * words for the sweep that follows. Factors, their order and the
 * association are untouched, so the sixteen excluded products and the block
 * root are bit-identical. -DQSB_TOP16_ACC_REG=0 restores the shared reload
 * in both waves.
 */
#ifndef QSB_TOP16_ACC_REG
#define QSB_TOP16_ACC_REG 1
#endif

/* QSB_TOP16_ACC_NOSTORE: once the two intermediate waves hand their product
 * over in a register, their stores into the exclusion arena have no reader
 * left. Slot excluded[e+t] is written by wave B, wave C and wave D, in that
 * order, always by lane t; the only reads of it inside the merged top are
 * the same lane's left operands in C and D, which QSB_TOP16_ACC_REG now
 * takes from the register, and the first read from outside the merged top is
 * the exclusion sweep's width-32 wave, which runs after wave D has
 * overwritten the slot (lanes 0..15 take that parent from the register
 * hand-off, lanes 16..31 read excluded[e+(tid&15)], i.e. wave D's value).
 * Dropping the B and C stores therefore removes eight shared stores per
 * active lane while leaving every word any reader observes untouched; the
 * four-limb product-side publications (products[p4..], products[p8..]) are
 * kept, because those are read across lanes by the next wave.
 * -DQSB_TOP16_ACC_NOSTORE=0 restores both intermediate publications. */
#ifndef QSB_TOP16_ACC_NOSTORE
#define QSB_TOP16_ACC_NOSTORE 1
#endif
#if QSB_TOP16_ACC_NOSTORE && !QSB_TOP16_ACC_REG
#error "the dropped intermediate publication relies on the register hand-off"
#endif

/* QSB_TOP16_SEED_UP: the merged top's first wave reads its left operand from
 * products[x+tid] for tid<8, and x+tid is exactly the node lane tid published
 * in the last up-sweep level -- with QSB_TREE_UP_REG that lane still holds
 * those four words in registers (the peeled seed level for N==32, the
 * width-32 level for N>=64; both write products[x+tid] for tid<16, and x is
 * 2N-32). Taking the operand from the register drops four shared loads for
 * the eight lanes of the wave and lets their multiply issue without waiting
 * on the load return; the right operand is the partner lane's node and still
 * comes from shared memory behind the publication barrier, and the node is
 * still published for wave B, which reads it across lanes. Same factors,
 * same order. -DQSB_TOP16_SEED_UP=0 restores the shared load. */
#ifndef QSB_TOP16_SEED_UP
#define QSB_TOP16_SEED_UP 1
#endif
#if QSB_TOP16_SEED_UP && !(QSB_TREE_UP_REG && QSB_TREE_SEED_REG)
#error "the seeded first wave takes the up-sweep's register node"
#endif

// The caller supplies nonzero effective leaves (identity for unusable lanes).
// Preserve immutable products and accumulate exclusion products separately.
// All N lanes participate in every barrier; one block publishes one raw root.
#if QSB_TREE_TOP16
/* Merged top of the cofactor tree (ported from public submission 464fad51's
 * top-16 merge, Codex/GPT 6 Astra design, onto the named-level traversal).
 * The last sixteen subtree roots x[0..15] live at products[2N-32 .. 2N-17].
 * The rolled traversal spends four up-sweep waves and three exclusion waves
 * on them; here each of four waves carries both roles, so one warp covers
 * the region in four multiply waves instead of seven. The factor sets and
 * the association ((x * P2) * P4) * P8 are the ones the down-sweep produces,
 * so every excluded product and the block root are bit-identical.
 * -DQSB_TREE_TOP16=0 restores the QSB_TREE_TOP2 top pair and the rolled
 * exclusion waves. */
template<int N> __device__ __forceinline__ void qsb_cofactor_top16(
    uint64_t *roots, uint64_t (*products)[2*N], uint64_t (*excluded)[N]
#if QSB_TREE_DOWN_REG
    , uint64_t *dn
#endif
#if QSB_TOP16_SEED_UP
    , const uint64_t *upn
#endif
    ) {
    static_assert(N>=32 && !(N&(N-1)), "power-of-two tree, N>=32");
    const int tid=threadIdx.x;
    const int x=2*N-32, p2=2*N-16, p4=2*N-8, p8=2*N-4, e=N-32;
#if QSB_TOP16_ACC_REG
    uint64_t acc[4];   /* this lane's product from the previous wave */
#endif
    if(tid<8) {                                    /* A: eight pair products */
        uint64_t a[5],b[5],o[5];
        #pragma unroll
#if QSB_TOP16_SEED_UP
        for(int k=0;k<4;k++){a[k]=upn[k];b[k]=products[k][x+tid+8];}
#else
        for(int k=0;k<4;k++){a[k]=products[k][x+tid];b[k]=products[k][x+tid+8];}
#endif
        a[4]=b[4]=0;qsb_field_mul_sc(o,a,b);
        #pragma unroll
        for(int k=0;k<4;k++)products[k][p2+tid]=o[k];
    }
    __syncwarp();
    if(tid<20) {                                   /* B: 16 exclusions + 4 quad products */
        uint64_t a[5],b[5],o[5];
        #pragma unroll
        for(int k=0;k<4;k++) {
            if(tid<16){a[k]=products[k][x+(tid^8)];b[k]=products[k][p2+((tid&7)^4)];}
            else{a[k]=products[k][p2+tid-16];b[k]=products[k][p2+tid-16+4];}
        }
        a[4]=b[4]=0;qsb_field_mul_sc(o,a,b);
        #pragma unroll
#if QSB_TOP16_ACC_NOSTORE
        for(int k=0;k<4;k++){if(tid>=16)products[k][p4+tid-16]=o[k];}
#else
        for(int k=0;k<4;k++){if(tid<16)excluded[k][e+tid]=o[k];else products[k][p4+tid-16]=o[k];}
#endif
#if QSB_TOP16_ACC_REG
        #pragma unroll
        for(int k=0;k<4;k++)acc[k]=o[k];
#endif
    }
    __syncwarp();
    if(tid<18) {                                   /* C: extend exclusions + two half roots */
        uint64_t a[5],b[5],o[5];
        #pragma unroll
        for(int k=0;k<4;k++) {
#if QSB_TOP16_ACC_REG
            if(tid<16){a[k]=acc[k];b[k]=products[k][p4+((tid&3)^2)];}
            else{a[k]=acc[k];b[k]=products[k][p4+tid-16+2];}
#else
            if(tid<16){a[k]=excluded[k][e+tid];b[k]=products[k][p4+((tid&3)^2)];}
            else{a[k]=products[k][p4+tid-16];b[k]=products[k][p4+tid-16+2];}
#endif
        }
        a[4]=b[4]=0;qsb_field_mul_sc(o,a,b);
        #pragma unroll
#if QSB_TOP16_ACC_NOSTORE
        for(int k=0;k<4;k++){if(tid>=16)products[k][p8+tid-16]=o[k];}
#else
        for(int k=0;k<4;k++){if(tid<16)excluded[k][e+tid]=o[k];else products[k][p8+tid-16]=o[k];}
#endif
#if QSB_TOP16_ACC_REG
        #pragma unroll
        for(int k=0;k<4;k++)acc[k]=o[k];
#endif
    }
    __syncwarp();
    if(tid<17) {                                   /* D: final exclusions + the root */
        uint64_t a[5],b[5],o[5];
        #pragma unroll
        for(int k=0;k<4;k++) {
#if QSB_TOP16_ACC_REG
            if(tid<16){a[k]=acc[k];b[k]=products[k][p8+((tid&1)^1)];}
            else{a[k]=acc[k];b[k]=products[k][p8+1];}
#else
            if(tid<16){a[k]=excluded[k][e+tid];b[k]=products[k][p8+((tid&1)^1)];}
            else{a[k]=products[k][p8];b[k]=products[k][p8+1];}
#endif
        }
        a[4]=b[4]=0;qsb_field_mul_sc(o,a,b);
        #pragma unroll
        for(int k=0;k<4;k++){
            if(tid<16){excluded[k][e+tid]=o[k];
#if QSB_TREE_DOWN_REG
                dn[k]=o[k];   /* seed the register hand-off for the count=32 wave */
#endif
            } else roots[(size_t)blockIdx.x*4+k]=o[k];
        }
    }
    __syncwarp();
}
#endif

template<int N> __device__ __forceinline__ void qsb_cofactor_prepare(
    uint64_t *value,uint64_t *roots,uint64_t (*products)[2*N],uint64_t (*excluded)[N]) {
    static_assert(N>=16 && !(N&(N-1)),"power-of-two tree");
    int tid=threadIdx.x;
    #pragma unroll
    for(int k=0;k<4;k++)products[k][tid]=value[k];
    __syncthreads();
#if QSB_TREE_UP_REG
    /* Level L of the up-sweep reads its left operand from products[offset+tid],
     * which is exactly the node lane tid published one level earlier (the next
     * level's base is the previous level's base plus its width). Carrying that
     * node in registers removes four shared loads per active lane per level;
     * the node is still published for the partner lanes and for the down-sweep,
     * so every operand, its order and every shared word are unchanged. */
    uint64_t up[4];
#endif
#if QSB_TREE_SEED_REG
    /* Level 0's left operand at lane tid is products[k][tid], which is the
     * value this lane has just published from its own registers. Peeling that
     * level lets it multiply the register copy directly: four shared loads per
     * lane disappear and only the right operand, which belongs to the partner
     * lane, depends on the publication barrier. The operands, their order and
     * the destination slots are the ones the rolled level used. */
#if QSB_TREE_LEAF_KEEP
    /* For tid < N/2 the sibling the exclusion finish needs is
     * products[tid^(N/2)] == products[(N>>1)+tid], which is precisely the
     * right operand this lane reads one line below at the peeled seed level.
     * Leaf slots 0..N-1 are never written again -- every up-sweep store lands
     * at index N or above, and the exclusion sweep writes the separate
     * `excluded` arena -- so the four words are still the ones the finish
     * would have reloaded, and keeping them takes four shared loads per lane
     * of the lower half out of the tail of the traversal. */
    uint64_t sib[4];
#endif
    if(tid<(N>>1)) {
        uint64_t a[5],b[5],out[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=value[k];b[k]=products[k][(N>>1)+tid];}
#if QSB_TREE_LEAF_KEEP
        #pragma unroll
        for(int k=0;k<4;k++)sib[k]=b[k];
#endif
        a[4]=b[4]=0;qsb_field_mul_sc(out,a,b);
        #pragma unroll
        for(int k=0;k<4;k++)products[k][N+tid]=out[k];
#if QSB_TREE_UP_REG
        #pragma unroll
        for(int k=0;k<4;k++)up[k]=out[k];
#endif
    }
    if((N>>1)>32)__syncthreads();else __syncwarp();
    int offset=N;
    const int first=N>>1;
#else
    int offset=0;
#if QSB_TREE_UP_REG
    /* Without the peeled level the first left operand is products[tid], the
     * word this lane has just published from `value`. */
    #pragma unroll
    for(int k=0;k<4;k++)up[k]=value[k];
#endif
    const int first=N;
#endif
#if QSB_TREE_LEVEL_NAMED
    #pragma unroll
#else
    #pragma unroll 1
#endif
#if QSB_TREE_TOP16
    for(int count=first;count>16;count>>=1) {  /* stop above the top 16: four merged waves cover them */
#elif QSB_TREE_TOP2
    for(int count=first;count>2;count>>=1) {   /* stop below the root: the top pair is merged into the down-sweep */
#else
    for(int count=first;count>1;count>>=1) {
#endif
        int half=count>>1;
        if(tid<half) {
            uint64_t a[5],b[5],out[5];
            #pragma unroll
#if QSB_TREE_UP_REG
            for(int k=0;k<4;k++){a[k]=up[k];b[k]=products[k][offset+half+tid];}
#else
            for(int k=0;k<4;k++){a[k]=products[k][offset+tid];b[k]=products[k][offset+half+tid];}
#endif
            a[4]=b[4]=0;qsb_field_mul_sc(out,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)products[k][offset+count+tid]=out[k];
#if QSB_TREE_UP_REG
            #pragma unroll
            for(int k=0;k<4;k++)up[k]=out[k];
#endif
        }
        offset+=count;
        if(count>2){if(half>32)__syncthreads();else __syncwarp();}
    }
#if QSB_TREE_DOWN_REG
    /* The exclusion sweep's parent slot for lane tid at width `count` is
     * excluded[offset-N+(tid mod half)], and the level above wrote exactly
     * that slot from lane (tid mod half): the lower half of every level
     * therefore re-reads the four words it has just produced. Holding them in
     * registers drops those loads; the publication is kept because the upper
     * half of the next level still reads them across lanes, so every barrier,
     * operand and shared word is unchanged. */
    uint64_t dn[4];
#endif
#if QSB_TREE_TOP16
    qsb_cofactor_top16<N>(roots,products,excluded
#if QSB_TREE_DOWN_REG
        ,dn
#endif
#if QSB_TOP16_SEED_UP
        ,up
#endif
        );
    offset=2*N-64;
#if QSB_TREE_LEVEL_NAMED
    #pragma unroll
#else
    #pragma unroll 1
#endif
    for(int count=32;count<N;count<<=1) {
#elif QSB_TREE_TOP2
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
#if QSB_TREE_DOWN_REG
            #pragma unroll
            for(int k=0;k<4;k++)dn[k]=out[k];
#endif
        } else {
            #pragma unroll
            for(int k=0;k<4;k++)roots[(size_t)blockIdx.x*4+k]=out[k];
        }
    }
    __syncwarp();
    offset=2*N-16;
#if QSB_TREE_LEVEL_NAMED
    #pragma unroll
#else
    #pragma unroll 1
#endif
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
#if QSB_TREE_LEVEL_NAMED
    #pragma unroll
#else
    #pragma unroll 1
#endif
    for(int count=2;count<N;count<<=1) {
#endif
        int half=count>>1;
        if(tid<count) {
            uint64_t parent[5],sibling[5],out[5];
#if QSB_TREE_DOWN_REG
            if(tid<half) {
                #pragma unroll
                for(int k=0;k<4;k++)parent[k]=dn[k];
            } else {
                #pragma unroll
                for(int k=0;k<4;k++)parent[k]=excluded[k][offset+count-N+(tid&(half-1))];
            }
            #pragma unroll
            for(int k=0;k<4;k++)sibling[k]=products[k][offset+(tid^half)];
#else
            #pragma unroll
            for(int k=0;k<4;k++) {
                parent[k]=excluded[k][offset+count-N+(tid&(half-1))];
                sibling[k]=products[k][offset+(tid^half)];
            }
#endif
            parent[4]=sibling[4]=0;
#if QSB_TREE_DEADCUT
            /* The merged top starts this sweep at count==8 (TOP2) or count==32
             * (TOP16), so the count==2 copy leaf of the rolled form is unreachable. */
            qsb_field_mul_sc(out,parent,sibling);
#else
            if(count==2){Load256(out,sibling);}else{qsb_field_mul_sc(out,parent,sibling);}
#endif
            #pragma unroll
            for(int k=0;k<4;k++)excluded[k][offset-N+tid]=out[k];
#if QSB_TREE_DOWN_REG
            #pragma unroll
            for(int k=0;k<4;k++)dn[k]=out[k];
#endif
        }
        offset-=count<<1;
        if((count<<1)>32)__syncthreads();else __syncwarp();
    }
    uint64_t parent[5],sibling[5];
#if QSB_TREE_DOWN_REG
    /* The last sweep level leaves offset == N, so its outputs are
     * excluded[0..N/2-1] and the final parent index tid&(N/2-1) is this lane's
     * own output for the lower half of the block. */
    if(tid<(N>>1)) {
        #pragma unroll
        for(int k=0;k<4;k++)parent[k]=dn[k];
#if QSB_TREE_LEAF_KEEP
        #pragma unroll
        for(int k=0;k<4;k++)sibling[k]=sib[k];
#endif
    } else {
        #pragma unroll
        for(int k=0;k<4;k++)parent[k]=excluded[k][tid&(N/2-1)];
#if QSB_TREE_LEAF_KEEP
        #pragma unroll
        for(int k=0;k<4;k++)sibling[k]=products[k][tid^(N/2)];
#endif
    }
#if !QSB_TREE_LEAF_KEEP
    #pragma unroll
    for(int k=0;k<4;k++)sibling[k]=products[k][tid^(N/2)];
#endif
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
