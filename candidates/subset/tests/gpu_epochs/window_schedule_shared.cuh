// Derived from odinfree's GPU-epoch consumer in submission 0db6e203.
// Only the first block depends on the epoch remainder. The second block's
// expanded schedule is shared by every epoch with the same window choice.
#pragma once
#ifndef QSB_950_PACK
#define QSB_950_PACK 1
#endif
#if QSB_FIRST_COMPACT8 && QSB_SE_WINDOWS == 128
#define QSB_FIRST_SLOTS 8
#else
#define QSB_FIRST_SLOTS (QSB_SE_WINDOWS==256?64:16)
#endif
#ifndef QSB_SHA_UNROLL_CONST
#define QSB_SHA_UNROLL_CONST 1
#endif   /* first-block classes per epoch in d_first */
__device__ uint32_t QSB_WINDOW_FIRST[14][QSB_SE_PER_EPOCH];
#if QSB_SECOND_COMPACT64 && QSB_SHA_SCHED_V4 && !QSB_SHA_WROLL_PIPE && QSB_SE_WINDOWS == 128
#define QSB_WINDOW_SECOND_CAPACITY 64
#else
#define QSB_WINDOW_SECOND_CAPACITY QSB_SE_PER_EPOCH
#endif
#define QSB_WINDOW_SECOND_PITCH (QSB_WINDOW_SECOND_CAPACITY + QSB_WINDOW_ROW_PAD)
#if QSB_SHA_SCHED_V4
/* QSB_SHA_SCHED_V4: W+K word r of second-block slot s at [r/4][s].{x,y,z,w}, so a block's 8 rounds
 * read two 16 B words per lane instead of eight 4 B words; same 32 KiB, same values. */
#if QSB_SHA_WROLL_PIPE
__device__ uint4 QSB_WINDOW_SECOND[17][QSB_WINDOW_SECOND_PITCH];   /* QSB_SHA_WROLL_PIPE: row 16 is zero padding the pipelined roll reads once and discards (the host uploads rows 0..15) */
#else
__device__ uint4 QSB_WINDOW_SECOND[16][QSB_WINDOW_SECOND_PITCH];
#endif
#define QSB_WSEC_V4(r, slot) (QSB_WINDOW_SECOND[(r) >> 2][slot])
#else
__device__ uint32_t QSB_WINDOW_SECOND[64][QSB_SE_PER_EPOCH];
#endif
__device__ uint32_t QSB_WINDOW_CLASS[QSB_SE_PER_EPOCH];
__device__ uint32_t QSB_FIRST_CLASS[QSB_SE_PER_EPOCH];
/* Public PR950: lossless (first_slot << 16) | second_slot. */
__device__ uint32_t QSB_LANE_CLASS[QSB_SE_PER_EPOCH];
__device__ uint32_t QSB_FIRST_UNIQUE[14][QSB_SE_WINDOWS==256?256:QSB_FIRST_SLOTS];
__device__ __constant__ int QSB_FIRST_COUNT;
static int qsb_first_class_count=0;
/* Host copy of the distinct first-block words (words 2..15 per class), kept for the host producers. */
static uint32_t qsb_first_unique_host[QSB_FIRST_SLOTS][14];

static uint32_t qsb_window_second_key(const uint8_t w[3]) {
    uint32_t key=0;
    for(int i=12,n=0;i>=0 && n<5;i--)
        if(i!=w[0]-137 && i!=w[1]-137 && i!=w[2]-137){key=(key<<4)|i;n++;}
    return key;
}

static uint32_t qsb_window_first_key(const uint8_t w[3]) {
    uint32_t key=0;
    for(int i=0,n=0;i<13 && n<6;i++)
        if(i!=w[0]-137 && i!=w[1]-137 && i!=w[2]-137){key=(key<<4)|i;n++;}
    return key;
}

static int qsb_prepare_window_schedule(const uint8_t *rows,
        const uint8_t windows[QSB_SE_PER_EPOCH][3], const uint32_t *constant) {
    uint32_t first[14][QSB_SE_PER_EPOCH], second[64][QSB_WINDOW_SECOND_CAPACITY]={}, round_k[64];
    uint32_t classes[QSB_SE_PER_EPOCH], unique[QSB_SE_PER_EPOCH][16];
    uint32_t first_classes[QSB_SE_PER_EPOCH], first_unique[QSB_SE_PER_EPOCH][14], transposed[14][QSB_SE_WINDOWS==256?256:QSB_FIRST_SLOTS]={};
    int first_distinct=0;
    int distinct=0;
    if (QSB_FROM_SYMBOL(round_k, K, sizeof(round_k)) != cudaSuccess) return 1;
    for (int lane=0; lane<QSB_SE_PER_EPOCH; lane++) {
        uint8_t bytes[128]={};
        int pos=8, sel=0;
        for (int i=137; i<150; i++) {
            if (sel<3 && windows[lane][sel]==i) { sel++; continue; }
            memcpy(bytes+pos, rows+10*i, 10); pos+=10;
        }
        if (pos!=108 || sel!=3) return 1;
        for (int j=0; j<5; j++)
            for (int b=0; b<4; b++) bytes[pos++]=(uint8_t)(constant[j]>>(24-8*b));
        if (pos!=128) return 1;
        uint32_t words[32], expanded[64];
        for (int j=0; j<32; j++)
            words[j]=((uint32_t)bytes[4*j]<<24)|((uint32_t)bytes[4*j+1]<<16)|
                     ((uint32_t)bytes[4*j+2]<<8)|bytes[4*j+3];
        for (int j=0; j<14; j++) first[j][lane]=words[j+2];
        int first_slot=0;
        while(first_slot<first_distinct && memcmp(first_unique[first_slot],words+2,56))first_slot++;
        if(first_slot==first_distinct){memcpy(first_unique[first_distinct],words+2,56);first_distinct++;}
        first_classes[lane]=first_slot;
        int slot=0;
        while(slot<distinct && memcmp(unique[slot],words+16,64))slot++;
        if(slot==distinct){
            if(distinct>=QSB_WINDOW_SECOND_CAPACITY)return 1;
            memcpy(unique[distinct],words+16,64);distinct++;
        }
        classes[lane]=slot;
        for (int j=0; j<16; j++) expanded[j]=words[j+16];
        for (int j=16; j<64; j++) {
            uint32_t x=expanded[j-15], y=expanded[j-2];
            uint32_t a=qsb_host_rotr(x,7)^qsb_host_rotr(x,18)^(x>>3);
            uint32_t b=qsb_host_rotr(y,17)^qsb_host_rotr(y,19)^(y>>10);
            expanded[j]=expanded[j-16]+a+expanded[j-7]+b;
        }
        for (int j=0; j<64; j++) second[j][slot]=expanded[j]+round_k[j];
    }
    printf("Window schedule classes: first=%d second=%d of %d\n",first_distinct,distinct,QSB_SE_PER_EPOCH);
    qsb_first_class_count=first_distinct;
    if(first_distinct>QSB_FIRST_SLOTS)return 1;
    for(int slot=0;slot<first_distinct;slot++)memcpy(qsb_first_unique_host[slot],first_unique[slot],56);
    for(int slot=0;slot<first_distinct;slot++)
        for(int j=0;j<14;j++)transposed[j][slot]=first_unique[slot][j];
    if(QSB_TO_SYMBOL(QSB_FIRST_COUNT,&first_distinct,sizeof(first_distinct))!=cudaSuccess)return 1;
    if(QSB_TO_SYMBOL(QSB_FIRST_CLASS,first_classes,sizeof(first_classes))!=cudaSuccess)return 1;
    if(QSB_TO_SYMBOL(QSB_FIRST_UNIQUE,transposed,sizeof(transposed))!=cudaSuccess)return 1;
#if QSB_950_PACK
    {   uint32_t packed[QSB_SE_PER_EPOCH];
        for(int lane=0;lane<QSB_SE_PER_EPOCH;lane++)
            packed[lane]=(first_classes[lane]<<16)|classes[lane];
        if(QSB_TO_SYMBOL(QSB_LANE_CLASS,packed,sizeof(packed))!=cudaSuccess)return 1;
    }
#endif
    if (QSB_TO_SYMBOL(QSB_WINDOW_CLASS,classes,sizeof(classes))!=cudaSuccess) return 1;
    if (QSB_TO_SYMBOL(QSB_WINDOW_FIRST,first,sizeof(first))!=cudaSuccess) return 1;
#if QSB_SHA_SCHED_V4
    {   /* QSB_SHA_SCHED_V4: the same words in the [r/4][slot].{x,y,z,w} layout */
        static uint4 second4[16][QSB_WINDOW_SECOND_PITCH];
        for(int r=0;r<64;r+=4)
            for(int slot=0;slot<QSB_WINDOW_SECOND_CAPACITY;slot++){
                second4[r/4][slot].x=second[r][slot];   second4[r/4][slot].y=second[r+1][slot];
                second4[r/4][slot].z=second[r+2][slot]; second4[r/4][slot].w=second[r+3][slot];
            }
        return QSB_TO_SYMBOL(QSB_WINDOW_SECOND,second4,sizeof(second4))==cudaSuccess?0:1;
    }
#else
    return QSB_TO_SYMBOL(QSB_WINDOW_SECOND,second,sizeof(second))==cudaSuccess?0:1;
#endif
}

/* First-block states for every (epoch, class) of a launch, computed as its own
 * producer stage: one block per epoch, one thread per first-block class. The
 * consumer used to spend a thread barrier plus one compression of block time
 * per epoch while 54 leader lanes built these states and the other lanes
 * waited; now each lane reads its class state (32 bytes). */
/* Flat mapping: one thread per (epoch, class) over full 256-thread blocks, instead of one
 * 54-thread block per epoch (two warps, 10 idle lanes, and a block launch per epoch). */
__global__ void __launch_bounds__(256) kernel_build_first_flat(const epoch_desc_t * __restrict__ d_epochs,
        uint32_t * __restrict__ d_first, unsigned n_epochs, unsigned classes) {
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned e = t / classes, c = t - e * classes;
    if (e >= n_epochs) return;
    const epoch_desc_t *ep = d_epochs + e;
    uint32_t st[8], W[16];
    #pragma unroll
    for(int j=0;j<8;j++)st[j]=ep->mid[j];
    W[0]=ep->remW[0];W[1]=ep->remW[1];
    #pragma unroll
    for(int j=2;j<16;j++)W[j]=QSB_FIRST_UNIQUE[j-2][c];
    _SHA256Transform(st,W);
    const size_t base=((size_t)e*QSB_FIRST_SLOTS+(size_t)c)*8;
    #pragma unroll
    for(int j=0;j<8;j++)d_first[base+j]=st[j];
}
#if 0   /* superseded by kernel_build_first_flat; kept out of the JIT-compiled module */
__global__ void kernel_build_first(const epoch_desc_t * __restrict__ d_epochs,
        uint32_t * __restrict__ d_first) {
    const epoch_desc_t *ep = d_epochs + blockIdx.x;
    const int c = threadIdx.x;
    uint32_t st[8], W[16];
    #pragma unroll
    for(int j=0;j<8;j++)st[j]=ep->mid[j];
    W[0]=ep->remW[0];W[1]=ep->remW[1];
    #pragma unroll
    for(int j=2;j<16;j++)W[j]=QSB_FIRST_UNIQUE[j-2][c];
    _SHA256Transform(st,W);
    const size_t base=((size_t)blockIdx.x*QSB_FIRST_SLOTS+(size_t)c)*8;
    #pragma unroll
    for(int j=0;j<8;j++)d_first[base+j]=st[j];
}
#endif

__device__ __forceinline__ void qsb_scheduled_window_hash(uint32_t *state,
        const epoch_desc_t *epoch, int lane, const uint32_t *first) {
    (void)epoch;
    const int first_slot=QSB_FIRST_CLASS[lane];
    #pragma unroll
    for(int j=0;j<8;j++)state[j]=first[first_slot*8+j];
    int slot=QSB_WINDOW_CLASS[lane];
    uint32_t a=state[0],b=state[1],c=state[2],d=state[3];
    uint32_t e=state[4],f=state[5],g=state[6],h=state[7],t1,t2;
    #pragma unroll 1
    for (int r=0; r<64; r+=8) {
#if QSB_SHA_SCHED_V4
        const uint4 wa=QSB_WSEC_V4(r,slot), wb=QSB_WSEC_V4(r+4,slot);
        S2Round(a,b,c,d,e,f,g,h,0,wa.x);
        S2Round(h,a,b,c,d,e,f,g,0,wa.y);
        S2Round(g,h,a,b,c,d,e,f,0,wa.z);
        S2Round(f,g,h,a,b,c,d,e,0,wa.w);
        S2Round(e,f,g,h,a,b,c,d,0,wb.x);
        S2Round(d,e,f,g,h,a,b,c,0,wb.y);
        S2Round(c,d,e,f,g,h,a,b,0,wb.z);
        S2Round(b,c,d,e,f,g,h,a,0,wb.w);
#else
        S2Round(a,b,c,d,e,f,g,h,0,QSB_WINDOW_SECOND[r][slot]);
        S2Round(h,a,b,c,d,e,f,g,0,QSB_WINDOW_SECOND[r+1][slot]);
        S2Round(g,h,a,b,c,d,e,f,0,QSB_WINDOW_SECOND[r+2][slot]);
        S2Round(f,g,h,a,b,c,d,e,0,QSB_WINDOW_SECOND[r+3][slot]);
        S2Round(e,f,g,h,a,b,c,d,0,QSB_WINDOW_SECOND[r+4][slot]);
        S2Round(d,e,f,g,h,a,b,c,0,QSB_WINDOW_SECOND[r+5][slot]);
        S2Round(c,d,e,f,g,h,a,b,0,QSB_WINDOW_SECOND[r+6][slot]);
        S2Round(b,c,d,e,f,g,h,a,0,QSB_WINDOW_SECOND[r+7][slot]);
#endif
    }
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;
    state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
#if QSB_SHA_UNROLL_CONST
    // Fully unrolled constant blocks: K+W becomes a constant-bank operand of the
    // round adds (no LDC, no loop counter). Same arithmetic, same order.
    qsb_compress_constant<0>(state);
    qsb_compress_constant<1>(state);
    qsb_compress_constant<2>(state);
    qsb_compress_constant<3>(state);
#else
    qsb_compress_constant_rolled(state);
#endif
}

#if ZLAB_DUAL_EPOCH_SHA
#ifndef QSB_PAIR_SHA_UNROLL_WINDOW
#if defined(QSB_SHA_WROLL_PIPE) && QSB_SHA_WROLL_PIPE
#define QSB_PAIR_SHA_UNROLL_WINDOW 0   /* the pipelined roll replaces the unrolled window loop */
#else
#define QSB_PAIR_SHA_UNROLL_WINDOW 1
#endif
#endif
#ifndef QSB_PAIR_SHA_UNROLL_CONST
#define QSB_PAIR_SHA_UNROLL_CONST 1
#endif
/* Keep the four-block loop unrolled; roll only its 8-round inner loop. */
#ifndef QSB_PAIR_SHA_UNROLL_CONST_INNER
#define QSB_PAIR_SHA_UNROLL_CONST_INNER 0
#endif
#if QSB_SHA_WROLL_PIPE && QSB_PAIR_SHA_UNROLL_WINDOW
#error "QSB_SHA_WROLL_PIPE replaces the window loop: set QSB_PAIR_SHA_UNROLL_WINDOW 0 with it"
#endif
#ifndef QSB_SHA_UEXIT
#define QSB_SHA_UEXIT 0   /* lane SHA probe (X-SHA): 0 = the base loop header, byte for byte */
#endif
#if QSB_CONST_CALLEE
/* QSB_CONST_CALLEE (tree.cu): the four constant blocks in their own __noinline__ callee
 * (16 state words in and out as ABI registers), block loop unrolled, 8-round loop rolled and C-indexed, so ptxas
 * keeps the K+W walk on the uniform datapath (ULDC) instead of per-thread LDC. Same words, rounds and order. */
struct QsbPairS16 { uint32_t w[16]; };
#if QSB_SHA_W0FOLD
/* QSB_SHA_W0FOLD (tree.cu): w[0] and w[8] come back without block 154's word-0 feed-forward; w[16] and w[17] carry
 * the final a of A and B, which the literal-K outer block adds inside round 0 and W16. */
struct QsbPairS18 { uint32_t w[18]; };
#define QSB_CC_RET QsbPairS18
#else
#define QSB_CC_RET QsbPairS16
#endif
__device__ __noinline__ QSB_CC_RET qsb_pair_const4(QsbPairS16 s) {
    uint32_t a0,b0,c0,d0,e0,f0,g0,h0,a1,b1,c1,d1,e1,f1,g1,h1,t1,t2;
#if QSB_SHA_W0FOLD
    uint32_t fa0=0,fa1=0;
#endif
    const uint4 *K=reinterpret_cast<const uint4*>(&QSB_CONST_SCHEDULE[0][0]);
    #pragma unroll
    for(int block=0;block<4;block++){
        a0=s.w[0];b0=s.w[1];c0=s.w[2];d0=s.w[3];e0=s.w[4];f0=s.w[5];g0=s.w[6];h0=s.w[7];
        a1=s.w[8];b1=s.w[9];c1=s.w[10];d1=s.w[11];e1=s.w[12];f1=s.w[13];g1=s.w[14];h1=s.w[15];
#if QSB_SHA_UEXIT == 1
        /* QSB_SHA_UEXIT 1 (lane SHA probe): the byte offset is the only induction variable, so the exit test can
         * sit on the uniform datapath with the ULDC address. Same words, rounds and order. */
        #pragma unroll 1
        for(uint32_t off=0;off!=256u;off+=32u){
            const uint4 *kr=reinterpret_cast<const uint4*>(reinterpret_cast<const char*>(K)+block*256+off);
            const uint4 ka=kr[0], kb=kr[1];
#elif QSB_SHA_UEXIT == 2
        /* QSB_SHA_UEXIT 2 (lane SHA probe): do-while on the row index with an equality exit. */
        int q=0;
        #pragma unroll 1
        do{
            const uint4 ka=K[block*16+q], kb=K[block*16+q+1];
#elif QSB_SHA_UEXIT == 3
        /* QSB_SHA_UEXIT 3 (lane SHA probe): pointer walk with an equality exit on the end pointer. */
        #pragma unroll 1
        for(const uint4 *kr=K+block*16;kr!=K+block*16+16;kr+=2){
            const uint4 ka=kr[0], kb=kr[1];
#elif QSB_SHA_UEXIT == 4
        /* QSB_SHA_UEXIT 4 (lane SHA probe): the 8-round loop fully unrolled inside the callee, so K+W are
         * immediate constant-bank operands (no ULDC, no loop control, no per-block copies). Code-size arm. */
        #pragma unroll
        for(int q=0;q<16;q+=2){
            const uint4 ka=K[block*16+q], kb=K[block*16+q+1];
#else
        #pragma unroll 1
        for(int q=0;q<16;q+=2){
            const uint4 ka=K[block*16+q], kb=K[block*16+q+1];
#endif
            {const uint32_t w=ka.x;S2Round(a0,b0,c0,d0,e0,f0,g0,h0,0,w);S2Round(a1,b1,c1,d1,e1,f1,g1,h1,0,w);}
            {const uint32_t w=ka.y;S2Round(h0,a0,b0,c0,d0,e0,f0,g0,0,w);S2Round(h1,a1,b1,c1,d1,e1,f1,g1,0,w);}
            {const uint32_t w=ka.z;S2Round(g0,h0,a0,b0,c0,d0,e0,f0,0,w);S2Round(g1,h1,a1,b1,c1,d1,e1,f1,0,w);}
            {const uint32_t w=ka.w;S2Round(f0,g0,h0,a0,b0,c0,d0,e0,0,w);S2Round(f1,g1,h1,a1,b1,c1,d1,e1,0,w);}
            {const uint32_t w=kb.x;S2Round(e0,f0,g0,h0,a0,b0,c0,d0,0,w);S2Round(e1,f1,g1,h1,a1,b1,c1,d1,0,w);}
            {const uint32_t w=kb.y;S2Round(d0,e0,f0,g0,h0,a0,b0,c0,0,w);S2Round(d1,e1,f1,g1,h1,a1,b1,c1,0,w);}
            {const uint32_t w=kb.z;S2Round(c0,d0,e0,f0,g0,h0,a0,b0,0,w);S2Round(c1,d1,e1,f1,g1,h1,a1,b1,0,w);}
            {const uint32_t w=kb.w;S2Round(b0,c0,d0,e0,f0,g0,h0,a0,0,w);S2Round(b1,c1,d1,e1,f1,g1,h1,a1,0,w);}
#if QSB_SHA_UEXIT == 2
            q+=2;
        }while(q!=16);
#else
        }
#endif
#if QSB_SHA_W0FOLD
        if(block<3){ s.w[0]+=a0; s.w[8]+=a1; } else { fa0=a0; fa1=a1; }
        s.w[1]+=b0;s.w[2]+=c0;s.w[3]+=d0;s.w[4]+=e0;s.w[5]+=f0;s.w[6]+=g0;s.w[7]+=h0;
        s.w[9]+=b1;s.w[10]+=c1;s.w[11]+=d1;s.w[12]+=e1;s.w[13]+=f1;s.w[14]+=g1;s.w[15]+=h1;
#else
        s.w[0]+=a0;s.w[1]+=b0;s.w[2]+=c0;s.w[3]+=d0;s.w[4]+=e0;s.w[5]+=f0;s.w[6]+=g0;s.w[7]+=h0;
        s.w[8]+=a1;s.w[9]+=b1;s.w[10]+=c1;s.w[11]+=d1;s.w[12]+=e1;s.w[13]+=f1;s.w[14]+=g1;s.w[15]+=h1;
#endif
    }
#if QSB_SHA_W0FOLD
    QsbPairS18 o;
    #pragma unroll
    for(int j=0;j<16;j++) o.w[j]=s.w[j];
    o.w[16]=fa0; o.w[17]=fa1;
    return o;
#else
    return s;
#endif
}
#endif
/* Paired epoch SHA from dukemawex 4cea5476 (origin e771d5c7 / e9812a9). The paired consumer has the same lane (and therefore the same scheduled
 * second block and constant suffix) in both epochs.  Load each schedule word
 * once and advance two independent SHA-256 states with it. */
#if QSB_ROOT_FILL
/* QSB_ROOT_FILL (tree.cu): PART 0 = the whole paired hash below; 1 = the first-state load and the window block only
 * (stateA/stateB leave with the window block's chaining values); 2 = the four constant blocks only, from stateA/stateB. */
template<int PART=0>
#endif
__device__ __forceinline__ void qsb_scheduled_window_hash_pair(
    uint32_t *stateA, uint32_t *stateB, int lane,
    const uint32_t *firstA, const uint32_t *firstB
#if QSB_SHA_W0FOLD
    , uint32_t *w0x   /* QSB_SHA_W0FOLD: block 154's final a for A and B; stateA[0], stateB[0] exclude it */
#endif
    ) {
#if QSB_950_PACK
    const uint32_t lane_rec=QSB_LANE_CLASS[lane];
    const int first_slot=(int)(lane_rec>>16);
    const int slot=(int)(lane_rec&0xffffu);
#else
    const int first_slot=QSB_FIRST_CLASS[lane];
    const int slot=QSB_WINDOW_CLASS[lane];
#endif
#if QSB_ROOT_FILL
    if(PART!=2){
#endif
#if QSB_950_PACK
    {   const uint4 *pA=reinterpret_cast<const uint4*>(firstA+first_slot*8);
        const uint4 *pB=reinterpret_cast<const uint4*>(firstB+first_slot*8);
        const uint4 vA0=pA[0], vA1=pA[1], vB0=pB[0], vB1=pB[1];
        stateA[0]=vA0.x;stateA[1]=vA0.y;stateA[2]=vA0.z;stateA[3]=vA0.w;
        stateA[4]=vA1.x;stateA[5]=vA1.y;stateA[6]=vA1.z;stateA[7]=vA1.w;
        stateB[0]=vB0.x;stateB[1]=vB0.y;stateB[2]=vB0.z;stateB[3]=vB0.w;
        stateB[4]=vB1.x;stateB[5]=vB1.y;stateB[6]=vB1.z;stateB[7]=vB1.w;
    }
#else
    #pragma unroll
    for(int j=0;j<8;j++){
        stateA[j]=firstA[first_slot*8+j];
        stateB[j]=firstB[first_slot*8+j];
    }
#endif
#if QSB_ROOT_FILL
    }
#endif
    uint32_t a0,b0,c0,d0,e0,f0,g0,h0;
    uint32_t a1,b1,c1,d1,e1,f1,g1,h1,t1,t2;
#define QSB_PAIR_STATE_LOAD() do { \
    a0=stateA[0];b0=stateA[1];c0=stateA[2];d0=stateA[3]; \
    e0=stateA[4];f0=stateA[5];g0=stateA[6];h0=stateA[7]; \
    a1=stateB[0];b1=stateB[1];c1=stateB[2];d1=stateB[3]; \
    e1=stateB[4];f1=stateB[5];g1=stateB[6];h1=stateB[7]; \
} while(0)
#define QSB_PAIR_STATE_ADD() do { \
    stateA[0]+=a0;stateA[1]+=b0;stateA[2]+=c0;stateA[3]+=d0; \
    stateA[4]+=e0;stateA[5]+=f0;stateA[6]+=g0;stateA[7]+=h0; \
    stateB[0]+=a1;stateB[1]+=b1;stateB[2]+=c1;stateB[3]+=d1; \
    stateB[4]+=e1;stateB[5]+=f1;stateB[6]+=g1;stateB[7]+=h1; \
} while(0)
#if QSB_ROOT_FILL
    if(PART!=2){
#endif
    QSB_PAIR_STATE_LOAD();
#if QSB_SHA_WROLL_PIPE
    /* QSB_SHA_WROLL_PIPE (tree.cu): the rolled window block with its loads one
     * half-trip ahead. wb (rounds r+4..r+7) is issued at the top of the trip, the next trip's wa (rounds r+8..r+11)
     * after rounds r..r+3 have consumed wa, so each load leads its first use by four rounds, as in the unrolled
     * form. The pointer walks the rows; the last trip's extra wa load reads padding row 16 (discarded). Same
     * words, same rounds, same order. */
    {
        const uint4 *wp=&QSB_WINDOW_SECOND[0][slot];
        uint4 wa=wp[0];
        #pragma unroll 1
        for(int r=0;r<64;r+=8){
            const uint4 wb=wp[QSB_SE_PER_EPOCH];
            {const uint32_t w=wa.x;S2Round(a0,b0,c0,d0,e0,f0,g0,h0,0,w);S2Round(a1,b1,c1,d1,e1,f1,g1,h1,0,w);}
            {const uint32_t w=wa.y;S2Round(h0,a0,b0,c0,d0,e0,f0,g0,0,w);S2Round(h1,a1,b1,c1,d1,e1,f1,g1,0,w);}
            {const uint32_t w=wa.z;S2Round(g0,h0,a0,b0,c0,d0,e0,f0,0,w);S2Round(g1,h1,a1,b1,c1,d1,e1,f1,0,w);}
            {const uint32_t w=wa.w;S2Round(f0,g0,h0,a0,b0,c0,d0,e0,0,w);S2Round(f1,g1,h1,a1,b1,c1,d1,e1,0,w);}
            wa=(wp+=2*QSB_SE_PER_EPOCH, *wp);
            {const uint32_t w=wb.x;S2Round(e0,f0,g0,h0,a0,b0,c0,d0,0,w);S2Round(e1,f1,g1,h1,a1,b1,c1,d1,0,w);}
            {const uint32_t w=wb.y;S2Round(d0,e0,f0,g0,h0,a0,b0,c0,0,w);S2Round(d1,e1,f1,g1,h1,a1,b1,c1,0,w);}
            {const uint32_t w=wb.z;S2Round(c0,d0,e0,f0,g0,h0,a0,b0,0,w);S2Round(c1,d1,e1,f1,g1,h1,a1,b1,0,w);}
            {const uint32_t w=wb.w;S2Round(b0,c0,d0,e0,f0,g0,h0,a0,0,w);S2Round(b1,c1,d1,e1,f1,g1,h1,a1,0,w);}
        }
    }
    if(0)   /* the rolled loop replaces the loop below, which stays in the source as dead code (this form is the
             * measured one; an #else form reorders two moves) */
#endif
#if QSB_WINDOW_GROUP_ROUNDS
    /* Keep a fixed group of rounds inline, and roll between groups. The inner
     * group always ends on an eight-round state rotation boundary. This uses
     * all original schedule words once, in their original order. */
    #pragma unroll 1
    for(int group=0;group<64;group+=QSB_WINDOW_GROUP_ROUNDS){
        #pragma unroll
        for(int offset=0;offset<QSB_WINDOW_GROUP_ROUNDS;offset+=8){
            const int r=group+offset;
#else
#if QSB_PAIR_SHA_UNROLL_WINDOW   /* exact: same rounds, no loop counter, loads can be hoisted */
    #pragma unroll
#else
    #pragma unroll 1
#endif
    for(int r=0;r<64;r+=8){
#endif
#if QSB_SHA_SCHED_V4
        /* QSB_SHA_SCHED_V4: two 16 B loads carry the 8 rounds' W+K words; the same words in the same order */
        const uint4 wa=QSB_WSEC_V4(r,slot), wb=QSB_WSEC_V4(r+4,slot);
        {const uint32_t w=wa.x;S2Round(a0,b0,c0,d0,e0,f0,g0,h0,0,w);S2Round(a1,b1,c1,d1,e1,f1,g1,h1,0,w);}
        {const uint32_t w=wa.y;S2Round(h0,a0,b0,c0,d0,e0,f0,g0,0,w);S2Round(h1,a1,b1,c1,d1,e1,f1,g1,0,w);}
        {const uint32_t w=wa.z;S2Round(g0,h0,a0,b0,c0,d0,e0,f0,0,w);S2Round(g1,h1,a1,b1,c1,d1,e1,f1,0,w);}
        {const uint32_t w=wa.w;S2Round(f0,g0,h0,a0,b0,c0,d0,e0,0,w);S2Round(f1,g1,h1,a1,b1,c1,d1,e1,0,w);}
        {const uint32_t w=wb.x;S2Round(e0,f0,g0,h0,a0,b0,c0,d0,0,w);S2Round(e1,f1,g1,h1,a1,b1,c1,d1,0,w);}
        {const uint32_t w=wb.y;S2Round(d0,e0,f0,g0,h0,a0,b0,c0,0,w);S2Round(d1,e1,f1,g1,h1,a1,b1,c1,0,w);}
        {const uint32_t w=wb.z;S2Round(c0,d0,e0,f0,g0,h0,a0,b0,0,w);S2Round(c1,d1,e1,f1,g1,h1,a1,b1,0,w);}
        {const uint32_t w=wb.w;S2Round(b0,c0,d0,e0,f0,g0,h0,a0,0,w);S2Round(b1,c1,d1,e1,f1,g1,h1,a1,0,w);}
#else
        {const uint32_t w=QSB_WINDOW_SECOND[r][slot];S2Round(a0,b0,c0,d0,e0,f0,g0,h0,0,w);S2Round(a1,b1,c1,d1,e1,f1,g1,h1,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+1][slot];S2Round(h0,a0,b0,c0,d0,e0,f0,g0,0,w);S2Round(h1,a1,b1,c1,d1,e1,f1,g1,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+2][slot];S2Round(g0,h0,a0,b0,c0,d0,e0,f0,0,w);S2Round(g1,h1,a1,b1,c1,d1,e1,f1,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+3][slot];S2Round(f0,g0,h0,a0,b0,c0,d0,e0,0,w);S2Round(f1,g1,h1,a1,b1,c1,d1,e1,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+4][slot];S2Round(e0,f0,g0,h0,a0,b0,c0,d0,0,w);S2Round(e1,f1,g1,h1,a1,b1,c1,d1,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+5][slot];S2Round(d0,e0,f0,g0,h0,a0,b0,c0,0,w);S2Round(d1,e1,f1,g1,h1,a1,b1,c1,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+6][slot];S2Round(c0,d0,e0,f0,g0,h0,a0,b0,0,w);S2Round(c1,d1,e1,f1,g1,h1,a1,b1,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+7][slot];S2Round(b0,c0,d0,e0,f0,g0,h0,a0,0,w);S2Round(b1,c1,d1,e1,f1,g1,h1,a1,0,w);}
#endif
    }
#if QSB_WINDOW_GROUP_ROUNDS
    }
#endif
    QSB_PAIR_STATE_ADD();
#if QSB_ROOT_FILL
    }
    if(PART!=1){
#endif
#if QSB_CONST_CALLEE
    {   /* QSB_CONST_CALLEE (tree.cu): the four constant blocks in the __noinline__ callee above */
        QsbPairS16 s16;
        #pragma unroll
        for(int j=0;j<8;j++){ s16.w[j]=stateA[j]; s16.w[8+j]=stateB[j]; }
#if QSB_SHA_W0FOLD
        const QsbPairS18 r18=qsb_pair_const4(s16);
        #pragma unroll
        for(int j=0;j<8;j++){ stateA[j]=r18.w[j]; stateB[j]=r18.w[8+j]; }
        w0x[0]=r18.w[16]; w0x[1]=r18.w[17];
#else
        s16=qsb_pair_const4(s16);
        #pragma unroll
        for(int j=0;j<8;j++){ stateA[j]=s16.w[j]; stateB[j]=s16.w[8+j]; }
#endif
    }
#elif QSB_SHA_CONST_IV
    /* QSB_SHA_CONST_IV: the four constant blocks on one induction variable. q indexes the 16 B rows
     * of QSB_CONST_SCHEDULE viewed as uint4[64] (block b, rounds r..r+3 at q = 16 b + r/4) and runs on across
     * the block boundaries, so the inner loop needs no block term in its address (the base LEA and IMAD per
     * 8 rounds go) and the block loop needs no counter of its own. Same words, same rounds, same order. */
    {
        /* The constant-bank byte address itself is the induction variable (a 32-bit register; LLVM's loop
         * strength reduction otherwise splits a pointer walk into a trip counter plus an index): kp walks the
         * 64 rows, 32 B per 8 rounds, and the loops end on its value. */
        uint32_t kp, kw0,kw1,kw2,kw3,kw4,kw5,kw6,kw7;
        asm("mov.u32 %0, QSB_CONST_SCHEDULE;" : "=r"(kp));
        const uint32_t kall=kp+1024u;
/* the step as asm too, so LLVM cannot re-derive the walk as a trip counter plus an index */
#define QSB_R2B_KSTEP() asm("add.u32 %0, %0, 32;" : "+r"(kp))
#define QSB_R2B_KLOAD() asm("ld.const.v4.u32 {%0,%1,%2,%3}, [%8];\n\tld.const.v4.u32 {%4,%5,%6,%7}, [%8+16];" \
            : "=r"(kw0),"=r"(kw1),"=r"(kw2),"=r"(kw3),"=r"(kw4),"=r"(kw5),"=r"(kw6),"=r"(kw7) : "r"(kp))
        #pragma unroll 1
        do{
#if !QSB_SHA_CONST_PEEL
            QSB_PAIR_STATE_LOAD();
#endif
            uint32_t kend; asm("add.u32 %0, %1, 256;" : "=r"(kend) : "r"(kp));
#if QSB_SHA_CONST_PEEL
            /* QSB_SHA_CONST_PEEL: rounds 0..7 of the block read the saved state (stateA/stateB) and write the
             * working registers, so the 16 working-state copies per block per pair go; costs one more
             * 8-round body of code. */
            {
                QSB_R2B_KLOAD();
                uint32_t pa0=stateA[0],pb0=stateA[1],pc0=stateA[2],pd0=stateA[3],pe0=stateA[4],pf0=stateA[5],pg0=stateA[6],ph0=stateA[7];
                uint32_t pa1=stateB[0],pb1=stateB[1],pc1=stateB[2],pd1=stateB[3],pe1=stateB[4],pf1=stateB[5],pg1=stateB[6],ph1=stateB[7];
                {const uint32_t w=kw0;S2Round(pa0,pb0,pc0,pd0,pe0,pf0,pg0,ph0,0,w);S2Round(pa1,pb1,pc1,pd1,pe1,pf1,pg1,ph1,0,w);}
                {const uint32_t w=kw1;S2Round(ph0,pa0,pb0,pc0,pd0,pe0,pf0,pg0,0,w);S2Round(ph1,pa1,pb1,pc1,pd1,pe1,pf1,pg1,0,w);}
                {const uint32_t w=kw2;S2Round(pg0,ph0,pa0,pb0,pc0,pd0,pe0,pf0,0,w);S2Round(pg1,ph1,pa1,pb1,pc1,pd1,pe1,pf1,0,w);}
                {const uint32_t w=kw3;S2Round(pf0,pg0,ph0,pa0,pb0,pc0,pd0,pe0,0,w);S2Round(pf1,pg1,ph1,pa1,pb1,pc1,pd1,pe1,0,w);}
                {const uint32_t w=kw4;S2Round(pe0,pf0,pg0,ph0,pa0,pb0,pc0,pd0,0,w);S2Round(pe1,pf1,pg1,ph1,pa1,pb1,pc1,pd1,0,w);}
                {const uint32_t w=kw5;S2Round(pd0,pe0,pf0,pg0,ph0,pa0,pb0,pc0,0,w);S2Round(pd1,pe1,pf1,pg1,ph1,pa1,pb1,pc1,0,w);}
                {const uint32_t w=kw6;S2Round(pc0,pd0,pe0,pf0,pg0,ph0,pa0,pb0,0,w);S2Round(pc1,pd1,pe1,pf1,pg1,ph1,pa1,pb1,0,w);}
                {const uint32_t w=kw7;S2Round(pb0,pc0,pd0,pe0,pf0,pg0,ph0,pa0,0,w);S2Round(pb1,pc1,pd1,pe1,pf1,pg1,ph1,pa1,0,w);}
                a0=pa0;b0=pb0;c0=pc0;d0=pd0;e0=pe0;f0=pf0;g0=pg0;h0=ph0;
                a1=pa1;b1=pb1;c1=pc1;d1=pd1;e1=pe1;f1=pf1;g1=pg1;h1=ph1;
                QSB_R2B_KSTEP();
            }
#endif
            #pragma unroll 1
            do{
                QSB_R2B_KLOAD();
                {const uint32_t w=kw0;S2Round(a0,b0,c0,d0,e0,f0,g0,h0,0,w);S2Round(a1,b1,c1,d1,e1,f1,g1,h1,0,w);}
                {const uint32_t w=kw1;S2Round(h0,a0,b0,c0,d0,e0,f0,g0,0,w);S2Round(h1,a1,b1,c1,d1,e1,f1,g1,0,w);}
                {const uint32_t w=kw2;S2Round(g0,h0,a0,b0,c0,d0,e0,f0,0,w);S2Round(g1,h1,a1,b1,c1,d1,e1,f1,0,w);}
                {const uint32_t w=kw3;S2Round(f0,g0,h0,a0,b0,c0,d0,e0,0,w);S2Round(f1,g1,h1,a1,b1,c1,d1,e1,0,w);}
                {const uint32_t w=kw4;S2Round(e0,f0,g0,h0,a0,b0,c0,d0,0,w);S2Round(e1,f1,g1,h1,a1,b1,c1,d1,0,w);}
                {const uint32_t w=kw5;S2Round(d0,e0,f0,g0,h0,a0,b0,c0,0,w);S2Round(d1,e1,f1,g1,h1,a1,b1,c1,0,w);}
                {const uint32_t w=kw6;S2Round(c0,d0,e0,f0,g0,h0,a0,b0,0,w);S2Round(c1,d1,e1,f1,g1,h1,a1,b1,0,w);}
                {const uint32_t w=kw7;S2Round(b0,c0,d0,e0,f0,g0,h0,a0,0,w);S2Round(b1,c1,d1,e1,f1,g1,h1,a1,0,w);}
                QSB_R2B_KSTEP();
            }while(kp!=kend);
            QSB_PAIR_STATE_ADD();
        }while(kp!=kall);
#undef QSB_R2B_KLOAD
#undef QSB_R2B_KSTEP
    }
#else
#if QSB_PAIR_SHA_UNROLL_CONST
    #pragma unroll
#else
    #pragma unroll 1
#endif
    for(int block=0;block<4;block++){
        QSB_PAIR_STATE_LOAD();
#if QSB_PAIR_SHA_UNROLL_CONST_INNER
        #pragma unroll
#else
        #pragma unroll 1
#endif
        for(int r=0;r<64;r+=8){
#if QSB_SHA_SCHED_V4
            /* QSB_SHA_SCHED_V4: the 16 B-aligned constant rows read as two uint4 (LDC.128) per 8 rounds */
            const uint4 ka=*reinterpret_cast<const uint4*>(&QSB_CONST_SCHEDULE[block][r]);
            const uint4 kb=*reinterpret_cast<const uint4*>(&QSB_CONST_SCHEDULE[block][r+4]);
            {const uint32_t w=ka.x;S2Round(a0,b0,c0,d0,e0,f0,g0,h0,0,w);S2Round(a1,b1,c1,d1,e1,f1,g1,h1,0,w);}
            {const uint32_t w=ka.y;S2Round(h0,a0,b0,c0,d0,e0,f0,g0,0,w);S2Round(h1,a1,b1,c1,d1,e1,f1,g1,0,w);}
            {const uint32_t w=ka.z;S2Round(g0,h0,a0,b0,c0,d0,e0,f0,0,w);S2Round(g1,h1,a1,b1,c1,d1,e1,f1,0,w);}
            {const uint32_t w=ka.w;S2Round(f0,g0,h0,a0,b0,c0,d0,e0,0,w);S2Round(f1,g1,h1,a1,b1,c1,d1,e1,0,w);}
            {const uint32_t w=kb.x;S2Round(e0,f0,g0,h0,a0,b0,c0,d0,0,w);S2Round(e1,f1,g1,h1,a1,b1,c1,d1,0,w);}
            {const uint32_t w=kb.y;S2Round(d0,e0,f0,g0,h0,a0,b0,c0,0,w);S2Round(d1,e1,f1,g1,h1,a1,b1,c1,0,w);}
            {const uint32_t w=kb.z;S2Round(c0,d0,e0,f0,g0,h0,a0,b0,0,w);S2Round(c1,d1,e1,f1,g1,h1,a1,b1,0,w);}
            {const uint32_t w=kb.w;S2Round(b0,c0,d0,e0,f0,g0,h0,a0,0,w);S2Round(b1,c1,d1,e1,f1,g1,h1,a1,0,w);}
#else
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r];S2Round(a0,b0,c0,d0,e0,f0,g0,h0,0,w);S2Round(a1,b1,c1,d1,e1,f1,g1,h1,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+1];S2Round(h0,a0,b0,c0,d0,e0,f0,g0,0,w);S2Round(h1,a1,b1,c1,d1,e1,f1,g1,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+2];S2Round(g0,h0,a0,b0,c0,d0,e0,f0,0,w);S2Round(g1,h1,a1,b1,c1,d1,e1,f1,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+3];S2Round(f0,g0,h0,a0,b0,c0,d0,e0,0,w);S2Round(f1,g1,h1,a1,b1,c1,d1,e1,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+4];S2Round(e0,f0,g0,h0,a0,b0,c0,d0,0,w);S2Round(e1,f1,g1,h1,a1,b1,c1,d1,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+5];S2Round(d0,e0,f0,g0,h0,a0,b0,c0,0,w);S2Round(d1,e1,f1,g1,h1,a1,b1,c1,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+6];S2Round(c0,d0,e0,f0,g0,h0,a0,b0,0,w);S2Round(c1,d1,e1,f1,g1,h1,a1,b1,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+7];S2Round(b0,c0,d0,e0,f0,g0,h0,a0,0,w);S2Round(b1,c1,d1,e1,f1,g1,h1,a1,0,w);}
#endif
        }
        QSB_PAIR_STATE_ADD();
    }
#endif /* QSB_SHA_CONST_IV */
#if QSB_ROOT_FILL
    }
#endif
#undef QSB_PAIR_STATE_LOAD
#undef QSB_PAIR_STATE_ADD
}
#endif
