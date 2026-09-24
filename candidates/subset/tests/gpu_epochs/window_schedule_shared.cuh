// Derived from odinfree's GPU-epoch consumer in submission 0db6e203.
// Only the first block depends on the epoch remainder. The second block's
// expanded schedule is shared by every epoch with the same window choice.
#pragma once
#ifndef QSB_950_PACK
#define QSB_950_PACK 1
#endif
#define QSB_FIRST_SLOTS (QSB_SE_WINDOWS==256?64:16)
#ifndef QSB_SHA_UNROLL_CONST
#define QSB_SHA_UNROLL_CONST 1
#endif   /* first-block classes per epoch in d_first */
__device__ uint32_t QSB_WINDOW_FIRST[14][QSB_SE_PER_EPOCH];
__device__ uint32_t QSB_WINDOW_SECOND[64][QSB_SE_PER_EPOCH];
__device__ uint32_t QSB_WINDOW_CLASS[QSB_SE_PER_EPOCH];
__device__ uint32_t QSB_FIRST_CLASS[QSB_SE_PER_EPOCH];
/* Public PR950: lossless (first_slot << 16) | second_slot. */
__device__ uint32_t QSB_LANE_CLASS[QSB_SE_PER_EPOCH];
__device__ uint32_t QSB_FIRST_UNIQUE[14][QSB_SE_WINDOWS==256?256:QSB_FIRST_SLOTS];
__device__ __constant__ int QSB_FIRST_COUNT;
static int qsb_first_class_count=0;

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
    uint32_t first[14][QSB_SE_PER_EPOCH], second[64][QSB_SE_PER_EPOCH]={}, round_k[64];
    uint32_t classes[QSB_SE_PER_EPOCH], unique[QSB_SE_PER_EPOCH][16];
    uint32_t first_classes[QSB_SE_PER_EPOCH], first_unique[QSB_SE_PER_EPOCH][14], transposed[14][QSB_SE_WINDOWS==256?256:QSB_FIRST_SLOTS]={};
    int first_distinct=0;
    int distinct=0;
    if (cudaMemcpyFromSymbol(round_k, K, sizeof(round_k)) != cudaSuccess) return 1;
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
        if(slot==distinct){memcpy(unique[distinct],words+16,64);distinct++;}
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
    for(int slot=0;slot<first_distinct;slot++)
        for(int j=0;j<14;j++)transposed[j][slot]=first_unique[slot][j];
    if(cudaMemcpyToSymbol(QSB_FIRST_COUNT,&first_distinct,sizeof(first_distinct))!=cudaSuccess)return 1;
    if(cudaMemcpyToSymbol(QSB_FIRST_CLASS,first_classes,sizeof(first_classes))!=cudaSuccess)return 1;
    if(cudaMemcpyToSymbol(QSB_FIRST_UNIQUE,transposed,sizeof(transposed))!=cudaSuccess)return 1;
#if QSB_950_PACK
    {   uint32_t packed[QSB_SE_PER_EPOCH];
        for(int lane=0;lane<QSB_SE_PER_EPOCH;lane++)
            packed[lane]=(first_classes[lane]<<16)|classes[lane];
        if(cudaMemcpyToSymbol(QSB_LANE_CLASS,packed,sizeof(packed))!=cudaSuccess)return 1;
    }
#endif
    if (cudaMemcpyToSymbol(QSB_WINDOW_CLASS,classes,sizeof(classes))!=cudaSuccess) return 1;
    if (cudaMemcpyToSymbol(QSB_WINDOW_FIRST,first,sizeof(first))!=cudaSuccess) return 1;
    return cudaMemcpyToSymbol(QSB_WINDOW_SECOND,second,sizeof(second))==cudaSuccess?0:1;
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
        S2Round(a,b,c,d,e,f,g,h,0,QSB_WINDOW_SECOND[r][slot]);
        S2Round(h,a,b,c,d,e,f,g,0,QSB_WINDOW_SECOND[r+1][slot]);
        S2Round(g,h,a,b,c,d,e,f,0,QSB_WINDOW_SECOND[r+2][slot]);
        S2Round(f,g,h,a,b,c,d,e,0,QSB_WINDOW_SECOND[r+3][slot]);
        S2Round(e,f,g,h,a,b,c,d,0,QSB_WINDOW_SECOND[r+4][slot]);
        S2Round(d,e,f,g,h,a,b,c,0,QSB_WINDOW_SECOND[r+5][slot]);
        S2Round(c,d,e,f,g,h,a,b,0,QSB_WINDOW_SECOND[r+6][slot]);
        S2Round(b,c,d,e,f,g,h,a,0,QSB_WINDOW_SECOND[r+7][slot]);
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
#define QSB_PAIR_SHA_UNROLL_WINDOW 1
#endif
#ifndef QSB_PAIR_SHA_UNROLL_CONST
#define QSB_PAIR_SHA_UNROLL_CONST 1
#endif
/* Keep the four-block loop unrolled; roll only its 8-round inner loop. */
#ifndef QSB_PAIR_SHA_UNROLL_CONST_INNER
#define QSB_PAIR_SHA_UNROLL_CONST_INNER 0
#endif
/* Paired epoch SHA from dukemawex 4cea5476 (origin e771d5c7 / e9812a9). The paired consumer has the same lane (and therefore the same scheduled
 * second block and constant suffix) in both epochs.  Load each schedule word
 * once and advance two independent SHA-256 states with it. */
__device__ __forceinline__ void qsb_scheduled_window_hash_pair(
    uint32_t *stateA, uint32_t *stateB, int lane,
    const uint32_t *firstA, const uint32_t *firstB) {
#if QSB_950_PACK
    const uint32_t lane_rec=QSB_LANE_CLASS[lane];
    const int first_slot=(int)(lane_rec>>16);
    const int slot=(int)(lane_rec&0xffffu);
#else
    const int first_slot=QSB_FIRST_CLASS[lane];
    const int slot=QSB_WINDOW_CLASS[lane];
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
    QSB_PAIR_STATE_LOAD();
#if QSB_PAIR_SHA_UNROLL_WINDOW   /* exact: same rounds, no loop counter, loads can be hoisted */
    #pragma unroll
#else
    #pragma unroll 1
#endif
    for(int r=0;r<64;r+=8){
        {const uint32_t w=QSB_WINDOW_SECOND[r][slot];S2Round(a0,b0,c0,d0,e0,f0,g0,h0,0,w);S2Round(a1,b1,c1,d1,e1,f1,g1,h1,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+1][slot];S2Round(h0,a0,b0,c0,d0,e0,f0,g0,0,w);S2Round(h1,a1,b1,c1,d1,e1,f1,g1,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+2][slot];S2Round(g0,h0,a0,b0,c0,d0,e0,f0,0,w);S2Round(g1,h1,a1,b1,c1,d1,e1,f1,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+3][slot];S2Round(f0,g0,h0,a0,b0,c0,d0,e0,0,w);S2Round(f1,g1,h1,a1,b1,c1,d1,e1,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+4][slot];S2Round(e0,f0,g0,h0,a0,b0,c0,d0,0,w);S2Round(e1,f1,g1,h1,a1,b1,c1,d1,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+5][slot];S2Round(d0,e0,f0,g0,h0,a0,b0,c0,0,w);S2Round(d1,e1,f1,g1,h1,a1,b1,c1,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+6][slot];S2Round(c0,d0,e0,f0,g0,h0,a0,b0,0,w);S2Round(c1,d1,e1,f1,g1,h1,a1,b1,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+7][slot];S2Round(b0,c0,d0,e0,f0,g0,h0,a0,0,w);S2Round(b1,c1,d1,e1,f1,g1,h1,a1,0,w);}
    }
    QSB_PAIR_STATE_ADD();
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
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r];S2Round(a0,b0,c0,d0,e0,f0,g0,h0,0,w);S2Round(a1,b1,c1,d1,e1,f1,g1,h1,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+1];S2Round(h0,a0,b0,c0,d0,e0,f0,g0,0,w);S2Round(h1,a1,b1,c1,d1,e1,f1,g1,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+2];S2Round(g0,h0,a0,b0,c0,d0,e0,f0,0,w);S2Round(g1,h1,a1,b1,c1,d1,e1,f1,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+3];S2Round(f0,g0,h0,a0,b0,c0,d0,e0,0,w);S2Round(f1,g1,h1,a1,b1,c1,d1,e1,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+4];S2Round(e0,f0,g0,h0,a0,b0,c0,d0,0,w);S2Round(e1,f1,g1,h1,a1,b1,c1,d1,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+5];S2Round(d0,e0,f0,g0,h0,a0,b0,c0,0,w);S2Round(d1,e1,f1,g1,h1,a1,b1,c1,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+6];S2Round(c0,d0,e0,f0,g0,h0,a0,b0,0,w);S2Round(c1,d1,e1,f1,g1,h1,a1,b1,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+7];S2Round(b0,c0,d0,e0,f0,g0,h0,a0,0,w);S2Round(b1,c1,d1,e1,f1,g1,h1,a1,0,w);}
        }
        QSB_PAIR_STATE_ADD();
    }
#undef QSB_PAIR_STATE_LOAD
#undef QSB_PAIR_STATE_ADD
}
#endif
