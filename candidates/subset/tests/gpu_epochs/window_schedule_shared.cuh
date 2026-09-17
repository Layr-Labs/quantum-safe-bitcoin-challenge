// Derived from odinfree's GPU-epoch consumer in submission 0db6e203.
// Only the first block depends on the epoch remainder. The second block's
// expanded schedule is shared by every epoch with the same window choice.
#pragma once
__device__ uint32_t QSB_WINDOW_FIRST[14][256];
__device__ uint32_t QSB_WINDOW_SECOND[64][256];
__device__ uint32_t QSB_WINDOW_CLASS[256];
__device__ uint32_t QSB_FIRST_CLASS[256];
__device__ uint32_t QSB_FIRST_UNIQUE[14][256];
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
        const uint8_t windows[256][3], const uint32_t *constant) {
    uint32_t first[14][256], second[64][256]={}, round_k[64];
    uint32_t classes[256], unique[256][16];
    uint32_t first_classes[256], first_unique[256][14], transposed[14][256]={};
    int first_distinct=0;
    int distinct=0;
    if (cudaMemcpyFromSymbol(round_k, K, sizeof(round_k)) != cudaSuccess) return 1;
    for (int lane=0; lane<256; lane++) {
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
    printf("Window schedule classes: first=%d second=%d of 256\n",first_distinct,distinct);
    qsb_first_class_count=first_distinct;
    for(int slot=0;slot<first_distinct;slot++)
        for(int j=0;j<14;j++)transposed[j][slot]=first_unique[slot][j];
    if(cudaMemcpyToSymbol(QSB_FIRST_COUNT,&first_distinct,sizeof(first_distinct))!=cudaSuccess)return 1;
    if(cudaMemcpyToSymbol(QSB_FIRST_CLASS,first_classes,sizeof(first_classes))!=cudaSuccess)return 1;
    if(cudaMemcpyToSymbol(QSB_FIRST_UNIQUE,transposed,sizeof(transposed))!=cudaSuccess)return 1;
    if (cudaMemcpyToSymbol(QSB_WINDOW_CLASS,classes,sizeof(classes))!=cudaSuccess) return 1;
    if (cudaMemcpyToSymbol(QSB_WINDOW_FIRST,first,sizeof(first))!=cudaSuccess) return 1;
    return cudaMemcpyToSymbol(QSB_WINDOW_SECOND,second,sizeof(second))==cudaSuccess?0:1;
}

/* PRODUCER STAGE (sub_prod2). The per-epoch first-block SHA-256 midstate of
 * first-class `cls` used to be compressed by leader lane `cls` of the digest
 * block into a __shared__ uint32_t first_states[8][256] (8192 B) behind a
 * __syncthreads(): 54 of 256 lanes worked, 202 idled, and every lane paid the
 * barrier. kernel_build_first now computes exactly these states one launch
 * earlier and parks them in global memory; the digest's consumer lanes read
 * their class's 32 bytes directly. The arithmetic below is character-for-
 * character the old leader-lane body, so every produced state is bit-identical.
 *
 * Layout: class-major, 8 words (32 B) per class, QSB_FIRST_COUNT classes per
 * epoch. Offsets are therefore always 32 B multiples, so the 16-byte vector
 * accessors below are always correctly aligned for a cudaMalloc'd base. */
__device__ __forceinline__ void qsb_first_state_class(const epoch_desc_t *epoch,
        int cls, uint32_t out[8]) {
    uint32_t W[16];
    #pragma unroll
    for(int j=0;j<8;j++)out[j]=epoch->mid[j];
    W[0]=epoch->remW[0];W[1]=epoch->remW[1];
    #pragma unroll
    for(int j=2;j<16;j++)W[j]=QSB_FIRST_UNIQUE[j-2][cls];
    _SHA256Transform(out,W);   /* mutates W; out is the epoch midstate */
}

/* 2 x 128-bit accesses instead of 8 x 32-bit: the store is a fully coalesced
 * 32 B/lane run, and one consumer warp needs 32 sectors rather than 256. */
__device__ __forceinline__ void qsb_store_first_state(uint32_t * __restrict__ dst,
        const uint32_t in[8]) {
    uint4 a,b;
    a.x=in[0];a.y=in[1];a.z=in[2];a.w=in[3];
    b.x=in[4];b.y=in[5];b.z=in[6];b.w=in[7];
    uint4 *v=reinterpret_cast<uint4*>(dst);
    v[0]=a;v[1]=b;
}
__device__ __forceinline__ void qsb_load_first_state(const uint32_t * __restrict__ src,
        uint32_t out[8]) {
    const uint4 *v=reinterpret_cast<const uint4*>(src);
    uint4 a=v[0],b=v[1];
    out[0]=a.x;out[1]=a.y;out[2]=a.z;out[3]=a.w;
    out[4]=b.x;out[5]=b.y;out[6]=b.z;out[7]=b.w;
}

/* first_epoch points at this block's epoch record: d_first + blockIdx.x*stride
 * with stride = 8*QSB_FIRST_COUNT words. */
__device__ __forceinline__ void qsb_scheduled_window_hash(uint32_t *state,
        const uint32_t * __restrict__ first_epoch, int lane) {
    // No barrier and no shared staging: the first-block states were produced
    // by kernel_build_first on the previous launch of the same stream.
    int first_slot=QSB_FIRST_CLASS[lane];
    qsb_load_first_state(first_epoch+(size_t)first_slot*8,state);
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
    qsb_compress_constant_rolled(state);
}
