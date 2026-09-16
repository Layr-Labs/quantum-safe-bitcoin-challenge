// Derived from odinfree's GPU-epoch consumer in submission 0db6e203.
// Only the first block depends on the epoch remainder. The second block's
// expanded schedule is shared by every epoch with the same window choice.
#pragma once

/* There are C(13,3)=286 valid window triples but one consumer block has 256
 * lanes.  Which 256 triples we use is therefore a performance choice, not a
 * correctness constraint.  The old lexicographic prefix happened to retain
 * all 84 distinct first-block tails.  Prefer triples from large equivalence
 * classes instead: this leaves only 54 first-block tails, so two warps rather
 * than three build the per-epoch first states.  Finally group equal second
 * blocks so adjacent lanes read adjacent/broadcast schedule slots. */
#define QSB_WINDOW_ALL 286
#define QSB_WINDOW_CLASS_CAP 64

typedef struct {
    uint8_t skip[3];
    uint8_t kept[100];
    int first_freq;
    int selected;
    int emitted;
} qsb_window_choice_t;

static int qsb_choice_less(const qsb_window_choice_t *a,
                           const qsb_window_choice_t *b) {
    int d = memcmp(a->kept + 56, b->kept + 56, 44);
    if (d) return d < 0;
    d = memcmp(a->kept, b->kept, 56);
    if (d) return d < 0;
    return memcmp(a->skip, b->skip, 3) < 0;
}

static int qsb_select_window_schedule(const uint8_t *rows,
        uint8_t windows[QSB_SE_PER_EPOCH][QSB_SE_TWIN]) {
    qsb_window_choice_t choices[QSB_WINDOW_ALL];
    int count = 0;
    for (int a = 0; a < 13; a++)
        for (int b = a + 1; b < 13; b++)
            for (int c = b + 1; c < 13; c++) {
                qsb_window_choice_t *q = &choices[count++];
                q->skip[0] = (uint8_t)a;
                q->skip[1] = (uint8_t)b;
                q->skip[2] = (uint8_t)c;
                q->first_freq = q->selected = q->emitted = 0;
                int pos = 0;
                for (int i = 0; i < 13; i++) {
                    if (i == a || i == b || i == c) continue;
                    memcpy(q->kept + pos, rows + (QSB_SE_CUT + i) * SIG_PUSH_SIZE,
                           SIG_PUSH_SIZE);
                    pos += SIG_PUSH_SIZE;
                }
                if (pos != 100) return 1;
            }
    if (count != QSB_WINDOW_ALL) return 1;

    for (int i = 0; i < count; i++)
        for (int j = 0; j < count; j++)
            choices[i].first_freq +=
                memcmp(choices[i].kept, choices[j].kept, 56) == 0;

    /* Take the 256 members of the largest first-block classes.  Equal-size
     * ties use lexicographic skip order, making the runtime choice stable. */
    for (int pick = 0; pick < QSB_SE_PER_EPOCH; pick++) {
        int best = -1;
        for (int i = 0; i < count; i++) {
            if (choices[i].selected) continue;
            if (best < 0 || choices[i].first_freq > choices[best].first_freq ||
                (choices[i].first_freq == choices[best].first_freq &&
                 memcmp(choices[i].skip, choices[best].skip, 3) < 0))
                best = i;
        }
        if (best < 0) return 1;
        choices[best].selected = 1;
    }

    /* Lane order is otherwise semantically irrelevant.  Sort by the second
     * message block first, then the first-block tail, to make schedule slots
     * and shared-state reads as warp-friendly as the selected set permits. */
    for (int lane = 0; lane < QSB_SE_PER_EPOCH; lane++) {
        int best = -1;
        for (int i = 0; i < count; i++) {
            if (!choices[i].selected || choices[i].emitted) continue;
            if (best < 0 || qsb_choice_less(&choices[i], &choices[best])) best = i;
        }
        if (best < 0) return 1;
        choices[best].emitted = 1;
        for (int j = 0; j < QSB_SE_TWIN; j++)
            windows[lane][j] = (uint8_t)(QSB_SE_CUT + choices[best].skip[j]);
    }
    return 0;
}

#ifndef QSB_SCHEDULE_SELECT_ONLY
__device__ uint32_t QSB_WINDOW_SECOND[64][QSB_WINDOW_CLASS_CAP];
__device__ uint8_t QSB_WINDOW_CLASS[256];
__device__ uint8_t QSB_FIRST_CLASS[256];
__device__ uint32_t QSB_FIRST_UNIQUE[14][QSB_WINDOW_CLASS_CAP];
__device__ __constant__ int QSB_FIRST_COUNT;
static int qsb_first_class_count=0;

static int qsb_prepare_window_schedule(const uint8_t *rows,
        const uint8_t windows[256][3], const uint32_t *constant) {
    uint32_t second[64][QSB_WINDOW_CLASS_CAP]={}, round_k[64];
    uint8_t classes[256], first_classes[256];
    uint32_t unique[QSB_WINDOW_CLASS_CAP][16];
    uint32_t first_unique[QSB_WINDOW_CLASS_CAP][14];
    uint32_t transposed[14][QSB_WINDOW_CLASS_CAP]={};
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
        int first_slot=0;
        while(first_slot<first_distinct && memcmp(first_unique[first_slot],words+2,56))first_slot++;
        if(first_slot==first_distinct){
            if(first_distinct>=QSB_WINDOW_CLASS_CAP)return 1;
            memcpy(first_unique[first_distinct],words+2,56);first_distinct++;
        }
        first_classes[lane]=first_slot;
        int slot=0;
        while(slot<distinct && memcmp(unique[slot],words+16,64))slot++;
        if(slot==distinct){
            if(distinct>=QSB_WINDOW_CLASS_CAP)return 1;
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
    printf("Window schedule classes: first=%d second=%d of 256\n",first_distinct,distinct);
    qsb_first_class_count=first_distinct;
    for(int slot=0;slot<first_distinct;slot++)
        for(int j=0;j<14;j++)transposed[j][slot]=first_unique[slot][j];
    if(cudaMemcpyToSymbol(QSB_FIRST_COUNT,&first_distinct,sizeof(first_distinct))!=cudaSuccess)return 1;
    if(cudaMemcpyToSymbol(QSB_FIRST_CLASS,first_classes,sizeof(first_classes))!=cudaSuccess)return 1;
    if(cudaMemcpyToSymbol(QSB_FIRST_UNIQUE,transposed,sizeof(transposed))!=cudaSuccess)return 1;
    if (cudaMemcpyToSymbol(QSB_WINDOW_CLASS,classes,sizeof(classes))!=cudaSuccess) return 1;
    return cudaMemcpyToSymbol(QSB_WINDOW_SECOND,second,sizeof(second))==cudaSuccess?0:1;
}

__device__ __forceinline__ void qsb_scheduled_window_hash(uint32_t *state,
        const epoch_desc_t *epoch, int lane) {
    // Every lane in this epoch's block reaches the shared-memory barrier.
    __shared__ uint32_t first_states[8][QSB_WINDOW_CLASS_CAP];
    if(lane<QSB_FIRST_COUNT) {
        uint32_t initial[8],W[16];
        #pragma unroll
        for(int j=0;j<8;j++)initial[j]=epoch->mid[j];
        W[0]=epoch->remW[0];W[1]=epoch->remW[1];
        #pragma unroll
        for(int j=2;j<16;j++)W[j]=QSB_FIRST_UNIQUE[j-2][lane];
        _SHA256Transform(initial,W);
        #pragma unroll
        for(int j=0;j<8;j++)first_states[j][lane]=initial[j];
    }
    __syncthreads();
    int first_slot=QSB_FIRST_CLASS[lane];
    #pragma unroll
    for(int j=0;j<8;j++)state[j]=first_states[j][first_slot];
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
#endif
