// Diagnostic split of the promoted paired consumer. Same pairing and tree order.
#pragma once
__device__ __forceinline__ void qsb_split_store(uint64_t *work, uint8_t *valid,
        unsigned stride, unsigned index, const QsbPairFront3 &front) {
    valid[index]=(uint8_t)front.ok;
    #pragma unroll
    for(int j=0;j<16;j++) work[(size_t)j*stride+index]=front.ok?front.words[j]:(j==0?1:0);
}
__global__ void __launch_bounds__(256,2) kernel_split_front(const uint32_t *first,
        const uint8_t *gtable, uint64_t *work, uint8_t *valid,
        unsigned epochs, unsigned stride) {
    const unsigned tid=threadIdx.x,lane=tid&(QSB_SE_WINDOWS-1);
    const unsigned ea=QSB_PAIR_MUL*blockIdx.x+2*(tid/QSB_SE_WINDOWS);
    if(ea>=epochs)return;
    const bool hasB=ea+1<epochs;
    const uint32_t *fa=first+(size_t)ea*QSB_FIRST_SLOTS*8;
    const uint32_t *fb=hasB?fa+QSB_FIRST_SLOTS*8:fa;
    const QsbPairEpochZ z=qsb_pair_epoch_z_value(fa,fb,lane);
    const unsigned ia=ea*QSB_SE_WINDOWS+lane;
    {
        QsbPairFront3 f=qsb_pair_front3_z_value(z.a[0],z.a[1],z.a[2],z.a[3],gtable,
            QSB_U2R[0],QSB_U2R[1],QSB_U2R[2],QSB_U2R[3],QSB_U2R[4],QSB_U2R[5],QSB_U2R[6],QSB_U2R[7]);
        qsb_split_store(work,valid,stride,ia,f);
    }
    if(hasB){
        QsbPairFront3 f=qsb_pair_front3_z_value(z.b[0],z.b[1],z.b[2],z.b[3],gtable,
            QSB_U2R[0],QSB_U2R[1],QSB_U2R[2],QSB_U2R[3],QSB_U2R[4],QSB_U2R[5],QSB_U2R[6],QSB_U2R[7]);
        qsb_split_store(work,valid,stride,ia+QSB_SE_WINDOWS,f);
    }
}
__global__ void __launch_bounds__(256,2) kernel_split_inverse(uint64_t *work,unsigned epochs,unsigned stride){
    const unsigned tid=threadIdx.x,lane=tid&(QSB_SE_WINDOWS-1);
    const unsigned ea=QSB_PAIR_MUL*blockIdx.x+2*(tid/QSB_SE_WINDOWS);
    const unsigned ia=ea*QSB_SE_WINDOWS+lane,ib=ia+QSB_SE_WINDOWS;
    uint64_t a[5]={1,0,0,0,0},b[5]={1,0,0,0,0},leaf[5],inv[5];
    #pragma unroll
    for(int j=0;j<4;j++){
        if(ea<epochs)a[j]=work[(size_t)j*stride+ia];
        if(ea+1<epochs)b[j]=work[(size_t)j*stride+ib];
    }
    QSB_TREE_MUL(leaf,a,b);
    qsb_block_inverse_tree(leaf);
    if(ea<epochs){
        QSB_TREE_MUL(inv,leaf,b);
        #pragma unroll
        for(int j=0;j<4;j++)work[(size_t)j*stride+ia]=inv[j];
    }
    if(ea+1<epochs){
        QSB_TREE_MUL(inv,leaf,a);
        #pragma unroll
        for(int j=0;j<4;j++)work[(size_t)j*stride+ib]=inv[j];
    }
}
__global__ void __launch_bounds__(256,2) kernel_split_tail(const uint64_t *work,
        const uint8_t *valid,unsigned stride,const epoch_desc_t *epochs,uint8_t *hitbuf){
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=stride || !valid[i])return;
    uint64_t n[12],v[4];
    #pragma unroll
    for(int j=0;j<12;j++)n[j]=work[(size_t)(j+4)*stride+i];
    #pragma unroll
    for(int j=0;j<4;j++)v[j]=work[(size_t)j*stride+i];
    const int encoded=qsb_pair_tail3_value(n[0],n[1],n[2],n[3],n[4],n[5],n[6],n[7],n[8],n[9],n[10],n[11],
        v[0],v[1],v[2],v[3],QSB_U2R[0],QSB_U2R[1],QSB_U2R[2],QSB_U2R[3],QSB_U2R[4],QSB_U2R[5],QSB_U2R[6],QSB_U2R[7]);
    if(encoded){
        const unsigned slot=atomicAdd((uint32_t*)hitbuf,1);
        if(slot<1024){
            uint8_t *record=hitbuf+4+(size_t)slot*ZLAB_HIT_REC;
            *(uint32_t*)record=i|((uint32_t)(encoded-1)<<30);
            const unsigned e=i/QSB_SE_WINDOWS,lane=i&(QSB_SE_WINDOWS-1);
            for(int j=0;j<6;j++)record[4+j]=epochs[e].early[j];
            for(int j=0;j<3;j++)record[10+j]=WIN3[lane][j];
        }
    }
}
