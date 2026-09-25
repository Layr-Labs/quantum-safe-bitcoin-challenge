__global__ void __launch_bounds__(256,2) kernel_split_inverse_ref(uint64_t *work,unsigned epochs,unsigned stride){
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
