// One work-efficient binary product tree per block. The caller supplies
// a power-of-two block size at most 256 and identity factors for inactive lanes.
#pragma once
/* Block inverse tree:
 *  0 = promoted heap tree: every product canonical, 18 barriers.
 *  1 = same heap layout, lazy canonicalization (internal nodes stay exact but
 *      possibly non-canonical residues in [0,2^256); only the root is
 *      normalized, right before _ModInv), the root product is computed by
 *      lane 0 immediately before its inversion and the first downward level
 *      right after it (no barriers in between), and every lane forms its own
 *      leaf inverse from its parent inverse and sibling product (no leaf write
 *      barrier). 15 barriers, 1 canonicalization per block, same 765 multiplies.
 *  2 = level-packed products/inverses (layout of the promoted pinning tree)
 *      with the same lazy/barrier schedule as 1; barrier kind chosen by the
 *      lanes that READ the next level (warp barrier only when all readers and
 *      writers sit in warp 0).
 * Leaves are returned as exact residues below 2^256, the same contract as
 * every _ModMult output that feeds the finish. */
__device__ __forceinline__ void qsb_block_inverse_tree(uint64_t *value){
    __shared__ uint64_t products[4][512];
    __shared__ uint64_t inverses[4][256];
    const int tid=threadIdx.x,n=blockDim.x;
    #pragma unroll
    for(int k=0;k<4;k++)products[k][tid]=value[k];
    __syncthreads();
    // Level (offset,count): (0,n),(n,n/2),...,(2n-4,2). Level `count` is
    // formed by lanes < count/2 and read by lanes < count/4.
    int offset=0;
    #pragma unroll 1
    for(int count=n;count>2;count>>=1){
        int half=count>>1;
        if(tid<half){
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=products[k][offset+tid];b[k]=products[k][offset+half+tid];}
            a[4]=b[4]=0;
            qsb_field_mul_raw(out,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)products[k][offset+count+tid]=out[k];
        }
        offset+=count;
        if(half>32)__syncthreads();else __syncwarp();
    }
    // offset == 2n-4: the two root children.
    if(tid==0){
        uint64_t a[5],b[5],root[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=products[k][offset];b[k]=products[k][offset+1];}
        a[4]=b[4]=0;
        qsb_field_mul_raw(root,a,b);
        qsb_field_normalize(root);
        _ModInv(root);
        root[4]=0;
        qsb_field_mul_raw(a,root,a);   /* 1/b */
        qsb_field_mul_raw(b,root,b);   /* 1/a */
        // inverse index = product index - n
        #pragma unroll
        for(int k=0;k<4;k++){inverses[k][offset-n]=b[k];inverses[k][offset-n+1]=a[k];}
    }
    __syncwarp();
    // Level count (4..n/2): lanes < count read parent inverses written by
    // lanes < count/2 and write inverses read by lanes < 2*count.
    offset-=4;   /* level count=4 */
    #pragma unroll 1
    for(int count=4;count<n;count<<=1){
        int half=count>>1;
        if(tid<count){
            uint64_t parent_inv[5],sibling[5],child_inv[5];
            #pragma unroll
            for(int k=0;k<4;k++){
                parent_inv[k]=inverses[k][offset+count-n+(tid&(half-1))];
                sibling[k]=products[k][offset+(tid^half)];
            }
            parent_inv[4]=sibling[4]=0;
            qsb_field_mul_raw(child_inv,parent_inv,sibling);
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-n+tid]=child_inv[k];
        }
        offset-=count<<1;
        if((count<<1)>32)__syncthreads();else __syncwarp();
    }
    // offset == 0 would be the leaf level; lanes form their own leaf inverse.
    {
        const int half=n>>1;
        uint64_t parent_inv[5],sibling[5];
        #pragma unroll
        for(int k=0;k<4;k++){
            parent_inv[k]=inverses[k][tid&(half-1)];
            sibling[k]=products[k][tid^half];
        }
        parent_inv[4]=sibling[4]=0;
        qsb_field_mul_raw(value,parent_inv,sibling);
    }
    value[4]=0;
}
