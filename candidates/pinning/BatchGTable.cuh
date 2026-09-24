// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Batch-normalize the instance-specific GLV table. The product-tree primitive
// is adapted from this candidate's existing qsb_block_inverse, with its
// recovery-only u^-1 factor deliberately removed. Existing authors retain
// credit for field arithmetic and the original tree. Each inactive/zero
// denominator is neutralized before the collective; output lanes stay masked.
__device__ __forceinline__ void qsb_gtable_inverse256(uint64_t *value) {
    __shared__ uint64_t products[4][512];
    __shared__ uint64_t inverses[4][256];
    int tid=threadIdx.x;

    #pragma unroll
    for(int k=0;k<4;k++)products[k][tid]=value[k];
    __syncthreads();

    // Level (offset,count) pairs are (0,256), (256,128), (384,64), ...,
    // (508,2), (510,1). The final root iteration is executed only by lane zero,
    // so it can invert the result immediately without another synchronization.
    int offset=0;
    #pragma unroll 1
    for(int count=256;count>1;count>>=1){
        int half=count>>1;
        if(tid<half){
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++){
                a[k]=products[k][offset+tid];
                b[k]=products[k][offset+half+tid];
            }
            a[4]=b[4]=0;
            qsb_field_mul(out,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)products[k][offset+count+tid]=out[k];
        }
        offset+=count;
        if(count>2)__syncthreads();
    }

    if(tid==0){
        uint64_t root[5];
        #pragma unroll
        for(int k=0;k<4;k++)root[k]=products[k][510];
        root[4]=0;
        qsb_field_normalize(root);
        _ModInv(root);
        #pragma unroll
        for(int k=0;k<4;k++)inverses[k][254]=root[k];
    }
    __syncthreads();

    // If I=1/(L*R), then I*R=1/L and I*L=1/R. One thread per child
    // therefore expands all internal inverse levels with 254 multiplies.
    // Internal inverse index = product index - 256.
    offset=508;
    #pragma unroll 1
    for(int count=2;count<256;count<<=1){
        int half=count>>1;
        if(tid<count){
            int local_parent=tid&(half-1);
            uint64_t parent_inv[5],sibling[5],child_inv[5];
            #pragma unroll
            for(int k=0;k<4;k++){
                parent_inv[k]=inverses[k][offset+count-256+local_parent];
                sibling[k]=products[k][offset+(tid^half)];
            }
            parent_inv[4]=sibling[4]=0;
            qsb_field_mul(child_inv,parent_inv,sibling);
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-256+tid]=child_inv[k];
        }
        offset-=count<<1;
        __syncthreads();
    }

    // The leaf level has no shared inverse destination or following barrier.
    // Its 256 child inverses can be returned directly to the callers.
    uint64_t parent_inv[5],sibling[5];
    #pragma unroll
    for(int k=0;k<4;k++){
        parent_inv[k]=inverses[k][tid&127];
        sibling[k]=products[k][tid^128];
    }
    parent_inv[4]=sibling[4]=0;
    qsb_field_mul(value,parent_inv,sibling);
    qsb_field_normalize(value);
}
__global__ void kernel_build_gtable(
    const uint64_t * __restrict__ d_L,
    const uint64_t * __restrict__ d_H,
    uint8_t * __restrict__ gTable)
{
    // All 256 lanes must enter the collective, including the final partial CTA.
    const uint64_t t=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    const bool active=t<GT_TOTAL_ENTRIES;
    int ch=-1;
    #pragma unroll
    for(int c=0;c<GT_CHUNKS;c++)
        if(active && t>=gt_offset(c) && t<(uint64_t)gt_offset(c)+gt_entries(c)) ch=c;
    uint64_t px[4]={0,0,0,0},py[4]={0,0,0,0},pz[5]={1,0,0,0,0};
    if(ch>=0) {
        const unsigned d=(unsigned)(t-gt_offset(ch));
        const unsigned m=ch==0?d:2u*d+1u;
#if QSB_BIGTBL
        const unsigned hi=m>>QSB_GT_RADIX_BITS,lo=m&(GT_LO-1u);
#else
        const unsigned hi=m>>8,lo=m&255u;
#endif
        const uint64_t *Hp=d_H+((size_t)ch*GT_HI+hi)*8;
        const uint64_t *Lp=d_L+((size_t)ch*GT_LO+lo)*8;
        if(hi==0) {
            #pragma unroll
            for(int k=0;k<4;k++){px[k]=Lp[k];py[k]=Lp[k+4];}
        } else {
            uint64_t qx[4],qy[4];
            #pragma unroll
            for(int k=0;k<4;k++){px[k]=Hp[k];py[k]=Hp[k+4];qx[k]=Lp[k];qy[k]=Lp[k+4];}
            _PointAddSecp256k1(px,py,pz,qx,qy);
        }
    }
    qsb_field_normalize(pz);
    const bool zero=!(pz[0]|pz[1]|pz[2]|pz[3]);
    if(zero)pz[0]=1;
    qsb_gtable_inverse256(pz);
    if(zero){pz[0]=pz[1]=pz[2]=pz[3]=0;}
    if(ch>=0) {
        _ModMult(px,pz); _ModMult(py,pz);
        qsb_field_normalize(px); qsb_field_normalize(py);
        uint64_t *dst=(uint64_t*)(gTable+t*64u);
        #pragma unroll
        for(int k=0;k<4;k++){dst[k]=px[k];dst[k+4]=py[k];}
    }
}
