// Isolated research helpers. GPL-3.0; inherited notices remain in candidate/COPYING.
#pragma once
__device__ __forceinline__ void qsb_block_product_only(
    uint64_t *value, uint64_t *roots
) {
    __shared__ uint64_t products[4][512];
    int tid=threadIdx.x;

    #pragma unroll
    for(int k=0;k<4;k++)products[k][tid]=value[k];
    __syncthreads();

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
            int node=offset+count+tid;
            #pragma unroll
            for(int k=0;k<4;k++){
                products[k][node]=out[k];
            }
        }
        offset+=count;
        if(count>2)__syncthreads();
    }

    if(tid==0){
        #pragma unroll
        for(int k=0;k<4;k++)roots[(size_t)blockIdx.x*4u+k]=products[k][510];
    }
}

__device__ __forceinline__ void qsb_block_inverse_rebuild(uint64_t *value,const uint64_t *roots) {
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
    for(int count=256;count>2;count>>=1){
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
        #pragma unroll
        for(int k=0;k<4;k++)inverses[k][254]=roots[(size_t)blockIdx.x*4u+k];
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

__device__ __forceinline__ uint32_t qsb_xyzz_finish_three(
    uint64_t *Y, uint64_t *W, uint64_t *ZZZ,
    uint64_t *inv, uint64_t *xR, uint64_t *yR,
    uint64_t *x1, uint64_t *x2
) {
    uint64_t delta[4],yb[4],m[4],t[4],s[4];
    // inv is 1/(W*B), with B=ZZZ. Since C*B^2=W^2,
    // C/W = W/B^2; h=B/W. No fourth checkpoint field is needed.
    _ModMult(yb,yR,ZZZ);
    _ModMult(delta,W,inv);      // 1/B
    _ModSqr(delta,delta);
    _ModMult(delta,W);          // W/B^2 = xR-xP
    _ModSqr(ZZZ,ZZZ);
    _ModMult(ZZZ,inv);          // B/W
    _ModAdd256(W,xR,xR);
    _ModSub256(W,delta);        // xP+xR

    _ModSub256(m, yb, Y);
    _ModMult(m, ZZZ);            /* lambda1 = (yR*B-Y)*h */
    _ModSqr(x1, m);
    _ModSub256(x1, W);
    _ModSub256(t, xR, x1);
    _ModMult(s, m, t);
    _ModSub256(s, yR);
    uint32_t parities = (uint32_t)(s[0] & 1ULL);

    _ModAdd256(m, yb, Y);
    _ModMult(m, ZZZ);            /* m2 = (yR*B+Y)*h = -lambda2 */
    _ModSqr(x2, m);
    _ModSub256(x2, W);
    _ModSub256(t, xR, x2);
    _ModMult(s, m, t);
    _ModSub256(s, yR);
    /* y2=-s. Since p is odd, field negation flips its parity. */
    parities |= (uint32_t)((((s[0]|s[1]|s[2]|s[3]) != 0) && !(s[0]&1ULL)) << 1);
    return parities;
}
