// One work-efficient binary product tree per block. Four neighboring lanes
// keep the two lowest levels in registers; the remaining tree uses shared RAM.
// Requires a power-of-two block size in [32,256], whole-warp participation,
// and nonzero inputs (inactive tail lanes supply identity factors).
#pragma once
__device__ __forceinline__ void qsb_block_inverse_tree(uint64_t *value){
    __shared__ uint64_t tree[4][128];
    const int tid=threadIdx.x,n=blockDim.x/4,lane=tid&31;
    uint64_t sibling[5]={0,0,0,0,0},pair[5]={0,0,0,0,0},other[5]={0,0,0,0,0};
    #pragma unroll
    for(int k=0;k<4;k++)
        sibling[k]=__shfl_sync(0xffffffffu,(unsigned long long)value[k],lane^1);
    if((lane&1)==0)qsb_field_mul(pair,value,sibling);
    // Every lane reaches every shuffle; only the two even lanes hold products.
    // Select the opposite pair's even lane within this group of four.
    #pragma unroll
    for(int k=0;k<4;k++)
        other[k]=__shfl_sync(0xffffffffu,(unsigned long long)pair[k],(lane&~1)^2);
    if((lane&3)==0){
        qsb_field_mul(pair,pair,other);
        #pragma unroll
        for(int k=0;k<4;k++)tree[k][n+tid/4]=pair[k];
    }
    __syncthreads();
    #pragma unroll 1
    for(int width=n>>1;width>0;width>>=1){
        if(tid<width){
            int node=width+tid;
            uint64_t a[5]={0,0,0,0,0},b[5]={0,0,0,0,0};
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=tree[k][2*node];b[k]=tree[k][2*node+1];}
            qsb_field_mul(a,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)tree[k][node]=a[k];
        }
        __syncthreads();
    }
    if(tid==0){
        uint64_t root[5]={0,0,0,0,0};
        #pragma unroll
        for(int k=0;k<4;k++)root[k]=tree[k][1];
        _ModInv(root);
        #pragma unroll
        for(int k=0;k<4;k++)tree[k][1]=root[k];
    }
    __syncthreads();
    #pragma unroll 1
    for(int width=1;width<n;width<<=1){
        if(tid<width){
            int node=width+tid;
            uint64_t parent[5]={0,0,0,0,0},left[5]={0,0,0,0,0},right[5]={0,0,0,0,0};
            #pragma unroll
            for(int k=0;k<4;k++){
                parent[k]=tree[k][node];
                left[k]=tree[k][2*node];right[k]=tree[k][2*node+1];
            }
            qsb_field_mul(right,parent,right);qsb_field_mul(left,parent,left);
            #pragma unroll
            for(int k=0;k<4;k++){
                tree[k][2*node]=right[k];tree[k][2*node+1]=left[k];
            }
        }
        __syncthreads();
    }
    if((lane&1)==0){
        #pragma unroll
        for(int k=0;k<4;k++)pair[k]=tree[k][n+tid/4];
        qsb_field_mul(pair,pair,other);
    }
    #pragma unroll
    for(int k=0;k<4;k++)
        value[k]=__shfl_sync(0xffffffffu,(unsigned long long)pair[k],lane&~1);
    qsb_field_mul(value,value,sibling);
}
