// One work-efficient binary product tree per block. The caller supplies
// a power-of-two block size at most 256 and identity factors for inactive lanes.
#pragma once
__device__ __forceinline__ void qsb_block_inverse_tree(uint64_t *value){
    __shared__ uint64_t tree[4][512];
    const int tid=threadIdx.x,n=blockDim.x;
    #pragma unroll
    for(int k=0;k<4;k++)tree[k][n+tid]=value[k];
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
        if(width>32)__syncthreads();else __syncwarp();
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
    for(int width=1;width<(n>=4 ? n>>2 : n);width<<=1){
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
        int next=width<<1;
        if(next>32)__syncthreads();else __syncwarp();
    }

    // Fuse the last two down-sweep levels. Each owner reads the four original
    // leaf products before replacing them; neighboring owners never access
    // this subtree. The intermediate pair inverses stay in registers.
    if(n>=4){
        if(tid<(n>>2)){
            int node=(n>>2)+tid,leaf=4*node;
            uint64_t parent[5]={},left[5]={},right[5]={},other[5]={};
            #pragma unroll
            for(int k=0;k<4;k++){
                parent[k]=tree[k][node];left[k]=tree[k][2*node];right[k]=tree[k][2*node+1];
            }
            qsb_field_mul(right,parent,right); // inverse of the left pair
            qsb_field_mul(left,parent,left);   // inverse of the right pair
            #pragma unroll
            for(int k=0;k<4;k++){parent[k]=tree[k][leaf];other[k]=tree[k][leaf+1];}
            qsb_field_mul(other,right,other);qsb_field_mul(parent,right,parent);
            #pragma unroll
            for(int k=0;k<4;k++){tree[k][leaf]=other[k];tree[k][leaf+1]=parent[k];}
            #pragma unroll
            for(int k=0;k<4;k++){parent[k]=tree[k][leaf+2];other[k]=tree[k][leaf+3];}
            qsb_field_mul(other,left,other);qsb_field_mul(parent,left,parent);
            #pragma unroll
            for(int k=0;k<4;k++){tree[k][leaf+2]=other[k];tree[k][leaf+3]=parent[k];}
        }
        if(n>32)__syncthreads();else __syncwarp();
    }
    #pragma unroll
    for(int k=0;k<4;k++)value[k]=tree[k][n+tid];
    value[4]=0;
}
