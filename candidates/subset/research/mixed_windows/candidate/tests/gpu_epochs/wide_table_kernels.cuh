// Chunked GPU table construction; uses the inherited corrected product-tree helpers.
// Each table entry is H[hi]+L[lo], where m=2*index+1=hi*256+lo.
#pragma once
__host__ __device__ __forceinline__ void wide_decode_entry(uint32_t t,int *ch,uint32_t *hi,uint32_t *lo){
    *ch=t<(1u<<17)?0:1+(int)((t-(1u<<17))>>16);
    uint32_t m=2*(t-gt_offset(*ch))+1;
    *hi=m>>8;*lo=m&255u;
}
__device__ __forceinline__ void wide_finish_affine(uint64_t *x,uint64_t *y,const uint64_t *H,const uint64_t *L,uint64_t *inv){
    uint64_t slope[4];
    _ModSub256(slope,(uint64_t*)L+4,(uint64_t*)H+4);
    _ModMult(slope,inv);
    _ModSqr(x,slope);
    _ModSub256(x,(uint64_t*)H);
    _ModSub256(x,(uint64_t*)L);
    _ModSub256(y,(uint64_t*)H,x);
    _ModMult(y,slope);
    _ModSub256(y,(uint64_t*)H+4);
}
__global__ void __launch_bounds__(256,2) wide_table_prepare(
    const uint64_t *L,const uint64_t *H,uint8_t *table,uint32_t start,int count,
    uint64_t *roots,uint64_t *tree){
    int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);bool active=i<count;
    uint64_t d[5]={1,0,0,0,0};
    if(active){
        uint32_t t=start+(uint32_t)i,hi,lo;int ch;wide_decode_entry(t,&ch,&hi,&lo);
        if(hi){
            const uint64_t *hp=H+((size_t)ch*GT_HI+hi)*8;
            const uint64_t *lp=L+((size_t)ch*GT_LO+lo)*8;
            _ModSub256(d,(uint64_t*)lp,(uint64_t*)hp);
        }
        // Preserve each denominator in the not-yet-built X coordinate.
        memcpy(table+(size_t)t*64,d,32);
    }
    qsb_block_product_checkpoint(d,roots,tree);
}
__global__ void __launch_bounds__(256,2) wide_table_finish(
    const uint64_t *L,const uint64_t *H,uint8_t *table,uint32_t start,int count,
    const uint64_t *roots,const uint64_t *tree){
    int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);bool active=i<count;
    uint32_t t=start+(uint32_t)i;uint64_t inv[5]={1,0,0,0,0};
    if(active)memcpy(inv,table+(size_t)t*64,32);
    qsb_block_inverse_checkpoint(inv,roots,tree);
    if(!active)return;
    uint32_t hi,lo;int ch;wide_decode_entry(t,&ch,&hi,&lo);
    const uint64_t *lp=L+((size_t)ch*GT_LO+lo)*8;
    uint64_t x[4],y[4];
    if(hi){
        const uint64_t *hp=H+((size_t)ch*GT_HI+hi)*8;
        wide_finish_affine(x,y,hp,lp,inv);
    }else{Load256(x,lp);Load256(y,lp+4);}
    memcpy(table+(size_t)t*64,x,32);memcpy(table+(size_t)t*64+32,y,32);
}
