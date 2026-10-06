#pragma once
#ifndef QSB_INVERSE_CORRECTION
#define QSB_INVERSE_CORRECTION 1
#endif
#ifndef QSB_INVERSE_BIAS
#define QSB_INVERSE_BIAS 1
#endif
#ifndef QSB_INVERSE_SIGNMAC
#define QSB_INVERSE_SIGNMAC 0
#endif
#if QSB_INVERSE_SIGNMAC < 0 || QSB_INVERSE_SIGNMAC > 1
#error "QSB_INVERSE_SIGNMAC must be 0 or 1"
#endif
#if QSB_INVERSE_SIGNMAC
/* Signed32 coefficients times raw unsigned32 limbs, exact modulo 2^64.
 * Accumulate unsigned products first; then apply both sign corrections to
 * high32. Actual selected LUT coefficients make the signed64 sum defined. */
__device__ __forceinline__ int64_t zi_signmac32(int32_t a,uint32_t x,int32_t b,uint32_t y){
    int64_t value;
    asm("{ .reg .u64 total; .reg .u32 lo,hi,am,bm;\n"
        "mul.wide.u32 total,%1,%2;\n"
        "mad.wide.u32 total,%3,%4,total;\n"
        "shr.s32 am,%1,31; and.b32 am,am,%2;\n"
        "shr.s32 bm,%3,31; and.b32 bm,bm,%4;\n"
        "mov.b64 {lo,hi},total;\n"
        "sub.u32 hi,hi,am; sub.u32 hi,hi,bm;\n"
        "mov.b64 %0,{lo,hi}; }"
        : "=l"(value) : "r"(a),"r"(x),"r"(b),"r"(y));
    return value;
}
#endif
/* One full warp per inverse: four signed rows, eight low limbs per row.
 * Each row's signed ninth limb is replicated. Normalize independent limb
 * products with ballot carry/borrow lookahead, then perform the exact >>30.
 * The original cap, isomorphic scale and independent fallback are retained. */
__device__ __forceinline__ bool zi_inverse_limbs_bounded(uint64_t *R,int lane){
    constexpr unsigned mask=0xffffffffu;
    const int digit=lane&7,row=lane>>3,start=lane&~7;
    const unsigned odd=row&1,rs=row>>1;
    const uint64_t rw=digit<2?R[0]:digit<4?R[1]:digit<6?R[2]:R[3];
    const uint32_t xl=(uint32_t)(rw>>(32*(digit&1)));
#if defined(QSB_ISO_FUSED_ROOT_SCALE) && QSB_ISO_FUSED_ROOT_SCALE
    const uint64_t sw=QSB_ISO_INVU[digit>>1];
    const uint32_t scaled=(uint32_t)(sw>>(32*(digit&1)));
#else
    const uint32_t scaled=(uint32_t)(digit==0);
#endif
    const uint32_t pl=digit==0?0xfffffc2fu:digit==1?0xfffffffeu:0xffffffffu;
    uint32_t x=odd?(rs?scaled:xl):(rs?0u:pl);
    int32_t xt=0,delta=1;
    unsigned batches=0;
#if defined(QSB_DIVSTEP_LOOKAHEAD) && QSB_DIVSTEP_LOOKAHEAD
    /* QSB_DIVSTEP_LOOKAHEAD (tree.cu): the decision for the first batch is peeled; inside the loop
     * the next batch's decision is formed as soon as acc exists, from the new low limbs of f and g, and
     * takes effect after the termination vote. Why the early low limb is exact (both QSB_INVERSE_BIAS
     * forms): on digit 0 prev is 0, so low0 = (u32)acc0 and carry bit 0 is 0; carry bit 1 is gen0 =
     * (low0 + 0 < low0) = 0, so low1 = (u32)biased1 + (u32)(biased0 >> 32) = (u32)acc1 + (u32)(acc0 >> 32)
     * mod 2^32 (the two 2^31 biases cancel); rows f and g (rs = 0) take no Montgomery correction (m = 0),
     * so their acc is final here. The new x on digit 0 is (low0 >> 30) | (low1 << 2). */
    int32_t top,bottom;
    {
        const uint32_t f0=__shfl_sync(mask,x,0),g0=__shfl_sync(mask,x,8);
        delta=zi_divstep30_column(delta,f0,g0,rs,&top,&bottom);
    }
    while(true){
        if(batches==ZI_ROOT_MAX_BATCHES)return false;
        ++batches;
        const int32_t selected=odd?bottom:top;
        const int32_t a=__shfl_sync(mask,selected,odd?24:0);
        const int32_t b=__shfl_sync(mask,selected,odd?8:16);
        const uint32_t y=__shfl_sync(mask,x,lane^8);
        const int32_t yt=__shfl_sync(mask,xt,lane^8);
#if QSB_INVERSE_SIGNMAC
        int64_t acc=zi_signmac32(a,x,b,y);
#else
        int64_t acc=(int64_t)a*(int64_t)x+(int64_t)b*(int64_t)y;
#endif
        const uint32_t la_acc1=__shfl_down_sync(mask,(uint32_t)acc,1,8);
        const uint32_t la_x0=((uint32_t)acc>>30)|((la_acc1+(uint32_t)(acc>>32))<<2);
        const uint32_t la_f0=__shfl_sync(mask,la_x0,0),la_g0=__shfl_sync(mask,la_x0,8);
        int32_t la_top,la_bottom;
        const int32_t la_delta=zi_divstep30_column(delta,la_f0,la_g0,rs,&la_top,&la_bottom);
        uint32_t m=__shfl_sync(mask,(uint32_t)acc,start);
#else
    while(true){
        if(batches==ZI_ROOT_MAX_BATCHES)return false;
        ++batches;
        const uint32_t f0=__shfl_sync(mask,x,0),g0=__shfl_sync(mask,x,8);
#if QSB_DIVSTEP_4LANE
        /* QSB_DIVSTEP_4LANE (tree.cu): the decision runs only on lanes 0, 8, 16
         * and 24, the four lanes whose top/bottom the shuffles below read; delta is read nowhere else. */
        int32_t top=0,bottom=0;
        if(digit==0)delta=zi_divstep30_column(delta,f0,g0,rs,&top,&bottom);
#else
        int32_t top,bottom;
        delta=zi_divstep30_column(delta,f0,g0,rs,&top,&bottom);
#endif
        const int32_t selected=odd?bottom:top;
        const int32_t a=__shfl_sync(mask,selected,odd?24:0);
        const int32_t b=__shfl_sync(mask,selected,odd?8:16);
        const uint32_t y=__shfl_sync(mask,x,lane^8);
        const int32_t yt=__shfl_sync(mask,xt,lane^8);
#if QSB_INVERSE_SIGNMAC
        int64_t acc=zi_signmac32(a,x,b,y);
#else
        int64_t acc=(int64_t)a*(int64_t)x+(int64_t)b*(int64_t)y;
#endif
        uint32_t m=__shfl_sync(mask,(uint32_t)acc,start);
#endif
        m=(m*ZI_MM32)&ZI_MASK30&(0u-rs);
#if QSB_INVERSE_CORRECTION
        // The sparse modulus correction is one unsigned 32x32 product.
        // All eight lanes follow the same instruction stream.
        const uint32_t factor=digit==0?977u:digit==1?1u:0u;
        acc-=(int64_t)((uint64_t)factor*m);
#else
        if(digit==0)acc-=(int64_t)977*m;
        if(digit==1)acc-=(int64_t)m;
#endif
        int64_t high=(int64_t)a*xt+(int64_t)b*yt+(int64_t)m;
#if QSB_INVERSE_BIAS
        /* Add K*B to limbs 0..7, subtract K from limbs 1..8. The
         * telescoping bias is zero, and each low-limb accumulator is unsigned.
         * K=2^31 exceeds the absolute signed carry bound of the divstep rows. */
        const uint64_t biased=(uint64_t)acc+0x8000000000000000ULL-
                              (digit?0x80000000ULL:0ULL);
        const uint32_t lo=(uint32_t)biased,hi=(uint32_t)(biased>>32);
        const uint32_t prev0=__shfl_up_sync(mask,hi,1,8);
        const uint32_t prev=digit?prev0:0u;
        const uint32_t sum=lo+prev;
        const unsigned gen=(__ballot_sync(mask,sum<lo)>>start)&255u;
        const unsigned prop=(__ballot_sync(mask,sum==0xffffffffu)>>start)&255u;
        const unsigned carry=((prop+(gen<<1))^prop);
        const uint32_t low=sum+((carry>>digit)&1u);
        high+=(int64_t)__shfl_sync(mask,hi,start+7)-0x80000000LL;
        high+=(int64_t)((carry>>8)&1u);
#else
        const uint32_t lo=(uint32_t)acc;
        const int32_t hi=(int32_t)(acc>>32);
        const int32_t prev0=__shfl_up_sync(mask,hi,1,8);
        const int32_t prev=digit?prev0:0;
        const uint32_t positive=prev>0?(uint32_t)prev:0u;
        const uint32_t negative=prev<0?0u-(uint32_t)prev:0u;
        const uint32_t plus=lo+positive;
        const unsigned pg=(__ballot_sync(mask,plus<lo)>>start)&255u;
        const unsigned pp=(__ballot_sync(mask,plus==0xffffffffu)>>start)&255u;
        const unsigned carry=((pp+(pg<<1))^pp);
        const uint32_t sum=plus+((carry>>digit)&1u);
        const uint32_t minus=sum-negative;
        const unsigned bg=(__ballot_sync(mask,sum<negative)>>start)&255u;
        const unsigned bp=(__ballot_sync(mask,minus==0u)>>start)&255u;
        const unsigned borrow=((bp+(bg<<1))^bp);
        const uint32_t low=minus-((borrow>>digit)&1u);
        high+=(int64_t)__shfl_sync(mask,hi,start+7);
        high+=(int64_t)((carry>>8)&1u)-(int64_t)((borrow>>8)&1u);
#endif
        const uint32_t next0=__shfl_down_sync(mask,low,1,8);
        const uint32_t next=digit==7?(uint32_t)high:next0;
        x=(low>>30)|(next<<2);
        xt=(int32_t)(high>>30);
        if((__ballot_sync(mask,x!=0 || xt!=0)&0x0000ff00u)==0)break;
#if defined(QSB_DIVSTEP_LOOKAHEAD) && QSB_DIVSTEP_LOOKAHEAD
        delta=la_delta;top=la_top;bottom=la_bottom;   /* the look-ahead becomes the next batch's decision */
#endif
    }
    uint32_t out[9];
    #pragma unroll
    for(int i=0;i<8;i++)out[i]=__shfl_sync(mask,x,16+i);
    out[8]=(uint32_t)__shfl_sync(mask,xt,16);
    const uint32_t neg=(uint32_t)(__shfl_sync(mask,xt,0)<0);
    zi_condneg(out,neg);
    zi_canon(out);
    #pragma unroll
    for(int i=0;i<4;i++)R[i]=(uint64_t)out[2*i]|((uint64_t)out[2*i+1]<<32);
    R[4]=0;
    return true;
}
__device__ __forceinline__ void zi_inverse_limbs(uint64_t *R,int lane){
    if(zi_inverse_limbs_bounded(R,lane))return;
    if(lane==0){
        QsbInverseWords out=qsb_root_fermat({R[0],R[1],R[2],R[3]});
        R[0]=out.a;R[1]=out.b;R[2]=out.c;R[3]=out.d;
    }
    #pragma unroll
    for(int i=0;i<4;i++)R[i]=__shfl_sync(0xffffffffu,R[i],0);
    R[4]=0;
}
