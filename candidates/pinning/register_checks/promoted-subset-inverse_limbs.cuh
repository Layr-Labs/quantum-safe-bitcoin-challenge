#pragma once
#ifndef QSB_INVERSE_CORRECTION
#define QSB_INVERSE_CORRECTION 1
#endif
#ifndef QSB_INVERSE_BIAS
#define QSB_INVERSE_BIAS 1
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
    while(true){
        if(batches==ZI_ROOT_MAX_BATCHES)return false;
        ++batches;
        const uint32_t f0=__shfl_sync(mask,x,0),g0=__shfl_sync(mask,x,8);
        int32_t top,bottom;
        delta=zi_divstep30_column(delta,f0,g0,rs,&top,&bottom);
        const int32_t selected=odd?bottom:top;
        const int32_t a=__shfl_sync(mask,selected,odd?24:0);
        const int32_t b=__shfl_sync(mask,selected,odd?8:16);
        const uint32_t y=__shfl_sync(mask,x,lane^8);
        const int32_t yt=__shfl_sync(mask,xt,lane^8);
        int64_t acc=(int64_t)a*(int64_t)x+(int64_t)b*(int64_t)y;
        uint32_t m=__shfl_sync(mask,(uint32_t)acc,start);
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
