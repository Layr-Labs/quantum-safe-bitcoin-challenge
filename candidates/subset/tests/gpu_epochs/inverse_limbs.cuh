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
#if QSB_C3_TREE_GLUE
    /* ==== cut3 R3: QSB_C3_TREE_GLUE (tree.cu) ==== the base loop (QSB_DIVSTEP_4LANE, QSB_INVERSE_CORRECTION and
     * QSB_INVERSE_BIAS forms below) with less per-batch glue; the same shuffles, products, votes and carries in the
     * same order. bit 0: no batch cap; bit 1: the decision lanes form `selected`; bit 2: the sparse correction as one
     * signed multiply-add with the loop-invariant -factor (see the knob comment in tree.cu). */
#if QSB_C3_TREE_GLUE & 4
    const int32_t nfactor=-(int32_t)(digit==0?977u:digit==1?1u:0u);   /* -factor, |factor| <= 977 */
#endif
#if QSB_C3_TREE_GLUE & 256
    /* bit 8: loop-invariant lane mask for the top digit's next word, and the base's per-lane carry bias */
    const uint32_t m7=digit==7?0xffffffffu:0u;
    const uint64_t Kb=0x8000000000000000ULL-(digit?0x80000000ULL:0ULL);
#endif
    while(true){
#if !(QSB_C3_TREE_GLUE & 1)
        if(batches==ZI_ROOT_MAX_BATCHES)return false;
        ++batches;
#endif
#if QSB_C3_TREE_GLUE & 8
        /* bit 3: the decision on lanes 0 and 8 only, lane 0 forming column 0 (u, q), lane 8 column 1 (v, r) of the same
         * 30-step matrix (the column form's columns are independent: the same integers as on lanes 0/8 and 16/24 of the
         * base). f0 and g0 are lane 0's and lane 8's x, which the two lanes already hold as (x, y): y is the base's own
         * partner shuffle (lane ^ 8), issued before the decision instead of after it; no extra shuffle (a partial-mask
         * shuffle inside the branch would make ptxas wrap every shuffle of the function in WARPSYNC calls).
         * a = u (even rows) / r (odd rows), b = v / q. */
        const uint32_t y=__shfl_sync(mask,x,lane^8);   /* the base's partner shuffle, issued before the decision */
        int32_t sela,selb;
        asm("" : "=r"(sela));        /* lanes other than 0 and 8: never read by the two shuffles below */
        asm("" : "=r"(selb));
        if((lane&~8)==0){
            /* lane 0: x = f0, y = g0; lane 8: x = g0, y = f0 (the base's two shuffles from lanes 0 and 8) */
            const uint32_t f0=odd?y:x,g0=odd?x:y;
            int32_t top,bottom;
            delta=zi_divstep30_column(delta,f0,g0,odd,&top,&bottom);
            sela=odd?bottom:top;
            selb=odd?top:bottom;
        }
        const int32_t a=__shfl_sync(mask,sela,odd?8:0);
        const int32_t b=__shfl_sync(mask,selb,odd?0:8);
#else
        const uint32_t f0=__shfl_sync(mask,x,0),g0=__shfl_sync(mask,x,8);
#if QSB_C3_TREE_GLUE & 2
        int32_t selected;
        asm("" : "=r"(selected));    /* lanes with digit != 0: never read by the two shuffles below */
        if(digit==0){
            int32_t top,bottom;
            delta=zi_divstep30_column(delta,f0,g0,rs,&top,&bottom);
            selected=odd?bottom:top;
        }
#else
        int32_t top=0,bottom=0;
        if(digit==0)delta=zi_divstep30_column(delta,f0,g0,rs,&top,&bottom);
        const int32_t selected=odd?bottom:top;
#endif
        const int32_t a=__shfl_sync(mask,selected,odd?24:0);
        const int32_t b=__shfl_sync(mask,selected,odd?8:16);
        const uint32_t y=__shfl_sync(mask,x,lane^8);
#endif
        const int32_t yt=__shfl_sync(mask,xt,lane^8);
#if QSB_C3_TREE_GLUE & 256
        /* bit 8: the accumulator biased from the start, Kb + a*x + b*y + (-factor)*m mod 2^64 = the base's `biased` word;
         * the row-start lanes (digit 0) have Kb's low word 0, so the low word shuffled for m is acc's own; the correction
         * is one PTX signed multiply-add (m < 2^30, |factor| <= 977: exact). */
        uint64_t accb=Kb+(uint64_t)((int64_t)a*(int64_t)x)+(uint64_t)((int64_t)b*(int64_t)y);
        uint32_t m=__shfl_sync(mask,(uint32_t)accb,start);
        m=(m*ZI_MM32)&ZI_MASK30&(0u-rs);
        {
            int64_t t;
            asm("mad.wide.s32 %0, %1, %2, %3;" : "=l"(t) : "r"(nfactor), "r"((int32_t)m), "l"((int64_t)accb));
            accb=(uint64_t)t;
        }
        /* (d): the top limb's -2^31 enters with m: (int32)(m | 2^31) = m - 2^31 for m < 2^30 */
        int64_t high=(int64_t)a*xt+(int64_t)b*yt+(int64_t)(int32_t)(m|0x80000000u);
        const uint64_t biased=accb;
#else
        int64_t acc=(int64_t)a*(int64_t)x+(int64_t)b*(int64_t)y;
        uint32_t m=__shfl_sync(mask,(uint32_t)acc,start);
        m=(m*ZI_MM32)&ZI_MASK30&(0u-rs);
#if QSB_C3_TREE_GLUE & 4
        acc+=(int64_t)nfactor*(int64_t)(int32_t)m;   /* m < 2^30: = acc - factor*m exactly (no wrap) */
#else
        {
            const uint32_t factor=digit==0?977u:digit==1?1u:0u;
            acc-=(int64_t)((uint64_t)factor*m);
        }
#endif
        int64_t high=(int64_t)a*xt+(int64_t)b*yt+(int64_t)m;
        const uint64_t biased=(uint64_t)acc+0x8000000000000000ULL-
                              (digit?0x80000000ULL:0ULL);
#endif
        const uint32_t lo=(uint32_t)biased,hi=(uint32_t)(biased>>32);
        const uint32_t prev0=__shfl_up_sync(mask,hi,1,8);
        const uint32_t prev=digit?prev0:0u;
        const uint32_t sum=lo+prev;
        const unsigned gen=(__ballot_sync(mask,sum<lo)>>start)&255u;
        const unsigned prop=(__ballot_sync(mask,sum==0xffffffffu)>>start)&255u;
        const unsigned carry=((prop+(gen<<1))^prop);
        const uint32_t low=sum+((carry>>digit)&1u);
#if QSB_C3_TREE_GLUE & 256
        high+=(int64_t)__shfl_sync(mask,hi,start+7);   /* bit 8 (d): its -2^31 was added with m */
#else
        high+=(int64_t)__shfl_sync(mask,hi,start+7)-0x80000000LL;
#endif
        high+=(int64_t)((carry>>8)&1u);
        const uint32_t next0=__shfl_down_sync(mask,low,1,8);
#if QSB_C3_TREE_GLUE & 256
        uint32_t next;   /* bit 8: digit == 7 ? high : next0 as one lop3 with the invariant lane mask m7 */
        asm("lop3.b32 %0, %1, %2, %3, 0xE4;" : "=r"(next) : "r"((uint32_t)high), "r"(next0), "r"(m7));
#else
        const uint32_t next=digit==7?(uint32_t)high:next0;
#endif
        x=(low>>30)|(next<<2);
        xt=(int32_t)(high>>30);
        if((__ballot_sync(mask,x!=0 || xt!=0)&0x0000ff00u)==0)break;
    }
    (void)batches;
#else   /* QSB_C3_TREE_GLUE 0: the base loop byte for byte */
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
        int64_t acc=(int64_t)a*(int64_t)x+(int64_t)b*(int64_t)y;
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
        int64_t acc=(int64_t)a*(int64_t)x+(int64_t)b*(int64_t)y;
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
#endif   /* QSB_C3_TREE_GLUE */
    uint32_t out[9];
    #pragma unroll
    for(int i=0;i<8;i++)out[i]=__shfl_sync(mask,x,16+i);
    out[8]=(uint32_t)__shfl_sync(mask,xt,16);
    const uint32_t neg=(uint32_t)(__shfl_sync(mask,xt,0)<0);
#if QSB_C3_TREE_GLUE & 16
    /* cut3 R3, QSB_C3_TREE_GLUE bit 4: the canonical form on lane 0 only; every caller reads lane 0's R alone (the
     * wave top broadcasts it: root[k] = __shfl_sync(.., root[k], 0)), so lanes 1..31 computed an unread copy. */
    if(lane==0){zi_condneg(out,neg);zi_canon(out);}
#else
    zi_condneg(out,neg);
    zi_canon(out);
#endif
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
