#pragma once
/* Standalone root64 admission only. Four radix-2^64 digits and a replicated
 * signed top per row; lanes16..31 duplicate the first16 lanes. The original
 * D30 controller and its table are unchanged. No built-in signed128 arithmetic
 * or signed overflowing operations are used below. */
struct ZiRoot64Pair { uint64_t lo,hi; };
#ifndef QSB_ROOT64_SHARED_PRODUCTS
#define QSB_ROOT64_SHARED_PRODUCTS 1
#endif
#if QSB_ROOT64_SHARED_PRODUCTS < 0 || QSB_ROOT64_SHARED_PRODUCTS > 1
#error "QSB_ROOT64_SHARED_PRODUCTS must be0 or1"
#endif
#if QSB_ROOT64_SHARED_PRODUCTS
#if ZI_ROOT_MAX_BATCHES < 0 || ZI_ROOT_MAX_BATCHES > 32
#error "bounded signed-top proof requires a decision cap in[0,32]"
#endif
#include "root64_products.cuh"
#endif

__device__ __forceinline__ ZiRoot64Pair zi_root64_add(ZiRoot64Pair a,ZiRoot64Pair b){
    const uint64_t lo=a.lo+b.lo;
    return {lo,a.hi+b.hi+(uint64_t)(lo<a.lo)};
}
__device__ __forceinline__ ZiRoot64Pair zi_root64_sub(ZiRoot64Pair a,ZiRoot64Pair b){
    return {a.lo-b.lo,a.hi-b.hi-(uint64_t)(a.lo<b.lo)};
}
__device__ __forceinline__ ZiRoot64Pair zi_root64_mul_uu(uint64_t a,uint64_t b){
    return {a*b,__umul64hi(a,b)};
}
/* a is a signed64 bit pattern, x an unsigned radix digit. */
__device__ __forceinline__ ZiRoot64Pair zi_root64_mul_su(uint64_t a,uint64_t x){
#if QSB_ROOT64_SHARED_PRODUCTS
    ZiRoot64Pair r=zi_root64_shared_uu(a,x);
    r.hi-=((a>>63)?x:0ULL); // exact original signed64*unsigned64 correction.
    return r;
#else
    return {a*x,__umul64hi(a,x)-((a>>63)?x:0ULL)};
#endif
}
/* Both inputs are signed64 bit patterns; apply BOTH sign corrections. */
__device__ __forceinline__ ZiRoot64Pair zi_root64_mul_ss(uint64_t a,uint64_t b){
    return {a*b,__umul64hi(a,b)-((a>>63)?b:0ULL)-((b>>63)?a:0ULL)};
}

template<int bits>
__device__ __forceinline__ void zi_root64_row(uint64_t &x,uint64_t &xt,
                                             uint64_t selected,int lane){
    static_assert(bits==30 || bits==60,"root64 update width");
    constexpr unsigned mask=0xffffffffu;
    constexpr uint64_t mm=0x0838091dd2253531ULL; // (2^32+977)^-1 mod2^60
    constexpr uint64_t digit_mask=(1ULL<<bits)-1ULL;
    constexpr uint64_t K=1ULL<<(bits+1);
    const int digit=lane&3,row=(lane&15)>>2,start=lane&~3;
    const unsigned odd=row&1,rs=row>>1;
    const uint64_t a=__shfl_sync(mask,selected,odd?12:0);
    const uint64_t b=__shfl_sync(mask,selected,odd?4:8);
    const uint64_t y=__shfl_sync(mask,x,lane^4);
    const uint64_t yt=__shfl_sync(mask,xt,lane^4);
    ZiRoot64Pair acc=zi_root64_add(zi_root64_mul_su(a,x),zi_root64_mul_su(b,y));
    uint64_t m=__shfl_sync(mask,acc.lo,start);
    m=(m*mm)&digit_mask&(0ULL-(uint64_t)rs);
    // p=2^256-(2^32+977): sparse low subtraction plus signed-top +m.
    acc=zi_root64_sub(acc,zi_root64_mul_uu(m,digit==0?0x1000003d1ULL:0ULL));
#if QSB_ROOT64_SHARED_PRODUCTS
    ZiRoot64Pair top=zi_root64_add(zi_root64_bounded_st(a,xt),zi_root64_bounded_st(b,yt));
#else
    ZiRoot64Pair top=zi_root64_add(zi_root64_mul_ss(a,xt),zi_root64_mul_ss(b,yt));
#endif
    top=zi_root64_add(top,{m,0ULL});
    /* K*B on every digit, -K on digits1..3 and the top telescopes to0.
     * Biased highs lie in[0,2^(bits+2)), so one binary carry scan suffices. */
    acc.hi+=K;
    acc=zi_root64_sub(acc,{digit?K:0ULL,0ULL});
    top=zi_root64_sub(top,{K,0ULL});
    const uint64_t prev0=__shfl_up_sync(mask,acc.hi,1,4);
    const uint64_t prev=digit?prev0:0ULL;
    const uint64_t sum=acc.lo+prev;
    const unsigned gen=(__ballot_sync(mask,sum<acc.lo)>>start)&15u;
    const unsigned prop=(__ballot_sync(mask,sum==0xffffffffffffffffULL)>>start)&15u;
    const unsigned carry=(prop+(gen<<1))^prop;
    const uint64_t low=sum+((carry>>digit)&1u);
    const uint64_t end=__shfl_sync(mask,acc.hi,start+3)+((carry>>4)&1u);
    top=zi_root64_add(top,{end,0ULL});
    const uint64_t next0=__shfl_down_sync(mask,low,1,4);
    const uint64_t next=digit==3?top.lo:next0;
    x=(low>>bits)|(next<<(64-bits));
    // Exact signed128 floor shift; the bounded normalized top fits signed64.
    xt=(top.lo>>bits)|(top.hi<<(64-bits));
}

__device__ __forceinline__ int32_t zi_root64_decision60(
    int32_t delta,uint64_t x,unsigned column,unsigned odd,int lane,uint64_t *selected){
    constexpr unsigned mask=0xffffffffu;
    const uint64_t f=__shfl_sync(mask,x,0),g=__shfl_sync(mask,x,4);
    int32_t t1,b1;
    delta=zi_divstep30_column(delta,(uint32_t)f,(uint32_t)g,column,&t1,&b1);
    const int32_t e1=odd?b1:t1;
    // Signed32 coefficient is sign-extended BEFORE unsigned wrapping multiply.
    const uint64_t contribution=(uint64_t)(int64_t)e1*(column?g:f);
    const uint64_t bridge=contribution+__shfl_sync(mask,contribution,lane^8);
    const uint32_t next=(uint32_t)(bridge>>30);
    const uint32_t nf=__shfl_sync(mask,next,0),ng=__shfl_sync(mask,next,4);
    int32_t t2,b2;
    delta=zi_divstep30_column(delta,nf,ng,column,&t2,&b2);
    const int32_t e2=odd?b2:t2;
    const int32_t left=__shfl_sync(mask,e2,odd?4:0);
    const int32_t right=__shfl_sync(mask,e2,odd?12:8);
    // Row L1<=2^60 makes both signed64 products and their sum representable.
    const int64_t composed=(int64_t)left*(int64_t)t1+(int64_t)right*(int64_t)b1;
    *selected=(uint64_t)composed;
    return delta;
}

#if QSB_ROOT64_TEST_STATS
// Test only: calls, successes, consumed D30 batches, caps, odd-tail successes.
static __device__ unsigned qsb_root64_test_stats[5];
#endif

__device__ __forceinline__ bool zi_inverse_root64_bounded(uint64_t *R,int lane){
    constexpr unsigned mask=0xffffffffu;
    const int h=lane&15,digit=h&3,row=h>>2;
    const unsigned odd=row&1,column=row>>1;
#if defined(QSB_ISO_FUSED_ROOT_SCALE) && QSB_ISO_FUSED_ROOT_SCALE
    const uint64_t scaled=QSB_ISO_INVU[digit];
#else
    const uint64_t scaled=(uint64_t)(digit==0);
#endif
    const uint64_t pl=digit==0?0xfffffffefffffc2fULL:0xffffffffffffffffULL;
    const uint64_t rw=digit==0?R[0]:digit==1?R[1]:digit==2?R[2]:R[3];
    uint64_t x=odd?(column?scaled:rw):(column?0ULL:pl);
    uint64_t xt=0ULL;
    int32_t delta=1;
    unsigned batches=0;
#if QSB_ROOT64_TEST_STATS
    if(lane==0)atomicAdd(qsb_root64_test_stats,1u);
#endif
    while(batches+2u<=(unsigned)ZI_ROOT_MAX_BATCHES){
        uint64_t selected;
        delta=zi_root64_decision60(delta,x,column,odd,lane,&selected);
        zi_root64_row<60>(x,xt,selected,lane);
        batches+=2u;
        if((__ballot_sync(mask,x!=0ULL || xt!=0ULL)&0x000000f0u)==0u)goto success;
    }
#if (ZI_ROOT_MAX_BATCHES & 1)
    {
        const uint32_t f0=(uint32_t)__shfl_sync(mask,x,0),g0=(uint32_t)__shfl_sync(mask,x,4);
        int32_t top,bottom;
        delta=zi_divstep30_column(delta,f0,g0,column,&top,&bottom);
        const uint64_t selected=(uint64_t)(int64_t)(odd?bottom:top);
        zi_root64_row<30>(x,xt,selected,lane);
        ++batches;
        if((__ballot_sync(mask,x!=0ULL || xt!=0ULL)&0x000000f0u)==0u)goto success;
    }
#endif
#if QSB_ROOT64_TEST_STATS
    if(lane==0){atomicAdd(qsb_root64_test_stats+2,batches);atomicAdd(qsb_root64_test_stats+3,1u);}
#endif
    return false; // R is untouched: the existing scaled Fermat sees the root.
success:
    uint32_t out[9];
    #pragma unroll
    for(int i=0;i<4;i++){
        const uint64_t v=__shfl_sync(mask,x,8+i);
        out[2*i]=(uint32_t)v;out[2*i+1]=(uint32_t)(v>>32);
    }
    out[8]=(uint32_t)__shfl_sync(mask,xt,8);
    const uint32_t neg=(uint32_t)(__shfl_sync(mask,xt,0)>>63);
    zi_condneg(out,neg);
    zi_canon(out);
    #pragma unroll
    for(int i=0;i<4;i++)R[i]=(uint64_t)out[2*i]|((uint64_t)out[2*i+1]<<32);
    R[4]=0;
#if QSB_ROOT64_TEST_STATS
    if(lane==0){
        atomicAdd(qsb_root64_test_stats+1,1u);atomicAdd(qsb_root64_test_stats+2,batches);
        if(batches&1u)atomicAdd(qsb_root64_test_stats+4,1u);
    }
#endif
    return true;
}
