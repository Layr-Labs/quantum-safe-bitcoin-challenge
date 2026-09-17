/* GPL-3.0-only. Distributed-limb ownership of VanitySearch's root inverse.
 * Original divstep arithmetic: Jean Luc Pons, GPUMath.h. Exactly one full
 * warp participates; groups of eight lanes own U,V,R,S, six live words each. */
#pragma once
#ifndef HM43_WARP_ROOT
#define HM43_WARP_ROOT 1
#endif
// A finite cooperative budget closes the fixed-width accumulator proof.
// Smaller limits are supported for forced-fallback tests; zero uses the baseline.
#ifndef HM43_WARP_MAX_BATCHES
#define HM43_WARP_MAX_BATCHES 16
#endif
#if HM43_WARP_MAX_BATCHES < 0 || HM43_WARP_MAX_BATCHES > 16
#error "HM43_WARP_MAX_BATCHES must be between 0 and 16"
#endif
#ifdef HM43_HOST_ORACLE
uint64_t hm43_exchange(uint64_t value,int source);
uint32_t hm43_ballot(bool value);
__device__ __forceinline__ uint64_t hm43_mulhi(uint64_t a,uint64_t b){return (uint64_t)(((__uint128_t)a*b)>>64);}
#else
__device__ __forceinline__ uint64_t hm43_exchange(uint64_t value,int source){return __shfl_sync(0xffffffffu,(unsigned long long)value,source);}
__device__ __forceinline__ uint32_t hm43_ballot(bool value){return __ballot_sync(0xffffffffu,value);}
__device__ __forceinline__ uint64_t hm43_mulhi(uint64_t a,uint64_t b){return __umul64hi(a,b);}
#endif

// Full-mask operations must be called by all 32 lanes, including unused words.
__device__ __forceinline__ uint64_t hm43_add(uint64_t a,uint64_t b,int lane){
    const int word=lane&7;
    uint64_t sum=a+b;
    uint32_t g=hm43_ballot(word<5 && sum<a);
    uint32_t p=hm43_ballot(word<6 && sum==~0ULL);
    uint32_t carry=((g<<1)+p)^p;
    return sum+((carry>>lane)&1u);
}
__device__ __forceinline__ uint64_t hm43_negate(uint64_t x,int lane){
    uint32_t zeros=hm43_ballot(x==0);
    const uint32_t lower=((1u<<(lane&7))-1u)<<(lane&~7);
    return ~x+(uint64_t)((zeros&lower)==lower);
}
__device__ __forceinline__ uint64_t hm43_unsigned_mul(uint64_t x,uint64_t k,int lane){
    uint64_t lo=x*k,hi=hm43_mulhi(x,k);
    uint64_t previous=hm43_exchange(hi,(lane&~7)|((lane-1)&7));
    if((lane&7)==0)previous=0;
    return hm43_add(lo,previous,lane);
}
__device__ __forceinline__ uint64_t hm43_signed_mul(uint64_t x,int64_t k,int lane){
    uint64_t magnitude=k<0?(uint64_t)(-k):(uint64_t)k;
    uint64_t out=hm43_unsigned_mul(x,magnitude,lane);
    uint64_t negative=hm43_negate(out,lane);
    return k<0?negative:out;
}

// m < 2^62. For c=2^32+977, m*p = m*2^256 - m*c. Since c is
// odd and 0<=m<2^64, the low product is zero iff m==0; that determines
// the entire borrow chain without another distributed multiplication.
__device__ __forceinline__ uint64_t hm43_multiple_p(uint64_t m,int word){
    uint64_t al=m*0x1000003D1ULL,ah=hm43_mulhi(m,0x1000003D1ULL);
    uint64_t borrow=(m!=0);
    if(word==0)return 0ULL-al;
    if(word==1)return 0ULL-ah-borrow;
    if(word==2 || word==3)return 0ULL-borrow;
    if(word==4)return m-borrow;
    return 0;
}

// All32 lanes return the same completion flag. On false, result is untouched.
// True means completed arithmetic; noninvertible inputs retain the zero result.
__device__ __noinline__ bool hm43_warp_inverse(uint64_t result[5],int lane){
    const int group=lane>>3,word=lane&7,base=lane&~7;
    uint64_t state=0;
    #pragma unroll
    for(int j=0;j<5;j++){
        uint64_t root=hm43_exchange(result[j],0);
        if(group==1 && word==j)state=root;
    }
    if(group==0){
        if(word==0)state=0xFFFFFFFEFFFFFC2FULL;
        else if(word<4)state=~0ULL;
    }
    if(group==3 && word==0)state=1;
    uint32_t nonzero=0;
    unsigned completed_batches=0;
    while(true){
        // Uniform return before any next-batch collective. No shared root write.
        if(completed_batches==HM43_WARP_MAX_BATCHES)return false;
        ++completed_batches;
        uint32_t nz=hm43_ballot(word<5 && state!=0);
        uint32_t uv=(nz&31u)|((nz>>8)&31u);
        int pos=63-__clzll((uint64_t)uv);
        uint64_t u0=hm43_exchange(state,0),v0=hm43_exchange(state,8);
        uint64_t uh=hm43_exchange(state,pos),vh=hm43_exchange(state,8+pos);
        if(pos>0){
            uint64_t ul=hm43_exchange(state,pos-1),vl=hm43_exchange(state,7+pos);
            unsigned shift=__clzll(uh|vh);
            if(shift){uh=(uh<<shift)|(ul>>(64-shift));vh=(vh<<shift)|(vl>>(64-shift));}
        }
        int64_t uu=1,uvc=0,vu=0,vv=1;
        uint32_t bits=62;
        while(true){
            uint32_t zeros=_CTZ(v0|(1ULL<<bits));
            v0>>=zeros;vh>>=zeros;
            uu=(int64_t)((uint64_t)uu<<zeros);
            uvc=(int64_t)((uint64_t)uvc<<zeros);
            bits-=zeros;
            if(!bits)break;
            if(vh<uh){
                uint64_t w=uh;uh=vh;vh=w;w=u0;u0=v0;v0=w;
                int64_t t=uu;uu=vu;vu=t;t=uvc;uvc=vv;vv=t;
            }
            vh-=uh;v0-=u0;vv-=uvc;vu-=uu;
        }
        uint64_t partner=hm43_exchange(state,lane^8);
        uint64_t left=(group&1)?partner:state,right=(group&1)?state:partner;
        uint64_t a=hm43_signed_mul(left,(group&1)?vu:uu,lane);
        uint64_t b=hm43_signed_mul(right,(group&1)?vv:uvc,lane);
        uint64_t t=hm43_add(a,b,lane);
        bool flip=(int64_t)hm43_exchange(t,(group&1)*8+4)<0;
        uint64_t negated=hm43_negate(t,lane);
        if(flip)t=negated;
        uint64_t low=hm43_exchange(t,base);
        uint64_t m=group>=2?((low*MM64)&MSK62):0;
        uint64_t correction=hm43_multiple_p(m,word);
        t=hm43_add(t,correction,lane);
        uint64_t next=hm43_exchange(t,base|((word+1)&7));
        state=(t>>62)|(next<<2);
        uint64_t sign=hm43_exchange(state,base+4);
        if(word==5)state=(int64_t)sign<0?~0ULL:0;
        if(word>=6)state=0;
        nonzero=hm43_ballot(word<5 && state!=0);
        if((nonzero&0x1f00u)==0)break;
    }
    uint64_t u0=hm43_exchange(state,0);
    bool invertible=u0==1 && (nonzero&0x1eu)==0;
    #pragma unroll
    for(int j=0;j<5;j++)result[j]=hm43_exchange(state,16+j);
    if(lane==0){
        if(!invertible){for(int j=0;j<5;j++)result[j]=0;}
        else{
            while(_IsNegative(result))AddP(result);
            while(!_IsNegative(result))SubP(result);
            AddP(result);
        }
    }
    #pragma unroll
    for(int j=0;j<5;j++)result[j]=hm43_exchange(result[j],0);
    return true;
}
