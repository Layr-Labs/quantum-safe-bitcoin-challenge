// A[0..7] must contain all eight limbs, identically in each subgroup lane.
// Eight-lane cooperative tree primitive.
// Eight contiguous lanes cooperate on one raw C31 field product.
// Every lane named in mask must execute every shuffle; each 8-lane subgroup
// is complete. Inputs/outputs are one little-endian uint32 limb per lane.
#pragma once
__device__ __forceinline__ unsigned qsb_c8_map(unsigned f,unsigned c) {
    return (f&1u) | (((f>>1)&1u)&c);
}
__device__ __forceinline__ unsigned qsb_c8_compose(unsigned f,unsigned g) {
    return ((f&1u) | (((f>>1)&1u)&(g&1u))) | (f&g&2u);
}
__device__ __forceinline__ unsigned qsb_c8_prefix(unsigned f,unsigned mask,unsigned lane) {
    #pragma unroll
    for(int d=1;d<8;d<<=1) {
        unsigned prev=__shfl_up_sync(mask,f,d,8);
        if(lane>=(unsigned)d) f=qsb_c8_compose(f,prev);
    }
    return f;
}
__device__ __forceinline__ unsigned qsb_c8_transition(unsigned long long z) {
    return (unsigned)(z>>32) | (((unsigned)z==0xffffffffu)?2u:0u);
}
__device__ __forceinline__ unsigned qsb_coop8_raw_preloaded(
    const unsigned *a,unsigned b,unsigned mask) {
    const unsigned lane=threadIdx.x&7u;
    unsigned long long lo=0,hi=0;
    unsigned lx=0,hx=0;
    #pragma unroll
    for(unsigned j=0;j<8;j++) {
        unsigned aj=a[j];
        unsigned bj=__shfl_sync(mask,b,(lane-j)&7u,8);
        unsigned long long t=(unsigned long long)aj*bj;
        unsigned long long tl=j<=lane?t:0,th=j>lane?t:0;
        unsigned long long nl=lo+tl,nh=hi+th;
        lx+=(nl<lo);hx+=(nh<hi);lo=nl;hi=nh;
    }
    unsigned prev1=__shfl_up_sync(mask,(unsigned)(lo>>32),1,8);
    unsigned prev2=__shfl_up_sync(mask,lx,2,8);
    unsigned long long zlo=(unsigned)lo;
    if(lane>=1) zlo+=prev1;
    if(lane>=2) zlo+=prev2;
    unsigned hp1=__shfl_up_sync(mask,(unsigned)(hi>>32),1,8);
    unsigned hp2=__shfl_up_sync(mask,hx,2,8);
    unsigned cross1=__shfl_sync(mask,(unsigned)(lo>>32),7,8);
    unsigned cross2=__shfl_sync(mask,lx,(lane+6)&7u,8);
    unsigned long long zhi=(unsigned)hi;
    zhi+=lane>=1?hp1:cross1;
    zhi+=lane>=2?hp2:cross2;
    // Move every coefficient's integer high part to the next word first.
    // The resulting carry maps have domain {0,1}, including former carry2.
    unsigned zlp=__shfl_up_sync(mask,(unsigned)(zlo>>32),1,8);
    unsigned zhp=__shfl_up_sync(mask,(unsigned)(zhi>>32),1,8);
    unsigned zcross=__shfl_sync(mask,(unsigned)(zlo>>32),7,8);
    zlo=(unsigned)zlo+(unsigned long long)(lane?zlp:0);
    zhi=(unsigned)zhi+(unsigned long long)(lane?zhp:zcross);
    unsigned flo=qsb_c8_prefix(qsb_c8_transition(zlo),mask,lane);
    unsigned fhi=qsb_c8_prefix(qsb_c8_transition(zhi),mask,lane);
    unsigned lowcarry=qsb_c8_map(flo,0);
    unsigned seed=__shfl_sync(mask,lowcarry,7,8);
    unsigned highcarry=qsb_c8_map(fhi,seed);
    unsigned cinlo=__shfl_up_sync(mask,lowcarry,1,8);
    unsigned cinhi=__shfl_up_sync(mask,highcarry,1,8);
    unsigned tlo=(unsigned)(zlo+(lane?cinlo:0));
    unsigned thi=(unsigned)(zhi+(lane?cinhi:seed));
    // Exact first pseudo-Mersenne fold: low + (2^32+977)*high.
    unsigned prevhi=__shfl_up_sync(mask,thi,1,8);
    unsigned long long fold=(unsigned long long)tlo+977ULL*thi;
    if(lane) fold+=prevhi;
    unsigned prevfold=__shfl_up_sync(mask,(unsigned)(fold>>32),1,8);
    unsigned long long z=(unsigned)fold;
    if(lane) z+=prevfold;
    unsigned ff=qsb_c8_prefix(qsb_c8_transition(z),mask,lane);
    unsigned cout=qsb_c8_map(ff,0);
    unsigned cin=__shfl_up_sync(mask,cout,1,8);
    unsigned r=(unsigned)(z+(lane?cin:0));
    // Lane7 owns the ninth fold word, then low32(h) is broadcast.
    unsigned h=thi+(unsigned)(fold>>32)+cout;
    h=__shfl_sync(mask,h,7,8);
    // Inherited C31 second fold updates low96 only; keep upper160 raw.
    unsigned long long tail0=(unsigned long long)r+977ULL*h;
    unsigned c0=__shfl_sync(mask,(unsigned)(tail0>>32),0,8);
    unsigned long long tail1=(unsigned long long)r+h+c0;
    unsigned c1=__shfl_sync(mask,(unsigned)(tail1>>32),1,8);
    if(lane==0) return (unsigned)tail0;
    if(lane==1) return (unsigned)tail1;
    if(lane==2) return r+c1;
    return r;
}
