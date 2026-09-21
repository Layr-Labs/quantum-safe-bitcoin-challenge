// Independently derived bounded product window. Research inspiration:
// EvanYan1024 PR885 and terrapinelf public Subset note 252f6ac.
// Only the speculative post3 path calls this; exact replay remains unchanged.
#pragma once
#ifndef QSB_PARITY_WINDOW
#define QSB_PARITY_WINDOW 1
#endif
// Quotient of h*(2^32+977)+term*2^32+epsilon by 2^64.
// term <= 2^33+1; all intermediate values fit uint64_t.
__device__ __forceinline__ uint64_t qsb_pw_quot(uint64_t h, uint64_t term, uint32_t epsilon) {
    const uint64_t lo=(uint32_t)h, hi=h>>32;
    return hi+((lo+977ULL*hi+term+((977ULL*lo+epsilon)>>32))>>32);
}
// Returns canonical parity(a*b-y mod p) only when interval bounds certify it.
// Arbitrary 256-bit a,b; canonical y. False requests the inherited full path.
__device__ __forceinline__ bool qsb_pw_sub(uint32_t *parity,
    const uint64_t *a, const uint64_t *b, const uint64_t *y) {
    const uint64_t mask=0xffffffffffffffffULL;
    const uint64_t p0=0xfffffffefffffc2fULL;
    if(y[3]==mask && y[2]==mask && y[1]==mask && y[0]>p0) return false;
    // Only the low bit and top word of p-y are needed. y==0 is allowed:
    // the addend p has the same modular meaning as zero.
    const uint32_t borrow=(y[0]>p0 && y[1]==mask && y[2]==mask && (uint32_t)y[3]==0xffffffffU);
    const uint32_t y7=0xffffffffU-(uint32_t)(y[3]>>32)-borrow;
    uint32_t aa[8],bb[8];
    #pragma unroll
    for(int i=0;i<8;i++) { aa[i]=(uint32_t)(a[i/2]>>(32*(i&1))); bb[i]=(uint32_t)(b[i/2]>>(32*(i&1))); }
    uint64_t low=0,high=0,v,old;
#define QSB_PW_ACC(i,j) do { v=(uint64_t)aa[i]*bb[j]; old=low; low+=v; high+=(low<old); } while(0)
    QSB_PW_ACC(0,6);
    QSB_PW_ACC(1,5);
    QSB_PW_ACC(2,4);
    QSB_PW_ACC(3,3);
    QSB_PW_ACC(4,2);
    QSB_PW_ACC(5,1);
    QSB_PW_ACC(6,0);
    const uint64_t d6q=(high<<32)|(low>>32);
    const uint32_t d6r=(uint32_t)low;
    low=0; high=0;
    QSB_PW_ACC(0,7);
    QSB_PW_ACC(1,6);
    QSB_PW_ACC(2,5);
    QSB_PW_ACC(3,4);
    QSB_PW_ACC(4,3);
    QSB_PW_ACC(5,2);
    QSB_PW_ACC(6,1);
    QSB_PW_ACC(7,0);
    const uint64_t d7q=(high<<32)|(low>>32);
    const uint32_t d7r=(uint32_t)low;
    low=0; high=0;
    QSB_PW_ACC(6,7);
    QSB_PW_ACC(7,6);
    const uint64_t d13q=(high<<32)|(low>>32);
    const uint32_t d13r=(uint32_t)low;
#undef QSB_PW_ACC
    const uint64_t c7lo=d6q;
    const uint64_t c7hi=d6q+(((uint64_t)d6r+6ULL*0xffffffffULL)>>32);
    const uint64_t c8lo=d7q+(((uint64_t)d7r+c7lo)>>32);
    const uint64_t c8hi=d7q+(((uint64_t)d7r+c7hi)>>32);
    if(c8lo!=c8hi) return false;
    const uint32_t llo=(uint32_t)((uint64_t)d7r+c7lo);
    const uint32_t lhi=(uint32_t)((uint64_t)d7r+c7hi);
    const uint64_t hlo=(uint64_t)aa[7]*bb[7]+d13q;
    const uint64_t delta=((uint64_t)d13r+8ULL*0xffffffffULL)>>32;
    // Avoid both the conservative upper-bound overflow and hhi+1 overflow.
    if(hlo>=mask-delta) return false;
    const uint64_t hhi=hlo+delta;
    const uint64_t qlo=qsb_pw_quot(hlo,(uint64_t)llo+y7,0);
    const uint64_t qhi=qsb_pw_quot(hhi+1,(uint64_t)lhi+y7+2,1);
    if(qlo!=qhi) return false;
    uint32_t middle=0;
    #pragma unroll
    for(int i=1;i<8;i++) middle^=aa[i]&bb[8-i];
    *parity=(uint32_t)(((aa[0]&bb[0])^middle^c8lo^y[0]^1ULL^qlo)&1ULL);
    return true;
}
