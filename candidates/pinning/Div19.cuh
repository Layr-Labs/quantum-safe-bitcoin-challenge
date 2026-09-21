// GLV608 exact base-2^32 long-division digit. Input rem is in [0,18].
#pragma once
__device__ __forceinline__ uint32_t qsb_div19_digit(uint32_t x,uint32_t &rem) {
 const uint32_t c=226050910u; //2^32=19*c+6
 uint32_t h=__umulhi(x,c);
 uint32_t z=x-19u*h+6u*rem; //0<=z<=145
 uint32_t adjust=(z*216u)>>12; //exact floor(z/19) on this bounded range
 uint32_t q=rem*c+h+adjust;
 rem=z-19u*adjust;
 return q;
}
// Little-endian bounded magnitude. Use uniform round-specific L=3,2,1.
template<int L>
__device__ __forceinline__ uint32_t qsb_udiv19_words(uint32_t *q,const uint32_t *x) {
 static_assert(L>=1 && L<=4,"bounded magnitude width");
 uint32_t rem=0;
 #pragma unroll
 for(int i=L-1;i>=0;--i)q[i]=qsb_div19_digit(x[i],rem);
 return rem;
}
