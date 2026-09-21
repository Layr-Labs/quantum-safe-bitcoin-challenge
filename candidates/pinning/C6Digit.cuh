#pragma once
struct qsb_c6_digit { uint32_t code; int32_t dx,dy; };
__device__ __forceinline__ void qsb_c6_offer(uint32_t x,uint32_t y,uint32_t tag,
 uint32_t &bx,uint32_t &by,uint32_t &bt) {
 if(x<bx || (x==bx && y<by)){bx=x;by=y;bt=tag;}
}
template<int M> __device__ __forceinline__ qsb_c6_digit qsb_c6_radix(uint32_t a,uint32_t b) {
 static_assert(M==512 || M==608,"audited non-multiple-of3 radices");
 uint32_t na=a?M-a:0,nb=b?M-b:0,d=a>=b?a-b:a+M-b,nd=d?M-d:0;
 uint32_t bx=a,by=b,tag=0;
 // Ties retain the preferred inverse unit j=0,1,2, then positive sign.
 qsb_c6_offer(na,nb,4,bx,by,tag);
 qsb_c6_offer(nd,na,1,bx,by,tag);
 qsb_c6_offer(d,a,5,bx,by,tag);
 qsb_c6_offer(nb,d,2,bx,by,tag);
 qsb_c6_offer(b,nd,6,bx,by,tag);
 uint32_t rank=bx==0?by:M/2+1+(bx-1)*M-3*(bx-1)*bx/2+(by-2*bx);
 int32_t x=(int32_t)bx,y=2*by<=M+bx?(int32_t)by:(int32_t)by-M;
 int32_t dx=x,dy=y;
 if((tag&3)==1){dx=-y;dy=x-y;}
 if((tag&3)==2){dx=y-x;dy=-x;}
 if(tag&4){dx=-dx;dy=-dy;}
 return {rank|((tag&3)<<16)|((tag>>2)<<18),dx,dy};
}
