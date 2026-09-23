// SPDX-License-Identifier: GPL-3.0-only
// Executes the actual generated seed and anchor point bodies over exact F_p.
// This audits point equations and aliases, not CUDA PTX reduction semantics.
#include <stdint.h>
#include <boost/multiprecision/cpp_int.hpp>
using boost::multiprecision::cpp_int;
#define __device__
#define __forceinline__ inline
#define QSB_NEG_Y_MAC 1
#define QSB_YOFF 1
static const cpp_int fp=(cpp_int(1)<<256)-(cpp_int(1)<<32)-977;
static cpp_int load(const uint64_t *x){
  cpp_int n=0; for(int i=3;i>=0;--i){n<<=64;n+=x[i];}return n;
}
static void save(uint64_t *r,cpp_int n){
  n%=fp;if(n<0)n+=fp;
  for(int i=0;i<4;++i){r[i]=(uint64_t)(n&((cpp_int(1)<<64)-1));n>>=64;}
}
static void Load256(uint64_t*r,const uint64_t*a){for(int i=0;i<4;++i)r[i]=a[i];}
static void _ModMult(uint64_t*r,const uint64_t*a,const uint64_t*b){save(r,load(a)*load(b));}
static void _ModMult(uint64_t*r,const uint64_t*b){save(r,load(r)*load(b));}
static void _ModSqr(uint64_t*r,const uint64_t*a){save(r,load(a)*load(a));}
static void _ModSub256(uint64_t*r,const uint64_t*a,const uint64_t*b){save(r,load(a)-load(b));}
static void _ModAddLazy(uint64_t*r,const uint64_t*a,const uint64_t*b){save(r,load(a)+load(b));}
static void _ModAddLazyOff(uint64_t*r,const uint64_t*a,const uint64_t*b){save(r,load(a)+load(b)-0x1000003d0ULL);}
static void _ModSqrAddSub3(uint64_t*r,const uint64_t*a,const uint64_t*e,const uint64_t*q){save(r,load(a)*load(a)+load(e)-3*load(q));}
static void qsb_muladd_seed(uint64_t*r,const uint64_t*a,const uint64_t*b,const uint64_t*c){save(r,load(a)*load(b)+load(c));}
#include "anchor.cuh"
#include "seed_extracted.cuh"
extern "C" void seed(uint64_t*s,const uint64_t*a,const uint64_t*b){qsb_anchor_seed(s,s+4,s+8,s+12,a,a+4,b,b+4);}
extern "C" void step(uint64_t*s,const uint64_t*b,const uint64_t*a,int last){qsb_anchor_add(s,s+4,s+8,s+12,b,b+4,a,a+4,last!=0);}
