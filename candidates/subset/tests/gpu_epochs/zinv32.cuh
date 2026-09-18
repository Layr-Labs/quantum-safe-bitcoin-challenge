/* zinv32: 32-bit-word delayed binary-GCD field inverse for secp256k1 p.
 * Same algorithm family as VanitySearch's _ModInv (Jean Luc Pons, GPL-3.0; see GPUMath.h):
 * delayed right-shift divsteps on low/high approximations, r/s tracked with a Montgomery-style
 * m*p correction. Differences: 30-bit batches on 32-bit registers (int32 matrix), 9x32-bit
 * signed limbs, sparse m*p, sentinel-terminated decision loop. Output: canonical inverse in
 * [0,p), or 0 when gcd != 1 (input 0), bit-identical to _ModInv on canonical inputs.
 *
 * The cooperative form below (lanes owning U, V, R and S of one root inverse and exchanging
 * only the matrix rows, the sign flag, the zero flag and the partner vector) follows the
 * structure of @AbdelStark's warp-cooperative root inverse (PR 189 / submission db248c65,
 * hm41/hm43 ablations); the algorithm, the 32-bit decision core and all arithmetic here are
 * ours. Measured on an RTX 4090: one root inverse costs 39,851 cycles with the serial lane-0
 * _ModInv and 28,155 with this form (-29.4%); 85-90% of what remains is the ~190-iteration
 * divstep decision chain, which is serial in any design. */
#pragma once
#ifdef __CUDA_ARCH__
#define ZI_DEV __device__ __forceinline__
ZI_DEV uint32_t zi_ctz32(uint32_t x){
    uint32_t n;
    asm("{\n\t .reg .u32 tmp;\n\t brev.b32 tmp, %1;\n\t clz.b32 %0, tmp;\n\t}" : "=r"(n) : "r"(x));
    return n;
}
ZI_DEV uint32_t zi_clz32(uint32_t x){
    uint32_t n;
    asm("{\n\t clz.b32 %0, %1;\n\t}" : "=r"(n) : "r"(x));
    return n;
}
#else
#define ZI_DEV static inline
static inline uint32_t zi_ctz32(uint32_t x){return (uint32_t)__builtin_ctz(x);}
static inline uint32_t zi_clz32(uint32_t x){return x?(uint32_t)__builtin_clz(x):32u;}
#endif
#define ZI_B 30
#define ZI_MM32 0xD2253531u            /* -p^-1 mod 2^32 */
#define ZI_MASK30 0x3FFFFFFFu
#ifndef QSB_ZI_SKIP_ZERO_NEGATION
#define QSB_ZI_SKIP_ZERO_NEGATION 1
#endif

/* Decision loop: 30 delayed divsteps on (u0,v0) low words and (uh,vh) aligned heads.
 * Returns matrix rows (a,b) for u and (c,d) for v, each row l1-norm <= 2^30. */
ZI_DEV void zi_divstep30(uint32_t u0,uint32_t v0,uint32_t uh,uint32_t vh,
                         int32_t *ra,int32_t *rb,int32_t *rc,int32_t *rd){
    uint32_t a=1,b=0,c=0,d=1,S=1u<<ZI_B;
    while(true){
        uint32_t z=zi_ctz32(v0|S);
        v0>>=z; vh>>=z; a<<=z; b<<=z; S>>=z;
        if(S==1u)break;
        if(vh<uh){
            uint32_t t;
            t=uh;uh=vh;vh=t; t=u0;u0=v0;v0=t; t=a;a=c;c=t; t=b;b=d;d=t;
        }
        vh-=uh; v0-=u0; d-=b; c-=a;
    }
    *ra=(int32_t)a;*rb=(int32_t)b;*rc=(int32_t)c;*rd=(int32_t)d;
}

/* ======================= 4-lane cooperative form =======================
 * Lanes 0..3 of one warp run the SAME instruction stream except the decision loop
 * (lanes 0,1 only; they hold identical u,v so they take identical branches).
 * lane 0 owns u, lane 1 v, lane 2 r, lane 3 s; P = own vector, Q = pair partner's.
 * Row rule: lane even  new P = a*P + b*Q ; lane odd new P = d*P + c*Q  (same as c*u+d*v).
 * Collectives per batch: 2 coefficients + 1 sign + 1 zero flag + 9 partner limbs. */
#ifdef __CUDA_ARCH__
ZI_DEV uint32_t zi_x(uint32_t v,int src){return (uint32_t)__shfl_sync(0xFu,(unsigned int)v,src);}
#else
uint32_t zi_x(uint32_t v,int src);
#endif
/* p limbs, little-endian 32-bit (limb 8 = 0). Kept as an initialised local array in every
 * user so it folds to immediates in device code instead of a constant-bank load. */
#define ZI_PL_INIT {0xFFFFFC2Fu,0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0u}

/* In place: X = (a*X + b*Y [+ m*p]) >> 30. */
ZI_DEV void zi_row_ip(uint32_t *X,const uint32_t *Y,int32_t a,int32_t b,uint32_t modp){
    int64_t acc=(int64_t)a*(int64_t)X[0]+(int64_t)b*(int64_t)Y[0];
    uint32_t m=((uint32_t)acc*ZI_MM32)&ZI_MASK30&(0u-modp);
    acc-=(int64_t)977*(int64_t)m;
    X[0]=(uint32_t)acc; acc>>=32;
    acc+=(int64_t)a*(int64_t)X[1]+(int64_t)b*(int64_t)Y[1]-(int64_t)m;
    X[1]=(uint32_t)acc; acc>>=32;
    for(int i=2;i<8;i++){
        acc+=(int64_t)a*(int64_t)X[i]+(int64_t)b*(int64_t)Y[i];
        X[i]=(uint32_t)acc; acc>>=32;
    }
    acc+=(int64_t)a*(int64_t)(int32_t)X[8]+(int64_t)b*(int64_t)(int32_t)Y[8]+(int64_t)m;
    X[8]=(uint32_t)acc;
    for(int i=0;i<8;i++)X[i]=(X[i]>>ZI_B)|(X[i+1]<<(32-ZI_B));
    X[8]=(uint32_t)(acc>>ZI_B);
}
ZI_DEV void zi_condneg(uint32_t *X,uint32_t neg){
    const uint32_t msk=0u-neg; uint64_t c=neg;
    for(int i=0;i<9;i++){c+=(uint64_t)(X[i]^msk);X[i]=(uint32_t)c;c>>=32;}
}
/* X signed 288-bit with |X| < 2^261 and X == r (mod p) -> canonical r in X[0..7]. */
ZI_DEV void zi_canon(uint32_t *X){
    const uint32_t ZI_PL[9]=ZI_PL_INIT;
    const int32_t hi=(int32_t)X[8];
    int64_t acc=(int64_t)X[0]+(int64_t)hi*977;
    X[0]=(uint32_t)acc; acc>>=32;
    acc+=(int64_t)X[1]+(int64_t)hi;
    X[1]=(uint32_t)acc; acc>>=32;
    for(int i=2;i<8;i++){acc+=(int64_t)X[i];X[i]=(uint32_t)acc;acc>>=32;}
    X[8]=(uint32_t)acc;                               /* -1, 0 or 1 */
    const uint32_t mneg=(uint32_t)((int32_t)X[8]>>31);
    uint64_t c=0;
    for(int i=0;i<9;i++){c+=(uint64_t)X[i]+(uint64_t)(ZI_PL[i]&mneg);X[i]=(uint32_t)c;c>>=32;}
    uint32_t T[9]; c=1;
    for(int i=0;i<9;i++){c+=(uint64_t)X[i]+(uint64_t)(uint32_t)~ZI_PL[i];T[i]=(uint32_t)c;c>>=32;}
    const uint32_t keep=(uint32_t)((int32_t)T[8]>>31);
    for(int i=0;i<8;i++)X[i]=(X[i]&keep)|(T[i]&~keep);
}
/* All four lanes pass the same canonical root in R[0..3]; all return the canonical inverse
 * (0 for root 0: v starts at 0, one batch gives r=0, canon(0)=0 -- no gcd test needed since p is prime). */
ZI_DEV void zi_inverse_quad(uint64_t *R,int lane){
    const uint32_t ZI_PL[9]=ZI_PL_INIT;
    uint32_t P[9],Q[9];
    const uint32_t odd=(uint32_t)(lane&1),rs=(uint32_t)((lane>>1)&1);
    #pragma unroll
    for(int i=0;i<9;i++){
        const uint32_t xl=i<8?(uint32_t)(R[i>>1]>>(32*(i&1))):0u;
        const uint32_t own=rs?(uint32_t)(i==0):xl;        /* s=1 / x */
        const uint32_t oth=rs?0u:ZI_PL[i];                 /* r=0 / p */
        P[i]=odd?own:oth; Q[i]=odd?oth:own;
    }
    int pos=7;
    while(true){
        int32_t a=0,b=0,c=0,d=0;
        if(lane<2){
            while(pos>0 && (P[pos]|Q[pos])==0)pos--;
            uint32_t ph=P[pos],qh=Q[pos];
            if(pos>0){
                const uint32_t sh=zi_clz32(ph|qh);
                if(sh){ph=(ph<<sh)|(P[pos-1]>>(32-sh));qh=(qh<<sh)|(Q[pos-1]>>(32-sh));}
            }
            /* one call, selected operands: both lanes see the same (u0,v0,uh,vh), so the
             * decision loop is a single uniform instruction stream (no divergence). */
            const uint32_t u0=odd?Q[0]:P[0],v0=odd?P[0]:Q[0],uh=odd?qh:ph,vh=odd?ph:qh;
            zi_divstep30(u0,v0,uh,vh,&a,&b,&c,&d);
        }
        int32_t ka=odd?d:a,kb=odd?c:b;
        ka=(int32_t)zi_x((uint32_t)ka,lane&1);
        kb=(int32_t)zi_x((uint32_t)kb,lane&1);
        zi_row_ip(P,Q,ka,kb,rs);
        uint32_t neg=(uint32_t)((int32_t)P[8]<0);
        neg=zi_x(neg,lane&1);
#if QSB_ZI_SKIP_ZERO_NEGATION
        if(neg)zi_condneg(P,1u);
#else
        zi_condneg(P,neg);
#endif
        uint32_t nz=0;
        for(int i=0;i<9;i++)nz|=P[i];
        nz=zi_x(nz,1);
        if(nz==0)break;
        for(int i=0;i<9;i++)Q[i]=zi_x(P[i],lane^1);
    }
    zi_canon(P);
    for(int i=0;i<8;i++)P[i]=zi_x(P[i],2);
    for(int i=0;i<4;i++)R[i]=(uint64_t)P[2*i]|((uint64_t)P[2*i+1]<<32);
    R[4]=0;
}
