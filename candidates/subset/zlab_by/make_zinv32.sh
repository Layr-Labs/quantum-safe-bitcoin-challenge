#!/bin/bash
# Rebuild tests/gpu_epochs/zinv32.cuh from the frontier-subset6 original:
# every line of the original is kept verbatim (sed line ranges); the
# Bernstein-Yang path is spliced in under #if ZLAB_BY.
set -e
ORIG=${1:-/root/qsb/frontier-subset6/tests/gpu_epochs/zinv32.cuh}
OUT=${2:-/root/qsb/exp/sub-by/tests/gpu_epochs/zinv32.cuh}
LUT=$(dirname "$0")/lut_body.inc
T=$(mktemp)
sed -n '1,14p' "$ORIG" >> "$T"           # provenance header (verbatim)
cat >> "$T" <<'EOF'
/* ZLAB_BY (default 1) replaces that decision chain with table-driven
 * Bernstein-Yang divsteps: 7 four-step constant-memory lookups plus 2
 * branchless single steps per 30-bit batch, no ctz, no brev/flo, no branch and
 * no head alignment, at the cost of ~44% more batches (18.10 vs 12.54 mean).
 * ZLAB_BY=0 selects the Stein path below unchanged and compiles byte-identical
 * to the base.  Both produce the same canonical inverse: 314,081 cross-checked
 * inversions agree bit for bit and with OpenSSL BN_mod_inverse (zlab_by/). */
EOF
sed -n '15,35p' "$ORIG" >> "$T"          # zi_ctz32/zi_clz32, ZI_B/ZI_MM32/ZI_MASK30
cat >> "$T" <<'EOF'

/* ZLAB_BY (kill switch): 1 = table-driven Bernstein-Yang divsteps, 0 = the
 * Stein/magnitude-comparison decision loop below (byte-identical to the base).
 *
 * A Bernstein-Yang divstep
 *     delta>0 and g odd : (delta,f,g) <- (1-delta, g, (g-f)/2)
 *     delta<=0, g odd   : (delta,f,g) <- (1+delta, f, (g+f)/2)
 *     g even            : (delta,f,g) <- (1+delta, f, g/2)
 * depends only on (delta, f mod 2^k, g mod 2^k), so k=4 steps collapse into one
 * lookup: f is always odd (8 values of f mod 16), 16 values of g mod 16 and
 * delta clamped to [-4,4] (9 values) = 1152 entries, 4608 B, one 32-bit word
 * each.  The clamp is exact: over 4 steps a delta above 4 stays positive until
 * the first swap and cannot return above 0 afterwards, and a delta below -4
 * cannot reach 0, so the clamped run makes the same decisions; the affine delta
 * recurrence then gives delta' = (-1)^swaps * delta + C with C in [-2,4] a
 * function of the clamped state alone (proved exhaustively in zlab_by/).
 *
 * Entry layout: a:6 b:6 c:6 d:6 (signed, in [-8,16]) | C:4 (signed) | bit31 =
 * parity of the swaps.  f' = (a*f+b*g)>>4, g' = (c*f+d*g)>>4 exactly.
 * The table is problem-independent, so it is a __constant__ initialiser: no
 * host upload, no shared-memory budget (the kernel is at 32,768 B shared with
 * exactly 2 resident blocks) and lanes 0 and 1 always present the same address,
 * so the LDC is a broadcast out of the constant cache. */
#ifndef ZLAB_BY
#define ZLAB_BY 1
#endif
#if ZLAB_BY
#ifdef __CUDA_ARCH__
#define ZI_CONST __constant__
#else
#define ZI_CONST static const
#endif
ZI_CONST uint32_t ZI_BY_LUT[1152]={
EOF
cat "$LUT" >> "$T"
cat >> "$T" <<'EOF'
};
/* 30 Bernstein-Yang divsteps on the low words of f and g: 7 four-step lookups
 * plus 2 branchless single steps.  No ctz, no head alignment, no branch on the
 * decision path.  Returns the new delta and the matrix rows (a,b) for f and
 * (c,d) for g, each row l1-norm <= 2^30 (so zi_row_ip's int32 x uint32 -> int64
 * products stay below 2^62, exactly as for the Stein rows). */
ZI_DEV int32_t zi_divstep30_by(int32_t delta,uint32_t f,uint32_t g,
                               int32_t *ra,int32_t *rb,int32_t *rc,int32_t *rd){
    int32_t u=1,v=0,q=0,r=1;
    #pragma unroll
    for(int k=0;k<7;k++){
        const int32_t dc=delta<-4?-4:(delta>4?4:delta);
        const uint32_t e=ZI_BY_LUT[(uint32_t)((dc+4)<<7)|((f&14u)<<3)|(g&15u)];
        const int32_t a=(int32_t)(e<<26)>>26,b=(int32_t)(e<<20)>>26;
        const int32_t c=(int32_t)(e<<14)>>26,d=(int32_t)(e<<8)>>26;
        const uint32_t nf=((uint32_t)a*f+(uint32_t)b*g)>>4;
        g=((uint32_t)c*f+(uint32_t)d*g)>>4; f=nf;
        const int32_t nu=a*u+b*q,nv=a*v+b*r;
        q=c*u+d*q; r=c*v+d*r; u=nu; v=nv;
        const int32_t sm=(int32_t)e>>31;
        delta=((delta^sm)-sm)+((int32_t)(e<<4)>>28);
    }
    #pragma unroll
    for(int k=0;k<2;k++){
        const int32_t mg=-(int32_t)(g&1u);
        const int32_t sw=mg&-(int32_t)(delta>0);
        const uint32_t nf=f^((f^g)&(uint32_t)sw);
        int32_t t=(int32_t)f&mg; t=(t^sw)-sw;
        g=(g+(uint32_t)t)>>1; f=nf;
        int32_t tu=u&mg; tu=(tu^sw)-sw;
        int32_t tv=v&mg; tv=(tv^sw)-sw;
        const int32_t nu=((sw&(q^u))^u)*2,nv=((sw&(r^v))^v)*2;
        q+=tu; r+=tv; u=nu; v=nv;
        delta=((delta^sw)-sw)+1;
    }
    *ra=u;*rb=v;*rc=q;*rd=r;
    return delta;
}
#else
EOF
sed -n '36,53p' "$ORIG" >> "$T"          # blank line, comment, zi_divstep30
echo '#endif  /* ZLAB_BY */' >> "$T"
sed -n '54,108p' "$ORIG" >> "$T"         # coop comment, zi_x, ZI_PL_INIT, zi_row_ip, zi_condneg, zi_canon
cat >> "$T" <<'EOF'
#if ZLAB_BY
/* All four lanes pass the same canonical root in R[0..3]; all return the canonical inverse
 * (0 for root 0: g starts at 0, one batch gives R=0, canon(0)=0 -- no gcd test needed since
 * p is prime).  Lane 0 owns f (init p), lane 1 g (init x), lane 2 R (init 0), lane 3 S (init 1);
 * the invariant f == R*x and g == S*x (mod p) is preserved because (R,S) take the same matrix
 * as (f,g) with the m*p correction supplying the division by 2^30 modulo p.  The loop exits
 * when g == 0, at which point f == +-gcd(p,x) == +-1, so R == +-1/x: hence the single final
 * conditional negate of R driven by the sign of lane 0's f.
 * Unlike the Stein path there is NO per-batch zi_condneg: Bernstein-Yang needs signed f and g
 * (negating g alone changes the trajectory), and none of what remains wants them non-negative
 * -- zi_row_ip already reads limb 8 as int32 and treats X as a 288-bit two's-complement value,
 * head alignment (the only magnitude comparison) is gone, and max(|f|,|g|) is non-increasing
 * from p, while |R|,|S| grow by at most p per batch (|a|+|b| <= 2^30, 0 <= m < 2^30), so after
 * the worst case of 25 batches (Bernstein-Yang's 741-divstep bound for 256-bit inputs, 30 per
 * batch) |R| < 26p < 2^261, inside zi_canon's stated precondition. */
ZI_DEV void zi_inverse_quad(uint64_t *R,int lane){
    const uint32_t ZI_PL[9]=ZI_PL_INIT;
    uint32_t P[9],Q[9];
    const uint32_t odd=(uint32_t)(lane&1),rs=(uint32_t)((lane>>1)&1);
    #pragma unroll
    for(int i=0;i<9;i++){
        const uint32_t xl=i<8?(uint32_t)(R[i>>1]>>(32*(i&1))):0u;
        const uint32_t own=rs?(uint32_t)(i==0):xl;        /* S=1 / g=x */
        const uint32_t oth=rs?0u:ZI_PL[i];                 /* R=0 / f=p */
        P[i]=odd?own:oth; Q[i]=odd?oth:own;
    }
    int32_t delta=1;
    while(true){
        int32_t a=0,b=0,c=0,d=0;
        if(lane<2){
            /* both lanes see the same (delta,f0,g0), so the decision is a single
             * uniform instruction stream (no divergence, no cross-lane traffic). */
            const uint32_t f0=odd?Q[0]:P[0],g0=odd?P[0]:Q[0];
            delta=zi_divstep30_by(delta,f0,g0,&a,&b,&c,&d);
        }
        int32_t ka=odd?d:a,kb=odd?c:b;
        ka=(int32_t)zi_x((uint32_t)ka,lane&1);
        kb=(int32_t)zi_x((uint32_t)kb,lane&1);
        zi_row_ip(P,Q,ka,kb,rs);
        uint32_t nz=0;
        for(int i=0;i<9;i++)nz|=P[i];
        nz=zi_x(nz,1);
        if(nz==0)break;
        for(int i=0;i<9;i++)Q[i]=zi_x(P[i],lane^1);
    }
    uint32_t fneg=(uint32_t)((int32_t)P[8]<0);
    fneg=zi_x(fneg,0);
    zi_condneg(P,fneg);
    zi_canon(P);
    for(int i=0;i<8;i++)P[i]=zi_x(P[i],2);
    for(int i=0;i<4;i++)R[i]=(uint64_t)P[2*i]|((uint64_t)P[2*i+1]<<32);
    R[4]=0;
}
#else
EOF
sed -n '109,154p' "$ORIG" >> "$T"        # original comment + zi_inverse_quad
echo '#endif  /* ZLAB_BY */' >> "$T"
mv "$T" "$OUT"
chmod 644 "$OUT"
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
