/* Let B=2^256, p=B-K, K=2^32+977. A raw exact product is in [0,B). */
// If b[3]!=0 then b>=2^192>K, so -p<raw-b<p. Its canonical parity
// needs just the subtraction borrow. Small b retains normalization.
__device__ __forceinline__ void qsb_parity_boundary(uint64_t *raw,const uint64_t *b) {
    if(b[3]==0)qsb_field_normalize(raw);
}
// If a[3]!=UINT64_MAX then a<B-2^192<B-2K, hence raw+a<2p.
/* One conditional subtraction in _ModAdd256 is then sufficient. The extreme */
// upper fixed-a range retains the original canonical-product boundary.
__device__ __forceinline__ void qsb_add_boundary(uint64_t *raw,const uint64_t *a) {
    if(a[3]==UINT64_MAX)qsb_field_normalize(raw);
}

// Canonical inputs a,b<p. Adding odd p on borrow flips only the
/* parity we need; computing all four corrected output limbs is unnecessary. */
__device__ __forceinline__ uint32_t qsb_difference_parity(
    const uint64_t *a,const uint64_t *b) {
    const bool borrow=a[3]!=b[3] ? a[3]<b[3] :
        a[2]!=b[2] ? a[2]<b[2] : a[1]!=b[1] ? a[1]<b[1] : a[0]<b[0];
    return (uint32_t)((a[0]^b[0]^uint64_t(borrow))&1u);
}

#if QSB_PARITY_SUM
/* P9 parity of (+-(w+b)) mod p for raw w in [0,2^256) and canonical b != 0 (b = y(u2*R): secp256k1 */
// has no point with y = 0). With c the carry of w+b and S its low 256 bits, w+b = y + k*p where
/* y = (w+b) mod p and k in {0,1,2}: k = c unless S[1..3] are all ones (a 2^-192 event), where */
// k = c + [S >= (c ? 2^256-2K : p)] and y = 0 exactly when S equals that bound. p is odd, so
// par(y) = (w0^b0^k)&1 and par(-y mod p) = y ? 1^par(y) : 0.
__device__ __forceinline__ uint32_t qsb_sum_parity(const uint64_t *w,const uint64_t *b,uint32_t neg) {
    uint64_t s0,s1,s2,s3,c;
    asm("add.cc.u64 %0,%5,%9;\n\taddc.cc.u64 %1,%6,%10;\n\taddc.cc.u64 %2,%7,%11;\n\t"
        "addc.cc.u64 %3,%8,%12;\n\taddc.u64 %4,0,0;"
        : "=l"(s0),"=l"(s1),"=l"(s2),"=l"(s3),"=l"(c)
        : "l"(w[0]),"l"(w[1]),"l"(w[2]),"l"(w[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    uint32_t k=(uint32_t)c,zero=0u;
    if((s1&s2&s3)==UINT64_MAX) {
        const uint64_t lim=c ? 0xFFFFFFFDFFFFF85EULL : 0xFFFFFFFEFFFFFC2FULL;
        if(s0>=lim){k+=1u;zero=(s0==lim);}
    }
    const uint32_t par=((uint32_t)(w[0]^b[0])^k)&1u;
    return zero ? 0u : (par^neg);
}
#endif

/* Exact full-width residue; callers normalize before additions/parity. */
__device__ __forceinline__ void qsb_packed_raw_mul(
    uint64_t *out,const uint64_t *a,const uint64_t *b) {
    uint64_t tmp[5];qsb_field_mul_sc(tmp,const_cast<uint64_t*>(a),const_cast<uint64_t*>(b));
    Load256(out,tmp);
}

/* QSB_FIN_CAP_IMAD (GPUMath.h): the finish kernel's hot products use _ModMultCoreFin, whose
 * carry captures run on the multiply pipe. Same bits as qsb_packed_raw_mul for every input. */
#if QSB_FIN_CAP_IMAD
#if !QSB_SHA_ALU_ADD || !QSB_FIELD_SC
#error "QSB_FIN_CAP_IMAD needs pin_zero_add (QSB_SHA_ALU_ADD=1) and the short-carry field product (QSB_FIELD_SC=1)"
#endif
__device__ __forceinline__ void qsb_packed_raw_mul_fin(
    uint64_t *out,const uint64_t *a,const uint64_t *b) {
    uint64_t tmp[5];_ModMultCoreFin(tmp,a,b);tmp[4]=0;
    Load256(out,tmp);
}
#define QSB_FIN_RAW_MUL qsb_packed_raw_mul_fin
#else
#define QSB_FIN_RAW_MUL qsb_packed_raw_mul
#endif

/* QSB_FIN_BAL2 (GPUMath.h): the finish kernel's field adds use the finish-only copies, whose
 * selects, captures and K32 masks run on the multiply pipe. Same bits for every input. */
#if QSB_FIN_BAL2
#if !QSB_SHA_ALU_ADD
#error "QSB_FIN_BAL2 needs pin_zero_add (QSB_SHA_ALU_ADD=1)"
#endif
#define QSB_FIN_ADD _ModAdd256Fin
#define QSB_FIN_ADDL _ModAddLazyFin
#define QSB_FIN_SUB _ModSub256Fin
#else
#define QSB_FIN_ADD _ModAdd256
#define QSB_FIN_ADDL _ModAddLazy
#define QSB_FIN_SUB _ModSub256
#endif
#if QSB_FIN_BAL2 & 2
/* QSB_FIN_BAL2 bit 2: the two parities leave qsb_packed_finish as the two pubkey prefix bytes,
 * byte 0 = 2 + parity_u and byte 1 = 2 + parity_v (each parity is 0 or 1, so the bytes cannot
 * carry into each other). The kernel picks byte ri with the PRMT that builds pb[0], so the
 * pack (shift, or), the unpack (shift) and the two "| 2" ops are gone. Both adds run as IMAD. */
__device__ __forceinline__ uint32_t qsb_fin_prefix_bytes(uint32_t pu, uint32_t pv) {
    uint32_t y;
    asm("{\n.reg .u32 f;\nld.const.u32 f,[pin_pow2+32];\nmad.lo.u32 %0,%1,f,%2;\n}" : "=r"(y) : "r"(pv), "r"(pu));
    return qsb_fadd(y, pin_one_mul, 0x202u);
}
#endif

#ifndef QSB_PARITY_WINDOW
#define QSB_PARITY_WINDOW 1
#endif
#if QSB_PARITY_WINDOW
#include "ParityWindow.cuh"
#endif

/* Combine the public cofactor traversal with our existing exact/canonical */
/* recovery boundary and the odinfree square-free finish identity. */
__device__ __forceinline__ void qsb_packed_prepare(
    uint64_t *D, const uint64_t *U, const uint64_t *Y, const uint64_t *V,
    bool usable, bool active, int n, ulonglong2 *saved, uint64_t *roots) {
    // All lanes finish reading their digits/anchor before tree overwrites.
    __syncthreads();
#if QSB_TREE_GFILL
    /* QSB_TREE_GFILL (cofactor_checkpoint.h): D comes back as hc = U*cofactor, exact. */
    static_assert(QSB_RECOVERY_N==QSB_TREE_N && QSB_PREP_STATE,"QSB_TREE_GFILL geometry");
#if QSB_TREE_TOP5 && (QSB_POST_GLUE & 1)
    /* QSB_POST_GLUE bit 1 (cofactor_checkpoint.h): TOP5 with 16-byte limb pairs in shared memory. */
    qsb_cofactor_top5v<QSB_RECOVERY_N>(D,U,roots,(char *)qsb_digit_arena());
#elif QSB_TREE_TOP5
    /* QSB_TREE_TOP5 (cofactor_checkpoint.h): same contract, 24 warp-multiplies per block. */
    qsb_cofactor_top5<QSB_RECOVERY_N>(D,U,roots,(uint64_t (*)[QSB_GF_COLS])qsb_digit_arena());
#else
    qsb_cofactor_gfill<QSB_RECOVERY_N>(D,U,roots,(uint64_t (*)[QSB_GF_COLS])qsb_digit_arena());
#endif
    if(active) {
        uint64_t hc[4],vbar[4],tbar[4];
        #pragma unroll
        for(int k=0;k<4;k++)hc[k]=usable?D[k]:0ULL;
        qsb_packed_raw_mul(vbar,Y,hc);
        qsb_packed_raw_mul(tbar,V,hc);
#else
    uint64_t (*products)[2*QSB_RECOVERY_N]=(uint64_t (*)[2*QSB_RECOVERY_N])qsb_digit_arena();
    uint64_t (*excluded)[QSB_RECOVERY_N]=(uint64_t (*)[QSB_RECOVERY_N])(qsb_digit_arena()+8*QSB_TREE_N);
    qsb_cofactor_prepare<QSB_RECOVERY_N>(D,roots,products,excluded);
    if(active) {
        uint64_t hc[4],vbar[4],tbar[4];
#endif
#if QSB_TREE_GFILL
#elif QSB_PREP_STATE
        /* Zero state for unusable lanes from hc = 0 (QSB_PREP_STATE, pinning.cu). */
        qsb_packed_raw_mul(hc,U,D);
        #pragma unroll
        for(int k=0;k<4;k++)hc[k]=usable?hc[k]:0ULL;
        qsb_packed_raw_mul(vbar,Y,hc);
        qsb_packed_raw_mul(tbar,V,hc);
#else
        qsb_packed_raw_mul(hc,U,D);
        qsb_packed_raw_mul(vbar,Y,hc);
        qsb_packed_raw_mul(tbar,V,hc);
        if(!usable)for(int k=0;k<4;k++){vbar[k]=0;tbar[k]=0;}
#endif
#if QSB_PREP_STATE
        /* Block-major state (QSB_PREP_STATE, pinning.cu): plane p at st + p*QSB_RECOVERY_N. */
        (void)n;
        ulonglong2 *st=saved+(uint32_t)(blockIdx.x*(QSB_STATE_PLANES*QSB_RECOVERY_N)+threadIdx.x);
#if QSB_PREP_STATE == 2
        uint64_t *sw=(uint64_t *)st;
        qsb_st_u64(sw,vbar[0]); qsb_st_u64(sw+1,vbar[1]);
        qsb_st_u64(sw+2*QSB_RECOVERY_N,vbar[2]); qsb_st_u64(sw+2*QSB_RECOVERY_N+1,vbar[3]);
        qsb_st_u64(sw+4*QSB_RECOVERY_N,tbar[0]); qsb_st_u64(sw+4*QSB_RECOVERY_N+1,tbar[1]);
        qsb_st_u64(sw+6*QSB_RECOVERY_N,tbar[2]); qsb_st_u64(sw+6*QSB_RECOVERY_N+1,tbar[3]);
#else
        qsb_st_v2(st,vbar[0],vbar[1]);
        qsb_st_v2(st+QSB_RECOVERY_N,vbar[2],vbar[3]);
        qsb_st_v2(st+2*QSB_RECOVERY_N,tbar[0],tbar[1]);
        qsb_st_v2(st+3*QSB_RECOVERY_N,tbar[2],tbar[3]);
#endif
#else
        size_t i=(size_t)blockIdx.x*QSB_RECOVERY_N+threadIdx.x,s=(size_t)n;
#if QSB_STREAM2
        qsb_st_v2(&saved[0*s+i],vbar[0],vbar[1]);
        qsb_st_v2(&saved[1*s+i],vbar[2],vbar[3]);
        qsb_st_v2(&saved[2*s+i],tbar[0],tbar[1]);
        qsb_st_v2(&saved[3*s+i],tbar[2],tbar[3]);
#else
        saved[0*s+i]=make_ulonglong2(vbar[0],vbar[1]);
        saved[1*s+i]=make_ulonglong2(vbar[2],vbar[3]);
        saved[2*s+i]=make_ulonglong2(tbar[0],tbar[1]);
        saved[3*s+i]=make_ulonglong2(tbar[2],tbar[3]);
#endif
#endif
    }
}

/* Public PR993 QSB_FIN_RAWS, isolated on the PR976 source. Both slope products
 * may remain raw in [0,2^256): the parity window accepts congruent raw input,
 * while qsb_add_boundary preserves canonical x-coordinate addition. */
#ifndef QSB_FIN_RAWS
#define QSB_FIN_RAWS 1
#endif
/* QSB_XOUT_LAZY (kill switch, default 1): x_i = s + a through the carry-folding lazy add.
 * s is a raw product in [0,2^256) and a is canonical, so the folded sum is congruent and
 * lies in [0,2^256); it is non-canonical only when it lands in [p,2^256), a window of
 * 2^32+977 values (probability < 2^-223), or when the fold carries twice (a 2^-223 input).
 * Such an x_i only mis-hashes that candidate, which the host exact gate would reject.
 * 0 keeps the boundary normalisation and the reducing add. */
#ifndef QSB_XOUT_LAZY
#define QSB_XOUT_LAZY 1
#endif
/* QSB_FIN_MLATE (kill switch, default 1): m is not held across the x1 half of the recovery. After
 * sum = l + m (lazy, congruent), m == sum - l (mod p), so it is re-derived as QSB_FIN_SUB(m,sum,l)
 * once the x1 parity window has read l for the last time. Every consumer of m (the m - c
 * subtraction and the parity window) takes any representative in [0,2^256): the subtraction is
 * borrow-corrected and the window returns the parity of the reduced product. s = sum*(m - c) is
 * the same residue; its representative leaves [0,p) only in the 2^-223 window QSB_XOUT_LAZY
 * already accepts (the host exact gate rejects such a mis-hash). Four limbs (eight registers)
 * fewer are live through the first slope product and its parity window. 0 keeps m live. */
#ifndef QSB_FIN_MLATE
#define QSB_FIN_MLATE 1
#endif
/* QSB_FIN_SUMU (kill switch, default 1; needs QSB_FIN_MLATE): sum = l + m == (u + v) + (u - v)
 * == 2u (mod p) for either QSB_NEG_Y_MAC sign, so sum is formed as the lazy doubling u + u right
 * after the first slope product. The early m = u -/+ v subtraction is gone (MLATE re-derives m as
 * sum - l after the x1 window, as before), and sum no longer waits for the v product, so the
 * second product overlaps the doubling. sum only feeds the borrow-corrected subtractions l - c,
 * sum - c / m - c and raw products, all of which take any representative in [0,2^256); a
 * representative change of their outputs is confined to the 2^-223 window QSB_XOUT_LAZY already
 * accepts. 0 keeps sum = l + m. */
#ifndef QSB_FIN_SUMU
#define QSB_FIN_SUMU 1
#endif
#if QSB_FIN_SUMU && !(QSB_FIN_MLATE && QSB_PARITY_SUM && QSB_FIN_RAWS)
#error "QSB_FIN_SUMU re-derives m through QSB_FIN_MLATE"
#endif
#if QSB_STATE_LNUM && !(QSB_FIN_SUMU && QSB_LAZY_REC)
#error "QSB_STATE_LNUM is written for the QSB_FIN_SUMU / QSB_LAZY_REC finish"
#endif
__device__ __forceinline__ uint32_t qsb_packed_finish(
    const uint64_t *vbar,const uint64_t *tbar,const uint64_t *root_inv,
    const uint64_t *weighted_inv,
    uint64_t *a,uint64_t *b,uint64_t *c,uint64_t *x1,uint64_t *x2) {
    uint64_t u[4],v[4],l[4],m[4],sum[4],t[4],s[4];
#if QSB_LAZY_REC
    /* u, v, l, m and sum only feed multiplies and borrow-corrected subtractions, which
     * accept any representative in [0,2^256); only x1/x2 (hashed) and the parity inputs
     * need [0,p). So u and v stay raw and m, sum use the carry-folding lazy add
     * (congruent, [0,2^256); a second carry needs a 2^-223 input, as in the chain). */
#if QSB_FIN_SUMU && QSB_WROOT_DBL
    QSB_FIN_RAW_MUL(sum,tbar,weighted_inv);
#else
    QSB_FIN_RAW_MUL(u,tbar,weighted_inv);
#endif
#if QSB_FIN_SUMU
#if !QSB_WROOT_DBL
    QSB_FIN_ADDL(sum,u,u);
#endif
#if QSB_STATE_LNUM
#if !QSB_NEG_Y_MAC
#error "QSB_STATE_LNUM stores the numerator of l = u + v (QSB_NEG_Y_MAC)"
#endif
    QSB_FIN_RAW_MUL(l,vbar,root_inv);
#else
    QSB_FIN_RAW_MUL(v,vbar,root_inv);
#if QSB_NEG_Y_MAC
    QSB_FIN_ADDL(l,u,v);
#else
    QSB_FIN_SUB(l,u,v);
#endif
#endif
#else
    QSB_FIN_RAW_MUL(v,vbar,root_inv);
#if QSB_NEG_Y_MAC
    QSB_FIN_ADDL(l,u,v); QSB_FIN_SUB(m,u,v);
#else
    QSB_FIN_SUB(l,u,v); QSB_FIN_ADDL(m,u,v);
#endif
    QSB_FIN_ADDL(sum,l,m);
#endif
#else
    qsb_recovery_mul(u,tbar,weighted_inv);
    qsb_recovery_mul(v,vbar,root_inv);
#if QSB_NEG_Y_MAC
    _ModAdd256(l,u,v); _ModSub256(m,u,v);
#else
    _ModSub256(l,u,v); _ModAdd256(m,u,v);
#endif
    _ModAdd256(sum,l,m);
#endif
#if QSB_RAW_X
    /* P7: x1, x2 stay raw; raw + a < 2p whenever a[3] != 2^64-1, so the one conditional
     * subtraction in _ModAdd256 still yields canonical x (qsb_add_boundary keeps the
     * normalisation for the other case). */
    _ModSub256(t,l,c); qsb_packed_raw_mul(x1,sum,t); qsb_add_boundary(x1,a); _ModAdd256(x1,x1,a);
    _ModSub256(t,m,c); qsb_packed_raw_mul(x2,sum,t); qsb_add_boundary(x2,a); _ModAdd256(x2,x2,a);
#elif !QSB_PARITY_SUM
    _ModSub256(t,l,c); qsb_recovery_mul(x1,sum,t); _ModAdd256(x1,x1,a);
    _ModSub256(t,m,c); qsb_recovery_mul(x2,sum,t); _ModAdd256(x2,x2,a);
#endif
#if QSB_PARITY_SUM
    /* P9: r_i = x_i - a is the canonical product before "+a" (re-derived here with one
     * subtraction-free identity: x_i - a == sum*(l or m - c)), so a - x_i == -r_i and
     * s1 = l*(a-x1) == -(l*r1), s2 = m*(a-x2) == -(m*r2). See qsb_sum_parity. */
#if QSB_FIN_RAWS
    QSB_FIN_SUB(t,l,c); QSB_FIN_RAW_MUL(s,sum,t);
#if QSB_PARITY_WINDOW
    const uint32_t parity_u=qsb_parity_product_window(l,s,b,1u);
#else
    qsb_packed_raw_mul(u,l,s);
#endif
#if QSB_FIN_MLATE
    QSB_FIN_SUB(m,sum,l);
#endif
#if QSB_XOUT_LAZY
    QSB_FIN_ADDL(x1,s,a);
#else
    qsb_add_boundary(s,a); QSB_FIN_ADD(x1,s,a);
#endif
#if QSB_FIN_CHORD
    /* QSB_FIN_CHORD: x1 + x2 = 2a + sum*((l - c) + (m - c)) = 2a + (sum - c)^2 - c^2, since
     * sum == l + m. So s = x2 - a == (sum - c)^2 + E - x1 with E = a - c^2 (pin_chord_e), and
     * with v = x1 + E that is (sum - c)^2 + v - 2*x1: the chain's fused square
     * _ModSqrAddSub2(out, a, e, q) = a^2 + e - 2q (45 IMAD.WIDE, one reduction) in place of the
     * general product (73 IMAD.WIDE). _ModSqrAddSub2 takes any 256-bit representatives and
     * returns a congruent value in [0,2^256), the raw product's convention, so s is the same
     * residue; the parity window is congruence invariant, and s + a leaves [0,p) only in the
     * 2^-223 window QSB_XOUT_LAZY already accepts. */
    {
        const uint64_t e[4]={pin_chord_e[0],pin_chord_e[1],pin_chord_e[2],pin_chord_e[3]};
#if QSB_FIN_CHORD == 2
        /* variant 2: plain square (45 IMAD.WIDE, [0,2^256) like _ModMultCore), lazy add of E,
         * borrow-corrected subtraction of x1 (its +p correction underflows only for x1 >= p,
         * the same 2^-223 window). */
        QSB_FIN_SUB(t,sum,c); _ModSqr(u,t); QSB_FIN_ADDL(v,u,e); QSB_FIN_SUB(s,v,x1);
#else
        QSB_FIN_ADDL(v,x1,e); QSB_FIN_SUB(t,sum,c); _ModSqrAddSub2(s,t,v,x1);
#endif
    }
#else
    QSB_FIN_SUB(t,m,c); QSB_FIN_RAW_MUL(s,sum,t);
#endif
#if QSB_PARITY_WINDOW
    const uint32_t parity_v=qsb_parity_product_window(m,s,b,0u);
#else
    qsb_packed_raw_mul(v,m,s);
#endif
#if QSB_XOUT_LAZY
    QSB_FIN_ADDL(x2,s,a);
#else
    qsb_add_boundary(s,a); QSB_FIN_ADD(x2,s,a);
#endif
#else
    _ModSub256(t,l,c); qsb_recovery_mul(s,sum,t); _ModAdd256(x1,s,a);
#if QSB_PARITY_WINDOW
    const uint32_t parity_u=qsb_parity_product_window(l,s,b,1u);
#else
    qsb_packed_raw_mul(u,l,s);
#endif
    _ModSub256(t,m,c); qsb_recovery_mul(s,sum,t); _ModAdd256(x2,s,a);
#if QSB_PARITY_WINDOW
    const uint32_t parity_v=qsb_parity_product_window(m,s,b,0u);
#else
    qsb_packed_raw_mul(v,m,s);
#endif
#endif
#if QSB_PARITY_WINDOW && (QSB_FIN_BAL2 & 2)
    return qsb_fin_prefix_bytes(parity_u,parity_v);
#elif QSB_PARITY_WINDOW
    return parity_u|(parity_v<<1);
#else
    return qsb_sum_parity(u,b,1u)|(qsb_sum_parity(v,b,0u)<<1);
#endif
}
#else
    _ModSub256(t,a,x1); qsb_packed_raw_mul(s,l,t); qsb_parity_boundary(s,b);
    uint32_t parity=qsb_difference_parity(s,b);
    _ModSub256(t,a,x2); qsb_packed_raw_mul(s,m,t); qsb_parity_boundary(s,b);
    return parity|(qsb_difference_parity(b,s)<<1);
}
#endif
